//! Sequence-recall memory benchmark for the `wave_driven` engine (test-only). Asks whether the engine
//! can *memorize* a branching sequence set — reproduce deterministic continuations, match fork
//! marginals as calibrated readout mass, and resolve a prefix family that only a 3-token memory can
//! answer. Self-contained (its own 9-class readout; `wave_driven_bench`'s 2-class path is untouched).
//! Spec: docs/superpowers/specs/2026-07-15-wave-driven-sequence-memory-design.md

#[cfg(test)]
mod tests {
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{key, mix, TopologyLevel};
    use crate::wave_driven::training::{Edge, EligParams};
    use std::sync::{Arc, Mutex};

    /// Vocabulary size. Tokens {1,2,3,4,5,6,7,8,16} → ids 0..8.
    const V: usize = 9;
    /// Sequences are 4 tokens: a prefix of 1..=3, then the target.
    const MAX_PREFIX: usize = 3;
    /// Hash purpose tag for trial sampling (distinct stream from CUE_P / P_DFA).
    const P_SEQ: u64 = 0x5E9;
    /// Hash purpose tag for the fixed random DFA feedback weights.
    const P_DFA: u64 = 61;
    /// Hash purpose tag for token site codes. Distinct from `wave_driven_bench`'s `CUE_P` (0xC0E) —
    /// this is a different predicate over a different site set, and a different task.
    const CUE_P: u64 = 0xC0F;

    /// Operating point — **selected from the Phase A1 measurement**, not assumed. `density 1` ≈ 32
    /// of 256 sites per token; `r3/c48` ≈ 6 synapses per L1 neuron.
    ///
    /// A1 (FF, 4-set, adapt_bump 3, rate_reg 5, 3 seeds, trained to peak):
    ///
    /// | cell | fan-in | fidelity w/m | family worst | σ | dead |
    /// |---|---|---|---|---|---|
    /// | **d1/c48** | 6.0 | **0.818 / 0.880** | **1.000** | 1.094 | 0 |
    /// | d2/c40 | 10.0 | 0.844 / 0.878 | 0.500 | 0.825 | 0 |
    /// | d2/c48 | 12.0 | 0.798 / 0.800 | 0.500 | 0.938 | 0 |
    /// | d2/c32 | 8.0 | 0.591 / 0.706 | 1.000 | 0.464 | 0 |
    /// | d2/c16 | 4.0 | 0.136 (chance) | 0.000 | 0.006 | 3 |
    ///
    /// d1/c48 wins on fidelity *and* is the only high-fidelity cell with family accuracy 1.000 on
    /// every seed — d2/c40 and d2/c48 each have a seed that drops to the 0.500 Markov-2 ceiling,
    /// i.e. fails the memory test. It also uses half the input drive (32 sites).
    ///
    /// **Count is the lever, not radius**: r2/r3/r4 at c24 give σ 0.127/0.097/0.129, all with 2 dead
    /// layers — radius is noise, as AGENTS.md says. c16 is dead from L3 up at both densities.
    ///
    /// **`rate_reg` cannot rescue a *fully* dead layer.** c16 stays at chance (0.136) with 3 dead
    /// layers even after training: eligibility accrues only on *target* fire, so a silent layer has
    /// `e = 0` and `shadow += -lr·signal·0` is identically zero. AGENTS.md's "conclusive liveness
    /// rescue" has an unstated precondition — the layer must be weakly firing, not silent.
    ///
    /// The coincidence floor is ~6-8 synapses/neuron, not the ~2 the spec estimated: untrained
    /// ternary weights are random ±1/0, so of 8 synapses only ~2.7 are +1 and ~2.7 are −1 and they
    /// largely cancel.
    const OP_DENSITY: u32 = 1;
    const OP_UR: u32 = 3;
    const OP_UC: u32 = 48;
    /// Trial ceiling. Set from Phase A1: the largest measured `peak_at` across every live cell was
    /// 2600 (most peak 300-1400), so ~2× that. The plan's 12000 was ~5× over-provisioned.
    const OP_MAX_TRIALS: usize = 6000;

    /// The six sequences as token ids. Sets are nested: set 4 = SEQS[..4], set 5 = SEQS[..5], etc.
    /// ids: 0→"1" 1→"2" 2→"3" 3→"4" 4→"5" 5→"6" 6→"7" 7→"8" 8→"16"
    ///
    /// S5 and S6 deliberately extend the same `2→3` collision S4 introduced, so growing the set
    /// deepens the memory test rather than only adding capacity: the `[·,2,3]` family goes 2→3→4-way
    /// and the Markov-2 ceiling falls 50%→33%→25% while true memory stays at 100%.
    const SEQS: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // 1→2→3→4
        [0, 1, 3, 7], // 1→2→4→8
        [0, 3, 7, 8], // 1→4→8→16
        [1, 1, 2, 4], // 2→2→3→5
        [2, 1, 2, 5], // 3→2→3→6
        [3, 1, 2, 6], // 4→2→3→7
    ];

    /// Trial generator: sequence uniform over the set, prefix length uniform in 1..=MAX_PREFIX,
    /// target = the next token. Deterministic in `trial`; matches the harness convention
    /// `Fn(task_seed, trial) -> (prefix, target)`.
    ///
    /// Uniform sequence sampling is what produces the target conditionals for free: conditioned on
    /// prefix `[1]`, the sequence is uniform over {S1,S2,S3}, giving {2: 2/3, 4: 1/3}.
    fn seq_task(set_size: usize) -> impl Fn(u64, usize) -> (Vec<usize>, usize) {
        move |task_seed, trial| {
            let s = (mix(key(task_seed, trial as u32, 0, 0, P_SEQ)) % set_size as u64) as usize;
            let n = (mix(key(task_seed, trial as u32, 0, 1, P_SEQ)) % MAX_PREFIX as u64) as usize + 1;
            (SEQS[s][..n].to_vec(), SEQS[s][n])
        }
    }

    /// Every distinct prefix of length 1..=MAX_PREFIX in the set, in deterministic discovery order.
    fn prefixes(set_size: usize) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::new();
        for s in 0..set_size {
            for n in 1..=MAX_PREFIX {
                let p = SEQS[s][..n].to_vec();
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Closed-form P(next | prefix) over the V tokens, under uniform sampling of the set.
    fn conditional(set_size: usize, prefix: &[usize]) -> Vec<f32> {
        let mut counts = vec![0f32; V];
        let mut total = 0f32;
        for s in 0..set_size {
            if SEQS[s][..prefix.len()] == *prefix {
                counts[SEQS[s][prefix.len()]] += 1.0;
                total += 1.0;
            }
        }
        counts.iter().map(|c| c / total).collect()
    }

    /// The `[·,2,3]` disambiguation family: the length-3 prefixes sharing the (2,3) suffix and
    /// differing only in the first token. A Markov-2 model cannot separate these; only a 3-token
    /// memory can. Token ids: "2" = 1, "3" = 2.
    fn family(set_size: usize) -> Vec<Vec<usize>> {
        prefixes(set_size).into_iter().filter(|p| p.len() == MAX_PREFIX && p[1] == 1 && p[2] == 2).collect()
    }

    /// The last `min(k, len)` tokens of a prefix — a Markov-k model's context.
    fn ctx_of(p: &[usize], k: usize) -> Vec<usize> {
        let n = p.len().min(k);
        p[p.len() - n..].to_vec()
    }

    /// Sampling weight of a prefix: proportional to the number of sequences carrying it (the
    /// uniform-over-prefix-length factor is constant and cancels).
    fn prefix_weight(set_size: usize, p: &[usize]) -> f32 {
        (0..set_size).filter(|&s| SEQS[s][..p.len()] == *p).count() as f32
    }

    /// Expected accuracy of a Markov-k model on `targets`, under the model's own predictive
    /// distribution (so ties need no tie-breaking rule: a model spreading mass over k options scores
    /// exactly 1/k). The model is fit in closed form from the set: group every prefix by its
    /// length-k context, then average their conditionals weighted by sampling frequency.
    fn markov_k_accuracy(set_size: usize, k: usize, targets: &[Vec<usize>]) -> f32 {
        let all = prefixes(set_size);
        let mut acc = 0f32;
        for p in targets {
            let ctx = ctx_of(p, k);
            let mut counts = vec![0f32; V];
            let mut total = 0f32;
            for q in &all {
                if ctx_of(q, k) == ctx {
                    let w = prefix_weight(set_size, q);
                    let cond = conditional(set_size, q);
                    for t in 0..V {
                        counts[t] += w * cond[t];
                    }
                    total += w;
                }
            }
            let qdist: Vec<f32> = counts.iter().map(|c| c / total).collect();
            let truth = conditional(set_size, p);
            acc += (0..V).map(|t| truth[t] * qdist[t]).sum::<f32>();
        }
        acc / targets.len() as f32
    }

    /// A token's L0 site code: a fixed random subset of the grid, `density`/8 of all sites
    /// (density 1 ≈ 32 sites, density 2 ≈ 64 of 256).
    ///
    /// Population-coded, not place-coded, for two reasons. (1) Arithmetic: the engine's leak floor
    /// (`wave.rs`, `d.max(1)`) drains ≥1 per wave, so a single +1 synapse nets zero and a lone site
    /// can never fire anything — ≥2 coincident synapses is a precondition for any activity, and
    /// `sample_distinct_cells` caps a source at one synapse per target. (2) Science: random codes
    /// share no exploitable structure, so the net cannot interpolate geometrically instead of
    /// remembering. See the spec's Design analysis §1-2.
    fn token_sites(task_seed: u64, size: u32, token: usize, density: u32) -> Vec<u32> {
        let ls = size * size;
        (0..ls).filter(|&loc| (mix(key(task_seed, loc, token as i32, 0, CUE_P)) & 7) < density as u64).collect()
    }

    /// Max-subtract softmax over V logits.
    fn softmax_n(z: &[f32]) -> Vec<f32> {
        let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = z.iter().map(|x| (x - m).exp()).collect();
        let s = e.iter().sum::<f32>().max(1e-30);
        e.iter().map(|x| x / s).collect()
    }

    /// Per-class score: the dot product of each class's readout weights with the spike counts.
    fn score_n(w: &[Vec<f32>], a: &[f32]) -> Vec<f32> {
        w.iter().map(|wc| wc.iter().zip(a).map(|(x, y)| x * y).sum()).collect()
    }

    /// Argmax breaking ties toward the lower index — matching the 2-class harness's `(s1 > s0)`
    /// rule, which predicts class 0 on a tie. At init every weight is 0 and every score ties at 0.0,
    /// so the tie-break is reachable, not theoretical.
    fn argmax_first(z: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, &x) in z.iter().enumerate() {
            if x > z[best] {
                best = i;
            }
        }
        best
    }

    /// Total variation distance, ½·Σ|p−q|. Bounded and legible: 0 is perfect, 0.5 means a 50/50 fork
    /// collapsed onto one branch. For a one-hot `p`, `1 − TV` is exactly `q[target]`.
    fn total_variation(p: &[f32], q: &[f32]) -> f32 {
        0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f32>()
    }

    /// Plain feed-forward stack, `layers` deep. L0 is forced to a transducer by the engine
    /// (threshold i16::MAX, adapt_bump 0), so 5 layers is 4 computing layers. The top layer is read
    /// directly; its level-1 topology points past the stack and is inert, and `entries[top]` is empty
    /// so DFA never targets it. Membrane-only eligibility (`elig_beta 0`).
    fn make_ff_seq(seed: u64, size: u32, layers: usize, uc: u32, ur: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: ur, count: uc }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay,
        };
        let mut net = Network::new(Config { seed, size, layers: vec![lc; layers] });
        net.enable_training();
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 });
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: uc as usize, radius: ur }] })
            .collect();
        (net, entries)
    }

    /// Backward-fed side-car, 5 layers: L0→L1(+1); L1→L3(+2, skipping past the scratchpad);
    /// L2 self(0)+→L3(+1); L3→L2(−1)+→L4(+1); L4 read. The recurrent layer is isolated from the
    /// forward path. Spike-ψ εᵃ (`elig_beta 0.4`) is what makes the recurrence trainable, and it
    /// requires a non-zero `adapt_bump` to couple to.
    fn make_sidecar_seq(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let mk = |topology| LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay,
        };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]),
            mk(vec![TopologyLevel { level: 2, radius: ur, count: uc }]),
            mk(vec![TopologyLevel { level: 0, radius: r, count: n }, TopologyLevel { level: 1, radius: r, count: n }]),
            mk(vec![TopologyLevel { level: -1, radius: r, count: n }, TopologyLevel { level: 1, radius: ur, count: uc }]),
            mk(vec![]),
        ];
        let mut net = Network::new(Config { seed, size, layers });
        net.set_elig_params(EligParams { rec_tau: 20.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 });
        net.enable_training();
        let entries = vec![
            vec![Edge { level: 1, count: uc as usize, radius: ur }],
            vec![Edge { level: 2, count: uc as usize, radius: ur }],
            vec![Edge { level: 0, count: n as usize, radius: r }, Edge { level: 1, count: n as usize, radius: r }],
            vec![Edge { level: -1, count: n as usize, radius: r }, Edge { level: 1, count: uc as usize, radius: ur }],
            vec![],
        ];
        (net, entries)
    }

    /// Per-layer firing rate (%/neuron/wave) over a window, plus σ (mean consecutive-layer spike
    /// ratio). The dynamics diagnostic AGENTS.md requires: σ + profile is what separates *dynamics
    /// collapse* from *credit starvation* when a result disappoints.
    fn rate_profile_seq(net: &mut Network, size: u32, task_seed: u64, token: usize, density: u32, warmup: usize, waves: usize) -> (Vec<f64>, f64) {
        let l = net.layer_count();
        let counts = Arc::new(Mutex::new(vec![0u64; l]));
        for z in 0..l {
            let c = counts.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] += f.len() as u64));
        }
        net.reset_state();
        let sites = token_sites(task_seed, size, token, density);
        for _ in 0..warmup {
            net.wave(&sites);
        }
        counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
        for _ in 0..waves {
            net.wave(&sites);
        }
        net.clear_listeners();
        let counts = std::mem::take(&mut *counts.lock().unwrap());
        let denom = ((size as u64) * (size as u64) * waves as u64) as f64;
        let pct: Vec<f64> = counts.iter().map(|&s| (s as f64 / denom * 1000.0).round() / 10.0).collect();
        let mut ratios = Vec::new();
        for z in 1..l - 1 {
            if counts[z] > 0 {
                ratios.push(counts[z + 1] as f64 / counts[z] as f64);
            }
        }
        let sigma = if ratios.is_empty() { 0.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
        (pct, sigma)
    }

    /// Task/readout configuration. `rate_reg`/`rate_target` are bench-side: the engine only exposes
    /// `layer_spike_count`, and the learning rule lives here per AGENTS.md.
    struct SeqCfg {
        size: u32,
        density: u32,
        present: usize,
        delay: usize,
        read: usize,
        readout_lr: f32,
        hidden_lr: f32,
        rate_reg: f32,
        rate_target: f32,
    }

    /// The spec's operating point. A 3-token prefix spans 26 waves, leaving ~66% of the first token's
    /// adaptation trace alive at read time (ρ = 1 − 2⁻⁶ ≈ 0.984/wave at `adapt_decay 6`).
    fn seq_cfg() -> SeqCfg {
        SeqCfg { size: 16, density: OP_DENSITY, present: 6, delay: 4, read: 8, readout_lr: 0.02, hidden_lr: 0.004, rate_reg: 5.0, rate_target: 0.1 }
    }

    /// Present a prefix token by token, then read. Each token fires its site code for `present`
    /// waves, with `delay` empty waves between tokens; `act` integrates the top layer's spikes over
    /// the trailing `read` window only.
    fn run_seq_trial(net: &mut Network, cfg: &SeqCfg, prefix: &[usize], task_seed: u64) -> (Vec<f32>, usize) {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let top_spikes: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let ts = top_spikes.clone();
        net.on_layer(top, Box::new(move |_w, fired: &[u32]| ts.lock().unwrap().push(fired.to_vec())));
        net.reset_state();
        let mut ttot = 0usize;
        for (pos, &token) in prefix.iter().enumerate() {
            if pos > 0 {
                for _ in 0..cfg.delay {
                    net.wave(&[]);
                    ttot += 1;
                }
            }
            let sites = token_sites(task_seed, cfg.size, token, cfg.density);
            for _ in 0..cfg.present {
                net.wave(&sites);
                ttot += 1;
            }
        }
        let read_start = top_spikes.lock().unwrap().len();
        for _ in 0..cfg.read {
            net.wave(&[]);
            ttot += 1;
        }
        net.clear_listeners();
        let mut act = vec![0f32; ls];
        for wv in top_spikes.lock().unwrap().iter().skip(read_start) {
            for &loc in wv {
                act[loc as usize] += 1.0;
            }
        }
        (act, ttot)
    }

    /// Fixed random ±1 DFA feedback weight for (neuron, class).
    fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
        if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
    }

    /// Learning signal per computational layer/neuron: DFA task feedback plus the `rate_reg`
    /// homeostatic term, generalized over V classes.
    ///
    /// `signal[tz][j] = Σ_{c<V} b·err[c] + rate_reg·(rate_j − rate_target)`, with `b = w[c][j]` at
    /// the top layer (symmetric readout feedback) and a fixed random ±1 hash below.
    ///
    /// The rate term is a homeostatic controller: `dfa_update` applies
    /// `shadow += -lr · signal[tz][j] · e` with `e ≥ 0`, so a neuron above `rate_target` gets a
    /// positive signal and has its incoming weights pushed *down*, and one below gets them pushed
    /// *up*. It rescues liveness in deep stacks (no firing ⇒ no eligibility ⇒ no credit) at the cost
    /// of homogenizing rates and eroding the class signal — which is why `rate_reg` is a Phase B
    /// axis here rather than an inherited constant. Note it acts **per neuron**: a layer whose mean
    /// sits on `rate_target` is still pushed toward uniformity *across* its neurons, which is
    /// exactly the prefix-specific structure this task needs. See the spec's Design analysis §4.
    ///
    /// `rate` is normalized by `ttot`, which varies with prefix length in this task.
    fn build_signal_n(net: &Network, w: &[Vec<f32>], err: &[f32], seed: u64, ttot: usize, cfg: &SeqCfg) -> Vec<Vec<f32>> {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let denom = ttot.max(1) as f32;
        let mut signal = vec![vec![0f32; ls]; l];
        for tz in 1..l {
            let sc = net.layer_spike_count(tz);
            for j in 0..ls {
                let task_sig: f32 = (0..V)
                    .map(|c| {
                        let b = if tz == top { w[c][j] } else { dfa_weight(seed, (tz * ls + j) as u32, c) };
                        b * err[c]
                    })
                    .sum();
                let rate = sc[j] as f32 / denom;
                signal[tz][j] = task_sig + cfg.rate_reg * (rate - cfg.rate_target);
            }
        }
        signal
    }

    /// Metrics at one evaluation point. All computed from an exact enumeration — no sampling.
    struct SeqMetrics {
        /// Checkpoint scalar: mean over all prefixes of `1 − TV(truth, softmax)`. For a
        /// deterministic prefix this is exactly the softmax mass on the target; for a fork it is the
        /// calibration score. One number scoring both kinds uniformly.
        fidelity: f32,
        /// Top-1 accuracy over the deterministic (single-continuation) prefixes.
        det_acc: f32,
        /// Top-1 accuracy over the `[·,2,3]` family — compare against `markov_k_accuracy(·, 2, ·)`.
        family_acc: f32,
        /// Per-fork total variation between the true conditional and the readout softmax.
        fork_tv: Vec<(Vec<usize>, f32)>,
    }

    /// Enumerate every prefix once and score it. The engine is deterministic and resets per trial,
    /// so each prefix yields exactly one score vector — the evaluation is exact, with no sampling
    /// and no variance, at ~9 runs instead of a sampled holdout's 200.
    ///
    /// There is deliberately no holdout: the input universe *is* these 9/12/15 prefixes, and a
    /// held-out prefix's answer is arbitrary rather than derivable. The Markov-2 control does a
    /// holdout's actual job — ruling out the answer-from-recent-context shortcut. See the spec §6.
    fn eval_all_prefixes(net: &mut Network, cfg: &SeqCfg, w: &[Vec<f32>], set_size: usize, task_seed: u64) -> SeqMetrics {
        let fam = family(set_size);
        let (mut fid_sum, mut n_all) = (0f32, 0f32);
        let (mut det_hit, mut det_n) = (0f32, 0f32);
        let (mut fam_hit, mut fam_n) = (0f32, 0f32);
        let mut fork_tv = Vec::new();

        for p in prefixes(set_size) {
            let truth = conditional(set_size, &p);
            let (act, _) = run_seq_trial(net, cfg, &p, task_seed);
            let s = score_n(w, &act);
            let q = softmax_n(&s);
            let tv = total_variation(&truth, &q);

            fid_sum += 1.0 - tv;
            n_all += 1.0;

            let live: Vec<usize> = (0..V).filter(|&t| truth[t] > 0.0).collect();
            if live.len() == 1 {
                let hit = if argmax_first(&s) == live[0] { 1.0 } else { 0.0 };
                det_hit += hit;
                det_n += 1.0;
                if fam.contains(&p) {
                    fam_hit += hit;
                    fam_n += 1.0;
                }
            } else {
                fork_tv.push((p.clone(), tv));
            }
        }

        SeqMetrics {
            fidelity: fid_sum / n_all,
            det_acc: if det_n > 0.0 { det_hit / det_n } else { 0.0 },
            family_acc: if fam_n > 0.0 { fam_hit / fam_n } else { 0.0 },
            fork_tv,
        }
    }

    /// Train online, evaluating exactly every `eval_every` trials and returning the **peak**
    /// metrics plus the trial count they were reached at.
    ///
    /// Best-checkpointing is not optional: `rate_reg` over-trains into a non-monotonic accuracy
    /// collapse (recorded as transient at ~4 layers, permanent by ~12 — we sit at 5). Compare at the
    /// peak of the duration sweep, never at a fixed final trial count.
    ///
    /// The usual objection — that reporting the max over evals selects on the reported set — is much
    /// weaker here than in the sampled-holdout battery: this evaluation has *no sampling noise*, so
    /// the max reads the true peak of a deterministic curve rather than the top of the noise.
    fn train_and_eval_best_seq(
        net: &mut Network,
        entries: &[Vec<Edge>],
        seed: u64,
        task_seed: u64,
        cfg: &SeqCfg,
        set_size: usize,
        eval_every: usize,
        patience: usize,
        max_trials: usize,
    ) -> (SeqMetrics, usize) {
        let ls = (cfg.size * cfg.size) as usize;
        let task = seq_task(set_size);
        let mut w = vec![vec![0f32; ls]; V];
        let mut best = eval_all_prefixes(net, cfg, &w, set_size, task_seed);
        let (mut best_at, mut stale, mut t) = (0usize, 0usize, 0usize);

        while t < max_trials {
            let stop = (t + eval_every).min(max_trials);
            while t < stop {
                let (prefix, target) = task(task_seed, t);
                let (act, ttot) = run_seq_trial(net, cfg, &prefix, task_seed);
                let p = softmax_n(&score_n(&w, &act));
                let err: Vec<f32> = (0..V).map(|c| p[c] - if c == target { 1.0 } else { 0.0 }).collect();
                for c in 0..V {
                    for j in 0..ls {
                        w[c][j] -= cfg.readout_lr * err[c] * act[j];
                    }
                }
                if cfg.hidden_lr != 0.0 {
                    let signal = build_signal_n(net, &w, &err, seed, ttot, cfg);
                    net.dfa_update(entries, &signal, cfg.hidden_lr);
                }
                t += 1;
            }
            let m = eval_all_prefixes(net, cfg, &w, set_size, task_seed);
            if m.fidelity > best.fidelity {
                best = m;
                best_at = t;
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }
        (best, best_at)
    }

    #[test]
    fn eval_all_prefixes_scores_an_untrained_net() {
        let cfg = seq_cfg();
        let (mut net, _) = make_ff_seq(6, 16, 5, OP_UC, OP_UR, 3, 6);
        let w = vec![vec![0f32; 256]; V];

        // At init every weight is 0, so every score ties at 0.0 and softmax is uniform over V.
        let m = eval_all_prefixes(&mut net, &cfg, &w, 4, 7);

        // Fidelity of a uniform predictor: for a one-hot truth, 1 - TV = 1/V. The 7 deterministic
        // prefixes each score 1/9; the 2 forks score higher (uniform is closer to a spread truth
        // than to a one-hot). So fidelity sits just above 1/V.
        assert!(m.fidelity > 1.0 / V as f32 - 0.01, "uniform predictor scores ~1/V, got {}", m.fidelity);
        assert!(m.fidelity < 0.35, "an untrained net must not look good, got {}", m.fidelity);

        // argmax_first breaks the init tie toward token 0, which no deterministic prefix targets.
        assert_eq!(m.det_acc, 0.0, "untrained argmax ties to token 0; no prefix targets it");

        // Exactly the two forks are reported, both badly calibrated at init.
        assert_eq!(m.fork_tv.len(), 2);
        assert!(m.fork_tv.iter().all(|(_, tv)| *tv > 0.3), "uniform is far from both forks");

        // Determinism: the whole evaluation is a pure function.
        let m2 = eval_all_prefixes(&mut net, &cfg, &w, 4, 7);
        assert_eq!(m.fidelity, m2.fidelity);
        assert_eq!(m.det_acc, m2.det_acc);
    }

    #[test]
    fn seq_training_moves_off_chance() {
        // A cheap smoke test: does the loop learn *anything*? Not a result — the real runs are the
        // #[ignore]d experiments. Kept small enough for `cargo test`.
        let cfg = seq_cfg();
        let (mut net, entries) = make_ff_seq(9, 16, 5, OP_UC, OP_UR, 3, 6);
        let (best, best_at) = train_and_eval_best_seq(&mut net, &entries, 9, 7, &cfg, 4, 100, 4, 1200);

        // Chance fidelity for a uniform predictor is ~1/V ≈ 0.111.
        assert!(best.fidelity > 0.15, "training must beat a uniform predictor, got {}", best.fidelity);
        assert!(best_at > 0, "the peak must land somewhere");
        assert_eq!(best.fork_tv.len(), 2, "both forks reported at the peak");

        // Every reported metric is a well-formed fraction. Whether family_acc clears the Markov-2
        // ceiling is Phase B's question, not this smoke test's — 1200 trials is far too few to ask.
        assert!((0.0..=1.0).contains(&best.det_acc), "det_acc is a fraction, got {}", best.det_acc);
        assert!((0.0..=1.0).contains(&best.family_acc), "family_acc is a fraction, got {}", best.family_acc);
        assert!(best.fork_tv.iter().all(|(_, tv)| (0.0..=1.0).contains(tv)), "TV is bounded, got {:?}", best.fork_tv);
    }

    #[test]
    fn run_seq_trial_is_deterministic_and_alive() {
        let cfg = seq_cfg();
        let (mut net, _) = make_ff_seq(3, 16, 5, OP_UC, OP_UR, 3, 6);

        // A trial produces activity — otherwise `act` is all zeros and nothing can ever train.
        let (act, ttot) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);
        assert_eq!(act.len(), 256);
        assert!(act.iter().any(|&x| x > 0.0), "top layer must spike in the read window");

        // ttot is the full timeline: 3 tokens × present + 2 gaps × delay + read.
        assert_eq!(ttot, 3 * cfg.present + 2 * cfg.delay + cfg.read);

        // Prefix length changes the timeline — unlike the fixed-length battery, `ttot` varies here,
        // which is why build_signal's rate normalisation is load-bearing for this task.
        let (_, t1) = run_seq_trial(&mut net, &cfg, &[0], 7);
        assert_eq!(t1, cfg.present + cfg.read);
        assert!(t1 < ttot, "shorter prefix, shorter trial");

        // Determinism: the engine resets per trial, so a prefix yields exactly one score vector.
        // This is what makes the exact 9-prefix evaluation possible.
        let (a1, _) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);
        let (a2, _) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);
        assert_eq!(a1, a2, "same prefix → same activity, every time");

        // Different prefixes are distinguishable at the top layer, or the readout has nothing to use.
        let (b, _) = run_seq_trial(&mut net, &cfg, &[1, 1, 2], 7);
        assert_ne!(a1, b, "[1,2,3] and [2,2,3] must differ at the top layer");
    }

    #[test]
    fn build_signal_n_shape_and_rate_term() {
        let mut cfg = seq_cfg();
        let (mut net, _) = make_ff_seq(4, 16, 5, OP_UC, OP_UR, 3, 6);
        let ls = 256usize;
        let w = vec![vec![0f32; ls]; V];
        let err = vec![0f32; V];
        let (_, ttot) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);

        // Shape: one row per layer, L0's row left zeroed (DFA never targets the transducer).
        let sig = build_signal_n(&net, &w, &err, 5, ttot, &cfg);
        assert_eq!(sig.len(), 5);
        assert!(sig.iter().all(|r| r.len() == ls));
        assert!(sig[0].iter().all(|&x| x == 0.0), "L0 is never a DFA target");

        // With zero task error the signal is exactly the rate term, for every neuron.
        let sc = net.layer_spike_count(1);
        for j in 0..ls {
            let rate = sc[j] as f32 / ttot as f32;
            let expect = cfg.rate_reg * (rate - cfg.rate_target);
            assert!((sig[1][j] - expect).abs() < 1e-5, "signal is exactly the rate term at neuron {j}");
        }

        // The sign convention is the mechanism rate_reg works by: `dfa_update` applies
        // `-lr · signal · e` with `e ≥ 0`, so a below-target neuron must get a *negative* signal
        // (incoming weights pushed up, firing more) and an above-target one a *positive* signal.
        //
        // Note both sides must exist. At the c32 operating point no L1 neuron is ever silent — every
        // one fires at least once across a 34-wave trial at ~17.5% mean rate — so this deliberately
        // does not look for a zero-rate neuron; rate_reg still homogenizes *across* the spread.
        let below = (0..ls).find(|&j| (sc[j] as f32 / ttot as f32) < cfg.rate_target);
        let above = (0..ls).find(|&j| (sc[j] as f32 / ttot as f32) > cfg.rate_target);
        assert!(below.is_some() && above.is_some(), "the operating point straddles rate_target (not degenerate)");
        assert!(sig[1][below.unwrap()] < 0.0, "below-target neuron gets a negative (excitatory) signal");
        assert!(sig[1][above.unwrap()] > 0.0, "above-target neuron gets a positive (suppressing) signal");

        // At rate_reg 0 the term drops out entirely and rate_target becomes inert — the Phase B
        // rate_reg{0} cells depend on this.
        cfg.rate_reg = 0.0;
        let sig0 = build_signal_n(&net, &w, &err, 5, ttot, &cfg);
        assert!(sig0[1].iter().all(|&x| x == 0.0), "zero error + zero rate_reg → zero signal");

        // dfa_weight is a deterministic ±1 hash.
        assert_eq!(dfa_weight(5, 17, 3), dfa_weight(5, 17, 3));
        assert!(dfa_weight(5, 17, 3).abs() == 1.0);
    }

    #[test]
    fn builders_produce_live_five_layer_nets() {
        // FF: 5 layers (L0 transducer + 4 computing) at the operating point, r3/c32 density 2.
        let (mut net, entries) = make_ff_seq(1, 16, 5, OP_UC, OP_UR, 3, 6);
        assert_eq!(net.layer_count(), 5);
        assert_eq!(entries.len(), 5);
        assert!(entries[4].is_empty(), "top layer has no outgoing DFA edges");

        // EVERY computational layer must fire, untrained. A dead layer accrues no eligibility, so it
        // never trains — and a dead *top* layer means `act` is all zeros and the readout SGD
        // multiplies by zero, so nothing trains at all.
        //
        // This is not a formality: measured untrained, r3/c16 gives [22.3, 8.1, 1.3, 0.0, 0.0]
        // (σ 0.069) — dead from L3 up, unable to train. r3/c32 gives [22.3, 17.5, 13.8, 11.0, 10.2]
        // (σ 0.83). `c` is the lever, not radius (r4/c16 measures 0.070, indistinguishable from
        // r3/c16); c48 goes supercritical (σ 1.03, activity growing with depth).
        let (pct, sigma) = rate_profile_seq(&mut net, 16, 7, 0, OP_DENSITY, 8, 24);
        assert_eq!(pct.len(), 5);
        for z in 1..5 {
            assert!(pct[z] > 0.0, "L{z} must fire untrained, profile {pct:?} σ {sigma:.3}");
        }
        assert!(sigma > 0.4 && sigma < 1.2, "σ must be near-critical, got {sigma:.3}, profile {pct:?}");

        // Side-car: 5 layers, L2 the isolated recurrent scratchpad, L4 the read layer.
        let (mut net2, entries2) = make_sidecar_seq(1, 16, OP_UC, OP_UR, 8, 4, 3, 6);
        assert_eq!(net2.layer_count(), 5);
        assert_eq!(entries2.len(), 5);
        assert!(entries2[4].is_empty(), "side-car read layer has no outgoing DFA edges");
        assert_eq!(entries2[2].len(), 2, "L2 carries a self-loop and a forward edge");

        // The side-car must also reach its read layer, or the comparison is vacuous. Its forward
        // path is L0→L1→L3→L4 (L2 is the isolated scratchpad), so L2 may legitimately differ.
        let (pct2, _) = rate_profile_seq(&mut net2, 16, 7, 0, OP_DENSITY, 8, 24);
        assert!(pct2[4] > 0.0, "side-car read layer must fire untrained, profile {pct2:?}");

        // Determinism: same seed and config → identical dynamics.
        let (mut a, _) = make_ff_seq(2, 16, 5, OP_UC, OP_UR, 3, 6);
        let (mut b, _) = make_ff_seq(2, 16, 5, OP_UC, OP_UR, 3, 6);
        assert_eq!(rate_profile_seq(&mut a, 16, 7, 0, OP_DENSITY, 8, 24), rate_profile_seq(&mut b, 16, 7, 0, OP_DENSITY, 8, 24));
    }

    #[test]
    fn token_sites_density_and_determinism() {
        // Density 1/8 ≈ 32 sites, 2/8 ≈ 64 sites of 256 (binomial, so a generous band).
        for token in 0..V {
            let n32 = token_sites(11, 16, token, 1).len();
            let n64 = token_sites(11, 16, token, 2).len();
            assert!((18..=50).contains(&n32), "density 1 ≈ 32 sites, got {n32} for token {token}");
            assert!((44..=86).contains(&n64), "density 2 ≈ 64 sites, got {n64} for token {token}");
        }

        // Determinism: a pure function of its arguments.
        assert_eq!(token_sites(11, 16, 3, 2), token_sites(11, 16, 3, 2));

        // Distinct tokens get distinct codes (random population codes overlap, but not wholly).
        let a = token_sites(11, 16, 0, 2);
        let b = token_sites(11, 16, 1, 2);
        assert_ne!(a, b, "distinct tokens must not share a code");
        let shared = a.iter().filter(|s| b.contains(s)).count();
        assert!(shared < a.len() * 3 / 4, "token codes must stay separable, shared {shared} of {}", a.len());

        // Sites are in range and strictly ascending (the filter preserves order).
        let s = token_sites(11, 16, 5, 2);
        assert!(s.windows(2).all(|w| w[0] < w[1]), "sites ascend");
        assert!(s.iter().all(|&loc| loc < 256), "sites are in range");
    }

    #[test]
    fn readout_primitives_correct() {
        // softmax_n is a distribution, and max-subtraction keeps it finite on large inputs.
        let p = softmax_n(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0], "monotone in the logit");
        let big = softmax_n(&[1000.0, 1000.0]);
        assert!((big[0] - 0.5).abs() < 1e-6, "no overflow: {big:?}");

        // Uniform logits → uniform distribution.
        let u = softmax_n(&vec![0.0; V]);
        assert!(u.iter().all(|&x| (x - 1.0 / V as f32).abs() < 1e-6));

        // score_n is a per-class dot product.
        let w = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        assert_eq!(score_n(&w, &[3.0, 5.0]), vec![3.0, 10.0]);

        // argmax_first breaks ties toward the lower index (matching the 2-class `s1 > s0` rule).
        assert_eq!(argmax_first(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(argmax_first(&[1.0, 5.0, 5.0]), 1);

        // total_variation: 0 when identical; 1/2 when a 50/50 fork collapses onto one branch;
        // 1/3 when [1]'s 67/33 collapses. Both figures are quoted in the spec's metrics.
        let fork = vec![0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(total_variation(&fork, &fork).abs() < 1e-6);
        let collapsed = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((total_variation(&fork, &collapsed) - 0.5).abs() < 1e-6);
        let skew = conditional(4, &[0]); // {2: 2/3, 4: 1/3}
        let onto2 = {
            let mut v = vec![0.0; V];
            v[1] = 1.0;
            v
        };
        assert!((total_variation(&skew, &onto2) - 1.0 / 3.0).abs() < 1e-5);

        // For a deterministic (one-hot) truth, 1 - TV is exactly the mass on the target. This is the
        // identity the checkpoint scalar rests on.
        let truth = conditional(4, &[0, 1, 2]); // one-hot on token 3
        let q = softmax_n(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((1.0 - total_variation(&truth, &q) - q[3]).abs() < 1e-5);
    }

    /// Phase A1 — forward fan-in × density, 3 seeds, under training. Run manually in --release:
    ///   cargo test --release --lib seq_phase_a1_forward_sweep -- --ignored --nocapture
    ///
    /// The untrained probe (recorded on `OP_UC`) already established the σ landscape: count is the
    /// lever, c16 is dead from L3 up, c32 is near-critical, c48 is supercritical. This sweep asks
    /// the question that probe *cannot*: what survives **training**, where weights reshape and σ
    /// moves. c16 is kept in as the documented dead control.
    ///
    /// Selects on dynamics (σ near 1, no dead layer) with fidelity secondary — dynamics are the
    /// low-variance, seed-robust signal.
    ///
    /// **Also reports `peak_at`** — Phase B's `max_trials` must be set from that measurement, not
    /// from the 12000 ceiling guessed here.
    ///
    /// Density and `c` are not interchangeable: at equal input drive (4 synapses/neuron), d1/c32
    /// measured σ 0.473 while d2/c16 measured 0.069, because `c` alone sets *hidden*-layer drive and
    /// that job dominates. This sweep should show the two separating under training too.
    #[test]
    #[ignore]
    fn seq_phase_a1_forward_sweep() {
        const SEEDS: [u64; 3] = [1, 2, 3];

        println!("\n=== Phase A1: forward fan-in × density (FF, 4-set, adapt_bump 3, rate_reg 5) ===");
        println!("chance fidelity ~{:.3}; markov-2 family ceiling {:.3}", 1.0 / V as f32, markov_k_accuracy(4, 2, &family(4)));
        println!("\n-- count sweep at r3 (c ≤ (2r+1)² = 49) --");
        for density in [1u32, 2u32] {
            for uc in [16u32, 24, 32, 40, 48] {
                run_a1_cell(density, 3, uc, &SEEDS);
            }
        }
        // Radius at fixed count. c24 is the largest count r2 can carry: c ≤ (2·2+1)² = 25.
        println!("\n-- radius sweep at c24, density 2 --");
        for ur in [2u32, 3, 4] {
            run_a1_cell(2, ur, 24, &SEEDS);
        }
        println!("\n=== select on σ + profile; set OP_MAX_TRIALS from peak_at ===\n");
    }

    /// One Phase A1 cell: train 3 seeds at (density, ur, uc) and report worst+mean with dynamics.
    fn run_a1_cell(density: u32, ur: u32, uc: u32, seeds: &[u64]) {
        let (mut fid, mut fam, mut det, mut sig, mut peaks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut prof0 = Vec::new();
        for &seed in seeds {
            let mut cfg = seq_cfg();
            cfg.density = density;
            let (mut net, entries) = make_ff_seq(seed, 16, 5, uc, ur, 3, 6);
            let (best, best_at) = train_and_eval_best_seq(&mut net, &entries, seed, 7, &cfg, 4, 100, 10, 12000);
            let (pct, sigma) = rate_profile_seq(&mut net, 16, 7, 0, density, 8, 24);
            fid.push(best.fidelity);
            fam.push(best.family_acc);
            det.push(best.det_acc);
            sig.push(sigma);
            peaks.push(best_at);
            if prof0.is_empty() {
                prof0 = pct;
            }
        }
        let n = seeds.len() as f32;
        let wf = fid.iter().copied().fold(f32::INFINITY, f32::min);
        let wd = det.iter().copied().fold(f32::INFINITY, f32::min);
        let wfam = fam.iter().copied().fold(f32::INFINITY, f32::min);
        let ms = sig.iter().sum::<f64>() / seeds.len() as f64;
        let dead = prof0[1..].iter().filter(|&&x| x == 0.0).count();
        println!(
            "d{density}(~{:2} sites) r{ur}/c{uc:2} fan-in {:5.2} | fid w {wf:.3} m {:.3} | det w {wd:.3} m {:.3} | fam w {wfam:.3} m {:.3} | σ {ms:.3} dead {dead} | {prof0:?} | peak {peaks:?}",
            256 * density / 8,
            (256 * density / 8) as f32 * uc as f32 / 256.0,
            fid.iter().sum::<f32>() / n,
            det.iter().sum::<f32>() / n,
            fam.iter().sum::<f32>() / n,
        );
    }

    /// Phase A2 — recurrent fan-in, swept **separately** from the forward path (AGENTS.md), at the
    /// Phase A1 operating point. Run manually in --release:
    ///   cargo test --release --lib seq_phase_a2_recurrent_sweep -- --ignored --nocapture
    ///
    /// The recorded sweet spot is n=8 (σ collapses by n≥24), so this confirms it at *this* task's
    /// operating point rather than searching openly.
    #[test]
    #[ignore]
    fn seq_phase_a2_recurrent_sweep() {
        const SEEDS: [u64; 3] = [1, 2, 3];
        const NR: [(u32, u32); 3] = [(8, 3), (8, 4), (16, 4)];

        println!("\n=== Phase A2: recurrent fan-in (side-car, 4-set, adapt_bump 3, rate_reg 5) ===");
        println!("operating point: density {OP_DENSITY}/8, r{OP_UR}/c{OP_UC}, max_trials {OP_MAX_TRIALS}");
        println!("markov-2 family ceiling {:.3}", markov_k_accuracy(4, 2, &family(4)));

        for (n, r) in NR {
            let (mut fid, mut fam, mut det, mut sig, mut peaks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut prof0 = Vec::new();
            for seed in SEEDS {
                let cfg = seq_cfg();
                let (mut net, entries) = make_sidecar_seq(seed, 16, OP_UC, OP_UR, n, r, 3, 6);
                let (best, best_at) = train_and_eval_best_seq(&mut net, &entries, seed, 7, &cfg, 4, 100, 10, OP_MAX_TRIALS);
                let (pct, sigma) = rate_profile_seq(&mut net, 16, 7, 0, OP_DENSITY, 8, 24);
                fid.push(best.fidelity);
                fam.push(best.family_acc);
                det.push(best.det_acc);
                sig.push(sigma);
                peaks.push(best_at);
                if prof0.is_empty() {
                    prof0 = pct;
                }
            }
            let n_seeds = SEEDS.len() as f32;
            let wf = fid.iter().copied().fold(f32::INFINITY, f32::min);
            let wd = det.iter().copied().fold(f32::INFINITY, f32::min);
            let wfam = fam.iter().copied().fold(f32::INFINITY, f32::min);
            let ms = sig.iter().sum::<f64>() / SEEDS.len() as f64;
            println!(
                "rec n{n:2}/r{r} | fid w {wf:.3} m {:.3} | det w {wd:.3} m {:.3} | fam w {wfam:.3} m {:.3} | σ {ms:.3} | {prof0:?} | peak {peaks:?}",
                fid.iter().sum::<f32>() / n_seeds,
                det.iter().sum::<f32>() / n_seeds,
                fam.iter().sum::<f32>() / n_seeds,
            );
        }
        println!("=== end Phase A2 ===\n");
    }

    #[test]
    fn seq_conditionals_correct() {
        // Prefix enumeration: 9 / 12 / 15 distinct prefixes.
        assert_eq!(prefixes(4).len(), 9);
        assert_eq!(prefixes(5).len(), 12);
        assert_eq!(prefixes(6).len(), 15);

        // The two forks, in closed form.
        let c1 = conditional(4, &[0]);
        assert!((c1[1] - 2.0 / 3.0).abs() < 1e-6, "[1] → 2 with p=2/3");
        assert!((c1[3] - 1.0 / 3.0).abs() < 1e-6, "[1] → 4 with p=1/3");
        let c12 = conditional(4, &[0, 1]);
        assert!((c12[2] - 0.5).abs() < 1e-6, "[1,2] → 3 with p=1/2");
        assert!((c12[3] - 0.5).abs() < 1e-6, "[1,2] → 4 with p=1/2");

        // Deterministic prefixes, including the disambiguation pair.
        assert_eq!(conditional(4, &[0, 1, 2])[3], 1.0, "[1,2,3] → 4");
        assert_eq!(conditional(4, &[1, 1, 2])[4], 1.0, "[2,2,3] → 5");
        assert_eq!(conditional(4, &[0, 3, 7])[8], 1.0, "[1,4,8] → 16");

        // Every conditional is a distribution.
        for set_size in [4, 5, 6] {
            for p in prefixes(set_size) {
                let s: f32 = conditional(set_size, &p).iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "conditional sums to 1 for {p:?}");
            }
        }

        // The family grows 2 → 3 → 4-way, all sharing the (2,3) suffix.
        assert_eq!(family(4).len(), 2);
        assert_eq!(family(5).len(), 3);
        assert_eq!(family(6).len(), 4);

        // Markov-2 ceiling on the family is exactly 1/k — the control the whole task rests on.
        for (set_size, k) in [(4, 2.0), (5, 3.0), (6, 4.0)] {
            let m2 = markov_k_accuracy(set_size, 2, &family(set_size));
            assert!((m2 - 1.0 / k).abs() < 1e-6, "Markov-2 family ceiling is 1/{k} for set {set_size}, got {m2}");
        }

        // Markov-1 is never better than Markov-2 → Markov-2 is the discriminating control.
        for set_size in [4, 5, 6] {
            let m1 = markov_k_accuracy(set_size, 1, &family(set_size));
            let m2 = markov_k_accuracy(set_size, 2, &family(set_size));
            assert!(m1 <= m2 + 1e-6, "Markov-1 ({m1}) must not beat Markov-2 ({m2}) for set {set_size}");
        }

        // Markov-3 sees the whole prefix, so it is full memory: 100% on the family.
        for set_size in [4, 5, 6] {
            let m3 = markov_k_accuracy(set_size, 3, &family(set_size));
            assert!((m3 - 1.0).abs() < 1e-6, "Markov-3 == full memory for set {set_size}, got {m3}");
        }

        // seq_task only ever emits a real (prefix, target) pair from the set.
        let task = seq_task(4);
        for t in 0..500 {
            let (prefix, target) = task(7, t);
            assert!((1..=MAX_PREFIX).contains(&prefix.len()), "prefix length in 1..=3, got {prefix:?}");
            let cond = conditional(4, &prefix);
            assert!(cond[target] > 0.0, "target {target} must be reachable from {prefix:?}");
        }
    }
}
