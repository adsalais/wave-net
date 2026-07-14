//! FF training harness for the `wave_driven` engine (test-only). Ports the `wave_bitnet_bench` shape
//! onto the event-driven engine's online eligibility: no per-wave pot/eff recording, per-neuron rate
//! read from the engine's `spike_count`, and `Network::dfa_update` applied from the accumulated
//! eligibility. Proves the activity-scaled trainer learns end-to-end (FF single-cue above chance).

#[cfg(test)]
mod tests {
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{key, mix, TopologyLevel};
    use crate::wave_driven::training::{Edge, EligParams};
    use std::sync::{Arc, Mutex};

    const CUE_P: u64 = 0xC0E;
    const P_DFA: u64 = 61;

    fn cue_sites(task_seed: u64, size: u32, class: usize) -> Vec<u32> {
        let ls = (size * size) as u32;
        (0..ls).filter(|&loc| mix(key(task_seed, loc, class as i32, 0, CUE_P)) & 3 == 0).collect()
    }

    fn softmax2(z0: f32, z1: f32) -> (f32, f32) {
        let m = z0.max(z1);
        let (e0, e1) = ((z0 - m).exp(), (z1 - m).exp());
        let s = (e0 + e1).max(1e-30);
        (e0 / s, e1 / s)
    }

    fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
        if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
    }

    /// Run one trial; returns (top-layer read-window spike counts `act`, total waves `ttot`).
    fn run_trial(net: &mut Network, size: u32, classes: &[usize], task_seed: u64, present: usize, delay: usize, read: usize) -> (Vec<f32>, usize) {
        let l = net.layer_count();
        let ls = (size * size) as usize;
        let top = l - 1;
        let top_spikes: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let ts = top_spikes.clone();
        net.on_layer(top, Box::new(move |_w, fired: &[u32]| ts.lock().unwrap().push(fired.to_vec())));
        net.reset_state();
        let mut ttot = 0usize;
        for (pos, &class) in classes.iter().enumerate() {
            if pos > 0 {
                for _ in 0..delay {
                    net.wave(&[]);
                    ttot += 1;
                }
            }
            for _ in 0..present {
                let sites = cue_sites(task_seed, size, class);
                net.wave(&sites);
                ttot += 1;
            }
        }
        let read_start = top_spikes.lock().unwrap().len();
        for _ in 0..read {
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

    struct TaskCfg {
        size: u32,
        present: usize,
        delay: usize,
        read: usize,
        holdout: usize,
        readout_lr: f32,
        hidden_lr: f32,
        rate_reg: f32,
        rate_target: f32,
    }

    /// Learning signal per computational layer/neuron: DFA feedback + rate_reg (rate from the engine).
    fn build_signal(net: &Network, w: &[Vec<f32>], err: &[f32], seed: u64, ttot: usize, cfg: &TaskCfg) -> Vec<Vec<f32>> {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let denom = ttot.max(1) as f32;
        let mut signal = vec![vec![0f32; ls]; l];
        for tz in 1..l {
            let sc = net.layer_spike_count(tz);
            for j in 0..ls {
                let task_sig: f32 = (0..2)
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

    fn train_and_eval_best(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize), eval_every: usize, patience: usize, max_trials: usize) -> (u64, usize) {
        const EVAL_OFFSET: usize = 10_000_000;
        let ls = (cfg.size * cfg.size) as usize;
        let mut w = vec![vec![0f32; ls]; 2];
        let score = |w: &[Vec<f32>], a: &[f32]| -> (f32, f32) {
            (w[0].iter().zip(a).map(|(x, y)| x * y).sum(), w[1].iter().zip(a).map(|(x, y)| x * y).sum())
        };
        let (mut best, mut best_at, mut stale, mut t) = (0u64, 0usize, 0usize, 0usize);
        while t < max_trials {
            let stop = (t + eval_every).min(max_trials);
            while t < stop {
                let (classes, label) = task(task_seed, t);
                let (act, ttot) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                let (p0, p1) = softmax2(s0, s1);
                let err = [p0 - if label == 0 { 1.0 } else { 0.0 }, p1 - if label == 1 { 1.0 } else { 0.0 }];
                for c in 0..2 {
                    for j in 0..ls {
                        w[c][j] -= cfg.readout_lr * err[c] * act[j];
                    }
                }
                if cfg.hidden_lr != 0.0 {
                    let signal = build_signal(net, &w, &err, seed, ttot, cfg);
                    net.dfa_update(entries, &signal, cfg.hidden_lr);
                }
                t += 1;
            }
            let mut correct = 0usize;
            for i in 0..cfg.holdout {
                let (classes, label) = task(task_seed, EVAL_OFFSET + i);
                let (act, _) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                if ((s1 > s0) as usize) == label {
                    correct += 1;
                }
            }
            let acc = (correct as u64 * 1000) / cfg.holdout as u64;
            if acc > best {
                best = acc;
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

    fn single_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let c = (mix(key(seed, t as u32, 0, 0, 71)) & 1) as usize;
        (vec![c], c)
    }

    fn make_ff(seed: u64, size: u32, layers: usize, up_count: u32, up_radius: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
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
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: up_count as usize, radius: up_radius }] })
            .collect();
        (net, entries)
    }

    fn ff_cfg() -> TaskCfg {
        TaskCfg { size: 16, present: 6, delay: 4, read: 6, holdout: 200, readout_lr: 0.02, hidden_lr: 0.004, rate_reg: 5.0, rate_target: 0.1 }
    }

    // Backward-fed side-car (ported from benches/throughput_bitnet.rs make_sidecar):
    // L0→L1(+1); L1→L3(+2 skip); L2 self(0)+→L3(+1); L3→L2(−1)+→L4(+1); L4 read.
    fn make_sidecar(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let mk = |topology| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump, adapt_decay };
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

    /// N-bit sequential parity: N deterministic cue bits, label = their XOR. (N=2 is temporal XOR.)
    fn task_parity(seed: u64, t: usize, n: usize) -> (Vec<usize>, usize) {
        let bits: Vec<usize> = (0..n).map(|i| (mix(key(seed, t as u32, 0, i as u32, 51)) & 1) as usize).collect();
        let label = bits.iter().fold(0usize, |a, &b| a ^ b);
        (bits, label)
    }

    /// `[a, distractor, b]` where the middle is a label-irrelevant cue (class 2); label = a XOR b (ignore D).
    fn task_distractor(seed: u64, trial: usize) -> (Vec<usize>, usize) {
        let a = (mix(key(seed, trial as u32, 0, 0, 51)) & 1) as usize;
        let b = (mix(key(seed, trial as u32, 0, 0, 53)) & 1) as usize;
        (vec![a, 2, b], a ^ b)
    }

    /// `n_ops` set(class 0)/reset(class 1) ops; label = final state (set -> on 1, reset -> off 0).
    fn task_flipflop(seed: u64, trial: usize, n_ops: usize) -> (Vec<usize>, usize) {
        let ops: Vec<usize> = (0..n_ops).map(|i| (mix(key(seed, trial as u32, 0, i as u32, 57)) & 1) as usize).collect();
        let last = *ops.last().unwrap();
        (ops, if last == 0 { 1 } else { 0 })
    }

    #[test]
    fn task_labels_correct() {
        for trial in 0..25 {
            let (bits, label) = task_parity(42, trial, 4);
            assert_eq!(bits.len(), 4);
            assert!(bits.iter().all(|&b| b <= 1));
            assert_eq!(label, bits.iter().fold(0, |a, &b| a ^ b), "parity label is the XOR of the bits");

            let (classes, dlabel) = task_distractor(42, trial);
            assert_eq!(classes.len(), 3);
            assert_eq!(classes[1], 2, "middle cue is the class-2 distractor");
            assert!(classes[0] <= 1 && classes[2] <= 1);
            assert_eq!(dlabel, classes[0] ^ classes[2], "distractor label ignores the middle cue");

            let (ops, flabel) = task_flipflop(42, trial, 4);
            assert_eq!(ops.len(), 4);
            assert!(ops.iter().all(|&o| o <= 1));
            let last = *ops.last().unwrap();
            assert_eq!(flabel, if last == 0 { 1 } else { 0 }, "flip-flop label is the final state");
        }
    }

    /// Per-layer firing rate (%/neuron/wave) over a window, and a coarse σ (mean consecutive-layer spike
    /// ratio) — the dynamics diagnostic that separates σ-supercritical collapse from credit collapse.
    fn rate_profile(net: &mut Network, size: u32, task_seed: u64, class: usize, warmup: usize, waves: usize) -> (Vec<f64>, f64) {
        let l = net.layer_count();
        let counts = Arc::new(Mutex::new(vec![0u64; l]));
        for z in 0..l {
            let c = counts.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] += f.len() as u64));
        }
        net.reset_state();
        let sites = cue_sites(task_seed, size, class);
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

    #[test]
    fn wave_driven_ff_trains_above_chance() {
        // 4-layer FF, size 16, generous fan-in; single-cue 2-class must beat chance. Integration proof
        // that online eligibility + shadow update + repack + readout trains end-to-end.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 5, 6);
        let cfg = ff_cfg();
        let (best, _at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 3, 1000);
        assert!(best > 600, "wave_driven FF should train above chance: {best}");
    }

    #[test]
    #[ignore] // smoke: run manually in --release
    fn wave_driven_ff_depth8_smoke() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        eprintln!("== wave_driven FF depth-8 pure ternary smoke (r4/c48, adapt=5, online elig) ==");
        let mut bests = Vec::new();
        for &s in &seeds {
            let (mut net, entries) = make_ff(s, 32, 8, 48, 4, 5, 6);
            let mut cfg = ff_cfg();
            cfg.size = 32;
            cfg.present = 8;
            cfg.read = 8;
            cfg.holdout = 300;
            let (best, at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, single_task, 300, 3, 3000);
            eprintln!("seed {s:#x}: best {best}@{at}");
            bests.push(best);
        }
        let worst = *bests.iter().min().unwrap();
        eprintln!("worst {worst} (target ~1000)");
        assert!(worst >= 900, "pure ternary FF depth-8 should hold ~1000 (worst {worst})");
    }

    #[test]
    #[ignore] // experiment: online vs offline-eligibility trainer throughput (run in --release)
    fn wave_driven_training_throughput() {
        use crate::wave_driven::training::dense_eligibility;
        use std::time::Instant;
        for &size in &[16u32, 32u32] {
            let seed = 0xC0FFEEu64;
            let (mut net, entries) = make_ff(seed, size, 4, 32, 3, 5, 6);
            let mut cfg = ff_cfg();
            cfg.size = size;
            let trials = 200usize;

            // online: run trials, accrual happens inside wave(); dfa_update reads engine elig
            let w = vec![vec![0f32; (size * size) as usize]; 2];
            let t0 = Instant::now();
            for t in 0..trials {
                let (classes, _label) = single_task(seed, t);
                let (_act, ttot) = run_trial(&mut net, size, &classes, seed, cfg.present, cfg.delay, cfg.read);
                let err = [0.1f32, -0.1f32];
                let signal = build_signal(&net, &w, &err, seed, ttot, &cfg);
                net.dfa_update(&entries, &signal, cfg.hidden_lr);
            }
            let online = t0.elapsed().as_secs_f64();

            // offline: record fired every wave, compute dense_eligibility per trial (the size-bound path)
            let (mut net2, entries2) = make_ff(seed, size, 4, 32, 3, 5, 6);
            let t1 = Instant::now();
            for t in 0..trials {
                let (classes, _label) = single_task(seed, t);
                let l = net2.layer_count();
                let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
                for z in 0..l {
                    let r = rec.clone();
                    net2.on_layer(z, Box::new(move |_w, f: &[u32]| r.lock().unwrap()[z].push(f.to_vec())));
                }
                net2.reset_state();
                for _ in 0..(cfg.present + cfg.read) {
                    let sites = cue_sites(seed, size, classes[0]);
                    net2.wave(&sites);
                }
                net2.clear_listeners();
                let fired = rec.lock().unwrap().clone();
                let _e = dense_eligibility(&net2, &entries2, &fired, &EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 });
            }
            let offline = t1.elapsed().as_secs_f64();
            eprintln!("size {size}: online {:.1} trials/s, offline-eligibility {:.1} trials/s ({:.1}x)", trials as f64 / online, trials as f64 / offline, offline / online);
        }
    }

    #[test]
    #[ignore] // experiment: does spike-ψ εᵃ unlock recurrence? (run in --release; minutes)
    fn wave_driven_sidecar_vs_ff() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678];
        eprintln!("== wave_driven side-car vs FF (spike-ψ εᵃ, β=0.4) — width × rec_count, σ-instrumented ==");
        for &parity_n in &[2usize, 4usize] {
            eprintln!("-- parity N={parity_n} ({}) --", if parity_n == 2 { "temporal XOR" } else { "parity-4" });
            let task = move |s: u64, t: usize| task_parity(s, t, parity_n);
            for &size in &[16u32, 32u32] {
                // FF baseline (β=0, membrane) at this width
                let mut ff_bests = Vec::new();
                for &s in &seeds {
                    let (mut net, entries) = make_ff(s, size, 5, 32, 3, 5, 6);
                    let mut cfg = ff_cfg();
                    cfg.size = size;
                    cfg.present = 6;
                    cfg.delay = 8;
                    cfg.read = 8;
                    cfg.holdout = 200;
                    let (best, _at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, &task, 300, 3, 2400);
                    ff_bests.push(best);
                }
                eprintln!("  size {size} FF          : worst {} mean {}", ff_bests.iter().min().unwrap(), ff_bests.iter().sum::<u64>() / ff_bests.len() as u64);

                // side-car at a sweep of rec_count (into/beyond the historical bump-ψ cliff ~12)
                for &rec_count in &[8u32, 16u32, 24u32] {
                    let mut sc_bests = Vec::new();
                    let mut sigmas = Vec::new();
                    for &s in &seeds {
                        let (mut net, entries) = make_sidecar(s, size, 32, 3, rec_count, 4, 5, 6);
                        let mut cfg = ff_cfg();
                        cfg.size = size;
                        cfg.present = 6;
                        cfg.delay = 8;
                        cfg.read = 8;
                        cfg.holdout = 200;
                        let (best, _at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, &task, 300, 3, 2400);
                        sc_bests.push(best);
                        let (_pct, sigma) = rate_profile(&mut net, size, s, 0, 16, 64);
                        sigmas.push(sigma);
                    }
                    let profile = {
                        let (mut net, _e) = make_sidecar(seeds[0], size, 32, 3, rec_count, 4, 5, 6);
                        rate_profile(&mut net, size, seeds[0], 0, 16, 64).0
                    };
                    eprintln!(
                        "  size {size} side rec{rec_count:>2}: worst {} mean {} | σ≈{:.2} | rate% {:?}",
                        sc_bests.iter().min().unwrap(),
                        sc_bests.iter().sum::<u64>() / sc_bests.len() as u64,
                        sigmas.iter().sum::<f64>() / sigmas.len() as f64,
                        profile
                    );
                }
            }
        }
        // No hard assertion: this is the research readout. Interpret per the spec's convergence ladder
        // (σ super-critical ⇒ dynamics collapse, density too high; healthy σ + poor acc ⇒ credit-limited).
    }

    #[test]
    #[ignore] // validation: multi-seed, all-benchmark, matched-FF-baseline recurrence confirmation (--release, ~tens of min)
    fn wave_driven_recurrence_confirmation() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let size = 32u32;
        let rec = 8u32;
        eprintln!("== wave_driven recurrence confirmation — size {size}, rec {rec}, spike-ψ εᵃ (β=0.4), {} seeds ==", seeds.len());
        eprintln!("   {:<15} | FF w/mean | side-car w/mean | σ", "task");

        struct B {
            name: &'static str,
            present: usize,
            delay: usize,
            read: usize,
            task: Box<dyn Fn(u64, usize) -> (Vec<usize>, usize)>,
        }
        let benches: Vec<B> = vec![
            B { name: "temporal-XOR", present: 6, delay: 8, read: 8, task: Box::new(|s, t| task_parity(s, t, 2)) },
            B { name: "parity-4", present: 6, delay: 8, read: 8, task: Box::new(|s, t| task_parity(s, t, 4)) },
            B { name: "distractor-XOR", present: 6, delay: 20, read: 8, task: Box::new(|s, t| task_distractor(s, t)) },
            B { name: "flip-flop", present: 6, delay: 12, read: 8, task: Box::new(|s, t| task_flipflop(s, t, 4)) },
        ];

        let mut sidecar_xor_worst = 0u64;
        let mut beats = 0usize;
        for b in &benches {
            let mkcfg = || {
                let mut c = ff_cfg();
                c.size = size;
                c.present = b.present;
                c.delay = b.delay;
                c.read = b.read;
                c.holdout = 200;
                c
            };
            let (mut ff_bests, mut sc_bests, mut sigmas) = (Vec::new(), Vec::new(), Vec::new());
            for &s in &seeds {
                // FF baseline (5-layer, membrane β=0), best-checkpointed to its ceiling
                let (mut ffn, fe) = make_ff(s, size, 5, 32, 3, 5, 6);
                let (fb, _) = train_and_eval_best(&mut ffn, &fe, s, s, &mkcfg(), b.task.as_ref(), 300, 3, 2400);
                ff_bests.push(fb);
                // side-car (rec 8, spike-ψ εᵃ β=0.4), same budget
                let (mut scn, se) = make_sidecar(s, size, 32, 3, rec, 4, 5, 6);
                let (sb, _) = train_and_eval_best(&mut scn, &se, s, s, &mkcfg(), b.task.as_ref(), 300, 3, 2400);
                sc_bests.push(sb);
                let (_p, sigma) = rate_profile(&mut scn, size, s, 0, 16, 64);
                sigmas.push(sigma);
            }
            let ffw = *ff_bests.iter().min().unwrap();
            let ffm = ff_bests.iter().sum::<u64>() / ff_bests.len() as u64;
            let scw = *sc_bests.iter().min().unwrap();
            let scm = sc_bests.iter().sum::<u64>() / sc_bests.len() as u64;
            let sig = sigmas.iter().sum::<f64>() / sigmas.len() as f64;
            eprintln!("   {:<15} | {ffw:>4}/{ffm:<4} | {scw:>4}/{scm:<4}      | {sig:.2}", b.name);
            if scw >= ffw {
                beats += 1;
            }
            if b.name == "temporal-XOR" {
                sidecar_xor_worst = scw;
            }
        }
        eprintln!("== recurrence beats FF (worst-seed) on {beats}/{} tasks ==", benches.len());
        // plumbing sanity gate only: the side-car demonstrably solves temporal XOR
        assert!(sidecar_xor_worst > 700, "side-car should clear chance on temporal XOR (worst {sidecar_xor_worst}); harness broken?");
    }

    #[test]
    #[ignore] // diagnostic: does a denser/tighter recurrent scratchpad revive activity-starved flip-flop? (--release)
    fn wave_driven_flipflop_rec_sweep() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let size = 32u32;
        let mkcfg = || {
            let mut c = ff_cfg();
            c.size = size;
            c.present = 6;
            c.delay = 12;
            c.read = 8;
            c.holdout = 200;
            c
        };
        let task = |s: u64, t: usize| task_flipflop(s, t, 4);
        eprintln!("== flip-flop side-car recurrent-topology sweep (size {size}, spike-ψ εᵃ β=0.4, {} seeds) ==", seeds.len());

        // FF baseline (no recurrence; unchanged by the recurrent params)
        let ff: Vec<u64> = seeds
            .iter()
            .map(|&s| {
                let (mut n, e) = make_ff(s, size, 5, 32, 3, 5, 6);
                train_and_eval_best(&mut n, &e, s, s, &mkcfg(), &task, 300, 3, 2400).0
            })
            .collect();
        eprintln!("  FF (no recurrence): worst {} mean {}", ff.iter().min().unwrap(), ff.iter().sum::<u64>() / ff.len() as u64);

        // side-car recurrent scratchpad (count n, radius r): current r4/c8 vs the two requested denser/tighter layouts
        for &(n, r) in &[(8u32, 4u32), (16u32, 3u32), (32u32, 3u32)] {
            let mut bests = Vec::new();
            let mut sigmas = Vec::new();
            for &s in &seeds {
                let (mut net, e) = make_sidecar(s, size, 32, 3, n, r, 5, 6);
                bests.push(train_and_eval_best(&mut net, &e, s, s, &mkcfg(), &task, 300, 3, 2400).0);
                sigmas.push(rate_profile(&mut net, size, s, 0, 16, 64).1);
            }
            let profile = {
                let (mut net, _e) = make_sidecar(seeds[0], size, 32, 3, n, r, 5, 6);
                rate_profile(&mut net, size, seeds[0], 0, 16, 64).0
            };
            eprintln!(
                "  side-car rec r{r}/c{n:<2}: worst {} mean {} | σ≈{:.2} | rate% {:?}",
                bests.iter().min().unwrap(),
                bests.iter().sum::<u64>() / bests.len() as u64,
                sigmas.iter().sum::<f64>() / sigmas.len() as f64,
                profile
            );
        }
    }

    #[test]
    #[ignore] // diagnostic: flip-flop is adaptation-quenched, not density-starved — sweep the side-car's adaptation (--release)
    fn wave_driven_flipflop_adapt_sweep() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let size = 32u32;
        let mkcfg = || {
            let mut c = ff_cfg();
            c.size = size;
            c.present = 6;
            c.delay = 12;
            c.read = 8;
            c.holdout = 200;
            c
        };
        let task = |s: u64, t: usize| task_flipflop(s, t, 4);
        eprintln!("== flip-flop adaptation sweep (side-car r4/c8, size {size}, spike-ψ εᵃ β=0.4, {} seeds) ==", seeds.len());
        eprintln!("  reference: FF baseline 525/561; side-car bump5/decay6 = 670/810 (L2 silent, σ≈0.05)");

        // Lower adapt_bump relaxes the per-fire quench; lower adapt_decay speeds relaxation between ops.
        // Both keep the recurrent scratchpad (L2) alive over the long flip-flop op sequence.
        for &(bump, decay) in &[(5i16, 6u8), (3, 6), (2, 6), (1, 6), (3, 4), (2, 4)] {
            let mut bests = Vec::new();
            let mut sigmas = Vec::new();
            for &s in &seeds {
                let (mut net, e) = make_sidecar(s, size, 32, 3, 8, 4, bump, decay);
                bests.push(train_and_eval_best(&mut net, &e, s, s, &mkcfg(), &task, 300, 3, 2400).0);
                sigmas.push(rate_profile(&mut net, size, s, 0, 16, 64).1);
            }
            let profile = {
                let (mut net, _e) = make_sidecar(seeds[0], size, 32, 3, 8, 4, bump, decay);
                rate_profile(&mut net, size, seeds[0], 0, 16, 64).0
            };
            eprintln!(
                "  bump{bump} decay{decay}: worst {} mean {} | σ≈{:.2} | rate% {:?}",
                bests.iter().min().unwrap(),
                bests.iter().sum::<u64>() / bests.len() as u64,
                sigmas.iter().sum::<f64>() / sigmas.len() as f64,
                profile
            );
        }
    }
}
