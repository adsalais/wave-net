//! `rsnn` — training on the LIVE `wave_net` engine. Stage 1: a trained linear readout on the FULL
//! reservoir state (all computational layers) — a reliable Liquid State Machine. Stage 2 (Task 4): e-prop
//! on the hidden weights. Evaluated held-out + multi-seed — the bar the threshold-only approach failed.

use crate::bench::eprop::pick_class;
use crate::bench::store_recall::{cue_realization, probe_pattern};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::{Config, LayerConfig};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, local_of, map_range24, mix, wrap, xy_of, TopologyLevel, P_TARGET};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct RsnnConfig {
    pub seed: u64,
    pub task_seed: u64,
    pub size: u32,
    pub layers: usize,
    pub k: usize,
    pub present_waves: usize,
    pub delay: usize,
    pub read_waves: usize,
    pub base_q16: u32,
    pub keep_q16: u32,
    pub noise_q16: u32,
    pub probe_q16: u32,
    pub up_count: u32,
    pub up_radius: u32,
    pub trials: usize,
    pub readout_lr: f32,
    pub hidden_lr: f32,
    pub rec_count: u32,   // level-0 lateral synapses per neuron (0 = feed-forward, no recurrence)
    pub rec_radius: u32,  // level-0 recurrence radius
    pub rec_tau: f32,     // presynaptic-trace decay time constant (waves) for the temporal eligibility
    pub adapt_bump: i16,  // ALIF adaptation strength (0 = LIF; adaptation is a per-neuron memory)
    pub adapt_decay: u8,  // ALIF adaptation decay shift
    pub rec_init: i8,     // initial recurrent weight (0 = keep procedural ±1; >0 bootstraps self-excitation)
    pub multi_layer: bool, // train every feed-forward layer (DFA credit), not just the last
    pub back_count: u32,   // level −1/−2 backward synapses per neuron (0 = feed-forward only)
    pub back_radius: u32,  // backward recurrence radius
    pub subthreshold_psi: bool, // temporal-eligibility ψ from decide-time potential, not just spikes
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
    pub rate_reg: f32,             // firing-rate regularization coefficient c_reg (0.0 = off)
    pub rate_target_permille: u32, // target per-neuron firing rate r_target, permille (e.g. 100 = 10%)
    pub xor_layers: usize,         // depth of the sequence-task stack: forward layers under a recurrent top (2 = the original L0→L1)
    pub rec_stab: f32,             // per-LAYER recurrent stabilizer (uniform bias toward r_target on recurrent levels; class-preserving, unlike per-neuron rate_reg). 0 = off
    pub elig_beta: f32,      // ALIF adaptation-eligibility coupling β (0.0 = off → membrane-only, byte-identical). Active only when adapt_bump > 0.
    pub elig_bump_psi: bool, // use normalized bump pseudo-derivative ψ instead of spike/ramp post-factor (ablation: bump-ψ without the εᵃ term)
    pub elig_psi_width: f32, // half-width W of the bump pseudo-derivative ψ = γ·max(0, 1−|v−eff|/W) (default 16; θ≈1 baseline is too narrow)
    pub hidden_rec_depth: usize, // depth of the hidden-recurrent stack in engine_config_hidden_rec (default 4: L0→L1→L2→L3, recurrence on the second-from-top layer)
}

impl RsnnConfig {
    pub fn demo() -> RsnnConfig {
        let seed = 0xE9_0B_0A17;
        RsnnConfig {
            seed,
            task_seed: seed,
            size: 8,
            layers: 3,
            k: 2,
            present_waves: 6,
            delay: 4,
            read_waves: 6,
            base_q16: 18000,
            keep_q16: 60000,
            noise_q16: 1500,
            probe_q16: 20000,
            up_count: 16,
            up_radius: 3,
            trials: 1500,
            readout_lr: 0.02,
            hidden_lr: 0.004,
            rec_count: 0,
            rec_radius: 2,
            rec_tau: 4.0,
            adapt_bump: 20,
            adapt_decay: 6,
            rec_init: 0,
            multi_layer: false,
            back_count: 0,
            back_radius: 2,
            subthreshold_psi: false,
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
            rate_reg: 0.0,
            rate_target_permille: 100,
            xor_layers: 2,
            rec_stab: 0.0,
            elig_beta: 0.0,
            elig_bump_psi: false,
            elig_psi_width: PSI_WIDTH,
            hidden_rec_depth: 4,
        }
    }

    fn engine_config(&self) -> Config {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
        };
        Config { seed: self.seed, size: self.size, layers: vec![layer; self.layers] }
    }

    /// Sequence-task net: L0 input transducer → `xor_layers-2` forward layers → a **recurrent top layer**
    /// (level 0, read by the readout). `rec_count == 0` gives the top an empty topology — the feed-forward
    /// baseline. `xor_layers = 2` is the original L0→L1 (recurrent hidden = the only computational layer).
    fn engine_config_xor(&self) -> Config {
        let fwd = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        let n = self.xor_layers.max(2);
        // L0 transducer + forward layers all project +1; the top layer carries the level-0 recurrence.
        let mut layers = vec![fwd.clone(); n - 1];
        let top_topo = if self.rec_count > 0 {
            vec![TopologyLevel { level: 0, radius: self.rec_radius, count: self.rec_count }]
        } else {
            vec![]
        };
        layers.push(LayerConfig { topology: top_topo, ..fwd });
        Config { seed: self.seed, size: self.size, layers }
    }

    /// Multi-layer net with a uniform [+1, −1, −2] topology (backward levels only when back_count>0).
    /// Off-stack targets (top's +1, L0's −1/−2) are dropped by the router — harmless.
    fn engine_config_recurrent(&self) -> Config {
        let mut topo = vec![TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count }];
        if self.back_count > 0 {
            topo.push(TopologyLevel { level: -1, radius: self.back_radius, count: self.back_count });
            topo.push(TopologyLevel { level: -2, radius: self.back_radius, count: self.back_count });
        }
        let layer = LayerConfig {
            topology: topo,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        Config { seed: self.seed, size: self.size, layers: vec![layer; self.layers] }
    }

    /// The user's backward-fed recurrent side-car (4 layers): L0(input)→L1; L1 **skips** to L3 (+2);
    /// L2 is a recurrent scratchpad (0 self + +1 to L3); L3 feeds L2 back (−1) and is read. `rec_count`
    /// sizes the side-car's synapses per level; `up_count`/`up_radius` the forward/skip path.
    fn engine_config_sidecar(&self) -> Config {
        let mk = |topology| LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        let (uc, ur) = (self.up_count, self.up_radius);
        let (n, r) = (self.rec_count, self.rec_radius);
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]), // L0 input transducer → L1
            mk(vec![TopologyLevel { level: 2, radius: ur, count: uc }]), // L1 → L3 (skip L2)
            mk(vec![
                TopologyLevel { level: 0, radius: r, count: n }, // L2 self-recurse
                TopologyLevel { level: 1, radius: r, count: n }, // L2 → L3 (forward from side-car)
            ]),
            mk(vec![
                TopologyLevel { level: -1, radius: r, count: n },  // L3 → L2 (backward loop)
                TopologyLevel { level: 1, radius: ur, count: uc },  // L3 → L4 (forward to the read layer)
            ]),
            mk(vec![]), // L4 read (top) — separates recurrent computation (L3) from the readout
        ];
        Config { seed: self.seed, size: self.size, layers }
    }

    /// Deeper side-car (6 layers): forward path L0→L1→L4→L5 (L1 **skips** the side-car via +3), with the
    /// **2-layer** recurrent scratchpad L2↔L3 hanging off the **pre-read layer L4** — L4 drives it backward
    /// (L4 → L2, −2), it loops internally (L2→L3, L3→L2), and writes back forward (L3 → L4, +1). L5 is read.
    /// This mirrors the original single-layer side-car (scratchpad hangs off the read-adjacent layer, driven
    /// backward, writes forward), extended to a 2-layer loop with a longer forward skip.
    fn engine_config_sidecar_deep(&self) -> Config {
        let mk = |topology| LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        let (uc, ur) = (self.up_count, self.up_radius);
        let (n, r) = (self.rec_count, self.rec_radius);
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]), // L0 input → L1
            mk(vec![TopologyLevel { level: 3, radius: ur, count: uc }]), // L1 → L4 (forward, skip the side-car)
            mk(vec![
                TopologyLevel { level: 0, radius: r, count: n }, // L2 self-recurse
                TopologyLevel { level: 1, radius: r, count: n }, // L2 → L3
            ]),
            mk(vec![
                TopologyLevel { level: 0, radius: r, count: n },  // L3 self-recurse (holds state, like L2)
                TopologyLevel { level: 1, radius: r, count: n },  // L3 → L4 (side-car writes back to forward)
                TopologyLevel { level: -1, radius: r, count: n }, // L3 → L2 (loop back)
            ]),
            mk(vec![
                TopologyLevel { level: 1, radius: ur, count: uc }, // L4 → L5 (forward to read)
                TopologyLevel { level: -1, radius: r, count: n },  // L4 → L3 (drive the side-car top backward)
            ]),
            mk(vec![]), // L5 read (top)
        ];
        Config { seed: self.seed, size: self.size, layers }
    }

    /// Cleaner variant (4 layers): L0(input)→L1→**L2** with the **L2↔L3 loop** — L1 feeds L2 forward (+1),
    /// L2 self-recurses (0) and feeds L3 (+1), L3 feeds L2 back (−1). Read L3. The forward signal now flows
    /// *through* L2 (no skip), so L3 is driven along the main path (no dead top layer).
    fn engine_config_l2l3loop(&self) -> Config {
        let mk = |topology| LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        let (uc, ur) = (self.up_count, self.up_radius);
        let (n, r) = (self.rec_count, self.rec_radius);
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]), // L0 → L1
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]), // L1 → L2 (forward, no skip)
            mk(vec![
                TopologyLevel { level: 0, radius: r, count: n }, // L2 self-recurse
                TopologyLevel { level: 1, radius: r, count: n }, // L2 → L3
            ]),
            mk(vec![TopologyLevel { level: -1, radius: r, count: n }]), // L3 → L2 (loop); L3 is read
        ];
        Config { seed: self.seed, size: self.size, layers }
    }

    /// Clean hidden recurrent stack of `hidden_rec_depth` layers (default 4: L0→L1→L2→L3(read)), with
    /// **level-0 recurrence on the second-from-top layer** when `rec_count > 0`. The signal flows *through*
    /// that layer's recurrence; the other computational layers are plain forward. Forward path revived by
    /// `rate_reg`, the loop stabilized by `rec_stab`. (`hidden_rec_depth = 4` is byte-identical to the
    /// original L0→L1→L2→L3.)
    fn engine_config_hidden_rec(&self) -> Config {
        let mk = |topology| LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        let fwd = TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count };
        let rec = if self.rec_count > 0 {
            vec![fwd.clone(), TopologyLevel { level: 0, radius: self.rec_radius, count: self.rec_count }]
        } else {
            vec![fwd.clone()]
        };
        let d = self.hidden_rec_depth.max(3);
        // layers 0..d-2 are plain forward (+1); layer d-2 carries the recurrence; layer d-1 is the read top.
        let layers: Vec<LayerConfig> = (0..d)
            .map(|z| {
                if z == d - 1 {
                    mk(vec![]) // read top (no outgoing)
                } else if z == d - 2 {
                    mk(rec.clone()) // forward + level-0 recurrence
                } else {
                    mk(vec![fwd.clone()]) // plain forward
                }
            })
            .collect();
        Config { seed: self.seed, size: self.size, layers }
    }
}

/// Run one trial (reset → cue → delay → probe); return the FULL reservoir's per-neuron spike counts over
/// the computational layers `1..L` — the LSM feature vector (reads all reservoir neurons, so the class
/// signal is reliably present regardless of how far it propagates upward).
fn reservoir_activity(net: &mut Network, cfg: &RsnnConfig, class: usize, trial: usize) -> Vec<f32> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![0u32; l * ls]));
    for z in 1..l {
        let c = counts.clone();
        net.on_layer(
            z,
            Box::new(move |_w, fired: &[u32]| {
                let mut g = c.lock().unwrap();
                for &loc in fired {
                    g[z * ls + loc as usize] += 1;
                }
            }),
        );
    }
    net.reset_state();
    for w in 0..cfg.present_waves {
        let sites = cue_realization(cfg.task_seed, cfg.size, class, trial, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
        net.wave(&sites);
    }
    for _ in 0..cfg.delay {
        net.wave(&[]);
    }
    let probe = probe_pattern(cfg.task_seed, cfg.size, cfg.probe_q16);
    for _ in 0..cfg.read_waves {
        net.wave(&probe);
    }
    net.clear_listeners();
    let g = counts.lock().unwrap();
    g[ls..].iter().map(|&x| x as f32).collect() // skip L0 (input transducer)
}

pub(crate) fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum::<f32>().max(1e-30);
    e.iter().map(|v| v / s).collect()
}

/// Train a K×N linear readout (delta rule) on the reservoir; return held-out test accuracy permille.
pub fn train_readout(cfg: &RsnnConfig) -> u64 {
    let mut net = Network::new(cfg.engine_config());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let n = (cfg.layers - 1) * (cfg.size * cfg.size) as usize;
    let mut w = vec![vec![0f32; n]; cfg.k]; // readout weights (bench-side f32; int8 later)
    for t in 0..cfg.trials {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a = reservoir_activity(&mut net, cfg, class, t);
        let scores: Vec<f32> = (0..cfg.k).map(|c| w[c].iter().zip(&a).map(|(wi, ai)| wi * ai).sum()).collect();
        let p = softmax(&scores);
        for c in 0..cfg.k {
            let err = p[c] - if c == class { 1.0 } else { 0.0 };
            for j in 0..n {
                w[c][j] -= cfg.readout_lr * err * a[j];
            }
        }
    }
    // held-out: frozen readout, disjoint trial indices (unseen cue realisations)
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a = reservoir_activity(&mut net, cfg, class, t);
        let scores: Vec<f32> = (0..cfg.k).map(|c| w[c].iter().zip(&a).map(|(wi, ai)| wi * ai).sum()).collect();
        let pred = (0..cfg.k).max_by(|&x, &y| scores[x].total_cmp(&scores[y])).unwrap();
        if pred == class {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}

/// Regenerate the procedural target of source neuron `src_local`'s level-`+1` slot `k` (matches
/// `generate_into`), so the e-prop update can pair a stored weight with its postsynaptic neuron.
/// The topology entries (level, count, radius) in the same order as `engine_config_recurrent`, so the
/// training loop can walk out_weights slots and know each slot's level/target.
fn topo_entries(cfg: &RsnnConfig) -> Vec<(i32, usize, u32)> {
    let mut e = vec![(1i32, cfg.up_count as usize, cfg.up_radius)];
    if cfg.back_count > 0 {
        e.push((-1, cfg.back_count as usize, cfg.back_radius));
        e.push((-2, cfg.back_count as usize, cfg.back_radius));
    }
    e
}

/// Temporal-XOR trial (reset → cue(a) → delay → cue(b) → read) on a multi-layer net. Records every
/// computational layer's per-wave fired-set and returns (top-layer read-window spike counts, spikes[z][t]).
fn xor_trial_layers(net: &mut Network, cfg: &RsnnConfig, a: usize, b: usize, trial: usize) -> (Vec<f32>, Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<i16>>>, Vec<Vec<Vec<i32>>>) {
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 1..l {
        let r = rec.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| r.lock().unwrap()[z].push(fired.to_vec())));
    }
    net.reset_state();
    let mut pots: Vec<Vec<Vec<i16>>> = vec![Vec::new(); l];
    let mut effs: Vec<Vec<Vec<i32>>> = vec![Vec::new(); l];
    let record = |net: &Network, pots: &mut Vec<Vec<Vec<i16>>>, effs: &mut Vec<Vec<Vec<i32>>>| {
        for z in 1..l {
            pots[z].push(net.layer_decide_potential(z));
            effs[z].push(net.layer_decide_effective_threshold(z));
        }
    };
    for (class, phase) in [(a, 0usize), (b, 1)] {
        // (delay sits between the two cue presentations)
        if phase == 1 {
            for _ in 0..cfg.delay {
                net.wave(&[]);
                record(net, &mut pots, &mut effs);
            }
        }
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * 2 + phase, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
            record(net, &mut pots, &mut effs);
        }
    }
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
        record(net, &mut pots, &mut effs);
    }
    net.clear_listeners();
    let spikes = rec.lock().unwrap().clone();
    let ttot = spikes[l - 1].len();
    let mut act = vec![0f32; ls];
    for wv in spikes[l - 1].iter().skip(ttot - cfg.read_waves) {
        for &loc in wv {
            act[loc as usize] += 1.0;
        }
    }
    (act, spikes, pots, effs)
}

/// Like `xor_trial_layers` but for an arbitrary cue sequence `classes` (a `delay` gap before every cue
/// except the first), recording *every* computational layer's per-wave fired-set, decide-potential, and
/// decide-time effective threshold. For `classes = [a, b]` (per-cue seed `trial·n + pos`) it is byte-identical
/// to `xor_trial_layers(a, b, trial)` — see `sequence_trial_layers_matches_xor`. Lets the deep multi-layer
/// trainer run any binary-labelled sequence task (parity, distractor, flip-flop), not just temporal XOR.
fn sequence_trial_layers(net: &mut Network, cfg: &RsnnConfig, classes: &[usize], trial: usize) -> (Vec<f32>, Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<i16>>>, Vec<Vec<Vec<i32>>>) {
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let n = classes.len();
    let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 1..l {
        let r = rec.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| r.lock().unwrap()[z].push(fired.to_vec())));
    }
    net.reset_state();
    let mut pots: Vec<Vec<Vec<i16>>> = vec![Vec::new(); l];
    let mut effs: Vec<Vec<Vec<i32>>> = vec![Vec::new(); l];
    let record = |net: &Network, pots: &mut Vec<Vec<Vec<i16>>>, effs: &mut Vec<Vec<Vec<i32>>>| {
        for z in 1..l {
            pots[z].push(net.layer_decide_potential(z));
            effs[z].push(net.layer_decide_effective_threshold(z));
        }
    };
    for (pos, &class) in classes.iter().enumerate() {
        if pos > 0 {
            for _ in 0..cfg.delay {
                net.wave(&[]);
                record(net, &mut pots, &mut effs);
            }
        }
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * n + pos, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
            record(net, &mut pots, &mut effs);
        }
    }
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
        record(net, &mut pots, &mut effs);
    }
    net.clear_listeners();
    let spikes = rec.lock().unwrap().clone();
    let ttot = spikes[l - 1].len();
    let mut act = vec![0f32; ls];
    for wv in spikes[l - 1].iter().skip(ttot - cfg.read_waves) {
        for &loc in wv {
            act[loc as usize] += 1.0;
        }
    }
    (act, spikes, pots, effs)
}

const P_DFA: u64 = 61; // fixed random Direct-Feedback-Alignment weights

/// Fixed random ±1 DFA feedback weight for (target neuron `neuron_global`, output class `class`) —
/// deterministic, hash-derived, stored-free. Broadcasts the output error to a deep layer.
fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
    if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
}

/// Dampening (γ) for the bump pseudo-derivative ψ = γ·max(0, 1−|v−eff|/W). LSNN uses 0.3.
const PSI_GAMMA: f32 = 0.3;
/// Half-width W (in i16 potential units) of the bump pseudo-derivative, centered at the decide-time
/// effective threshold. LSNN normalizes by v_th, but this substrate's calibration floors baseline θ to ~1
/// while integer ±1 drive makes potentials overshoot eff by O(2–26) at a spike — so a θ-normalized bump
/// collapses to ~0. A fixed band matched to the potential-fluctuation scale (the engine's own working
/// `elig_post` uses PSI_BAND=8; deeper layers overshoot more) keeps ψ non-degenerate near threshold.
const PSI_WIDTH: f32 = 16.0;

/// Σ_t of the ALIF eligibility trace `e_ij(t) = ψ_j(t)·(εᵛ_i(t) − β·εᵃ_ij(t))`, with the adaptation
/// eligibility εᵃ recursed at the slow rate ρ: `εᵃ(t+1) = ψ(t)·εᵛ(t) + (ρ − β·ψ(t))·εᵃ(t)`. β = 0 reduces
/// to the plain membrane trace `Σ_t ψ_j·εᵛ_i` (what the code did before). `psi(tt)` is ψ_j(tt), `ev(tt)`
/// is εᵛ_i(tt) (the presynaptic trace). Bellec et al. 2020, Eq. 24–25 (verified against the official
/// autodiff implementation: ψ multiplies both the membrane and the −β·εᵃ term).
fn elig_adapt_sum(ttot: usize, beta: f32, rho: f32, psi: impl Fn(usize) -> f32, ev: impl Fn(usize) -> f32) -> f32 {
    let mut eps_a = 0.0f32;
    let mut e = 0.0f32;
    for tt in 0..ttot {
        let p = psi(tt);
        let v = ev(tt);
        e += p * (v - beta * eps_a);
        eps_a = p * v + (rho - beta * p) * eps_a;
    }
    e
}

fn target_of(seed: u64, source_global: u32, src_local: u32, level: i32, k: u32, radius: u32, size: u32) -> u32 {
    let (sx, sy) = xy_of(src_local, size);
    let h = mix(key(seed, source_global, level, k, P_TARGET));
    let span = 2 * radius + 1;
    let dx = map_range24((h >> 40) as u32, span) as i32 - radius as i32;
    let dy = map_range24(((h >> 16) as u32) & 0x00FF_FFFF, span) as i32 - radius as i32;
    local_of(wrap(sx, dx, size), wrap(sy, dy, size), size)
}

/// Build + calibrate + train (readout + hidden e-prop weights); return the trained net and the top-layer
/// readout. Split out so callers can both evaluate held-out accuracy and probe the trained net's per-layer
/// firing rates. `hidden_lr = 0` leaves the reservoir fixed (readout-only baseline).
fn train_eprop_inner(cfg: &RsnnConfig) -> (Network, Vec<Vec<f32>>) {
    let mut net = Network::new(cfg.engine_config());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let top = l - 1;
    let up = cfg.up_count as usize;
    let mut w = vec![vec![0f32; ls]; cfg.k]; // readout on the TOP layer only
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..w.len()).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    for t in 0..cfg.trials {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a_full = reservoir_activity(&mut net, cfg, class, t);
        let a_top = a_full[(l - 2) * ls..].to_vec(); // top-layer chunk
        let p = softmax(&score(&w, &a_top));
        let err: Vec<f32> = (0..cfg.k).map(|c| p[c] - if c == class { 1.0 } else { 0.0 }).collect();
        for c in 0..cfg.k {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * a_top[j];
            }
        }
        if cfg.hidden_lr != 0.0 {
            let trained: Vec<usize> = if cfg.multi_layer { (0..top).collect() } else { vec![top - 1] };
            for z in trained {
                let tgt = z + 1;
                // learning signal L_j for each target-layer neuron j: symmetric readout feedback for the
                // top layer, random DFA feedback for deeper layers. Eligibility is factored pre_i·psi_j.
                let mut l_sig: Vec<f32> = (0..ls)
                    .map(|j| {
                        (0..cfg.k)
                            .map(|c| {
                                let b = if tgt == top { w[c][j] } else { dfa_weight(cfg.seed, (tgt * ls + j) as u32, c) };
                                b * err[c]
                            })
                            .sum()
                    })
                    .collect();
                // Firing-rate regularization (LSNN-style): keep each target neuron near r_target by adding
                // c_reg·(r_j − r_target) to its learning signal, carried by the SAME eligibility. A too-quiet
                // neuron (r_j < r_target) gets a negative signal → its incoming weights rise → it fires more.
                // Guarded, so rate_reg = 0 is byte-identical.
                if cfg.rate_reg != 0.0 {
                    let n_waves = (cfg.present_waves + cfg.delay + cfg.read_waves) as f32;
                    let r_target = cfg.rate_target_permille as f32 / 1000.0;
                    let post_pre = net.with_layer_mut(tgt, |x| x.elig_pre.clone());
                    for j in 0..ls {
                        let r_j = post_pre[j] as f32 / n_waves;
                        l_sig[j] += cfg.rate_reg * (r_j - r_target);
                    }
                }
                let pre = net.with_layer_mut(z, |x| x.elig_pre.clone());
                let psi = net.with_layer_mut(tgt, |x| x.elig_post.clone());
                net.with_layer_mut(z, |lz| {
                    for i in 0..ls {
                        let pre_i = pre[i] as f32;
                        if pre_i == 0.0 {
                            continue;
                        }
                        let sg = (z * ls + i) as u32;
                        for kk in 0..up {
                            let j = target_of(cfg.seed, sg, i as u32, 1, kk as u32, cfg.up_radius, cfg.size) as usize;
                            lz.out_shadow[i * up + kk] += -cfg.hidden_lr * l_sig[j] * pre_i * psi[j] as f32;
                        }
                    }
                    for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                        *wq = s.round().clamp(-127.0, 127.0) as i8;
                    }
                });
            }
        }
    }
    (net, w)
}

/// Train a TOP-layer readout AND (via e-prop) the hidden layers' level+1 weights. Returns held-out test
/// accuracy permille. With `hidden_lr = 0` this is the fixed-reservoir top-layer readout (the fragile
/// baseline); with `hidden_lr > 0`, e-prop shapes the reservoir so the top layer becomes separable.
pub fn train_eprop(cfg: &RsnnConfig) -> u64 {
    let (mut net, w) = train_eprop_inner(cfg);
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..w.len()).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a_full = reservoir_activity(&mut net, cfg, class, t);
        let a_top = a_full[(l - 2) * ls..].to_vec();
        let scores = score(&w, &a_top);
        let pred = (0..cfg.k).max_by(|&x, &y| scores[x].total_cmp(&scores[y])).unwrap();
        if pred == class {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}

/// Two independent input bits for trial `t` (deterministic).
/// The temporal-XOR task as a sequence task: two independent cue bits (a, b), label = a XOR b. Passing this
/// to the task-parameterized deep trainer reproduces the original temporal-XOR training exactly.
fn xor_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
    let (a, b) = pick_ab(seed, t);
    (vec![a, b], a ^ b)
}

fn pick_ab(seed: u64, t: usize) -> (usize, usize) {
    let a = (mix(key(seed, t as u32, 0, 0, 51)) & 1) as usize;
    let b = (mix(key(seed, t as u32, 0, 0, 53)) & 1) as usize;
    (a, b)
}

/// `n` deterministic bits from `(seed, trial)`; label = their XOR (parity — non-monotone, needs recurrence).
pub fn task_parity(seed: u64, trial: usize, n: usize) -> (Vec<usize>, usize) {
    let bits: Vec<usize> = (0..n).map(|i| (mix(key(seed, trial as u32, 0, i as u32, 51)) & 1) as usize).collect();
    let label = bits.iter().fold(0usize, |acc, &b| acc ^ b);
    (bits, label)
}

/// `[a, distractor, b]` where the middle is a label-irrelevant cue (class 2); label = a XOR b (ignore D).
pub fn task_distractor(seed: u64, trial: usize) -> (Vec<usize>, usize) {
    let a = (mix(key(seed, trial as u32, 0, 0, 51)) & 1) as usize;
    let b = (mix(key(seed, trial as u32, 0, 0, 53)) & 1) as usize;
    (vec![a, 2, b], a ^ b)
}

/// `n_ops` set(class 0)/reset(class 1) ops; label = final state (set -> on 1, reset -> off 0).
pub fn task_flipflop(seed: u64, trial: usize, n_ops: usize) -> (Vec<usize>, usize) {
    let ops: Vec<usize> = (0..n_ops).map(|i| (mix(key(seed, trial as u32, 0, i as u32, 57)) & 1) as usize).collect();
    let last = *ops.last().unwrap();
    (ops, if last == 0 { 1 } else { 0 })
}

/// reset → present cue(a) → delay → present cue(b) → read (silent). Records L1 per-wave fired-sets and
/// returns (read-window L1 spike counts, per-wave L1 fired-sets over the whole trial).
fn xor_trial(net: &mut Network, cfg: &RsnnConfig, a: usize, b: usize, trial: usize) -> (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<i16>>, Vec<Vec<i32>>) {
    let ls = (cfg.size * cfg.size) as usize;
    let rec: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let r = rec.clone();
        net.on_layer(1, Box::new(move |_w, fired: &[u32]| r.lock().unwrap().push(fired.to_vec())));
    }
    net.reset_state();
    // per-wave decide potential + effective threshold of the recorded/recurrent layer (layer 1 = top here),
    // aligned one-per-wave with `waves`, so recurrent_update can build the bump ψ centered at eff.
    let mut pots_top: Vec<Vec<i16>> = Vec::new();
    let mut eff_top: Vec<Vec<i32>> = Vec::new();
    let present = |net: &mut Network, class: usize, phase: usize, pots: &mut Vec<Vec<i16>>, effs: &mut Vec<Vec<i32>>| {
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * 2 + phase, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
            pots.push(net.layer_decide_potential(1));
            effs.push(net.layer_decide_effective_threshold(1));
        }
    };
    present(net, a, 0, &mut pots_top, &mut eff_top);
    for _ in 0..cfg.delay {
        net.wave(&[]);
        pots_top.push(net.layer_decide_potential(1));
        eff_top.push(net.layer_decide_effective_threshold(1));
    }
    present(net, b, 1, &mut pots_top, &mut eff_top);
    let read_start = rec.lock().unwrap().len();
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
        pots_top.push(net.layer_decide_potential(1));
        eff_top.push(net.layer_decide_effective_threshold(1));
    }
    net.clear_listeners();
    let waves = rec.lock().unwrap().clone();
    let mut act = vec![0f32; ls];
    for wave in waves.iter().skip(read_start) {
        for &loc in wave {
            act[loc as usize] += 1.0;
        }
    }
    (act, waves, pots_top, eff_top)
}

/// reset → for each class in `classes`: (a `delay` gap before every cue except the first) present
/// cue(class) for `present_waves` → `read_waves` silent. Records L1 per-wave fired-sets; returns
/// (read-window L1 spike counts, per-wave fired-sets). Generalizes `xor_trial`: `classes = [a, b]` with the
/// per-cue seed `trial·n + pos` reproduces `xor_trial(a, b)` exactly.
fn sequence_trial(net: &mut Network, cfg: &RsnnConfig, classes: &[usize], trial: usize) -> (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<i16>>, Vec<Vec<i32>>) {
    let ls = (cfg.size * cfg.size) as usize;
    let n = classes.len();
    let top = net.layer_count() - 1; // read (and, for the recurrent-top stack, train) the top computational layer
    let rec: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let r = rec.clone();
        net.on_layer(top, Box::new(move |_w, fired: &[u32]| r.lock().unwrap().push(fired.to_vec())));
    }
    net.reset_state();
    // per-wave decide potential + effective threshold of the recurrent (top) layer, aligned with `waves`.
    let mut pots_top: Vec<Vec<i16>> = Vec::new();
    let mut eff_top: Vec<Vec<i32>> = Vec::new();
    for (pos, &class) in classes.iter().enumerate() {
        if pos > 0 {
            for _ in 0..cfg.delay {
                net.wave(&[]);
                pots_top.push(net.layer_decide_potential(top));
                eff_top.push(net.layer_decide_effective_threshold(top));
            }
        }
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * n + pos, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
            pots_top.push(net.layer_decide_potential(top));
            eff_top.push(net.layer_decide_effective_threshold(top));
        }
    }
    let read_start = rec.lock().unwrap().len();
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
        pots_top.push(net.layer_decide_potential(top));
        eff_top.push(net.layer_decide_effective_threshold(top));
    }
    net.clear_listeners();
    let waves = rec.lock().unwrap().clone();
    let mut act = vec![0f32; ls];
    for wave in waves.iter().skip(read_start) {
        for &loc in wave {
            act[loc as usize] += 1.0;
        }
    }
    (act, waves, pots_top, eff_top)
}

/// Temporal e-prop on the L1 level-0 recurrent weights. Builds a decaying presynaptic trace per neuron
/// over the recorded waves, correlates it with postsynaptic spikes (`e_ij = Σ_t pre_trace_i(t)·fired_j(t)`),
/// and updates the stored weights via the symmetric-feedback learning signal.
fn recurrent_update(net: &mut Network, cfg: &RsnnConfig, w: &[Vec<f32>], err: &[f32], waves: &[Vec<u32>], pots_top: &[Vec<i16>], eff_top: &[Vec<i32>], rec_layer: usize) {
    let ls = (cfg.size * cfg.size) as usize;
    let up = cfg.rec_count as usize;
    let ttot = waves.len();
    let mut fired = vec![vec![0f32; ls]; ttot];
    for (t, wv) in waves.iter().enumerate() {
        for &loc in wv {
            fired[t][loc as usize] = 1.0;
        }
    }
    let decay = 1.0 - 1.0 / cfg.rec_tau.max(1.0);
    let mut tr = vec![vec![0f32; ls]; ttot];
    for i in 0..ls {
        let mut trace = 0f32;
        for t in 0..ttot {
            trace = trace * decay + fired[t][i];
            tr[t][i] = trace;
        }
    }
    let mut l_sig: Vec<f32> = (0..ls).map(|j| (0..2).map(|c| w[c][j] * err[c]).sum()).collect();
    // Firing-rate regularization: keep the recurrent layer near r_target so the loop stays alive through
    // the gap (carried by the same temporal eligibility). Guarded — rate_reg = 0 is byte-identical.
    if cfg.rate_reg != 0.0 {
        let r_target = cfg.rate_target_permille as f32 / 1000.0;
        for j in 0..ls {
            let r_j = (0..ttot).map(|t| fired[t][j]).sum::<f32>() / ttot.max(1) as f32;
            l_sig[j] += cfg.rate_reg * (r_j - r_target);
        }
    }
    // ALIF adaptation eligibility (guarded): β active only with adaptation; bump ψ implied when β>0.
    let beta = if cfg.adapt_bump > 0 { cfg.elig_beta } else { 0.0 };
    let use_adapt = beta != 0.0;
    let use_bump = cfg.elig_bump_psi || use_adapt;
    let rho = 1.0 - (2.0f32).powi(-(cfg.adapt_decay as i32));
    // ψ_j(t): fixed-width bump centered at the decide-time ADAPTIVE firing threshold, else the raw spike.
    let psi = |t: usize, j: usize| -> f32 {
        if use_bump {
            (PSI_GAMMA * (1.0 - (pots_top[t][j] as f32 - eff_top[t][j] as f32).abs() / cfg.elig_psi_width.max(1.0))).max(0.0)
        } else {
            fired[t][j]
        }
    };
    net.with_layer_mut(rec_layer, |l1| {
        for i in 0..ls {
            let sg = (rec_layer * ls + i) as u32; // recurrent layer's global neuron id
            for kk in 0..up {
                let j = target_of(cfg.seed, sg, i as u32, 0, kk as u32, cfg.rec_radius, cfg.size) as usize;
                let e = if use_adapt {
                    elig_adapt_sum(ttot, beta, rho, |t| psi(t, j), |t| tr[t][i])
                } else {
                    let mut s = 0f32;
                    for t in 0..ttot {
                        s += tr[t][i] * fired[t][j];
                    }
                    s
                };
                l1.out_shadow[i * up + kk] += -cfg.hidden_lr * l_sig[j] * e;
            }
        }
        for (wq, s) in l1.out_weights.iter_mut().zip(&l1.out_shadow) {
            *wq = s.round().clamp(-127.0, 127.0) as i8;
        }
    });
}

/// Build + calibrate + train (readout + level-0 recurrent weights) for temporal XOR; return the trained net
/// and the L1 readout. Split out so callers can probe the trained net's per-wave recurrent activity.
fn train_xor_inner(cfg: &RsnnConfig) -> (Network, Vec<Vec<f32>>) {
    let mut net = Network::new(cfg.engine_config_xor());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    if cfg.rec_count > 0 && cfg.rec_init != 0 {
        // bootstrap self-excitation so recurrent activity persists through the gap (else no eligibility)
        net.with_layer_mut(1, |l1| {
            for wq in l1.out_weights.iter_mut() {
                *wq = cfg.rec_init;
            }
            for (s, wq) in l1.out_shadow.iter_mut().zip(&l1.out_weights) {
                *s = *wq as f32;
            }
        });
    }
    let ls = (cfg.size * cfg.size) as usize;
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    for t in 0..cfg.trials {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let label = a ^ b;
        let (act, waves, pots_top, eff_top) = xor_trial(&mut net, cfg, a, b, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.rec_count > 0 {
            let rl = net.layer_count() - 1;
            recurrent_update(&mut net, cfg, &w, &err, &waves, &pots_top, &eff_top, rl);
        }
    }
    (net, w)
}

/// Train a readout on L1's read-window activity for temporal XOR; with `rec_count > 0` also trains the L1
/// level-0 recurrent weights. Returns held-out test accuracy permille.
pub fn train_xor(cfg: &RsnnConfig) -> u64 {
    let (mut net, w) = train_xor_inner(cfg);
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let (act, _, _, _) = xor_trial(&mut net, cfg, a, b, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == (a ^ b) {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}

/// Train a readout (+ level-0 recurrent weights when `rec_count > 0`) on an arbitrary sequence task, given
/// by a `task(task_seed, trial) -> (cue-class sequence, binary label)` closure. Returns held-out permille.
/// Calibration is a one-time sensible init; ALIF and rate reg come from `cfg`.
pub fn train_sequence(cfg: &RsnnConfig, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
    let mut net = Network::new(cfg.engine_config_xor());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let ls = (cfg.size * cfg.size) as usize;
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    for t in 0..cfg.trials {
        let (classes, label) = task(cfg.task_seed, t);
        let (act, waves, pots_top, eff_top) = sequence_trial(&mut net, cfg, &classes, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.rec_count > 0 {
            let rl = net.layer_count() - 1;
            recurrent_update(&mut net, cfg, &w, &err, &waves, &pots_top, &eff_top, rl);
        }
    }
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (classes, label) = task(cfg.task_seed, t);
        let (act, _, _, _) = sequence_trial(&mut net, cfg, &classes, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == label {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}

/// Train (readout + all synapses via one temporal eligibility over every topology level) on temporal XOR,
/// multi-layer net. `back_count = 0` is the feed-forward baseline. Returns held-out test accuracy permille.
pub fn train_recurrent(cfg: &RsnnConfig) -> u64 {
    let mut net = Network::new(cfg.engine_config_recurrent());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let top = l - 1;
    let entries = topo_entries(cfg);
    let total_slots: usize = entries.iter().map(|(_, c, _)| c).sum();
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    let decay = 1.0 - 1.0 / cfg.rec_tau.max(1.0);
    // per-neuron threshold (post-calibration, fixed during weight training) for the sub-threshold ψ ramp
    let theta: Vec<Vec<f32>> = (0..l)
        .map(|z| net.layer_thresholds(z.max(1)).iter().map(|&t| (t as f32).max(1.0)).collect())
        .collect();
    // ALIF adaptation eligibility (guarded): β active only with adaptation; bump ψ implied when β>0.
    let beta = if cfg.adapt_bump > 0 { cfg.elig_beta } else { 0.0 };
    let use_adapt = beta != 0.0;
    let use_bump = cfg.elig_bump_psi || use_adapt;
    let rho = 1.0 - (2.0f32).powi(-(cfg.adapt_decay as i32));
    for t in 0..cfg.trials {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let label = a ^ b;
        let (act, spikes, pots, effs) = xor_trial_layers(&mut net, cfg, a, b, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.hidden_lr == 0.0 {
            continue;
        }
        let ttot = spikes[top].len();
        let mut fired = vec![vec![vec![0f32; ls]; ttot]; l];
        let mut pretr = vec![vec![vec![0f32; ls]; ttot]; l];
        for z in 1..l {
            for (tt, wv) in spikes[z].iter().enumerate() {
                for &loc in wv {
                    fired[z][tt][loc as usize] = 1.0;
                }
            }
            for i in 0..ls {
                let mut tr = 0.0;
                for tt in 0..ttot {
                    tr = tr * decay + fired[z][tt][i];
                    pretr[z][tt][i] = tr;
                }
            }
        }
        // postsynaptic factor ψ: normalized bump (centered at eff), sub-threshold ramp, or spike-time (fired)
        let mut post = vec![vec![vec![0f32; ls]; ttot]; l];
        for z in 1..l {
            for tt in 0..ttot {
                for j in 0..ls {
                    post[z][tt][j] = if use_bump {
                        (PSI_GAMMA * (1.0 - (pots[z][tt][j] as f32 - effs[z][tt][j] as f32).abs() / cfg.elig_psi_width.max(1.0))).max(0.0)
                    } else if cfg.subthreshold_psi {
                        (pots[z][tt][j] as f32 / theta[z][j]).clamp(0.0, 1.0)
                    } else {
                        fired[z][tt][j]
                    };
                }
            }
        }
        // per-(layer, neuron) firing rate for the rate regularizer (fraction of recorded waves j fired)
        let rate: Vec<Vec<f32>> = (0..l)
            .map(|z| {
                (0..ls)
                    .map(|j| (0..ttot).map(|tt| fired[z][tt][j]).sum::<f32>() / ttot.max(1) as f32)
                    .collect()
            })
            .collect();
        let r_target = cfg.rate_target_permille as f32 / 1000.0;
        // Firing-rate regularization: keep every recurrent layer near r_target so the loop stays alive
        // through the gap, carried by the same temporal eligibility. Guarded — rate_reg = 0 is byte-identical.
        let l_sig = |tz: usize, j: usize| -> f32 {
            let task: f32 = (0..2)
                .map(|c| {
                    let bb = if tz == top { w[c][j] } else { dfa_weight(cfg.seed, (tz * ls + j) as u32, c) };
                    bb * err[c]
                })
                .sum();
            task + if cfg.rate_reg != 0.0 { cfg.rate_reg * (rate[tz][j] - r_target) } else { 0.0 }
        };
        for z in 0..l {
            let mut updates: Vec<(usize, f32)> = Vec::new();
            let mut slot = 0usize;
            for &(level, count, radius) in &entries {
                let tz_i = z as i32 + level;
                if tz_i < 1 || tz_i >= l as i32 {
                    slot += count; // off-stack or into-L0 target — untrainable
                    continue;
                }
                let tz = tz_i as usize;
                for i in 0..ls {
                    let sg = (z * ls + i) as u32;
                    for k in 0..count {
                        let j = target_of(cfg.seed, sg, i as u32, level, k as u32, radius, cfg.size) as usize;
                        let e = if use_adapt {
                            elig_adapt_sum(ttot, beta, rho, |tt| post[tz][tt][j], |tt| pretr[z][tt][i])
                        } else {
                            let mut s = 0f32;
                            for tt in 0..ttot {
                                s += pretr[z][tt][i] * post[tz][tt][j];
                            }
                            s
                        };
                        if e != 0.0 {
                            updates.push((i * total_slots + slot + k, -cfg.hidden_lr * l_sig(tz, j) * e));
                        }
                    }
                }
                slot += count;
            }
            net.with_layer_mut(z, |lz| {
                for (idx, d) in &updates {
                    lz.out_shadow[*idx] += *d;
                }
                for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                    *wq = s.round().clamp(-127.0, 127.0) as i8;
                }
            });
        }
    }
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let (act, _, _, _) = xor_trial_layers(&mut net, cfg, a, b, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == (a ^ b) {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}

/// Multi-layer temporal e-prop on temporal XOR for an arbitrary **per-layer** topology. `layer_entries[z]`
/// lists layer z's `(level, count, radius)` in the SAME order as its built topology, so slot indices align
/// with `out_weights`. Trains every trainable weight (forward, backward, lateral, skip) via the factored
/// temporal eligibility × (symmetric-top / DFA-deep) signal. Returns held-out permille. Used for custom
/// topologies like the side-car; `train_recurrent` keeps its own uniform loop.
fn train_multilayer(cfg: &RsnnConfig, mut net: Network, layer_entries: &[Vec<(i32, usize, u32)>], task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let top = l - 1;
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    let decay = 1.0 - 1.0 / cfg.rec_tau.max(1.0);
    let theta: Vec<Vec<f32>> = (0..l)
        .map(|z| net.layer_thresholds(z.max(1)).iter().map(|&t| (t as f32).max(1.0)).collect())
        .collect();
    // ALIF adaptation eligibility (guarded): β active only with adaptation; bump ψ implied when β>0.
    let beta = if cfg.adapt_bump > 0 { cfg.elig_beta } else { 0.0 };
    let use_adapt = beta != 0.0;
    let use_bump = cfg.elig_bump_psi || use_adapt;
    let rho = 1.0 - (2.0f32).powi(-(cfg.adapt_decay as i32));
    for t in 0..cfg.trials {
        let (classes, label) = task(cfg.task_seed, t);
        let (act, spikes, pots, effs) = sequence_trial_layers(&mut net, cfg, &classes, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.hidden_lr == 0.0 {
            continue;
        }
        let ttot = spikes[top].len();
        let mut fired = vec![vec![vec![0f32; ls]; ttot]; l];
        let mut pretr = vec![vec![vec![0f32; ls]; ttot]; l];
        for z in 1..l {
            for (tt, wv) in spikes[z].iter().enumerate() {
                for &loc in wv {
                    fired[z][tt][loc as usize] = 1.0;
                }
            }
            for i in 0..ls {
                let mut tr = 0.0;
                for tt in 0..ttot {
                    tr = tr * decay + fired[z][tt][i];
                    pretr[z][tt][i] = tr;
                }
            }
        }
        let mut post = vec![vec![vec![0f32; ls]; ttot]; l];
        for z in 1..l {
            for tt in 0..ttot {
                for j in 0..ls {
                    post[z][tt][j] = if use_bump {
                        // normalized bump pseudo-derivative ψ = γ·max(0, 1−|v−eff|/θ), centered at the
                        // ADAPTIVE firing threshold eff (so it stays alive when adaptation raises the bar)
                        (PSI_GAMMA * (1.0 - (pots[z][tt][j] as f32 - effs[z][tt][j] as f32).abs() / cfg.elig_psi_width.max(1.0))).max(0.0)
                    } else if cfg.subthreshold_psi {
                        (pots[z][tt][j] as f32 / theta[z][j]).clamp(0.0, 1.0)
                    } else {
                        fired[z][tt][j]
                    };
                }
            }
        }
        let rate: Vec<Vec<f32>> = (0..l)
            .map(|z| {
                (0..ls)
                    .map(|j| (0..ttot).map(|tt| fired[z][tt][j]).sum::<f32>() / ttot.max(1) as f32)
                    .collect()
            })
            .collect();
        let r_target = cfg.rate_target_permille as f32 / 1000.0;
        // per-layer mean rate — used by the class-preserving recurrent stabilizer (uniform bias, not per-neuron)
        let layer_mean: Vec<f32> = (0..l).map(|z| rate[z].iter().sum::<f32>() / ls as f32).collect();
        let task_sig = |tz: usize, j: usize| -> f32 {
            (0..2)
                .map(|c| {
                    let bb = if tz == top { w[c][j] } else { dfa_weight(cfg.seed, (tz * ls + j) as u32, c) };
                    bb * err[c]
                })
                .sum()
        };
        for z in 0..l {
            let total_slots_z: usize = layer_entries[z].iter().map(|(_, c, _)| c).sum();
            let mut updates: Vec<(usize, f32)> = Vec::new();
            let mut slot = 0usize;
            for &(level, count, radius) in &layer_entries[z] {
                let tz_i = z as i32 + level;
                if tz_i < 1 || tz_i >= l as i32 {
                    slot += count;
                    continue;
                }
                let tz = tz_i as usize;
                for i in 0..ls {
                    let sg = (z * ls + i) as u32;
                    for k in 0..count {
                        let j = target_of(cfg.seed, sg, i as u32, level, k as u32, radius, cfg.size) as usize;
                        let e = if use_adapt {
                            elig_adapt_sum(ttot, beta, rho, |tt| post[tz][tt][j], |tt| pretr[z][tt][i])
                        } else {
                            let mut s = 0f32;
                            for tt in 0..ttot {
                                s += pretr[z][tt][i] * post[tz][tt][j];
                            }
                            s
                        };
                        if e != 0.0 {
                            // forward levels → per-neuron rate_reg (liveness). Recurrent levels (0/−1/−2) →
                            // per-LAYER rec_stab uniform bias (class-preserving) when set, else fall back to
                            // standard per-neuron rate_reg (the homogenizing one) for comparison.
                            let reg = if level > 0 {
                                cfg.rate_reg * (rate[tz][j] - r_target)
                            } else if cfg.rec_stab != 0.0 {
                                cfg.rec_stab * (layer_mean[tz] - r_target)
                            } else {
                                cfg.rate_reg * (rate[tz][j] - r_target)
                            };
                            updates.push((i * total_slots_z + slot + k, -cfg.hidden_lr * (task_sig(tz, j) + reg) * e));
                        }
                    }
                }
                slot += count;
            }
            net.with_layer_mut(z, |lz| {
                for (idx, d) in &updates {
                    lz.out_shadow[*idx] += *d;
                }
                for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                    *wq = s.round().clamp(-127.0, 127.0) as i8;
                }
            });
        }
    }
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (classes, label) = task(cfg.task_seed, t);
        let (act, _, _, _) = sequence_trial_layers(&mut net, cfg, &classes, t);
        let s = score(&w, &act);
        if (if s[1] > s[0] { 1 } else { 0 }) == label {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}

/// Train the backward-fed recurrent side-car (see `engine_config_sidecar`) on temporal XOR; held-out permille.
pub fn train_sidecar(cfg: &RsnnConfig) -> u64 {
    train_sidecar_task(cfg, xor_task)
}

/// The backward-fed recurrent side-car (L1 skips to L3 via +2; L2 is a recurrent scratchpad; L3↔L2 loop, L3
/// read) on an arbitrary binary-labelled sequence task.
pub fn train_sidecar_task(cfg: &RsnnConfig, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
    let mut net = Network::new(cfg.engine_config_sidecar());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let (uc, ur) = (cfg.up_count as usize, cfg.up_radius);
    let (n, r) = (cfg.rec_count as usize, cfg.rec_radius);
    // per-layer entries, matching engine_config_sidecar's topology order exactly
    let layer_entries = vec![
        vec![(1i32, uc, ur)],       // L0: +1
        vec![(2i32, uc, ur)],       // L1: +2 skip
        vec![(0i32, n, r), (1i32, n, r)], // L2: self + forward
        vec![(-1i32, n, r), (1i32, uc, ur)], // L3: backward loop + forward to L4 read
        vec![],                     // L4: read
    ];
    train_multilayer(cfg, net, &layer_entries, task)
}

/// The deeper 6-layer side-car (see `engine_config_sidecar_deep`) on an arbitrary binary sequence task.
pub fn train_sidecar_deep_task(cfg: &RsnnConfig, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
    let mut net = Network::new(cfg.engine_config_sidecar_deep());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let (uc, ur) = (cfg.up_count as usize, cfg.up_radius);
    let (n, r) = (cfg.rec_count as usize, cfg.rec_radius);
    // per-layer entries, matching engine_config_sidecar_deep's topology order exactly
    let layer_entries = vec![
        vec![(1i32, uc, ur)],              // L0: +1 → L1
        vec![(3i32, uc, ur)],              // L1: +3 skip → L4
        vec![(0i32, n, r), (1i32, n, r)],  // L2: self, +1 → L3
        vec![(0i32, n, r), (1i32, n, r), (-1i32, n, r)], // L3: self, +1 → L4 (write back), -1 → L2 (loop)
        vec![(1i32, uc, ur), (-1i32, n, r)], // L4: +1 → L5, -1 → L3 (drive side-car top)
        vec![],                            // L5: read
    ];
    train_multilayer(cfg, net, &layer_entries, task)
}

/// Train the clean hidden-recurrent stack (see `engine_config_hidden_rec`) on temporal XOR; permille.
/// Forward weights get per-neuron `rate_reg` (liveness); L2's level-0 recurrence gets per-layer `rec_stab`.
pub fn train_hidden_rec(cfg: &RsnnConfig) -> u64 {
    train_hidden_rec_task(cfg, xor_task)
}

/// Same deep hidden-recurrent stack, on an arbitrary **binary-labelled sequence task** (parity, distractor,
/// flip-flop, XOR). All forward layers are trained via multi-layer DFA; L2 carries the trained level-0
/// recurrence. This is the deep-forward-trained + recurrence combination the recurrence-null doc flagged as
/// the missing fair test.
pub fn train_hidden_rec_task(cfg: &RsnnConfig, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
    let mut net = Network::new(cfg.engine_config_hidden_rec());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let (uc, ur) = (cfg.up_count as usize, cfg.up_radius);
    let (n, r) = (cfg.rec_count as usize, cfg.rec_radius);
    let rec_entry = if n > 0 { vec![(1i32, uc, ur), (0i32, n, r)] } else { vec![(1i32, uc, ur)] };
    let d = cfg.hidden_rec_depth.max(3);
    // must mirror engine_config_hidden_rec's per-layer topology order: forward (+1) up to layer d-2, which
    // also carries the level-0 recurrence; the read top (d-1) has no outgoing.
    let layer_entries: Vec<Vec<(i32, usize, u32)>> = (0..d)
        .map(|z| {
            if z == d - 1 {
                vec![]
            } else if z == d - 2 {
                rec_entry.clone()
            } else {
                vec![(1i32, uc, ur)]
            }
        })
        .collect();
    train_multilayer(cfg, net, &layer_entries, task)
}

/// Train the L0→L1→L2 stack with the L2↔L3 loop (see `engine_config_l2l3loop`) on temporal XOR; permille.
pub fn train_l2l3loop(cfg: &RsnnConfig) -> u64 {
    let mut net = Network::new(cfg.engine_config_l2l3loop());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let (uc, ur) = (cfg.up_count as usize, cfg.up_radius);
    let (n, r) = (cfg.rec_count as usize, cfg.rec_radius);
    // per-layer entries, matching engine_config_l2l3loop's topology order exactly
    let layer_entries = vec![
        vec![(1i32, uc, ur)],             // L0: +1
        vec![(1i32, uc, ur)],             // L1: +1 (forward into L2)
        vec![(0i32, n, r), (1i32, n, r)], // L2: self + forward
        vec![(-1i32, n, r)],              // L3: backward
    ];
    train_multilayer(cfg, net, &layer_entries, xor_task)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elig_adapt_sum_matches_closed_form_and_sign() {
        // β=0 ⇒ pure membrane trace Σ ψ·εᵛ.
        let psi = [0.5f32, 0.0, 0.3];
        let ev = [1.0f32, 2.0, 4.0];
        let membrane: f32 = psi.iter().zip(ev).map(|(p, v)| p * v).sum();
        assert!((elig_adapt_sum(3, 0.0, 0.9, |t| psi[t], |t| ev[t]) - membrane).abs() < 1e-6);
        // β>0 ⇒ the −β·εᵃ term makes the total SMALLER than the membrane-only trace (adaptation is
        // suppressive: firing now raises future threshold), and εᵃ carries a slow trace forward.
        let full = elig_adapt_sum(3, 0.2, 0.9, |t| psi[t], |t| ev[t]);
        assert!(full < membrane, "adaptation term subtracts: {full} !< {membrane}");
        // hand-rolled reference for the same recursion
        let (mut eps_a, mut e) = (0.0f32, 0.0f32);
        for t in 0..3 {
            e += psi[t] * (ev[t] - 0.2 * eps_a);
            eps_a = psi[t] * ev[t] + (0.9 - 0.2 * psi[t]) * eps_a;
        }
        assert!((full - e).abs() < 1e-6);
    }

    /// The recurrence-requiring temporal-XOR config: LIF (no adaptation memory) + a delay that outlasts the
    /// membrane leak, so only a recurrent loop can hold A across the gap. (ALIF adaptation alone solves XOR
    /// feed-forward — a real finding — which is why we strip it here.)
    fn xor_cfg(seed: u64) -> RsnnConfig {
        let mut cfg = RsnnConfig::demo();
        cfg.seed = seed;
        cfg.task_seed = seed;
        cfg.adapt_bump = 0;
        cfg.delay = 20;
        cfg.trials = 1500;
        cfg
    }

    #[test]
    fn recurrence_does_not_yet_beat_ff_on_temporal_xor() {
        // HONEST NULL (after tuning delay, rec_count/radius, rec_tau, lr, rec_init): level-0 recurrence with
        // a spike-timing eligibility does NOT beat the feed-forward baseline on temporal XOR. Where FF fails
        // (delay 20, ~chance) recurrence can't sustain A across the 20-wave silent gap either (~chance);
        // where recurrence *can* hold memory (delay ~12) the LIF membrane leak already gives FF that memory,
        // so FF wins there too. The trained recurrent memory horizon ≈ the membrane-leak horizon (~12 waves)
        // — the floored leak that fixed infinite-memory now caps recurrent memory. Extending it needs a
        // better pseudo-derivative, level −1 recurrence, or surrogate-gradient BPTT (all deferred).
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mut best_rec = 0u64;
        for &s in &seeds {
            let ff = xor_cfg(s);
            let mut rc = ff.clone();
            rc.rec_count = 24;
            rc.rec_tau = 20.0;
            let ff_acc = train_xor(&ff);
            let rc_acc = train_xor(&rc);
            eprintln!("seed {s:#x}  FF {ff_acc}  +recurrence {rc_acc}");
            best_rec = best_rec.max(rc_acc);
        }
        assert!(best_rec < 640, "recurrence does not (yet) crack the 20-wave temporal XOR (best {best_rec})");
    }

    #[test]
    fn temporal_xor_ff_is_near_chance() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mut best = 0u64;
        for &s in &seeds {
            let acc = train_xor(&xor_cfg(s));
            eprintln!("FF (LIF, delay 20) temporal-XOR seed {s:#x}  held-out {acc}");
            best = best.max(acc);
        }
        assert!(best < 640, "feed-forward (LIF, long delay) must NOT solve temporal XOR (best {best})");
    }

    #[test]
    fn multilayer_beats_single_layer_at_depth() {
        // Separation erodes with depth: training only the last layer should weaken on a deep net, while
        // training every layer (multi-layer DFA credit) keeps it reliable.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let depth = 4usize;
        let mut worst_single = 1000u64;
        let mut worst_multi = 1000u64;
        for &s in &seeds {
            let mut single = RsnnConfig::demo();
            single.seed = s;
            single.task_seed = s;
            single.layers = depth;
            single.trials = 1500;
            let mut multi = single.clone();
            multi.multi_layer = true;
            let sa = train_eprop(&single);
            let ma = train_eprop(&multi);
            eprintln!("depth {depth} seed {s:#x}  single {sa}  multi {ma}");
            worst_single = worst_single.min(sa);
            worst_multi = worst_multi.min(ma);
        }
        eprintln!("worst single {worst_single}  worst multi {worst_multi}");
        assert!(worst_multi > 640, "multi-layer learns reliably at depth (worst {worst_multi})");
        assert!(worst_multi >= worst_single, "multi-layer is at least as good as single-layer at depth");
    }

    #[test]
    #[ignore]
    fn _gap_activity_probe() {
        // Where does the signal die? Total spikes per layer over one XOR trial, for LIF (adapt_bump=0) vs
        // ALIF (adapt_bump=20) — everything else identical. Isolates whether adaptation is what keeps the
        // deep layers alive.
        let mut cfg = RsnnConfig::demo();
        cfg.seed = 0xE9_0B_0A17;
        cfg.task_seed = 0xE9_0B_0A17;
        cfg.size = 16;
        cfg.layers = 4;
        cfg.back_count = 8;
        cfg.adapt_bump = 0;
        cfg.delay = 20;
        cfg.up_count = 32; // alive-LIF drive
        cfg.present_waves = 12;
        cfg.base_q16 = 30000;
        let mut net = Network::new(cfg.engine_config_recurrent());
        net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let theta: Vec<Vec<f32>> = (0..l)
            .map(|z| net.layer_thresholds(z.max(1)).iter().map(|&t| (t as f32).max(1.0)).collect())
            .collect();
        let (_, spikes, pots, _) = xor_trial_layers(&mut net, &cfg, 1, 0, 0);
        let ttot = spikes[l - 1].len();
        // does sub-ψ (clamp(v/θ)) ever differ from spike-ψ (fired)? count charged-but-silent neurons
        for z in 1..l {
            let total: usize = spikes[z].iter().map(|w| w.len()).sum();
            let mut diff = 0usize;
            let mut charged = 0usize;
            let mut sub_sum = 0f64;
            for tt in 0..ttot {
                let fs: std::collections::HashSet<u32> = spikes[z][tt].iter().copied().collect();
                for j in 0..ls {
                    let f = if fs.contains(&(j as u32)) { 1.0 } else { 0.0 };
                    let s = (pots[z][tt][j] as f32 / theta[z][j]).clamp(0.0, 1.0);
                    if (s - f).abs() > 1e-6 {
                        diff += 1;
                    }
                    if f == 0.0 && s > 0.0 {
                        charged += 1;
                        sub_sum += s as f64;
                    }
                }
            }
            let mean_sub = if charged > 0 { sub_sum / charged as f64 } else { 0.0 };
            println!("L{z}: spikes {total:>5}  subψ≠spikeψ entries {diff:>6}  charged-silent {charged:>6}  mean-subψ {mean_sub:.3}");
        }
        // per-wave total activity across all layers: does anything survive the silent gap?
        println!("present-a 0..12, GAP 12..32, present-b 32..44, read 44..50");
        let act_per_wave: Vec<(usize, f32)> = (0..ttot)
            .map(|tt| {
                let sp: usize = (1..l).map(|z| spikes[z][tt].len()).sum();
                let mv: f32 = (1..l).map(|z| pots[z][tt].iter().map(|&v| v.max(0) as f32).sum::<f32>()).sum::<f32>() / (ls * (l - 1)) as f32;
                (sp, mv)
            })
            .collect();
        for (tt, (sp, mv)) in act_per_wave.iter().enumerate() {
            println!("wave {tt:>3}  spikes(all) {sp:>4}  mean+v {mv:.2}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn lateral_gap_survival() {
        // Does rate reg keep the level-0 recurrent loop (L1) alive through the 20-wave gap? Per-wave L1 spike
        // counts on a TRAINED net, reg off vs on. Off: activity dies in ~6 waves. On: should persist. Trial
        // phases (present 6, delay 20, present 6, read 6): present-A 0..6, GAP 6..26, present-B 26..32, read 32..38.
        for reg in [0.0f32, 5.0] {
            let mut c = RsnnConfig::demo();
            c.seed = 0xE9_0B_0A17;
            c.task_seed = 0xE9_0B_0A17;
            c.adapt_bump = 0;
            c.delay = 20;
            c.rec_count = 24;
            c.rec_radius = 2;
            c.rec_tau = 20.0;
            c.rec_init = 0;
            c.trials = 800;
            c.rate_reg = reg;
            c.rate_target_permille = 100;
            let (mut net, _w) = train_xor_inner(&c);
            let (_, waves, _, _) = xor_trial(&mut net, &c, 1, 0, 0);
            let per_wave: Vec<usize> = waves.iter().map(|wv| wv.len()).collect();
            eprintln!("rate_reg {reg}: L1 spikes/wave {per_wave:?}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn alif_recurrence_vs_ff() {
        // Field-standard: ALIF holds the delay memory, recurrence + e-prop compute on top. Does lateral
        // recurrence EXTEND ALIF's memory horizon on temporal XOR? ALIF-FF vs ALIF+recurrence at delays
        // bracketing ALIF's ~64-wave horizon. Calibration is a one-time sensible init (not a rate target
        // that must transfer); rate reg off (ALIF owns the operating point). Single seed to locate the
        // horizon first; go multi-seed if recurrence shows a gain. (demo has ALIF on: adapt_bump 20.)
        let s = 0xE9_0B_0A17u64;
        for delay in [40usize, 80, 120] {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.delay = delay;
            ff.trials = 1500;
            ff.rate_reg = 5.0;
            ff.rate_target_permille = 100;
            let mut rec = ff.clone();
            rec.rec_count = 24;
            rec.rec_radius = 2;
            rec.rec_tau = delay as f32; // eligibility trace spans the gap
            rec.rec_init = 0;
            let fa = train_xor(&ff);
            let ra = train_xor(&rec);
            eprintln!("delay {delay}  ALIF-FF {fa}  ALIF+rec {ra}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn backward_recurrence_vs_ff() {
        // Backward recurrence (level −1/−2) + rate reg vs feed-forward on temporal XOR (LIF, delay 20), on the
        // alive-LIF deep config (size 16, depth 4, up_count 32). Multi-seed. Compare to the lateral result.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let (mut best_ff, mut best_bw) = (0u64, 0u64);
        for &s in &seeds {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.size = 16;
            ff.layers = 4;
            ff.adapt_bump = 0;
            ff.delay = 20;
            ff.trials = 1500;
            ff.up_count = 32; // alive-LIF drive for the deep net
            ff.present_waves = 12;
            ff.base_q16 = 30000;
            let mut bw = ff.clone();
            bw.back_count = 8;
            bw.rate_reg = 5.0;
            bw.rate_target_permille = 100;
            let fa = train_recurrent(&ff);
            let ba = train_recurrent(&bw);
            eprintln!("backward seed {s:#x}  FF {fa}  +back+reg {ba}");
            best_ff = best_ff.max(fa);
            best_bw = best_bw.max(ba);
        }
        eprintln!("backward: best FF {best_ff}  best +back+reg {best_bw}");
    }

    #[test]
    fn train_sidecar_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 8;
        cfg.delay = 10;
        cfg.rec_count = 8;
        cfg.trials = 100;
        assert_eq!(train_sidecar(&cfg), train_sidecar(&cfg));
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_recurrence() {
        // The user's backward-fed recurrent side-car (L1 skips to L3 via +2; L3↔L2 loop; read L3) on temporal
        // XOR, ALIF, size 16, delay 20, modest side-car (rec_count 24). vs a matched trained deep FF (4 layers).
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for &s in &seeds {
            let mut sc = RsnnConfig::demo();
            sc.seed = s;
            sc.task_seed = s;
            sc.size = 16;
            sc.delay = 20;
            sc.trials = 1500;
            sc.rec_count = 24;
            sc.rec_radius = 4;
            sc.rate_reg = 5.0;
            sc.rate_target_permille = 100;
            let mut ff = sc.clone();
            ff.layers = 4;
            ff.back_count = 0;
            ff.rec_count = 0; // matched 4-layer trained deep FF baseline (uniform +1)
            let ff_acc = train_recurrent(&ff);
            let sc_acc = train_sidecar(&sc);
            eprintln!("side-car seed {s:#x}  deep-FF(4) {ff_acc}  side-car {sc_acc}");
        }
    }

    #[test]
    #[ignore] // diagnostic; fast (no training)
    fn adapt_vs_propagation() {
        // Does ALIF adaptation quench FORWARD propagation (starving deep layers → sub-critical)? Plain
        // 4-layer feed-forward stack; present a cue and count per-layer spikes over the trial, for
        // adapt_bump 0 (LIF), 20, 40. If deep-layer (L3) spikes fall as adaptation rises, adaptation is
        // suppressing the gain — the "ALIF fights recurrence/propagation" hypothesis.
        for ab in [0i16, 20, 40] {
            let mut c = RsnnConfig::demo();
            c.size = 16;
            c.layers = 4;
            c.up_count = 32;
            c.back_count = 0; // plain forward stack
            c.delay = 12;
            c.adapt_bump = ab;
            let mut net = Network::new(c.engine_config_recurrent());
            net.calibrate(&c.calib, &random_l0_input(c.seed ^ 0xE9, c.size, c.calib_fraction_q16));
            let (_, spikes, _, _) = xor_trial_layers(&mut net, &c, 1, 0, 0);
            let per_layer: Vec<usize> = (1..spikes.len()).map(|z| spikes[z].iter().map(|w| w.len()).sum()).collect();
            eprintln!("adapt_bump {ab}: L1..L3 total spikes over a cue trial {per_layer:?}");
        }
    }

    #[test]
    fn train_hidden_rec_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 8;
        cfg.delay = 10;
        cfg.rec_count = 8;
        cfg.rate_reg = 5.0;
        cfg.rec_stab = 5.0;
        cfg.trials = 100;
        assert_eq!(train_hidden_rec(&cfg), train_hidden_rec(&cfg));
    }

    #[test]
    fn train_multilayer_elig_off_is_unchanged_and_on_differs() {
        // Guard on an ALIVE config (so eligibility is non-zero and the εᵃ term can act): with the feature
        // at its defaults the trainer takes the pre-change branch and returns the frozen characterization
        // value (proves no regression); turning elig_beta on changes the trained result (proves it is wired).
        let mut base = RsnnConfig::demo();
        base.size = 8;
        base.up_count = 32; // alive drive — deep layers fire, so eligibility is non-trivial
        base.delay = 4;
        base.rec_count = 8;
        base.rate_reg = 5.0;
        base.rec_stab = 5.0;
        base.rec_radius = 2;
        base.trials = 400;
        let off = train_hidden_rec(&base); // elig defaults off (elig_beta 0, elig_bump_psi false)
        assert_eq!(off, 550, "elig-off byte-identical to the pre-change baseline");
        let mut on = base.clone();
        on.elig_beta = 0.4;
        let a = train_hidden_rec(&on);
        assert_ne!(a, off, "completed eligibility changes the trained result (feature is wired)");
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn fair_recurrence_test() {
        // The genuinely fair recurrence test: a WORKING deep-FF baseline (sparse drive up16 + forward rate_reg
        // → ~980 on XOR), with L2 recurrence stabilized by a CLASS-PRESERVING per-layer stabilizer (rec_stab),
        // NOT per-neuron rate_reg (which homogenizes). Does recurrence add on a live, un-poisoned baseline?
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111, 0x0F0F_2222];
        let (mut ffa, mut ra) = (Vec::new(), Vec::new());
        for &s in &seeds {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.size = 16;
            ff.up_count = 16; // sparse drive: deep FF is starved (dead) without rate_reg
            ff.delay = 20;
            ff.trials = 1500;
            ff.rate_reg = 5.0; // revive the forward path (per-neuron liveness)
            ff.rate_target_permille = 100;
            ff.rec_count = 0; // FF baseline: L2 has no recurrence
            let mut rec = ff.clone();
            rec.rec_count = 24;
            rec.rec_radius = 4;
            rec.rec_tau = 20.0;
            rec.rec_init = 0;
            rec.rec_stab = 5.0; // class-preserving per-layer recurrent stabilizer (not per-neuron rate_reg)
            let f = train_hidden_rec(&ff);
            let r = train_hidden_rec(&rec);
            eprintln!("fair-rec seed {s:#x}  deep-FF {f}  +hidden-rec(stab) {r}");
            ffa.push(f);
            ra.push(r);
        }
        eprintln!(
            "fair-rec: FF worst {} mean {}   +rec worst {} mean {}",
            ffa.iter().min().unwrap(),
            ffa.iter().sum::<u64>() / 5,
            ra.iter().min().unwrap(),
            ra.iter().sum::<u64>() / 5
        );
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn parity_fixed_vs_trained_recurrence() {
        // The informative parity test (size 16, alive read layer). Four columns: FF (no recurrence, trained
        // forward); FIXED-rec (±1 recurrence, readout only — classic LSM); CRUDE-rec (trained via spike-ψ);
        // COMPLETED-rec (trained via the ALIF εᵃ eligibility). Worst-seed over 3 seeds. Tests whether the
        // completed credit rule lets *trained* recurrence beat the *fixed* recurrence reservoir.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for n in [3usize, 4] {
            let mk = || {
                let mut c = RsnnConfig::demo();
                c.size = 16;
                c.up_count = 16;
                c.delay = 8;
                c.trials = 1500;
                c.rate_reg = 5.0;
                c.rate_target_permille = 100;
                c
            };
            let (mut wff, mut wfix, mut wcru, mut wcom) = (1000u64, 1000u64, 1000u64, 1000u64);
            for &s in &seeds {
                let run = |c: &RsnnConfig| train_sequence(c, |seed, t| task_parity(seed, t, n));
                let mut ff = mk();
                ff.seed = s; ff.task_seed = s; ff.rec_count = 0;
                let mut rec = mk();
                rec.seed = s; rec.task_seed = s; rec.rec_count = 24; rec.rec_radius = 4; rec.rec_tau = 20.0;
                let mut fix = rec.clone(); fix.hidden_lr = 0.0;             // fixed recurrence, readout only
                let mut cru = rec.clone(); cru.elig_beta = 0.0;            // crude spike-ψ eligibility
                let mut com = rec.clone(); com.elig_beta = 0.4;           // completed ALIF εᵃ eligibility
                let (fa, xa, ca, ma) = (run(&ff), run(&fix), run(&cru), run(&com));
                eprintln!("parity N={n} seed {s:#x}  FF {fa}  fixed-rec {xa}  crude-rec {ca}  completed-rec {ma}");
                wff = wff.min(fa); wfix = wfix.min(xa); wcru = wcru.min(ca); wcom = wcom.min(ma);
            }
            eprintln!("parity N={n} WORST:  FF {wff}  fixed-rec {wfix}  crude-rec {wcru}  completed-rec {wcom}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn deep_fixed_vs_trained_recurrence() {
        // The DEEP (4-layer) architecture where deep-FF + rate_reg reaches ~986: L0→L1→L2→L3, recurrence on
        // L2, ALL forward layers trained via multi-layer DFA (train_hidden_rec). Temporal XOR, delay 20,
        // size 16. Columns like the parity test — FF (no rec), FIXED-rec (hidden_lr 0 = frozen reservoir +
        // trained readout), CRUDE-rec (spike-ψ), COMPLETED-rec (ALIF εᵃ) — plus a rec_count sweep {8, 24} to
        // see whether a LIGHTER recurrence avoids the deep collapse the fair test showed at rec_count 24.
        // Worst-seed over 3 seeds. NOTE: the deep "fixed" column freezes *all* hidden weights (no per-level
        // freeze exists), so it reads the top of a frozen deep reservoir — not directly comparable to the
        // shallow fixed-rec (where the recurrent layer IS the read layer).
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let base = || {
            let mut c = RsnnConfig::demo();
            c.size = 16;
            c.up_count = 16;
            c.delay = 20;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c
        };
        let ff: Vec<u64> = seeds
            .iter()
            .map(|&s| {
                let mut c = base();
                c.seed = s;
                c.task_seed = s;
                c.rec_count = 0;
                train_hidden_rec(&c)
            })
            .collect();
        eprintln!("deep-FF (4 layers, temporal XOR): {ff:?}  worst {}", ff.iter().min().unwrap());
        for rc in [8u32, 24] {
            let (mut wx, mut wc, mut wm) = (1000u64, 1000u64, 1000u64);
            for &s in &seeds {
                let mk = || {
                    let mut c = base();
                    c.seed = s;
                    c.task_seed = s;
                    c.rec_count = rc;
                    c.rec_radius = 4;
                    c.rec_tau = 20.0;
                    c.rec_stab = 5.0;
                    c
                };
                let mut fix = mk();
                fix.hidden_lr = 0.0; // frozen reservoir + trained readout
                let mut cru = mk();
                cru.elig_beta = 0.0; // crude spike-ψ eligibility
                let mut com = mk();
                com.elig_beta = 0.4; // completed ALIF εᵃ eligibility
                let (xa, ca, ma) = (train_hidden_rec(&fix), train_hidden_rec(&cru), train_hidden_rec(&com));
                eprintln!("deep rec_count {rc} seed {s:#x}  fixed {xa}  crude {ca}  completed {ma}");
                wx = wx.min(xa);
                wc = wc.min(ca);
                wm = wm.min(ma);
            }
            eprintln!("deep rec_count {rc} WORST:  FF {}  fixed {wx}  crude {wc}  completed {wm}", ff.iter().min().unwrap());
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_on_easy_tasks() {
        // Does the side-car HURT where feed-forward is already enough? FF vs side-car on the FF-solvable tasks
        // (temporal XOR, distractor-XOR, flip-flop), size 32, 3 seeds, worst-seed. A robust topology should
        // match FF here, not wreck it.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mk = |s: u64| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 32;
            c.up_count = 16;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c.rec_radius = 4;
            c.rec_tau = 20.0;
            c.rec_stab = 5.0;
            c.rec_count = 8;
            c.elig_beta = 0.4;
            c
        };
        let run_pair = |tweak: &dyn Fn(&mut RsnnConfig), task: &dyn Fn(u64, usize) -> (Vec<usize>, usize)| -> (u64, u64) {
            let (mut wf, mut ws) = (1000u64, 1000u64);
            for &s in &seeds {
                let mut ff = mk(s);
                tweak(&mut ff);
                ff.rec_count = 0;
                let mut sc = mk(s);
                tweak(&mut sc);
                wf = wf.min(train_hidden_rec_task(&ff, task));
                ws = ws.min(train_sidecar_task(&sc, task));
            }
            (wf, ws)
        };
        let (f1, s1) = run_pair(&|c| c.delay = 20, &|seed, t| xor_task(seed, t));
        eprintln!("temporal-XOR (delay 20):  FF {f1}  sidecar {s1}");
        let (f2, s2) = run_pair(&|c| c.delay = 20, &|seed, t| task_distractor(seed, t));
        eprintln!("distractor-XOR (delay 20): FF {f2}  sidecar {s2}");
        let (f3, s3) = run_pair(&|c| { c.delay = 12; c.read_waves = 12; }, &|seed, t| task_flipflop(seed, t, 3));
        eprintln!("flip-flop (delay 12):     FF {f3}  sidecar {s3}");
    }

    #[test]
    fn train_sidecar_deep_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 8;
        cfg.delay = 8;
        cfg.rec_count = 8;
        cfg.elig_beta = 0.4;
        cfg.trials = 80;
        let run = || train_sidecar_deep_task(&cfg, |seed, t| task_parity(seed, t, 4));
        assert_eq!(run(), run());
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_deep_parity() {
        // Forward-count × width study on the (L4-read) compact side-car. Recurrence fixed at 16/r4 (best
        // read-layer config). parity N=4, single seed. Grid: size {16,32,64} × up_count {8,16,32,64}.
        let s = 0xE9_0B_0A17u64;
        let base = |size: u32, up: u32| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = size;
            c.up_count = up;
            c.up_radius = 3;
            c.delay = 8;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c.rec_count = 16;
            c.rec_radius = 4;
            c.rec_tau = 20.0;
            c.rec_stab = 5.0;
            c.elig_beta = 0.4;
            c
        };
        for size in [16u32, 32, 64] {
            for up in [8u32, 16, 32, 64] {
                let sc = train_sidecar_task(&base(size, up), |seed, t| task_parity(seed, t, 4));
                eprintln!("size {size:>3}  up {up:>3}  compact-sidecar {sc}");
            }
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_uniform_params() {
        // Both side-cars with UNIFORM synapse params for EVERY layer/group — count 8, radius 4 (forward and
        // side-car alike), the "everything sparse" unification that respects the recurrence's density
        // preference. parity N=4, size 32, 3 seeds. FF for reference.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let base = |s: u64| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 32;
            c.delay = 8;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c.rec_stab = 5.0;
            c.rec_tau = 20.0;
            c.elig_beta = 0.4;
            // uniform: same count + radius on forward and side-car groups
            c.up_count = 8;
            c.up_radius = 3;
            c.rec_count = 8;
            c.rec_radius = 3;
            c
        };
        let (mut wf, mut ws, mut wd) = (1000u64, 1000u64, 1000u64);
        for &s in &seeds {
            let mut ff = base(s);
            ff.rec_count = 0;
            let fa = train_hidden_rec_task(&ff, |seed, t| task_parity(seed, t, 4));
            let sa = train_sidecar_task(&base(s), |seed, t| task_parity(seed, t, 4));
            let da = train_sidecar_deep_task(&base(s), |seed, t| task_parity(seed, t, 4));
            eprintln!("seed {s:#x}  FF {fa}  sidecar {sa}  sidecar-deep {da}");
            wf = wf.min(fa);
            ws = ws.min(sa);
            wd = wd.min(da);
        }
        eprintln!("WORST (uniform 8/r3):  FF {wf}  sidecar {ws}  sidecar-deep {wd}");
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_verify() {
        // Verify the side-car win (one-seed sweep found sidecar 837 vs FF ~590 on parity N=4) across 3 seeds,
        // and the stacked config (side-car + β 1.2 + W 8, the three levers that each helped). vs FF and the
        // hidden-rec baseline. All size 32, parity N=4.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let base = |s: u64| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 32;
            c.up_count = 16;
            c.delay = 8;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c.rec_radius = 4;
            c.rec_tau = 20.0;
            c.rec_stab = 5.0;
            c.rec_count = 8;
            c.elig_beta = 0.4;
            c
        };
        let (mut wff, mut whr, mut wsc, mut wst) = (1000u64, 1000u64, 1000u64, 1000u64);
        for &s in &seeds {
            let mut ff = base(s);
            ff.rec_count = 0;
            let mut st = base(s);
            st.elig_beta = 1.2;
            st.elig_psi_width = 8.0;
            let fa = train_hidden_rec_task(&ff, |seed, t| task_parity(seed, t, 4));
            let ha = train_hidden_rec_task(&base(s), |seed, t| task_parity(seed, t, 4));
            let sa = train_sidecar_task(&base(s), |seed, t| task_parity(seed, t, 4));
            let ta = train_sidecar_task(&st, |seed, t| task_parity(seed, t, 4));
            eprintln!("seed {s:#x}  FF {fa}  hidden-rec {ha}  sidecar {sa}  sidecar+stack {ta}");
            wff = wff.min(fa);
            whr = whr.min(ha);
            wsc = wsc.min(sa);
            wst = wst.min(ta);
        }
        eprintln!("WORST:  FF {wff}  hidden-rec {whr}  sidecar {wsc}  sidecar+stack {wst}");
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn parity_improve_sweep() {
        // ONE-SEED exploration (the hard seed 0xE9_0B_0A17, where deep-parity N=4 had completed 560 < FF 620).
        // Goal: push trained recurrence ABOVE FF by varying depth / rec_count / β / bump-width / topology.
        // Each variant prints its completed-εᵃ held-out; FF baseline for reference.
        let s = 0xE9_0B_0A17u64;
        let base = || {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 32;
            c.up_count = 16;
            c.delay = 8;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c.rec_radius = 4;
            c.rec_tau = 20.0;
            c.rec_stab = 5.0;
            c.rec_count = 8;
            c.elig_beta = 0.4;
            c
        };
        let run = |c: &RsnnConfig| train_hidden_rec_task(c, |seed, t| task_parity(seed, t, 4));
        let mut ff = base();
        ff.rec_count = 0;
        eprintln!("FF baseline (d4 s32): {}", run(&ff));
        eprintln!("A baseline  (d4 s32 rc8 β0.4 W16): {}", run(&base()));
        let mut b = base();
        b.hidden_rec_depth = 5;
        eprintln!("B depth 5:   {}", run(&b));
        let mut d = base();
        d.rec_count = 12;
        eprintln!("C rc 12:     {}", run(&d));
        let mut e = base();
        e.rec_count = 16;
        eprintln!("D rc 16:     {}", run(&e));
        let mut f = base();
        f.elig_beta = 0.8;
        eprintln!("E β 0.8:     {}", run(&f));
        let mut g = base();
        g.elig_beta = 1.2;
        eprintln!("F β 1.2:     {}", run(&g));
        let mut h = base();
        h.elig_psi_width = 32.0;
        eprintln!("G W 32:      {}", run(&h));
        let mut i = base();
        i.elig_psi_width = 8.0;
        eprintln!("H W 8:       {}", run(&i));
        eprintln!("I sidecar (s32 rc8): {}", train_sidecar_task(&base(), |seed, t| task_parity(seed, t, 4)));
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn deep_parity() {
        // Parity N=3/4 on the DEEP 4-layer hidden-rec architecture (all forward layers trained via multi-layer
        // DFA, recurrence on L2), size 32, rec_count 8. Parity N≥3 is non-monotone — a task with HEADROOM that
        // ALIF feed-forward does not already saturate — so this is the test of whether trained recurrence can
        // BEAT feed-forward once the completed ALIF eligibility is used. FF / crude / completed, 3 seeds, worst.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for n in [3usize, 4] {
            let base = |s: u64| {
                let mut c = RsnnConfig::demo();
                c.seed = s;
                c.task_seed = s;
                c.size = 32;
                c.up_count = 16;
                c.delay = 8;
                c.trials = 1500;
                c.rate_reg = 5.0;
                c.rate_target_permille = 100;
                c.rec_radius = 4;
                c.rec_tau = 20.0;
                c.rec_stab = 5.0;
                c
            };
            let (mut wf, mut wc, mut wm) = (1000u64, 1000u64, 1000u64);
            for &s in &seeds {
                let mut ff = base(s);
                ff.rec_count = 0;
                let mut cru = base(s);
                cru.rec_count = 8;
                cru.elig_beta = 0.0;
                let mut com = base(s);
                com.rec_count = 8;
                com.elig_beta = 0.4;
                let fa = train_hidden_rec_task(&ff, |seed, t| task_parity(seed, t, n));
                let ca = train_hidden_rec_task(&cru, |seed, t| task_parity(seed, t, n));
                let ma = train_hidden_rec_task(&com, |seed, t| task_parity(seed, t, n));
                eprintln!("deep-parity N={n} seed {s:#x}  FF {fa}  crude {ca}  completed {ma}");
                wf = wf.min(fa);
                wc = wc.min(ca);
                wm = wm.min(ma);
            }
            eprintln!("deep-parity N={n} WORST:  FF {wf}  crude {wc}  completed {wm}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn deep_density_sweep() {
        // Deep 4-layer (train_hidden_rec), temporal XOR delay 20, size 16. Map trained recurrence vs
        // recurrence density (rec_count on L2): where is it trainable, where does it collapse, and where does
        // the completed ALIF eligibility (elig_beta 0.4) beat the crude spike-ψ (elig_beta 0)? 3 seeds, worst.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let base = |s: u64| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 16;
            c.up_count = 16;
            c.delay = 20;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c.rec_radius = 4;
            c.rec_tau = 20.0;
            c.rec_stab = 5.0;
            c
        };
        let ff: Vec<u64> = seeds
            .iter()
            .map(|&s| {
                let mut c = base(s);
                c.rec_count = 0;
                train_hidden_rec(&c)
            })
            .collect();
        eprintln!("deep-FF (size 16) worst {}  {ff:?}", ff.iter().min().unwrap());
        for rc in [4u32, 8, 12, 16, 24] {
            let (mut wc, mut wm) = (1000u64, 1000u64);
            for &s in &seeds {
                let mut cru = base(s);
                cru.rec_count = rc;
                cru.elig_beta = 0.0;
                let mut com = base(s);
                com.rec_count = rc;
                com.elig_beta = 0.4;
                let (ca, ma) = (train_hidden_rec(&cru), train_hidden_rec(&com));
                eprintln!("density rc {rc} seed {s:#x}  crude {ca}  completed {ma}");
                wc = wc.min(ca);
                wm = wm.min(ma);
            }
            eprintln!("density rc {rc} WORST:  crude {wc}  completed {wm}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn deep_size_sweep() {
        // Deep 4-layer, temporal XOR delay 20, rec_count 8 (the trainable density from deep_fixed_vs_trained).
        // Vary layer width (size 8/16/32 = 64/256/1024 neurons/layer). Does more width help TRAINED recurrence,
        // and does the completed eligibility's edge over crude hold/grow? FF / crude / completed, 3 seeds,
        // worst. Caveat: up_count (16) and rec_radius (4) are held fixed, so their *relative* density shrinks
        // with size (radius 4 is full-coverage at size 8, local at size 32) — an exploratory first look.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for size in [8u32, 16, 32] {
            let base = |s: u64| {
                let mut c = RsnnConfig::demo();
                c.seed = s;
                c.task_seed = s;
                c.size = size;
                c.up_count = 16;
                c.delay = 20;
                c.trials = 1500;
                c.rate_reg = 5.0;
                c.rate_target_permille = 100;
                c.rec_radius = 4;
                c.rec_tau = 20.0;
                c.rec_stab = 5.0;
                c
            };
            let (mut wf, mut wc, mut wm) = (1000u64, 1000u64, 1000u64);
            for &s in &seeds {
                let mut ff = base(s);
                ff.rec_count = 0;
                let mut cru = base(s);
                cru.rec_count = 8;
                cru.elig_beta = 0.0;
                let mut com = base(s);
                com.rec_count = 8;
                com.elig_beta = 0.4;
                let (fa, ca, ma) = (train_hidden_rec(&ff), train_hidden_rec(&cru), train_hidden_rec(&com));
                eprintln!("size {size} seed {s:#x}  FF {fa}  crude {ca}  completed {ma}");
                wf = wf.min(fa);
                wc = wc.min(ca);
                wm = wm.min(ma);
            }
            eprintln!("size {size} WORST:  FF {wf}  crude {wc}  completed {wm}");
        }
    }

    #[test]
    #[ignore] // temp diagnostic
    fn _diag_fair_hidden_lr_sensitivity() {
        // Is the fair +rec held-out sensitive to hidden training AT ALL? If (hidden_lr 0) == (hidden_lr>0) and
        // every β gives the same number, the config is INERT to hidden weights (collapse masks them) — which
        // fully explains byte-identical-across-β without a bug. If hidden_lr moves it but β doesn't, that would
        // point at a β bug. Positive control: parity (known β-sensitive) must move.
        let mk = |up: u32, delay: usize| {
            let mut c = RsnnConfig::demo();
            c.seed = 0xE9_0B_0A17;
            c.task_seed = 0xE9_0B_0A17;
            c.size = 16;
            c.up_count = up;
            c.delay = delay;
            c.trials = 1500;
            c.rate_reg = 5.0;
            c.rec_count = 24;
            c.rec_radius = 4;
            c.rec_tau = 20.0;
            c.rec_stab = 5.0;
            c
        };
        eprintln!("--- FAIR hidden-rec (up16, delay20) ---");
        for (hl, b) in [(0.0f32, 0.0f32), (0.004, 0.0), (0.004, 0.4), (0.004, 2.0), (0.02, 0.4)] {
            let mut c = mk(16, 20);
            c.hidden_lr = hl;
            c.elig_beta = b;
            eprintln!("hidden_lr {hl}  elig_beta {b}  ->  {}", train_hidden_rec(&c));
        }
        eprintln!("--- parity control (up16, delay8, N=3) positive control ---");
        for (hl, b) in [(0.0f32, 0.0f32), (0.004, 0.0), (0.004, 0.4)] {
            let mut c = mk(16, 8);
            c.hidden_lr = hl;
            c.elig_beta = b;
            eprintln!("hidden_lr {hl}  elig_beta {b}  ->  {}", train_sequence(&c, |seed, t| task_parity(seed, t, 3)));
        }
    }

    #[test]
    #[ignore] // temp diagnostic
    fn _diag_fair_elig_magnitude() {
        // Why is the fair hidden-rec +rec test byte-identical across β? Compare the eligibility e_ij under the
        // OLD rule (fired-ψ) vs the COMPLETED rule (bump-ψ + εᵃ) for L2's forward (+1) and recurrent (0)
        // synapses over one calibrated (untrained) trial. If both are ~0, there's nothing to train and β is
        // inert here (an uninformative config); if they differ substantially, the byte-identical training
        // result is convergence-to-the-same-collapse instead.
        let mut cfg = RsnnConfig::demo();
        cfg.seed = 0xE9_0B_0A17;
        cfg.task_seed = 0xE9_0B_0A17;
        cfg.size = 16;
        cfg.up_count = 16;
        cfg.delay = 20;
        cfg.rate_reg = 5.0;
        cfg.rec_count = 24;
        cfg.rec_radius = 4;
        cfg.rec_tau = 20.0;
        cfg.rec_stab = 5.0;
        let mut net = Network::new(cfg.engine_config_hidden_rec());
        net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let (uc, ur) = (cfg.up_count as usize, cfg.up_radius);
        let (n, r) = (cfg.rec_count as usize, cfg.rec_radius);
        let (_, spikes, pots, effs) = xor_trial_layers(&mut net, &cfg, 1, 0, 0);
        let ttot = spikes[l - 1].len();
        let mut fired = vec![vec![vec![0f32; ls]; ttot]; l];
        let mut pretr = vec![vec![vec![0f32; ls]; ttot]; l];
        let decay = 1.0 - 1.0 / cfg.rec_tau.max(1.0);
        for z in 1..l {
            for (tt, wv) in spikes[z].iter().enumerate() {
                for &loc in wv { fired[z][tt][loc as usize] = 1.0; }
            }
            for i in 0..ls {
                let mut tr = 0.0;
                for tt in 0..ttot { tr = tr * decay + fired[z][tt][i]; pretr[z][tt][i] = tr; }
            }
        }
        let bump = |z: usize, tt: usize, j: usize| (PSI_GAMMA * (1.0 - (pots[z][tt][j] as f32 - effs[z][tt][j] as f32).abs() / cfg.elig_psi_width.max(1.0))).max(0.0);
        let rho = 1.0 - (2.0f32).powi(-(cfg.adapt_decay as i32));
        // L2 (z=2): forward (+1 -> L3) and recurrent (0 -> L2). Compare e under fired-ψ vs bump-ψ+εᵃ.
        for &(level, count, radius, tgtz) in &[(1i32, uc, ur, 3usize), (0i32, n, r, 2usize)] {
            let (mut e_fired, mut e_full, mut n_nz_fired, mut n_nz_full) = (0f64, 0f64, 0usize, 0usize);
            for i in 0..ls {
                let sg = (2 * ls + i) as u32;
                for k in 0..count {
                    let j = target_of(cfg.seed, sg, i as u32, level, k as u32, radius, cfg.size) as usize;
                    let ef: f32 = (0..ttot).map(|tt| pretr[2][tt][i] * fired[tgtz][tt][j]).sum();
                    let eb = elig_adapt_sum(ttot, 0.4, rho, |tt| bump(tgtz, tt, j), |tt| pretr[2][tt][i]);
                    e_fired += ef.abs() as f64; e_full += eb.abs() as f64;
                    if ef != 0.0 { n_nz_fired += 1; }
                    if eb != 0.0 { n_nz_full += 1; }
                }
            }
            let tag = if level > 0 { "L2 forward +1" } else { "L2 recurrent 0" };
            eprintln!("{tag}: fired-ψ Σ|e| {e_fired:.1} nz {n_nz_fired}   bump+εᵃ Σ|e| {e_full:.1} nz {n_nz_full}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn fair_recurrence_eprop_complete() {
        // THE headline re-run: the airtight fair recurrence test (deep-FF ~986 vs +hidden-rec ~498/chance),
        // now comparing the OLD eligibility (elig_beta 0 — membrane-only, reproduces the null) against the
        // COMPLETED ALIF eligibility (elig_beta > 0 — adds the εᵃ adaptation-trace credit) on the recurrent
        // path. If +rec climbs back toward FF at some β, the null was an artifact of the incomplete rule.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111, 0x0F0F_2222];
        let base = |s: u64| {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.size = 16;
            ff.up_count = 16; // sparse drive: deep FF is starved without rate_reg
            ff.delay = 20;
            ff.trials = 1500;
            ff.rate_reg = 5.0; // revive the forward path (per-neuron liveness)
            ff.rate_target_permille = 100;
            ff.rec_count = 0; // FF baseline
            ff
        };
        // FF baseline is β-independent — compute once per seed.
        let ffa: Vec<u64> = seeds.iter().map(|&s| train_hidden_rec(&base(s))).collect();
        for (s, f) in seeds.iter().zip(&ffa) {
            eprintln!("deep-FF seed {s:#x}  {f}");
        }
        eprintln!("FF worst {} mean {}", ffa.iter().min().unwrap(), ffa.iter().sum::<u64>() / 5);
        for &beta in &[0.0f32, 0.1, 0.2, 0.4] {
            let mut ra = Vec::new();
            for &s in &seeds {
                let mut rec = base(s);
                rec.rec_count = 24;
                rec.rec_radius = 4;
                rec.rec_tau = 20.0;
                rec.rec_init = 0;
                rec.rec_stab = 5.0; // class-preserving per-layer stabilizer
                rec.elig_beta = beta;
                let r = train_hidden_rec(&rec);
                eprintln!("β {beta}  seed {s:#x}  +rec {r}");
                ra.push(r);
            }
            eprintln!("β {beta}: +rec worst {} mean {}", ra.iter().min().unwrap(), ra.iter().sum::<u64>() / 5);
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn parity_recurrence_eprop_complete() {
        // Parity (needs recurrent computation), ALIF, FF vs +lateral-recurrence with the completed ALIF
        // eligibility, β sweep. β=0 reproduces the documented "recurrence hurts parity" null.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for n in [3usize, 4] {
            for &beta in &[0.0f32, 0.2, 0.4] {
                let (mut wf, mut wr) = (1000u64, 1000u64);
                for &s in &seeds {
                    let mut ff = RsnnConfig::demo();
                    ff.seed = s;
                    ff.task_seed = s;
                    ff.delay = 8;
                    ff.trials = 1500;
                    ff.rate_reg = 5.0;
                    ff.rate_target_permille = 100;
                    let mut rec = ff.clone();
                    rec.rec_count = 24;
                    rec.rec_radius = 2;
                    rec.rec_tau = 20.0;
                    rec.rec_init = 0;
                    rec.elig_beta = beta;
                    let fa = train_sequence(&ff, |seed, t| task_parity(seed, t, n));
                    let ra = train_sequence(&rec, |seed, t| task_parity(seed, t, n));
                    eprintln!("parity N={n} β {beta} seed {s:#x}  FF {fa}  +rec {ra}");
                    wf = wf.min(fa);
                    wr = wr.min(ra);
                }
                eprintln!("parity N={n} β {beta}: WORST FF {wf}  +rec {wr}");
            }
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn fair_recurrence_lif() {
        // Final fair test with STANDARD LIF neurons (adapt_bump 0 — no ALIF adaptation quenching the
        // recurrent gain). Hidden-rec architecture, sparse drive + forward rate_reg. FF baseline, then
        // recurrence with (a) standard per-neuron rate_reg (rec_stab 0) vs (b) class-preserving per-layer
        // rec_stab. Does removing adaptation let recurrence survive with either stabilizer?
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111, 0x0F0F_2222];
        let mk = |s: u64| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 16;
            c.up_count = 16;
            c.delay = 20;
            c.trials = 1500;
            c.adapt_bump = 0; // LIF: no adaptation
            c.rate_reg = 5.0;
            c.rate_target_permille = 100;
            c.rec_count = 24;
            c.rec_radius = 4;
            c.rec_tau = 20.0;
            c.rec_init = 0;
            c
        };
        let (mut ff, mut rr, mut rs) = (Vec::new(), Vec::new(), Vec::new());
        for &s in &seeds {
            let mut base = mk(s);
            base.rec_count = 0; // FF baseline (LIF, no recurrence)
            ff.push(train_hidden_rec(&base));
            let mut a = mk(s);
            a.rec_stab = 0.0; // (a) recurrence + standard per-neuron rate_reg
            rr.push(train_hidden_rec(&a));
            let mut b = mk(s);
            b.rec_stab = 5.0; // (b) recurrence + class-preserving per-layer rec_stab
            rs.push(train_hidden_rec(&b));
        }
        let m = |v: &Vec<u64>| (*v.iter().min().unwrap(), v.iter().sum::<u64>() / 5);
        for (i, &s) in seeds.iter().enumerate() {
            eprintln!("LIF seed {s:#x}  FF {}  +rec(std rate_reg) {}  +rec(rec_stab) {}", ff[i], rr[i], rs[i]);
        }
        let (m1, m2, m3) = (m(&ff), m(&rr), m(&rs));
        eprintln!("LIF fair-rec: FF worst/mean {m1:?}  +rec(rate_reg) {m2:?}  +rec(rec_stab) {m3:?}");
    }

    #[test]
    fn train_l2l3loop_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 8;
        cfg.delay = 10;
        cfg.rec_count = 8;
        cfg.trials = 100;
        assert_eq!(train_l2l3loop(&cfg), train_l2l3loop(&cfg));
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn l2l3loop_recurrence() {
        // L0→L1→L2 (recurrent hidden) with the L2↔L3 loop, read L3. First CONFIRM no dead layer (per-layer
        // rates), then deep-FF(4) vs the loop on temporal XOR. ALIF + strong drive (up_count 32) for liveness.
        let mut base = RsnnConfig::demo();
        base.size = 16;
        base.up_count = 32;
        base.rec_count = 24;
        base.rec_radius = 4;
        base.delay = 12;
        base.rate_reg = 5.0;
        base.rate_target_permille = 100;
        // liveness probe: does every layer (esp. the read top L3) fire?
        let mut net = Network::new(base.engine_config_l2l3loop());
        net.calibrate(&base.calib, &random_l0_input(base.seed ^ 0xE9, base.size, base.calib_fraction_q16));
        let rates = net.measure_layer_rates(base.calib.warmup, base.calib.waves, &random_l0_input(base.seed ^ 0xE9, base.size, base.calib_fraction_q16));
        eprintln!("l2l3loop per-layer rates L0..L3: {:?}", rates.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());

        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for &s in &seeds {
            let mut lp = base.clone();
            lp.seed = s;
            lp.task_seed = s;
            lp.trials = 1500;
            let mut ff = lp.clone();
            ff.layers = 4;
            ff.back_count = 0;
            ff.rec_count = 0; // matched 4-layer trained deep FF
            let ff_acc = train_recurrent(&ff);
            let lp_acc = train_l2l3loop(&lp);
            eprintln!("l2l3loop seed {s:#x}  deep-FF(4) {ff_acc}  L2<->L3-loop {lp_acc}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn deep_ff_ratereg_robustness() {
        // Is the deep-FF + rate_reg XOR result (982) robust or a config fluke? PLAIN 4-layer feed-forward
        // stack (no recurrence: back_count=0, rec_count=0), temporal XOR, ALIF, size 16. Sweep delay ×
        // up_count, rate_reg OFF vs ON, over 5 seeds. The 982 was delay 20 / up_count 16 / rate_reg 5.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111, 0x0F0F_2222];
        for &delay in &[12usize, 20] {
            for &upc in &[16u32, 32] {
                for &rr in &[0.0f32, 5.0] {
                    let mut accs = Vec::new();
                    for &s in &seeds {
                        let mut c = RsnnConfig::demo();
                        c.seed = s;
                        c.task_seed = s;
                        c.size = 16;
                        c.layers = 4;
                        c.back_count = 0; // plain deep feed-forward
                        c.rec_count = 0;
                        c.delay = delay;
                        c.up_count = upc;
                        c.trials = 1500;
                        c.rate_reg = rr;
                        c.rate_target_permille = 100;
                        accs.push(train_recurrent(&c));
                    }
                    let worst = *accs.iter().min().unwrap();
                    let mean = accs.iter().sum::<u64>() / accs.len() as u64;
                    eprintln!("deep-FF delay {delay} up_count {upc} rate_reg {rr}  seeds {accs:?}  worst {worst} mean {mean}");
                }
            }
        }
    }

    #[test]
    fn train_recurrent_elig_on_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 16;
        cfg.layers = 4;
        cfg.back_count = 8;
        cfg.delay = 20;
        cfg.trials = 150;
        cfg.elig_beta = 0.2;
        assert_eq!(train_recurrent(&cfg), train_recurrent(&cfg));
    }

    #[test]
    fn rate_reg_backward_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 16;
        cfg.layers = 4;
        cfg.back_count = 8;
        cfg.adapt_bump = 0;
        cfg.delay = 20;
        cfg.rate_reg = 5.0;
        cfg.rate_target_permille = 100;
        cfg.trials = 150;
        assert_eq!(train_recurrent(&cfg), train_recurrent(&cfg));
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn parity_recurrence_sweep() {
        // Sequential parity (non-monotone -> adaptation can't fake it). ALIF, FF vs +lateral-recurrence, over N.
        // Expect FF to solve N=2 (that's XOR) and fail N>=3; does recurrence rescue it?
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for n in [2usize, 3, 4, 5] {
            let (mut worst_ff, mut worst_rec) = (1000u64, 1000u64);
            for &s in &seeds {
                let mut ff = RsnnConfig::demo(); // ALIF on (adapt_bump 20), rate_reg off
                ff.seed = s;
                ff.task_seed = s;
                ff.delay = 8;
                ff.trials = 1500;
                ff.rate_reg = 5.0;
                ff.rate_target_permille = 100;
                let mut rec = ff.clone();
                rec.rec_count = 24;
                rec.rec_radius = 2;
                rec.rec_tau = 20.0;
                rec.rec_init = 0;
                let fa = train_sequence(&ff, |seed, t| task_parity(seed, t, n));
                let ra = train_sequence(&rec, |seed, t| task_parity(seed, t, n));
                eprintln!("parity N={n} seed {s:#x}  FF {fa}  +rec {ra}");
                worst_ff = worst_ff.min(fa);
                worst_rec = worst_rec.min(ra);
            }
            eprintln!("parity N={n}: WORST FF {worst_ff}  +rec {worst_rec}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn parity_fair_regime() {
        // Fair test vs LSNN: scale up (size 16 = 256 neurons, in the field's 100-400 range) and DENSIFY the
        // recurrence (rec_count 96 ≈ 37% of the layer, full-layer radius) so recurrence is a real substrate,
        // not a thin sparse add-on. Does recurrence help parity now? (Input projection is still fixed
        // procedural — the remaining gap if density alone doesn't move it.)
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for n in [3usize, 4] {
            let (mut worst_ff, mut worst_rec) = (1000u64, 1000u64);
            for &s in &seeds {
                let mut ff = RsnnConfig::demo();
                ff.seed = s;
                ff.task_seed = s;
                ff.size = 16;
                ff.delay = 8;
                ff.trials = 1500;
                ff.rate_reg = 5.0;
                ff.rate_target_permille = 100;
                let mut rec = ff.clone();
                rec.rec_count = 96;
                rec.rec_radius = 8; // full-layer coverage on the 16×16 torus
                rec.rec_tau = 20.0;
                rec.rec_init = 0;
                let fa = train_sequence(&ff, |seed, t| task_parity(seed, t, n));
                let ra = train_sequence(&rec, |seed, t| task_parity(seed, t, n));
                eprintln!("parity-fair N={n} size16 dense-rec seed {s:#x}  FF {fa}  +rec {ra}");
                worst_ff = worst_ff.min(fa);
                worst_rec = worst_rec.min(ra);
            }
            eprintln!("parity-fair N={n}: WORST FF {worst_ff}  +rec {worst_rec}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn parity_deep_hidden_recurrence() {
        // The corrected "deeper + hidden recurrent layer" architecture: L0(input) → L1(forward) → L2(recurrent
        // top, read), with MODEST recurrence (rec_count 24 — not the super-dense 96 that collapsed). Does a
        // forward layer under a recurrent top layer help parity? Deep-FF vs deep-hidden-recurrent, size 16.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for n in [3usize, 4] {
            let (mut worst_ff, mut worst_rec) = (1000u64, 1000u64);
            for &s in &seeds {
                let mut ff = RsnnConfig::demo();
                ff.seed = s;
                ff.task_seed = s;
                ff.size = 16;
                ff.xor_layers = 3; // L0 input, L1 forward, L2 recurrent top
                ff.delay = 8;
                ff.trials = 1500;
                ff.rate_reg = 5.0;
                ff.rate_target_permille = 100;
                let mut rec = ff.clone();
                rec.rec_count = 24; // modest lateral density (9% of a 256-neuron layer) — off the super-critical cliff
                rec.rec_radius = 4;
                rec.rec_tau = 20.0;
                rec.rec_init = 0;
                let fa = train_sequence(&ff, |seed, t| task_parity(seed, t, n));
                let ra = train_sequence(&rec, |seed, t| task_parity(seed, t, n));
                eprintln!("parity-deep N={n} size16 3-layer seed {s:#x}  FF {fa}  +hidden-rec {ra}");
                worst_ff = worst_ff.min(fa);
                worst_rec = worst_rec.min(ra);
            }
            eprintln!("parity-deep N={n}: WORST FF {worst_ff}  +hidden-rec {worst_rec}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn distractor_xor_recurrence() {
        // Delayed XOR with an irrelevant distractor cue between A and B. ALIF, FF vs +recurrence, multi-seed.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for &s in &seeds {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.delay = 20;
            ff.trials = 1500;
            ff.rate_reg = 5.0;
            ff.rate_target_permille = 100;
            let mut rec = ff.clone();
            rec.rec_count = 24;
            rec.rec_radius = 2;
            rec.rec_tau = 20.0;
            rec.rec_init = 0;
            let fa = train_sequence(&ff, task_distractor);
            let ra = train_sequence(&rec, task_distractor);
            eprintln!("distractor-XOR seed {s:#x}  FF {fa}  +rec {ra}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn flipflop_recurrence() {
        // Set/reset flip-flop, state held across the read gap. ALIF, FF vs +recurrence, multi-seed.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for &s in &seeds {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.delay = 12;
            ff.read_waves = 12; // read after a gap so the state must be held
            ff.trials = 1500;
            ff.rate_reg = 5.0;
            ff.rate_target_permille = 100;
            let mut rec = ff.clone();
            rec.rec_count = 24;
            rec.rec_radius = 2;
            rec.rec_tau = 20.0;
            rec.rec_init = 0;
            let fa = train_sequence(&ff, |seed, t| task_flipflop(seed, t, 3));
            let ra = train_sequence(&rec, |seed, t| task_flipflop(seed, t, 3));
            eprintln!("flip-flop seed {s:#x}  FF {fa}  +rec {ra}");
        }
    }

    #[test]
    fn recurrent_update_elig_on_is_deterministic_and_differs() {
        // The recurrent-top path (train_sequence → recurrent_update) with the completed eligibility: it is a
        // pure function of (seed, config), and turning elig_beta on changes the trained result vs. off.
        let mut base = RsnnConfig::demo();
        base.delay = 8;
        base.rec_count = 8;
        base.rec_init = 0;
        base.rate_reg = 5.0;
        base.trials = 300;
        let run = |c: &RsnnConfig| train_sequence(c, |seed, t| task_parity(seed, t, 3));
        let off = run(&base);
        assert_eq!(off, run(&base), "default recurrent path deterministic");
        let mut on = base.clone();
        on.elig_beta = 0.4;
        let a = run(&on);
        assert_eq!(a, run(&on), "elig-on deterministic");
        assert_ne!(a, off, "completed eligibility changes the recurrent-top result (feature is wired)");
    }

    #[test]
    fn train_sequence_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.delay = 8;
        cfg.rec_count = 8;
        cfg.rec_init = 0;
        cfg.trials = 120;
        let run = || train_sequence(&cfg, |seed, t| task_parity(seed, t, 3));
        assert_eq!(run(), run());
    }

    #[test]
    fn task_labels_are_correct() {
        for trial in 0..25 {
            let (bits, label) = task_parity(42, trial, 4);
            assert_eq!(bits.len(), 4);
            assert!(bits.iter().all(|&b| b <= 1));
            assert_eq!(label, bits.iter().fold(0, |a, &b| a ^ b), "parity label is the XOR of the bits");

            let (classes, dlabel) = task_distractor(42, trial);
            assert_eq!(classes.len(), 3);
            assert_eq!(classes[1], 2, "middle cue is the label-irrelevant distractor class 2");
            assert_eq!(dlabel, classes[0] ^ classes[2], "distractor label is a XOR b, ignoring D");

            let (ops, flabel) = task_flipflop(42, trial, 3);
            assert_eq!(ops.len(), 3);
            assert!(ops.iter().all(|&o| o <= 1));
            let last = *ops.last().unwrap();
            assert_eq!(flabel, if last == 0 { 1 } else { 0 }, "state = set(0)->on(1), reset(1)->off(0)");
        }
    }

    #[test]
    fn sequence_trial_matches_xor_on_two_cues() {
        // The 2-cue sequence must reproduce xor_trial exactly (same gap structure, same per-cue seed scheme).
        let mut cfg = RsnnConfig::demo();
        cfg.delay = 20;
        cfg.rec_count = 0;
        let build = || {
            let mut net = Network::new(cfg.engine_config_xor());
            net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
            net
        };
        let mut n1 = build();
        let mut n2 = build();
        let (a1, w1, _, _) = xor_trial(&mut n1, &cfg, 1, 0, 3);
        let (a2, w2, _, _) = sequence_trial(&mut n2, &cfg, &[1, 0], 3);
        assert_eq!(a1, a2, "read-window activity matches xor_trial");
        assert_eq!(w1, w2, "per-wave fired-sets match xor_trial");
    }

    #[test]
    fn sequence_trial_layers_matches_xor_on_two_cues() {
        // The all-layer recorder must reproduce xor_trial_layers exactly for a 2-cue [a, b] sequence — this is
        // what makes train_hidden_rec(cfg) == train_hidden_rec_task(cfg, xor_task) byte-identical.
        let mut cfg = RsnnConfig::demo();
        cfg.size = 8;
        cfg.layers = 4;
        cfg.delay = 12;
        let build = || {
            let mut net = Network::new(cfg.engine_config_hidden_rec());
            net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
            net
        };
        let mut n1 = build();
        let mut n2 = build();
        let (a1, s1, p1, e1) = xor_trial_layers(&mut n1, &cfg, 1, 0, 3);
        let (a2, s2, p2, e2) = sequence_trial_layers(&mut n2, &cfg, &[1, 0], 3);
        assert_eq!(a1, a2, "read-window activity matches");
        assert_eq!(s1, s2, "per-wave fired-sets match across all layers");
        assert_eq!(p1, p2, "per-wave decide potentials match");
        assert_eq!(e1, e2, "per-wave decide effective thresholds match");
    }

    #[test]
    fn rate_reg_lateral_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.adapt_bump = 0;
        cfg.delay = 20;
        cfg.rec_count = 8;
        cfg.rec_init = 0;
        cfg.rate_reg = 5.0;
        cfg.rate_target_permille = 100;
        cfg.trials = 150;
        assert_eq!(train_xor(&cfg), train_xor(&cfg));
    }

    #[test]
    fn subthreshold_psi_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 16;
        cfg.layers = 4;
        cfg.back_count = 8;
        cfg.subthreshold_psi = true;
        cfg.adapt_bump = 0;
        cfg.delay = 20;
        cfg.trials = 300;
        assert_eq!(train_recurrent(&cfg), train_recurrent(&cfg));
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn subthreshold_psi_vs_spike_psi_on_temporal_xor() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mk = |s: u64| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 16;
            c.layers = 4;
            c.adapt_bump = 0; // LIF
            c.delay = 20;
            c.trials = 1500;
            // "alive LIF": raise feed-forward gain so the transient cue propagates through all 4 layers
            // (default drive leaves the readout layer dead — see _gap_activity_probe).
            c.up_count = 32;
            c.present_waves = 12;
            c.base_q16 = 30000;
            c
        };
        let (mut best_ff, mut best_spk, mut best_sub) = (0u64, 0u64, 0u64);
        for &s in &seeds {
            let ff = mk(s);
            let mut spk = ff.clone();
            spk.back_count = 8;
            let mut sub = spk.clone();
            sub.subthreshold_psi = true;
            let (fa, sa, ua) = (train_recurrent(&ff), train_recurrent(&spk), train_recurrent(&sub));
            eprintln!("seed {s:#x}  FF {fa}  backward+spikeψ {sa}  backward+subψ {ua}");
            best_ff = best_ff.max(fa);
            best_spk = best_spk.max(sa);
            best_sub = best_sub.max(ua);
        }
        eprintln!("best  FF {best_ff}  spikeψ {best_spk}  subψ {best_sub}");
        assert!(best_sub >= 485, "sanity; verdict is the printed comparison (Step 6)");
    }

    #[test]
    #[ignore] // expensive (6 deep trainings) + a documented null; run manually in --release
    fn backward_recurrence_vs_ff_on_temporal_xor() {
        // Backward recurrence (level −1/−2) + width vs feed-forward on temporal XOR (LIF, delay 20).
        // If +backward beats FF, recurrence earns its keep; if it nulls too, topology+capacity are
        // controlled out and ψ (spike-time-only) is the implicated blocker.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mut best_ff = 0u64;
        let mut best_bw = 0u64;
        for &s in &seeds {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.size = 16;
            ff.layers = 4;
            ff.adapt_bump = 0; // LIF
            ff.delay = 20;
            ff.trials = 1500;
            let mut bw = ff.clone();
            bw.back_count = 8; // backward recurrence on
            let ff_acc = train_recurrent(&ff);
            let bw_acc = train_recurrent(&bw);
            eprintln!("seed {s:#x}  FF {ff_acc}  +backward {bw_acc}");
            best_ff = best_ff.max(ff_acc);
            best_bw = best_bw.max(bw_acc);
        }
        eprintln!("best FF {best_ff}  best +backward {best_bw}");
        assert!(best_bw >= 485, "sanity: accuracy in range (verdict is the printed comparison)");
    }

    #[test]
    fn backward_recurrence_config_builds() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 16;
        cfg.layers = 4;
        cfg.back_count = 8;
        let net = Network::new(cfg.engine_config_recurrent());
        assert_eq!(net.layer_count(), 4);
        let e = topo_entries(&cfg);
        assert_eq!(e.iter().map(|(_, c, _)| c).sum::<usize>(), cfg.up_count as usize + 2 * 8);
        assert_eq!(e.iter().map(|(lv, _, _)| *lv).collect::<Vec<_>>(), vec![1, -1, -2]);
    }

    #[test]
    fn dfa_weights_are_deterministic_and_signed() {
        let f = |g, c| dfa_weight(7, g, c);
        assert_eq!(f(10, 0), f(10, 0));
        assert!([-1.0, 1.0].contains(&f(10, 0)) && [-1.0, 1.0].contains(&f(3, 1)));
        let vals: Vec<f32> = (0..20).map(|g| f(g, 0)).collect();
        assert!(vals.iter().any(|&v| v > 0.0) && vals.iter().any(|&v| v < 0.0), "both signs occur");
    }

    #[test]
    fn multilayer_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.layers = 4;
        cfg.multi_layer = true;
        cfg.trials = 600;
        assert_eq!(train_eprop(&cfg), train_eprop(&cfg));
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn rate_reg_depth_wall() {
        // Does keeping every layer alive push the depth-20 wall (doc's ceiling ~485)? Worst-seed held-out,
        // multi-layer, trial length scaled to depth, rate_reg off vs a c_reg sweep.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let depth = 20usize;
        // c_reg ~5 revives the deep layers (see rate_reg_revives_dead_layers); bracket that zone.
        for reg in [0.0f32, 5.0, 20.0] {
            let mut worst = 1000u64;
            for &s in &seeds {
                let mut c = RsnnConfig::demo();
                c.seed = s;
                c.task_seed = s;
                c.size = 16;
                c.layers = depth;
                c.multi_layer = true;
                c.trials = 1500;
                c.present_waves = depth; // scale trial length to depth
                c.read_waves = depth;
                c.delay = 4;
                c.rate_reg = reg;
                c.rate_target_permille = 100;
                let acc = train_eprop(&c);
                eprintln!("depth {depth} rate_reg {reg} seed {s:#x}  {acc}");
                worst = worst.min(acc);
            }
            eprintln!("depth {depth} rate_reg {reg}: WORST {worst}");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn rate_reg_revives_dead_layers() {
        // Per-layer firing rate of a TRAINED deep net, rate_reg off vs on. Off: deep layers dead (~0).
        // On: they should fire near the target through the full depth (liveness climbed the stack).
        // The reg term c_reg·(r_j−r_target) must be comparable to the task signal L_j^task ~ O(1), so
        // c_reg ~ 10s (r_target = 0.1); a small c_reg is negligible.
        for reg in [0.0f32, 5.0, 20.0] {
            let mut c = RsnnConfig::demo();
            c.seed = 0xE9_0B_0A17;
            c.task_seed = 0xE9_0B_0A17;
            c.size = 16;
            c.layers = 20; // match the depth-wall config, so revival and the accuracy null are the same net
            c.multi_layer = true;
            c.trials = 800;
            c.present_waves = 20;
            c.read_waves = 20;
            c.delay = 4;
            c.rate_reg = reg;
            c.rate_target_permille = 100;
            let (mut net, _w) = train_eprop_inner(&c);
            let rates = net.measure_layer_rates(
                c.calib.warmup,
                c.calib.waves,
                &random_l0_input(c.seed ^ 0xE9, c.size, c.calib_fraction_q16),
            );
            let r2: Vec<f64> = rates.iter().map(|x| (x * 100.0).round() / 100.0).collect();
            eprintln!("rate_reg {reg}: per-layer rates {r2:?}");
        }
    }

    #[test]
    fn rate_reg_path_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.layers = 4;
        cfg.multi_layer = true;
        cfg.trials = 200;
        cfg.rate_reg = 0.5;
        cfg.rate_target_permille = 100;
        assert_eq!(train_eprop(&cfg), train_eprop(&cfg));
    }

    #[test]
    fn eprop_hidden_learns_reliably() {
        // A TOP-layer readout on a FIXED reservoir is seed-fragile (the class doesn't reliably propagate up).
        // e-prop on the hidden L1→L2 weights should shape the reservoir so the top layer is reliably
        // separable across seeds — training WEIGHTS (unlike thresholds) generalizes.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111];
        let mut worst = 1000u64;
        for &s in &seeds {
            let mut base = RsnnConfig::demo();
            base.seed = s;
            base.task_seed = s;
            let mut nohid = base.clone();
            nohid.hidden_lr = 0.0;
            let baseline = train_eprop(&nohid);
            let eprop = train_eprop(&base);
            eprintln!("seed {s:#x}  fixed-reservoir top-readout {baseline}  +e-prop {eprop}");
            worst = worst.min(eprop);
        }
        assert!(worst > 600, "e-prop hidden weight training is reliable across seeds (worst {worst})");
    }

    #[test]
    fn readout_learns_and_generalizes() {
        let test = train_readout(&RsnnConfig::demo());
        eprintln!("readout held-out {test}");
        assert!(test > 650, "trained readout on the reservoir generalizes: {test}");
    }

    #[test]
    fn readout_is_seed_robust() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111];
        let mut worst = 1000u64;
        for &s in &seeds {
            let mut cfg = RsnnConfig::demo();
            cfg.seed = s;
            cfg.task_seed = s;
            let test = train_readout(&cfg);
            eprintln!("seed {s:#x} held-out {test}");
            worst = worst.min(test);
        }
        assert!(worst > 600, "worst seed still learns (reliable, unlike threshold-only): {worst}");
    }
}
