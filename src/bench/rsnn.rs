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

    /// Temporal-XOR net: L0 input transducer → L1 recurrent hidden (level 0), readout reads L1.
    /// `rec_count == 0` gives L1 an empty topology — the feed-forward baseline.
    fn engine_config_xor(&self) -> Config {
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        let l1_topo = if self.rec_count > 0 {
            vec![TopologyLevel { level: 0, radius: self.rec_radius, count: self.rec_count }]
        } else {
            vec![]
        };
        let l1 = LayerConfig { topology: l1_topo, ..l0.clone() };
        Config { seed: self.seed, size: self.size, layers: vec![l0, l1] }
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
fn xor_trial_layers(net: &mut Network, cfg: &RsnnConfig, a: usize, b: usize, trial: usize) -> (Vec<f32>, Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<i16>>>) {
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 1..l {
        let r = rec.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| r.lock().unwrap()[z].push(fired.to_vec())));
    }
    net.reset_state();
    let mut pots: Vec<Vec<Vec<i16>>> = vec![Vec::new(); l];
    let record = |net: &Network, pots: &mut Vec<Vec<Vec<i16>>>| {
        for z in 1..l {
            pots[z].push(net.layer_decide_potential(z));
        }
    };
    for (class, phase) in [(a, 0usize), (b, 1)] {
        // (delay sits between the two cue presentations)
        if phase == 1 {
            for _ in 0..cfg.delay {
                net.wave(&[]);
                record(net, &mut pots);
            }
        }
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * 2 + phase, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
            record(net, &mut pots);
        }
    }
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
        record(net, &mut pots);
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
    (act, spikes, pots)
}

const P_DFA: u64 = 61; // fixed random Direct-Feedback-Alignment weights

/// Fixed random ±1 DFA feedback weight for (target neuron `neuron_global`, output class `class`) —
/// deterministic, hash-derived, stored-free. Broadcasts the output error to a deep layer.
fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
    if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
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
fn pick_ab(seed: u64, t: usize) -> (usize, usize) {
    let a = (mix(key(seed, t as u32, 0, 0, 51)) & 1) as usize;
    let b = (mix(key(seed, t as u32, 0, 0, 53)) & 1) as usize;
    (a, b)
}

/// reset → present cue(a) → delay → present cue(b) → read (silent). Records L1 per-wave fired-sets and
/// returns (read-window L1 spike counts, per-wave L1 fired-sets over the whole trial).
fn xor_trial(net: &mut Network, cfg: &RsnnConfig, a: usize, b: usize, trial: usize) -> (Vec<f32>, Vec<Vec<u32>>) {
    let ls = (cfg.size * cfg.size) as usize;
    let rec: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let r = rec.clone();
        net.on_layer(1, Box::new(move |_w, fired: &[u32]| r.lock().unwrap().push(fired.to_vec())));
    }
    net.reset_state();
    let present = |net: &mut Network, class: usize, phase: usize| {
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * 2 + phase, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
        }
    };
    present(net, a, 0);
    for _ in 0..cfg.delay {
        net.wave(&[]);
    }
    present(net, b, 1);
    let read_start = rec.lock().unwrap().len();
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
    }
    net.clear_listeners();
    let waves = rec.lock().unwrap().clone();
    let mut act = vec![0f32; ls];
    for wave in waves.iter().skip(read_start) {
        for &loc in wave {
            act[loc as usize] += 1.0;
        }
    }
    (act, waves)
}

/// Temporal e-prop on the L1 level-0 recurrent weights. Builds a decaying presynaptic trace per neuron
/// over the recorded waves, correlates it with postsynaptic spikes (`e_ij = Σ_t pre_trace_i(t)·fired_j(t)`),
/// and updates the stored weights via the symmetric-feedback learning signal.
fn recurrent_update(net: &mut Network, cfg: &RsnnConfig, w: &[Vec<f32>], err: &[f32], waves: &[Vec<u32>]) {
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
    let l_sig: Vec<f32> = (0..ls).map(|j| (0..2).map(|c| w[c][j] * err[c]).sum()).collect();
    net.with_layer_mut(1, |l1| {
        for i in 0..ls {
            let sg = (ls + i) as u32; // L1 global id = layer 1 * ls + i
            for kk in 0..up {
                let j = target_of(cfg.seed, sg, i as u32, 0, kk as u32, cfg.rec_radius, cfg.size) as usize;
                let mut e = 0f32;
                for t in 0..ttot {
                    e += tr[t][i] * fired[t][j];
                }
                l1.out_shadow[i * up + kk] += -cfg.hidden_lr * l_sig[j] * e;
            }
        }
        for (wq, s) in l1.out_weights.iter_mut().zip(&l1.out_shadow) {
            *wq = s.round().clamp(-127.0, 127.0) as i8;
        }
    });
}

/// Train a readout on L1's read-window activity for temporal XOR; with `rec_count > 0` also trains the L1
/// level-0 recurrent weights. Returns held-out test accuracy permille.
pub fn train_xor(cfg: &RsnnConfig) -> u64 {
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
        let (act, waves) = xor_trial(&mut net, cfg, a, b, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.rec_count > 0 {
            recurrent_update(&mut net, cfg, &w, &err, &waves);
        }
    }
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let (act, _) = xor_trial(&mut net, cfg, a, b, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == (a ^ b) {
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
    for t in 0..cfg.trials {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let label = a ^ b;
        let (act, spikes, pots) = xor_trial_layers(&mut net, cfg, a, b, t);
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
        // postsynaptic factor ψ: spike-time (fired) or sub-threshold ramp clamp(decide_potential/θ, 0, 1)
        let mut post = vec![vec![vec![0f32; ls]; ttot]; l];
        for z in 1..l {
            for tt in 0..ttot {
                for j in 0..ls {
                    post[z][tt][j] = if cfg.subthreshold_psi {
                        (pots[z][tt][j] as f32 / theta[z][j]).clamp(0.0, 1.0)
                    } else {
                        fired[z][tt][j]
                    };
                }
            }
        }
        let l_sig = |tz: usize, j: usize| -> f32 {
            (0..2)
                .map(|c| {
                    let bb = if tz == top { w[c][j] } else { dfa_weight(cfg.seed, (tz * ls + j) as u32, c) };
                    bb * err[c]
                })
                .sum()
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
                        let mut e = 0f32;
                        for tt in 0..ttot {
                            e += pretr[z][tt][i] * post[tz][tt][j];
                        }
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
        let (act, _, _) = xor_trial_layers(&mut net, cfg, a, b, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == (a ^ b) {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (_, spikes, pots) = xor_trial_layers(&mut net, &cfg, 1, 0, 0);
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
        for reg in [0.0f32, 5.0, 20.0, 50.0] {
            let mut c = RsnnConfig::demo();
            c.seed = 0xE9_0B_0A17;
            c.task_seed = 0xE9_0B_0A17;
            c.size = 16;
            c.layers = 16;
            c.multi_layer = true;
            c.trials = 800;
            c.present_waves = 16;
            c.read_waves = 16;
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
