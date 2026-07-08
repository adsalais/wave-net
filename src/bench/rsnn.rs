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
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
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
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
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

/// Train a TOP-layer readout AND (via e-prop) the last hidden layer's level+1 weights. Returns held-out
/// test accuracy permille. With `hidden_lr = 0` this is the fixed-reservoir top-layer readout (the fragile
/// baseline); with `hidden_lr > 0`, e-prop shapes the reservoir so the top layer becomes separable.
pub fn train_eprop(cfg: &RsnnConfig) -> u64 {
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
                let l_sig: Vec<f32> = (0..ls)
                    .map(|j| {
                        (0..cfg.k)
                            .map(|c| {
                                let b = if tgt == top { w[c][j] } else { dfa_weight(cfg.seed, (tgt * ls + j) as u32, c) };
                                b * err[c]
                            })
                            .sum()
                    })
                    .collect();
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
