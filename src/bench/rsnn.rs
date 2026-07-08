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
fn target_of(seed: u64, source_global: u32, src_local: u32, k: u32, radius: u32, size: u32) -> u32 {
    let (sx, sy) = xy_of(src_local, size);
    let h = mix(key(seed, source_global, 1, k, P_TARGET));
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
    let z_below = l - 2;
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
            // symmetric-feedback learning signal per top neuron, and factored eligibility pre_i·psi_j
            let l_sig: Vec<f32> = (0..ls).map(|j| (0..cfg.k).map(|c| w[c][j] * err[c]).sum()).collect();
            let pre = net.with_layer_mut(z_below, |x| x.elig_pre.clone());
            let psi = net.with_layer_mut(top, |x| x.elig_post.clone());
            net.with_layer_mut(z_below, |l1| {
                for i in 0..ls {
                    let pre_i = pre[i] as f32;
                    if pre_i == 0.0 {
                        continue;
                    }
                    let sg = (z_below * ls + i) as u32;
                    for kk in 0..up {
                        let j = target_of(cfg.seed, sg, i as u32, kk as u32, cfg.up_radius, cfg.size) as usize;
                        l1.out_shadow[i * up + kk] += -cfg.hidden_lr * l_sig[j] * pre_i * psi[j] as f32;
                    }
                }
                for (wq, s) in l1.out_weights.iter_mut().zip(&l1.out_shadow) {
                    *wq = s.round().clamp(-127.0, 127.0) as i8;
                }
            });
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

#[cfg(test)]
mod tests {
    use super::*;

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
