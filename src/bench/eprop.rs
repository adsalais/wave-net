//! `eprop` — a gradient-free, e-prop-like learning rule (v1): per-neuron threshold updates driven by a
//! global reward × a per-neuron eligibility trace, on a K=2 held-category task with spiking output
//! neurons. Trains thresholds to beat a frozen-threshold control. Reuses `store_recall`'s cue/probe.

use crate::bench::store_recall::{cue_realization, probe_pattern};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::Config;
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix};
use std::sync::{Arc, Mutex};

/// Reward-prediction-error tracker: returns `R − R̄` and updates the running mean `R̄` (EMA).
struct RewardTracker {
    mean: f64,
    rate: f64,
}

impl RewardTracker {
    fn new(rate: f64) -> RewardTracker {
        RewardTracker { mean: 0.0, rate }
    }
    /// Signal for this reward (before absorbing it), then update the mean.
    fn step(&mut self, r: f64) -> f64 {
        let s = r - self.mean;
        self.mean += self.rate * s;
        s
    }
}

/// Configuration for the e-prop learning experiment.
#[derive(Clone, Debug)]
pub struct EpropConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub baseline_init: i16,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub k: usize, // classes = output neurons
    pub present_waves: usize,
    pub delay: usize,
    pub read_waves: usize,
    pub base_q16: u32,
    pub keep_q16: u32,
    pub noise_q16: u32,
    pub probe_q16: u32,
    pub trials: usize,
    pub block: usize,     // accuracy-curve window
    pub reward_rate: f64, // EMA rate for R̄
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
    pub readout: bool, // V2a: append a non-spiking readout layer and score from its potentials
}

impl EpropConfig {
    pub fn demo() -> EpropConfig {
        let seed = 0xE9_0B_0A17;
        EpropConfig {
            seed,
            size: 8,
            layers: 3,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
            k: 2,
            present_waves: 6,
            delay: 4,
            read_waves: 6,
            base_q16: 18000,
            keep_q16: 60000,
            noise_q16: 1500,
            probe_q16: 20000,
            trials: 2400,
            block: 200,
            reward_rate: 0.02,
            calib: CalibrateParams {
                warmup: 16,
                waves: 48,
                max_steps: 24,
                refine_passes: 3,
                ..CalibrateParams::default()
            },
            calib_fraction_q16: 20000,
            readout: false,
        }
    }

    /// V2a engine config: computational layers + an appended non-spiking readout layer (empty
    /// topology sink). Build the resulting `Config` with `Network::new_with_readout`.
    fn engine_config_readout(&self) -> Config {
        use crate::wave_net::config::LayerConfig;
        use crate::wave_net::synapse::TopologyLevel;
        let comp = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: 16 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: self.baseline_init,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        let readout = LayerConfig { topology: vec![], ..comp.clone() };
        let mut layers = vec![comp; self.layers];
        layers.push(readout);
        Config { seed: self.seed, size: self.size, layers }
    }

    fn engine_config(&self) -> Config {
        // Dense feed-forward ALIF (held memory needs dense fan-out; feed-forward isolates adaptation).
        crate::bench::stream::engine_config(
            self.seed, self.size, self.layers, self.baseline_init, self.adapt_bump, self.adapt_decay, 0, false,
        )
    }
}

/// f64 shadow of the computational-layer thresholds (`1..L`), read from the current engine thresholds.
fn read_shadow(net: &Network) -> Vec<Vec<f64>> {
    let l = net.layer_count();
    (1..l).map(|z| net.layer_thresholds(z).iter().map(|&t| t as f64).collect()).collect()
}

/// Write the rounded, clamped shadow back to the engine's integer thresholds (`1..L`).
fn write_thresholds(net: &Network, shadow: &[Vec<f64>]) {
    let l = net.layer_count();
    for z in 1..l {
        let s = &shadow[z - 1];
        net.with_layer_mut(z, |layer| {
            for (i, t) in layer.threshold.iter_mut().enumerate() {
                *t = s[i].round().clamp(1.0, i16::MAX as f64) as i16;
            }
        });
    }
}

/// Run one trial (reset → present cue → delay → probe) accumulating per-neuron spike counts over the
/// whole trial for the computational layers `1..L`. This is the per-neuron eligibility trace.
fn trial_eligibility(net: &mut Network, cfg: &EpropConfig, class: usize, trial: usize) -> Vec<Vec<u32>> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![vec![0u32; ls]; l]));
    for z in 1..l {
        let c = counts.clone();
        net.on_layer(
            z,
            Box::new(move |_w: usize, fired: &[u32]| {
                let mut g = c.lock().unwrap();
                for &loc in fired {
                    g[z][loc as usize] += 1;
                }
            }),
        );
    }
    net.reset_state();
    for w in 0..cfg.present_waves {
        let sites = cue_realization(cfg.seed, cfg.size, class, trial, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
        net.wave(&sites);
    }
    for _ in 0..cfg.delay {
        net.wave(&[]);
    }
    let probe = probe_pattern(cfg.seed, cfg.size, cfg.probe_q16);
    for _ in 0..cfg.read_waves {
        net.wave(&probe);
    }
    net.clear_listeners();
    let g = counts.lock().unwrap();
    (1..l).map(|z| g[z].clone()).collect()
}

/// Windowed accuracy trajectory over training.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnCurve {
    pub accuracy_permille: Vec<u64>,
}

/// Deterministic class for trial `t`.
fn pick_class(seed: u64, t: usize, k: usize) -> usize {
    (mix(key(seed, t as u32, 0, 0, 41)) % k as u64) as usize
}

/// Class scores from a readout layer's integrated potentials: K contiguous population sums.
fn readout_scores(net: &Network, readout_z: usize, k: usize) -> Vec<i64> {
    let ls = (net.size() * net.size()) as usize;
    let group = (ls / k).max(1);
    (0..k)
        .map(|c| ((c * group)..((c + 1) * group).min(ls)).map(|i| net.potential(readout_z, i) as i64).sum())
        .collect()
}

/// Train per-neuron thresholds by global-reward × eligibility. `lr = 0.0` freezes the thresholds
/// (the control). Returns block-windowed accuracy over training.
pub fn train(cfg: &EpropConfig, lr: f64) -> LearnCurve {
    let mut net = if cfg.readout {
        Network::new_with_readout(cfg.engine_config_readout())
    } else {
        Network::new(cfg.engine_config())
    };
    let input = random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);

    let mut shadow = read_shadow(&net);
    let mut rt = RewardTracker::new(cfg.reward_rate);
    let mut outcomes: Vec<bool> = Vec::with_capacity(cfg.trials);

    for t in 0..cfg.trials {
        let class = pick_class(cfg.seed, t, cfg.k);
        let elig = trial_eligibility(&mut net, cfg, class, t);
        // Population output coding: split the output surface into K contiguous groups; class score c is
        // the group total. V1 sums top-layer spike counts; V2a sums the readout layer's potentials.
        // (Single output neurons are too often silent to carry a signal.)
        let outs: Vec<i64> = if cfg.readout {
            readout_scores(&net, net.layer_count() - 1, cfg.k)
        } else {
            let top = &elig[elig.len() - 1];
            let group = (top.len() / cfg.k).max(1);
            (0..cfg.k).map(|c| top[c * group..(c + 1) * group].iter().map(|&x| x as i64).sum()).collect()
        };
        let pred = (0..cfg.k).max_by_key(|&i| outs[i]).unwrap();
        outcomes.push(pred == class);

        let correct = outs[class] as f64;
        let best_rival = (0..cfg.k).filter(|&i| i != class).map(|i| outs[i]).max().unwrap_or(0) as f64;
        let signal = rt.step(correct - best_rival);

        if lr != 0.0 {
            for (zi, layer_e) in elig.iter().enumerate() {
                for (i, &e) in layer_e.iter().enumerate() {
                    shadow[zi][i] += -lr * signal * e as f64;
                }
            }
            write_thresholds(&net, &shadow);
        }
    }

    let block = cfg.block.max(1);
    let accuracy_permille = outcomes
        .chunks(block)
        .map(|c| (c.iter().filter(|&&b| b).count() as u64 * 1000) / c.len() as u64)
        .collect();
    LearnCurve { accuracy_permille }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    fn tiny_net() -> Network {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 2, count: 6 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 50,
            adapt_bump: 8,
            adapt_decay: 6,
        };
        Network::new(Config { seed: 3, size: 4, layers: vec![layer; 3] })
    }

    #[test]
    fn reward_prediction_error_centers() {
        let mut rt = RewardTracker::new(0.1);
        let first = rt.step(5.0);
        assert!((first - 5.0).abs() < 1e-9, "first signal is R − 0");
        let mut last = first;
        for _ in 0..200 {
            last = rt.step(5.0);
        }
        assert!(last.abs() < 0.05, "constant reward should center to ~0, got {last}");
    }

    #[test]
    fn shadow_write_roundtrips_thresholds() {
        let net = tiny_net();
        let mut shadow = read_shadow(&net);
        let before = net.layer_thresholds(1)[0];
        shadow[0][0] += 0.6;
        write_thresholds(&net, &shadow);
        assert_eq!(net.layer_thresholds(1)[0], before + 1);
        // a +0.4 sub-unit nudge rounds to no change.
        let before1 = net.layer_thresholds(1)[1];
        shadow[0][1] += 0.4;
        write_thresholds(&net, &shadow);
        assert_eq!(net.layer_thresholds(1)[1], before1);
    }

    #[test]
    fn trial_eligibility_shape_and_determinism() {
        let cfg = EpropConfig::demo();
        let mut net = Network::new(cfg.engine_config());
        let e1 = trial_eligibility(&mut net, &cfg, 0, 0);
        let e2 = trial_eligibility(&mut net, &cfg, 0, 0);
        assert_eq!(e1.len(), cfg.layers - 1); // computational layers 1..L
        assert_eq!(e1[0].len(), (cfg.size * cfg.size) as usize);
        assert_eq!(e1, e2, "a trial (reset each time) must be deterministic");
    }

    /// Mean accuracy over the late (second) half of the training curve — a stable summary of where
    /// learning settled, robust to block-to-block noise.
    fn late_mean(curve: &[u64]) -> u64 {
        let h = curve.len() / 2;
        curve[h..].iter().sum::<u64>() / (curve.len() - h).max(1) as u64
    }

    #[test]
    fn eprop_readout_is_deterministic() {
        let mut cfg = EpropConfig::demo();
        cfg.readout = true;
        let a = train(&cfg, 0.3);
        let b = train(&cfg, 0.3);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    // Finding: with a non-spiking readout (no trainable output layer) and a GLOBAL scalar reward,
    // learning is all-internal (feedback-alignment). The fixed ±1 readout projection does not separate
    // the classes, so (R − R̄) → 0 and no learning happens — at any lr. (V1's spiking, *trainable* output
    // populations are what let global reward work; the readout needs a richer per-output error signal,
    // i.e. broadcast-error alignment — V2b.) The readout *engine* works (see wave.rs); the *learning*
    // does not, and this documents that null.
    #[test]
    fn eprop_readout_global_reward_does_not_learn() {
        let mut cfg = EpropConfig::demo();
        cfg.readout = true;
        let learn = train(&cfg, 0.3);
        let frozen = train(&cfg, 0.0);
        eprintln!("readout learn  {:?}", learn.accuracy_permille);
        eprintln!("readout frozen {:?}", frozen.accuracy_permille);
        let ll = late_mean(&learn.accuracy_permille);
        let lf = late_mean(&frozen.accuracy_permille);
        let chance = 1000 / cfg.k as u64;
        assert!(ll < chance + 80, "global-reward readout stays near chance (no learning): {ll}");
        assert!((ll as i64 - lf as i64).abs() < 120, "readout learning ≈ frozen — no gap: {ll} vs {lf}");
    }

    #[test]
    fn eprop_is_deterministic() {
        let cfg = EpropConfig::demo();
        let a = train(&cfg, 0.3);
        let b = train(&cfg, 0.3);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    #[test]
    fn eprop_learns_and_beats_frozen_control() {
        let cfg = EpropConfig::demo();
        let learn = train(&cfg, 0.3);
        let frozen = train(&cfg, 0.0);
        eprintln!("learn  {:?}", learn.accuracy_permille);
        eprintln!("frozen {:?}", frozen.accuracy_permille);
        let ll = late_mean(&learn.accuracy_permille);
        let lf = late_mean(&frozen.accuracy_permille);
        let chance = 1000 / cfg.k as u64;
        assert!(ll > chance + 80, "learning late accuracy {ll} should be above chance {chance}");
        assert!(ll > lf + 150, "learning {ll} should beat the frozen control {lf}");
    }
}
