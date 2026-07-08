//! `memory_capacity` — the Tier-1 Memory Capacity metric. A binary i.i.d. bit stream is fed to the
//! reservoir in bins of `B` waves (continuously, no reset); per-bin spike counts form the state
//! `x(t)`, and a ridge readout reconstructs `u(t-k)` for each lag `k`. `MC = Σ_k r²_k`.

use crate::bench::readout::{record_response, RidgeReadout};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::{Config, LayerConfig};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix, TopologyLevel};

const P_BIT: u64 = 23; // input bit per timestep
const P_STREAM: u64 = 29; // fixed L0 pattern injected on a "1" bit

/// The i.i.d. input bit for timestep `t`.
fn bit(bit_seed: u64, t: usize) -> bool {
    (mix(key(bit_seed, t as u32, 0, 0, P_BIT)) & 1) == 1
}

/// The fixed L0 pattern injected whenever the bit is 1 (same every timestep).
fn stream_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
    let ls = size * size;
    (0..ls).filter(|&s| ((mix(key(seed, s, 0, 0, P_STREAM)) & 0xFFFF) as u32) < density_q16).collect()
}

/// Configuration for the Memory Capacity experiment.
#[derive(Clone, Debug)]
pub struct McConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub baseline_init: i16,
    pub adapt_bump: i16, // ALIF value; LIF passes 0
    pub adapt_decay: u8,
    pub bit_seed: u64,
    pub stream_density_q16: u32,
    pub bin_waves: usize,
    pub warmup_bins: usize,
    pub collect_bins: usize,
    pub k_lags: usize,
    pub lambda: f64,
    pub train_frac_permille: u64,
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl McConfig {
    /// Small, fast, deterministic config for the inline test.
    pub fn demo() -> McConfig {
        let seed = 0x3EC0_DE5;
        McConfig {
            seed,
            size: 8,
            layers: 3,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
            bit_seed: seed ^ 0xB17,
            stream_density_q16: 20000,
            bin_waves: 3,
            warmup_bins: 100,
            collect_bins: 700,
            k_lags: 20,
            lambda: 1.0,
            train_frac_permille: 700,
            calib: CalibrateParams {
                warmup: 16,
                waves: 48,
                max_steps: 24,
                refine_passes: 3,
                ..CalibrateParams::default()
            },
            calib_fraction_q16: 20000,
        }
    }

    /// Build the engine config. `recurrent` adds level 0 / -1 coupling; feed-forward is level +1
    /// only. Both use the dense drive the floored leak requires.
    pub fn engine_config(&self, adapt_bump: i16, recurrent: bool) -> Config {
        let mut topology = vec![TopologyLevel { level: 1, radius: 3, count: 16 }];
        if recurrent {
            topology.push(TopologyLevel { level: 0, radius: 1, count: 3 });
            topology.push(TopologyLevel { level: -1, radius: 1, count: 3 });
        }
        let layer = LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: self.baseline_init,
            adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        Config { seed: self.seed, size: self.size, layers: vec![layer; self.layers] }
    }
}

/// Drive the continuous bit stream and collect per-bin state rows (per-neuron spike counts over the
/// bin, layers 1..L, ++ a bias 1.0) and the bit sequence. Warmup bins advance the reservoir but are
/// not collected. No reset between bins.
///
/// Note: we read *spike counts*, the standard reservoir readout. An earlier experiment also exposed
/// the raw adaptation state to the readout; it did not help (it slightly hurt, via overfitting) —
/// adaptation is a slow low-pass trace, not the delayed echo MC rewards. See the spec's revision.
fn collect_states(net: &mut Network, cfg: &McConfig) -> (Vec<Vec<f64>>, Vec<f64>) {
    let pattern = stream_pattern(cfg.seed, cfg.size, cfg.stream_density_q16);
    for t in 0..cfg.warmup_bins {
        let on = bit(cfg.bit_seed, t);
        for _ in 0..cfg.bin_waves {
            net.wave(if on { &pattern } else { &[] });
        }
    }
    let mut xs = Vec::with_capacity(cfg.collect_bins);
    let mut us = Vec::with_capacity(cfg.collect_bins);
    for i in 0..cfg.collect_bins {
        let on = bit(cfg.bit_seed, cfg.warmup_bins + i);
        let pat = if on { pattern.clone() } else { Vec::new() };
        let counts = record_response(net, cfg.bin_waves, move |_w| pat.clone());
        let mut row: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        row.push(1.0); // bias
        xs.push(row);
        us.push(if on { 1.0 } else { 0.0 });
    }
    (xs, us)
}

/// The memory curve: `r2[k-1]` for lag `k = 1..=K`, and their sum.
#[derive(Clone, Debug, PartialEq)]
pub struct McCurve {
    pub r2: Vec<f64>,
    pub total: f64,
}

/// Squared Pearson correlation between prediction and target, clamped to [0,1].
fn r2(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mp = pred.iter().sum::<f64>() / n;
    let mt = target.iter().sum::<f64>() / n;
    let (mut cov, mut vp, mut vt) = (0.0, 0.0, 0.0);
    for (&p, &t) in pred.iter().zip(target) {
        cov += (p - mp) * (t - mt);
        vp += (p - mp) * (p - mp);
        vt += (t - mt) * (t - mt);
    }
    if vp <= 0.0 || vt <= 0.0 {
        return 0.0;
    }
    let r = cov / (vp.sqrt() * vt.sqrt());
    (r * r).clamp(0.0, 1.0)
}

/// Build+calibrate one variant, stream the reservoir, and fit a ridge readout per lag to reconstruct
/// `u(t-k)`. `adapt_bump` selects ALIF (>0) vs LIF (0); `recurrent` selects the topology.
pub fn memory_capacity(cfg: &McConfig, adapt_bump: i16, recurrent: bool) -> McCurve {
    let mut net = Network::new(cfg.engine_config(adapt_bump, recurrent));
    let input = random_l0_input(cfg.seed ^ 0x3EC0, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);
    let (xs, us) = collect_states(&mut net, cfg);

    let n = xs.len();
    let k = cfg.k_lags;
    let split = (n as u64 * cfg.train_frac_permille / 1000) as usize;
    // Same design matrix for every lag; rows [k, split) train, [split, n) test.
    let x_train: Vec<Vec<f64>> = xs[k..split].to_vec();
    let x_test: Vec<Vec<f64>> = xs[split..n].to_vec();
    let ridge = RidgeReadout::fit(&x_train, cfg.lambda);

    let mut r2s = Vec::with_capacity(k);
    for lag in 1..=k {
        let y_train: Vec<f64> = (k..split).map(|i| us[i - lag]).collect();
        let w = ridge.weights(&x_train, &y_train);
        let pred = RidgeReadout::predict(&x_test, &w);
        let y_test: Vec<f64> = (split..n).map(|i| us[i - lag]).collect();
        r2s.push(r2(&pred, &y_test));
    }
    let total = r2s.iter().sum();
    McCurve { r2: r2s, total }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_stream_is_deterministic_and_balanced() {
        let n = 2000;
        let ones = (0..n).filter(|&t| bit(42, t)).count();
        assert_eq!(
            (0..n).map(|t| bit(42, t)).collect::<Vec<_>>(),
            (0..n).map(|t| bit(42, t)).collect::<Vec<_>>()
        );
        let frac = ones as f64 / n as f64;
        assert!((frac - 0.5).abs() < 0.05, "bit stream not ~balanced: {frac}");
    }

    #[test]
    fn collect_states_shape_and_determinism() {
        let cfg = McConfig::demo();
        let build = || {
            let mut net = Network::new(cfg.engine_config(cfg.adapt_bump, true));
            let input = random_l0_input(cfg.seed ^ 0x3EC0, cfg.size, cfg.calib_fraction_q16);
            net.calibrate(&cfg.calib, &input);
            collect_states(&mut net, &cfg)
        };
        let (xs, us) = build();
        assert_eq!(xs.len(), cfg.collect_bins);
        assert_eq!(us.len(), cfg.collect_bins);
        // per-neuron spike counts over the computational layers, + bias
        assert_eq!(xs[0].len(), (cfg.layers - 1) * (cfg.size * cfg.size) as usize + 1);
        let (xs2, _) = build();
        assert_eq!(xs, xs2, "state collection must be deterministic");
    }

    #[test]
    fn memory_capacity_is_deterministic() {
        let cfg = McConfig::demo();
        let a = memory_capacity(&cfg, cfg.adapt_bump, true);
        let b = memory_capacity(&cfg, cfg.adapt_bump, true);
        assert_eq!(a.r2, b.r2);
        assert_eq!(a.total, b.total);
    }

    // Finding: MC measures delayed *linear echo*. LIF's fading spike echo reconstructs recent bits
    // well; ALIF's adaptation is a slow low-pass trace that trades echo for held/nonlinear memory
    // (the memory that won store-recall). So plain LIF has substantially higher MC than ALIF — in
    // both regimes. (Complementary to Spec 1, where ALIF beat LIF on held-cue memory.)
    #[test]
    fn memory_capacity_lif_echo_beats_alif() {
        let cfg = McConfig::demo();
        let lif_ff = memory_capacity(&cfg, 0, false);
        let alif_ff = memory_capacity(&cfg, cfg.adapt_bump, false);
        eprintln!("ff  LIF {:.3}  ALIF {:.3}", lif_ff.total, alif_ff.total);
        // LIF echoes the recent bit (one-hop delay) near-perfectly at lag 1.
        assert!(lif_ff.r2[0] > 0.9, "LIF should echo the recent bit at lag 1, got {}", lif_ff.r2[0]);
        assert!(
            lif_ff.total > alif_ff.total + 0.5,
            "LIF MC {} should exceed ALIF MC {} — adaptation trades echo for held memory",
            lif_ff.total,
            alif_ff.total
        );

        let lif_rec = memory_capacity(&cfg, 0, true);
        let alif_rec = memory_capacity(&cfg, cfg.adapt_bump, true);
        eprintln!("rec LIF {:.3}  ALIF {:.3}", lif_rec.total, alif_rec.total);
        assert!(lif_rec.total > 1.0, "recurrent reservoir should hold >1 bit of linear memory");
        assert!(lif_rec.total > alif_rec.total, "LIF MC should exceed ALIF MC in the recurrent regime too");
    }
}
