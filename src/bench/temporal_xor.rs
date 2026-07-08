//! `temporal_xor` — the Tier-1 temporal XOR task `y(t) = u(t) ⊕ u(t-τ)`, swept over `τ`. A thresholded
//! ridge readout on the reservoir state classifies the (non-linearly-separable) XOR; accuracy vs `τ`
//! tests whether ALIF's nonlinearity buys nonlinear temporal computation. Reuses `bench::stream`.

use crate::bench::readout::RidgeReadout;
use crate::bench::stream::{self, StreamParams};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::network::Network;

/// XOR of two bits held as f64 `0.0`/`1.0`.
fn xor(a: f64, b: f64) -> f64 {
    ((a != 0.0) ^ (b != 0.0)) as u8 as f64
}

/// Configuration for the temporal XOR experiment.
#[derive(Clone, Debug)]
pub struct XorConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub baseline_init: i16,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub inhibitor_ratio: u32, // Q16 inhibitory fraction — inhibition sharply improves XOR separation
    pub bit_seed: u64,
    pub stream_density_q16: u32,
    pub bin_waves: usize,
    pub warmup_bins: usize,
    pub collect_bins: usize,
    pub taus: Vec<usize>,
    pub lambda: f64,
    pub train_frac_permille: u64,
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl XorConfig {
    pub fn demo() -> XorConfig {
        let seed = 0x0A17_C0DE;
        XorConfig {
            seed,
            size: 8,
            layers: 3,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
            inhibitor_ratio: 9830, // ~0.15
            bit_seed: seed ^ 0xB17,
            stream_density_q16: 20000,
            bin_waves: 3,
            warmup_bins: 100,
            collect_bins: 700,
            taus: vec![1, 2, 4, 8],
            lambda: 1.0,
            train_frac_permille: 700,
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
        }
    }

    fn stream_params(&self) -> StreamParams {
        StreamParams {
            seed: self.seed,
            size: self.size,
            stream_density_q16: self.stream_density_q16,
            bit_seed: self.bit_seed,
            bin_waves: self.bin_waves,
            warmup_bins: self.warmup_bins,
            collect_bins: self.collect_bins,
        }
    }
}

/// XOR classification accuracy (permille) at each `τ`, for one variant.
#[derive(Clone, Debug, PartialEq)]
pub struct XorCurve {
    pub taus: Vec<usize>,
    pub accuracy_permille: Vec<u64>,
}

/// Build+calibrate one variant, stream, and fit a thresholded ridge classifier per `τ` for
/// `u(t) ⊕ u(t-τ)`. `adapt_bump` selects ALIF (>0) vs LIF (0); `recurrent` selects the topology.
pub fn temporal_xor(cfg: &XorConfig, adapt_bump: i16, recurrent: bool) -> XorCurve {
    let mut net = Network::new(stream::engine_config(
        cfg.seed, cfg.size, cfg.layers, cfg.baseline_init, adapt_bump, cfg.adapt_decay,
        cfg.inhibitor_ratio, recurrent,
    ));
    let input = random_l0_input(cfg.seed ^ 0x0AB1, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);
    let (xs, us) = stream::collect_states(&mut net, &cfg.stream_params());

    let n = xs.len();
    let tau_max = *cfg.taus.iter().max().unwrap();
    let split = (n as u64 * cfg.train_frac_permille / 1000) as usize;
    // Same design matrix for every τ; rows [tau_max, split) train, [split, n) test.
    let x_train: Vec<Vec<f64>> = xs[tau_max..split].to_vec();
    let x_test: Vec<Vec<f64>> = xs[split..n].to_vec();
    let ridge = RidgeReadout::fit(&x_train, cfg.lambda);

    let mut accuracy_permille = Vec::with_capacity(cfg.taus.len());
    for &tau in &cfg.taus {
        let y_train: Vec<f64> = (tau_max..split).map(|i| xor(us[i], us[i - tau])).collect();
        let w = ridge.weights(&x_train, &y_train);
        let pred = RidgeReadout::predict(&x_test, &w);
        let mut correct = 0usize;
        for (j, i) in (split..n).enumerate() {
            let phat = if pred[j] >= 0.5 { 1.0 } else { 0.0 };
            if phat == xor(us[i], us[i - tau]) {
                correct += 1;
            }
        }
        let acc = if x_test.is_empty() { 0 } else { (correct as u64 * 1000) / x_test.len() as u64 };
        accuracy_permille.push(acc);
    }
    XorCurve { taus: cfg.taus.clone(), accuracy_permille }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_target_is_correct() {
        assert_eq!(xor(0.0, 0.0), 0.0);
        assert_eq!(xor(1.0, 0.0), 1.0);
        assert_eq!(xor(0.0, 1.0), 1.0);
        assert_eq!(xor(1.0, 1.0), 0.0);
    }

    #[test]
    fn temporal_xor_is_deterministic() {
        let cfg = XorConfig::demo();
        let a = temporal_xor(&cfg, cfg.adapt_bump, true);
        let b = temporal_xor(&cfg, cfg.adapt_bump, true);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    #[test]
    fn xor_solvable_above_chance_at_small_tau() {
        // Sanity: the feed-forward reservoir (with inhibition) + linear readout separates XOR at the
        // smallest lag, well above chance (500). XOR is not linearly separable in the raw inputs, so
        // this confirms the reservoir provides the nonlinear features and the readout works. (An
        // architecture sweep showed inhibition/sparsity are what make XOR clearly solvable; the dense
        // all-excitatory topology the floored leak favors is comparatively poor for it.)
        let cfg = XorConfig::demo();
        let lif = temporal_xor(&cfg, 0, false);
        assert!(
            lif.accuracy_permille[0] > 700,
            "feed-forward reservoir should solve tau=1 XOR well above chance (LIF {})",
            lif.accuracy_permille[0]
        );
    }

    // Finding (robust across an architecture sweep — width, depth, refractory, density, inhibition):
    // LIF solves temporal XOR (nonlinear temporal computation); ALIF stays near chance. Adaptation
    // does NOT buy nonlinear computation. Combined with MC (linear echo: LIF) and store-recall
    // (held-category: ALIF), this pins ALIF's benefit to *held-category memory across a delay* only.
    #[test]
    fn temporal_xor_lif_beats_alif() {
        let cfg = XorConfig::demo();
        let lif = temporal_xor(&cfg, 0, false);
        let alif = temporal_xor(&cfg, cfg.adapt_bump, false);
        eprintln!("ff  tau {:?}  LIF {:?}  ALIF {:?}", cfg.taus, lif.accuracy_permille, alif.accuracy_permille);
        assert!(
            lif.accuracy_permille[0] > alif.accuracy_permille[0] + 100,
            "LIF should solve XOR while ALIF stays near chance (LIF {} vs ALIF {})",
            lif.accuracy_permille[0],
            alif.accuracy_permille[0]
        );
    }
}
