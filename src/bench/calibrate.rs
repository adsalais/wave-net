//! Firing-rate calibration — **downgraded to a bench tool.** Lower per-layer thresholds so each layer
//! fires near a target rate on a driven input (bottom-up, then a few global refine passes). It is the
//! brittle-but-general fallback init: `wave_net::Network::critical_init` is the default for feed-forward
//! configs, and calibration remains here for recurrent/side-car configs until a recurrent-σ init exists.
//! A free function over the engine's public/crate API (no longer a `Network` method).

use crate::wave_net::network::Network;

#[derive(Clone, Copy, Debug)]
pub struct CalibrateParams {
    pub target_permille: u64, // desired per-layer firing rate, e.g. 100 = 10%
    pub tol_permille: u64,    // stop a layer when |rate - target| <= tol
    pub warmup: usize,        // waves discarded per measurement
    pub waves: usize,         // waves counted per measurement
    pub max_steps: usize,     // max adjust steps per layer (bottom-up)
    pub refine_passes: usize, // global all-layers passes after bottom-up
    pub step_shift: u32,      // geometric step = max_threshold >> step_shift
}

impl Default for CalibrateParams {
    fn default() -> CalibrateParams {
        CalibrateParams {
            target_permille: 100,
            tol_permille: 20,
            warmup: 32,
            waves: 128,
            max_steps: 48,
            refine_passes: 4,
            step_shift: 2,
        }
    }
}

/// Lower per-layer thresholds (layers 1..L; L0 is the input surface, left as-is) so each fires near
/// target on `input`. Mutates in place; preserves the caller's listeners.
pub fn calibrate(net: &mut Network, params: &CalibrateParams, input: &impl Fn(usize) -> Vec<u32>) {
    let l = net.layer_count();
    let target = params.target_permille as f64 / 1000.0;
    let tol = params.tol_permille as f64 / 1000.0;
    // Readout layers never fire, so their rate is always 0 — calibrating them just burns steps lowering
    // a threshold that can't change the rate. Skip them.
    let is_readout: Vec<bool> = (0..l).map(|z| net.with_layer_mut(z, |layer| layer.readout)).collect();

    // Phase 1 — bottom-up: fix each layer before moving up (its feeder is now firing).
    for z in 1..l {
        if is_readout[z] {
            continue;
        }
        for _ in 0..params.max_steps {
            let rates = net.measure_layer_rates(params.warmup, params.waves, input);
            let adjusted = net.with_layer_mut(z, |layer| layer.calibrate_step(rates[z], target, tol, params.step_shift));
            if !adjusted {
                break;
            }
        }
    }

    // Phase 2 — global refine: absorb the downward (level 0/-1) coupling.
    for _ in 0..params.refine_passes {
        let rates = net.measure_layer_rates(params.warmup, params.waves, input);
        let mut moved = false;
        for z in 1..l {
            if is_readout[z] {
                continue;
            }
            moved |= net.with_layer_mut(z, |layer| layer.calibrate_step(rates[z], target, tol, params.step_shift));
        }
        if !moved {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::critical_init::random_l0_input;
    use crate::wave_net::synapse::TopologyLevel;

    fn test_config() -> Config {
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 3, count: 16 },
                TopologyLevel { level: 0, radius: 1, count: 3 },
                TopologyLevel { level: -1, radius: 1, count: 3 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 64,
            baseline_init: 8,
            adapt_bump: 5,
            adapt_decay: 5,
        };
        Config { seed: 0x00C0_FFEE, size: 8, layers: vec![layer; 4] }
    }

    #[test]
    fn calibrate_settles_upper_layers() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(0xABC, 8, 20000);
        let params = CalibrateParams::default();
        let top = net.layer_count() - 1;
        let target = params.target_permille as f64 / 1000.0;
        calibrate(&mut net, &params, &input);
        let after = net.measure_layer_rates(params.warmup, params.waves, &input)[top];
        assert!(after > 0.0, "top should fire after calibration");
        assert!(after > target / 2.0 && after < target * 2.0, "top rate {after} not near {target}");
    }

    #[test]
    fn calibrate_moves_every_upper_layer_toward_target() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(7, 8, 20000);
        let params = CalibrateParams::default();
        let target = params.target_permille as f64 / 1000.0;
        let before: Vec<f64> = net.measure_layer_rates(params.warmup, params.waves, &input);
        calibrate(&mut net, &params, &input);
        let after: Vec<f64> = net.measure_layer_rates(params.warmup, params.waves, &input);
        for z in 1..net.layer_count() {
            let improved = (after[z] - target).abs() <= (before[z] - target).abs() + 1e-9;
            assert!(improved, "layer {z}: rate moved away from target ({} -> {})", before[z], after[z]);
        }
    }

    #[test]
    fn calibrate_hits_target_with_adaptation_live() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(42, 8, 20000);
        let params = CalibrateParams::default();
        let target = params.target_permille as f64 / 1000.0;
        calibrate(&mut net, &params, &input);
        let rates = net.measure_layer_rates(params.warmup, params.waves, &input);
        for z in 1..net.layer_count() {
            assert!(rates[z] > target / 3.0 && rates[z] < target * 3.0, "layer {z} self-regulated rate {} not near target {target}", rates[z]);
        }
    }

    #[test]
    fn calibrate_is_deterministic() {
        let input = random_l0_input(42, 8, 20000);
        let params = CalibrateParams::default();
        let run = || {
            let mut net = Network::new(test_config());
            calibrate(&mut net, &params, &input);
            (0..net.layer_count()).map(|z| net.layer_thresholds(z)).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn calibrate_skips_readout_layers() {
        let comp = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 120,
            adapt_bump: 0,
            adapt_decay: 6,
        };
        let readout = LayerConfig { topology: vec![], ..comp.clone() };
        let cfg = Config { seed: 5, size: 4, layers: vec![comp.clone(), comp.clone(), readout] };
        let mut net = Network::new_with_readout(cfg);
        let readout_z = net.layer_count() - 1;
        let readout_before = net.layer_thresholds(readout_z);
        let comp_before = net.layer_thresholds(1);
        let params = CalibrateParams { warmup: 8, waves: 24, max_steps: 12, refine_passes: 2, ..CalibrateParams::default() };
        calibrate(&mut net, &params, &random_l0_input(9, 4, 20000));
        assert_eq!(net.layer_thresholds(readout_z), readout_before, "readout layer must be untouched by calibration");
        assert_ne!(net.layer_thresholds(1), comp_before, "computational layer must still be calibrated");
    }

    #[test]
    fn calibrate_preserves_listeners() {
        let mut net = Network::new(test_config());
        let hits = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        {
            let h = hits.clone();
            net.on_layer(0, Box::new(move |_w, _f| *h.lock().unwrap() += 1));
        }
        let input = random_l0_input(3, 8, 20000);
        calibrate(&mut net, &CalibrateParams::default(), &input);
        *hits.lock().unwrap() = 0;
        net.wave(&input(0));
        assert!(*hits.lock().unwrap() >= 1, "user listener must survive calibration");
    }
}
