//! Firing-rate calibration: tune each cascade layer's `threshold_base` so the layer fires at a
//! target rate on a given input stream. Generic over any `IntConfig` and any drive function, so
//! bigger networks added later are calibrated with the same call. Generalizes the
//! `calibrate_on_stream` routine that used to live only in the `params_study` example.

use crate::wave_net::config::{spread_log2_for, IntConfig, MAX_SATURATION};
use crate::wave_net::pipeline::LayerNet;
use std::sync::{Arc, Mutex};

/// Knobs for [`calibrate`].
#[derive(Clone, Copy, Debug)]
pub struct CalibrateParams {
    /// Target firing rate in per-mille (e.g. `120` = 12%).
    pub target_permille: u64,
    /// Number of measure-and-adjust passes (absorbs inter-layer coupling by repetition).
    pub passes: usize,
    /// Bits streamed per measurement pass.
    pub bits: usize,
    /// Waves per bit.
    pub wpb: usize,
}

impl Default for CalibrateParams {
    fn default() -> CalibrateParams {
        CalibrateParams { target_permille: 120, passes: 10, bits: 40, wpb: 8 }
    }
}

/// Per-layer firing fraction (spikes / (layer_size · waves)) over `waves` waves of `drive`.
pub fn layer_rates(
    cfg: &IntConfig,
    waves: usize,
    drive: &(impl Fn(usize, &mut Vec<i16>) + Sync),
) -> Vec<f64> {
    let ls = (cfg.w * cfg.h) as u64;
    let spikes = Arc::new(Mutex::new(vec![0u64; cfg.l as usize]));
    let mut net = LayerNet::new(cfg.clone());
    net.reset_state();
    for z in 0..cfg.l as usize {
        let sp = spikes.clone();
        net.on_layer(z, Box::new(move |_w, fired| sp.lock().unwrap()[z] += fired.len() as u64));
    }
    net.run_stream(waves, 1, drive);
    let spikes = std::mem::take(&mut *spikes.lock().unwrap());
    let denom = (ls * waves as u64) as f64;
    spikes.iter().map(|&s| s as f64 / denom).collect()
}

/// Tune layers `1..L`'s `threshold_base` so each fires near `target_permille` on `drive`.
/// Layer 0 (the input surface) is left as-is. Mutates `cfg` in place. Layers couple (a layer's
/// firing changes its neighbours' input), so this measures and adjusts over `passes` iterations;
/// each step is `threshold >> 2` so it converges geometrically from any starting scale.
pub fn calibrate(
    cfg: &mut IntConfig,
    params: &CalibrateParams,
    drive: &(impl Fn(usize, &mut Vec<i16>) + Sync),
) {
    let target = params.target_permille as f64 / 1000.0;
    let waves = params.bits * params.wpb;
    for _ in 0..params.passes {
        let rates = layer_rates(cfg, waves, drive);
        for z in 1..cfg.l as usize {
            let tb = cfg.layers[z].threshold_base;
            let step = (tb >> 2).max(1);
            if rates[z] > target {
                cfg.layers[z].threshold_base = tb + step;
            } else if rates[z] < target {
                cfg.layers[z].threshold_base = (tb - step).max(1);
            }
            cfg.layers[z].spread_log2 = spread_log2_for(cfg.layers[z].threshold_base);
        }
        let max_t = cfg.layers.iter().map(|l| l.threshold_base).max().unwrap_or(1);
        cfg.saturation = max_t.saturating_mul(32).min(MAX_SATURATION as i32) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::stream::{fair_bit, BipolarInput};
    use crate::wave_net::index::Dims;

    fn bit_drive(cfg: &IntConfig, wpb: usize) -> impl Fn(usize, &mut Vec<i16>) + Sync + use<> {
        let dims = Dims::new(cfg.w, cfg.h, cfg.l);
        let inp = BipolarInput::scatter_bottom(&dims, 0x0B17_5EED, 24, 4);
        move |w, buf| inp.drive_into(buf, fair_bit(0x5EED_C0DE, (w / wpb) as u64))
    }

    #[test]
    fn calibration_cools_a_hot_config_toward_target() {
        // The raw demo runs hot in its upper layers (~45%); calibrating to 12% must pull the
        // top layer's rate closer to target and raise at least one threshold.
        let mut cfg = IntConfig::demo();
        let params = CalibrateParams::default();
        let target = params.target_permille as f64 / 1000.0;
        let top = cfg.l as usize - 1;

        let waves = params.bits * params.wpb;
        let drive = bit_drive(&cfg, params.wpb); // owns its input sites; independent of cfg after this
        let before = layer_rates(&cfg, waves, &drive)[top];
        let thresholds_before: Vec<i32> = cfg.layers.iter().map(|l| l.threshold_base).collect();

        calibrate(&mut cfg, &params, &drive);

        let after = layer_rates(&cfg, waves, &drive)[top];
        let raised = cfg.layers.iter().zip(&thresholds_before).any(|(l, &b)| l.threshold_base > b);

        assert!(before > target, "precondition: raw top should be hot ({before:.3} vs {target:.3})");
        assert!(
            (after - target).abs() < (before - target).abs(),
            "calibration should cool the top toward target: {before:.3} -> {after:.3} (target {target:.3})"
        );
        assert!(raised, "calibration should have raised at least one threshold");
    }
}
