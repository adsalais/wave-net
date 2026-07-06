//! Depth-utilization diagnostic: is a reservoir's depth usable, i.e. does input-dependent signal
//! survive the climb to the top layer? For each layer it reports the firing rate and how well the
//! input bit (current and delayed) can be linearly decoded from *that layer alone*. A dead or
//! saturated top layer shows up as decode accuracy near chance (0.5). This is the promoted,
//! reusable form of the one-off probe used to decide the conditioning scope.

use crate::wave_net::linalg::ridge_fit;
use crate::wave_net::stream::{fair_bit, BipolarInput};
use crate::wave_reservoir::config::IntConfig;
use crate::wave_reservoir::hash::mix;
use crate::wave_reservoir::index::Dims;
use crate::wave_reservoir::pipeline::LayerNet;
use std::sync::{Arc, Mutex};

/// Per-layer diagnostic result.
#[derive(Clone, Debug)]
pub struct LayerReport {
    /// Fraction of the layer's neurons firing per wave.
    pub firing_rate: f64,
    /// Accuracy of decoding the input bit delayed by `0..=max_delay` from this layer alone
    /// (chance = 0.5). `decode_acc[0]` is the current bit — "does input reach this layer".
    pub decode_acc: Vec<f32>,
}

/// Knobs for [`depth_report`].
#[derive(Clone, Debug)]
pub struct DepthParams {
    pub input: BipolarInput,
    pub bit_seed: u64,
    pub wpb: usize,
    pub washout: usize,
    pub train: usize,
    pub test: usize,
    pub per_layer_sample: usize,
    pub max_delay: usize,
    pub lambda: f64,
    pub sample_seed: u64,
}

impl DepthParams {
    /// Standard settings for a given config (matches the input encoding the examples use).
    pub fn standard(cfg: &IntConfig) -> DepthParams {
        let dims = Dims::new(cfg.w, cfg.h, cfg.l);
        DepthParams {
            input: BipolarInput::scatter_bottom(&dims, 0x0B17_5EED, 24, 4),
            bit_seed: 0x5EED_C0DE,
            wpb: 8,
            washout: 30,
            train: 400,
            test: 200,
            per_layer_sample: 48,
            max_delay: 2,
            lambda: 1.0,
            sample_seed: 0xA5A5,
        }
    }
}

/// Deterministically pick `k` distinct local indices in `0..ls`.
fn sample_locals(ls: usize, k: usize, seed: u64) -> Vec<usize> {
    use std::collections::BTreeSet;
    let mut set = BTreeSet::new();
    let mut i = 0u64;
    while set.len() < k.min(ls) {
        set.insert((mix(seed ^ i.wrapping_mul(0x9E37_79B9)) % ls as u64) as usize);
        i += 1;
    }
    set.into_iter().collect()
}

/// Accuracy of a linear readout `w` predicting binary targets `y` from rows `x` (threshold 0.5).
fn accuracy(w: &[f64], x: &[Vec<f64>], y: &[f64]) -> f32 {
    let mut correct = 0;
    for (xi, &yi) in x.iter().zip(y) {
        let pred: f64 = xi.iter().zip(w).map(|(a, b)| a * b).sum();
        if ((pred >= 0.5) as u8) as f64 == yi {
            correct += 1;
        }
    }
    correct as f32 / y.len() as f32
}

/// Run the depth diagnostic on `cfg`. Streams a random bit sequence through a fresh reservoir,
/// collects per-layer sampled spike features per bit, and fits a ridge readout per layer and
/// delay. Single-threaded (feature collection is order-independent, but this keeps it simple).
pub fn depth_report(cfg: &IntConfig, params: &DepthParams) -> Vec<LayerReport> {
    let l = cfg.l as usize;
    let ls = (cfg.w * cfg.h) as usize;
    let total = params.washout + params.train + params.test;
    let total_waves = total * params.wpb;
    let bhist: Vec<u8> = (0..total).map(|t| fair_bit(params.bit_seed, t as u64)).collect();

    // per-layer sampled locals -> feature column
    let mut col_of: Vec<Vec<Option<usize>>> = (0..l).map(|_| vec![None; ls]).collect();
    for (z, cols) in col_of.iter_mut().enumerate() {
        for (c, &loc) in sample_locals(ls, params.per_layer_sample, params.sample_seed ^ z as u64)
            .iter()
            .enumerate()
        {
            cols[loc] = Some(c);
        }
    }

    // features[z][bit][col] (last col is the bias), plus per-layer spike totals
    let width = params.per_layer_sample + 1;
    let features = Arc::new(Mutex::new(
        (0..l).map(|_| vec![vec![0.0f64; width]; total]).collect::<Vec<_>>(),
    ));
    let spikes = Arc::new(Mutex::new(vec![0u64; l]));
    let mut net = LayerNet::new(cfg.clone());
    net.reset_state();
    let wpb = params.wpb;
    for z in 0..l {
        let feats = features.clone();
        let sp = spikes.clone();
        let cmap = col_of[z].clone();
        net.on_layer(
            z,
            Box::new(move |wave, fired| {
                let bit = wave / wpb;
                sp.lock().unwrap()[z] += fired.len() as u64;
                let mut f = feats.lock().unwrap();
                for &local in fired {
                    if let Some(c) = cmap[local as usize] {
                        f[z][bit][c] += 1.0;
                    }
                }
            }),
        );
    }
    let input = &params.input;
    net.run_stream(total_waves, 1, |w, buf| input.drive_into(buf, bhist[w / wpb]));

    let features = std::mem::take(&mut *features.lock().unwrap());
    let spikes = std::mem::take(&mut *spikes.lock().unwrap());

    let mut out = Vec::with_capacity(l);
    for z in 0..l {
        let firing_rate = spikes[z] as f64 / (ls as f64 * total_waves as f64);
        let mut decode_acc = Vec::with_capacity(params.max_delay + 1);
        for delay in 0..=params.max_delay {
            let mut x = Vec::new();
            let mut y = Vec::new();
            for t in params.washout..total {
                if t < delay {
                    continue;
                }
                let mut row = features[z][t].clone();
                row[params.per_layer_sample] = 1.0; // bias
                x.push(row);
                y.push(bhist[t - delay] as f64);
            }
            let split = params.train.min(x.len().saturating_sub(params.test));
            let (trx, tex) = x.split_at(split);
            let (try_, tey) = y.split_at(split);
            let w = ridge_fit(trx, try_, params.lambda);
            decode_acc.push(accuracy(&w, tex, tey));
        }
        out.push(LayerReport { firing_rate, decode_acc });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::calibrate::{calibrate, CalibrateParams};

    #[test]
    fn calibrated_demo_transmits_input_to_the_top_layer() {
        // Regression guard for the finding that decided the conditioning scope: after standard
        // rate-calibration, the top layer decodes the current input bit well above chance
        // (~0.89 measured). Layer 0 (input surface) decodes near-perfectly.
        let mut cfg = IntConfig::demo();
        let params = DepthParams::standard(&cfg);
        let drive = {
            let input = params.input.clone();
            let wpb = params.wpb;
            move |w: usize, buf: &mut Vec<i16>| input.drive_into(buf, fair_bit(0x5EED_C0DE, (w / wpb) as u64))
        };
        calibrate(&mut cfg, &CalibrateParams::default(), &drive);

        let report = depth_report(&cfg, &params);
        let top = report.last().unwrap();
        assert!(report[0].decode_acc[0] > 0.95, "layer 0 should decode input near-perfectly: {}", report[0].decode_acc[0]);
        assert!(
            top.decode_acc[0] > 0.7,
            "calibrated top layer must transmit input (current-bit decode {} > 0.7)",
            top.decode_acc[0]
        );
        assert!(top.firing_rate < 0.2, "calibrated top should not be saturated: {}", top.firing_rate);
    }
}
