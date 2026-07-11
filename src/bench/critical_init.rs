//! `critical_init` — a homeostatic weight-training **initialization** that replaces the brittle
//! threshold calibration. Instead of tuning per-layer baselines toward a firing-rate *proxy*, it
//! trains the stored **weights** with an e-prop update driven **only** by the per-neuron rate error
//! (the `rate_reg` learning signal, no task), on random noise, **layer-wise greedy bottom-up**.
//!
//! Why weights, not thresholds: a threshold only *gates* a fixed projection — it cannot manufacture
//! drive that never arrives, so calibration cannot revive a sub-critical (cue-dies-with-depth) stack.
//! Training weights raises the *gain*: a too-quiet neuron's incoming weights rise until it fires. This
//! is homeostatic synaptic scaling (the mechanism cortex is thought to use to self-organize toward
//! criticality). Greedy bottom-up is load-bearing: each edge `(z-1)->z` is trained only once its
//! source `z-1` is already live, so the target receives input and has non-zero eligibility (ψ > 0) to
//! learn from — otherwise a fully-dead layer gets no gradient. Feed-forward only (no readout / no
//! recurrence) for this first spike.

use crate::bench::rsnn::target_of;
use crate::wave_net::network::Network;

#[derive(Clone, Copy, Debug)]
pub struct CriticalInitParams {
    pub target_permille: u64, // desired per-neuron firing rate (e.g. 100 = 10%)
    pub lr: f32,              // e-prop learning rate for the homeostatic weight update
    pub rounds: usize,        // max update rounds per layer edge
    pub warmup: usize,        // waves discarded per measurement (let adaptation settle)
    pub waves: usize,         // waves the eligibility/rate window integrates over
    pub tol_permille: u64,    // stop an edge when |rate - target| <= tol
}

impl Default for CriticalInitParams {
    fn default() -> CriticalInitParams {
        CriticalInitParams { target_permille: 100, lr: 0.02, rounds: 80, warmup: 32, waves: 96, tol_permille: 15 }
    }
}

/// Windowed per-neuron eligibility for every layer: the pre-trace (spike count) and ψ (near-threshold
/// count) accumulated over `waves` *after* a `warmup` transient, read via the difference of the
/// engine's running `elig_pre` / `elig_post` accumulators (so the boots-hot transient is excluded).
fn windowed_eligibility(
    net: &mut Network,
    warmup: usize,
    waves: usize,
    input: &impl Fn(usize) -> Vec<u32>,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let l = net.layer_count();
    net.reset_state();
    for w in 0..warmup {
        net.wave(&input(w));
    }
    let pre0: Vec<Vec<i32>> = (0..l).map(|z| net.with_layer(z, |x| x.elig_pre.clone())).collect();
    let psi0: Vec<Vec<i32>> = (0..l).map(|z| net.with_layer(z, |x| x.elig_post.clone())).collect();
    for w in 0..waves {
        net.wave(&input(warmup + w));
    }
    let pre: Vec<Vec<i32>> =
        (0..l).map(|z| net.with_layer(z, |x| x.elig_pre.iter().zip(&pre0[z]).map(|(a, b)| a - b).collect())).collect();
    let psi: Vec<Vec<i32>> =
        (0..l).map(|z| net.with_layer(z, |x| x.elig_post.iter().zip(&psi0[z]).map(|(a, b)| a - b).collect())).collect();
    (pre, psi)
}

/// Train the net to a live ~`target` regime by homeostatic weight scaling, layer-wise bottom-up.
/// `seed` must be the engine's construction seed (needed to recover each synapse's target). Assumes a
/// uniform single-entry feed-forward topology (level +1). Mutates the stored weights in place.
pub fn rate_reg_init(net: &mut Network, seed: u64, params: &CriticalInitParams, input: &impl Fn(usize) -> Vec<u32>) {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let size = net.size();
    let r_target = params.target_permille as f32 / 1000.0;
    let waves_f = params.waves as f32;
    // Forward fan-out (count) and radius, read from a computational layer's topology.
    let (up, up_radius) = net.with_layer(1, |lz| {
        let e = &lz.topology[0];
        (e.count as usize, e.radius)
    });

    // Edge (z-1) -> z: adjust the SOURCE layer's out-weights so target layer z fires near target.
    for z in 1..l {
        for _round in 0..params.rounds {
            let (pre, psi) = windowed_eligibility(net, params.warmup, params.waves, input);
            let rate_z = pre[z].iter().sum::<i32>() as f32 / (params.waves * ls) as f32;
            if ((rate_z - r_target).abs() * 1000.0) as u64 <= params.tol_permille {
                break;
            }
            // Per-target-neuron learning signal: the rate error (rate_reg with no task term).
            let l_sig: Vec<f32> = (0..ls).map(|j| pre[z][j] as f32 / waves_f - r_target).collect();
            let src = z - 1;
            net.with_layer_mut(src, |lz| {
                for i in 0..ls {
                    let pre_i = pre[src][i] as f32;
                    if pre_i == 0.0 {
                        continue;
                    }
                    let sg = (src * ls + i) as u32;
                    for kk in 0..up {
                        let j = target_of(seed, sg, i as u32, 1, kk as u32, up_radius, size) as usize;
                        // too-quiet target (l_sig<0) -> weights rise (fires more); mirrors rsnn rate_reg.
                        lz.out_shadow[i * up + kk] += -params.lr * l_sig[j] * pre_i * psi[z][j] as f32;
                    }
                }
                for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                    *wq = s.round().clamp(-127.0, 127.0) as i8;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::TopologyLevel;

    const SEED: u64 = 0xC0FFEE_1234_5678;

    /// The 32×32 × 5 uniform feed-forward config, parameterized by forward fan-out.
    fn ff_config(up_count: u32) -> Config {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: 5,
            adapt_decay: 6,
        };
        Config { seed: SEED, size: 32, layers: vec![layer; 5] }
    }

    fn pct(r: &[f64]) -> Vec<f64> {
        r.iter().map(|x| (x * 1000.0).round() / 10.0).collect()
    }

    /// Sanity (fast): on a small starved stack, rate-init must lift the top layer's firing well above
    /// the untrained ±1 net (i.e. it revives depth, which is the whole point).
    #[test]
    fn rate_init_revives_a_starved_stack() {
        let cfg = {
            let layer = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 2, count: 6 }], // low fan-out -> starves
                leak: (3, 5),
                cooldown_base: 2,
                inhibitor_ratio: 0,
                threshold_jitter: 16,
                baseline_init: 6,
                adapt_bump: 5,
                adapt_decay: 6,
            };
            Config { seed: SEED, size: 8, layers: vec![layer; 4] }
        };
        let input = random_l0_input(SEED, 8, 20000);
        let top = cfg.layers.len() - 1;

        let mut untrained = Network::new(cfg.clone());
        let before = untrained.measure_layer_rates(16, 64, &input)[top];

        let mut net = Network::new(cfg);
        let params = CriticalInitParams { rounds: 40, ..CriticalInitParams::default() };
        rate_reg_init(&mut net, SEED, &params, &input);
        let after = net.measure_layer_rates(16, 64, &input)[top];

        assert!(after > before + 0.02, "rate-init should revive the top layer: {before:.3} -> {after:.3}");
    }

    /// Experiment (run manually): calibration vs rate-init resulting regime, on a propagating config
    /// (up_count 32) and the config where calibration fails (up_count 16, cue dies with depth).
    ///   cargo test --release critical_init_vs_calibration -- --ignored --nocapture
    #[test]
    #[ignore]
    fn critical_init_vs_calibration() {
        let calib = CalibrateParams { target_permille: 100, ..CalibrateParams::default() };
        for up_count in [16u32, 32] {
            let cfg = ff_config(up_count);
            let input = random_l0_input(SEED, 32, 20000);

            let mut a = Network::new(cfg.clone());
            a.calibrate(&calib, &input);
            let ra = a.measure_layer_rates(32, 128, &input);
            println!("up_count={up_count}: calibration          ={:?}", pct(&ra));

            for lr in [0.008f32, 0.02, 0.05] {
                let mut b = Network::new(cfg.clone());
                rate_reg_init(&mut b, SEED, &CriticalInitParams { lr, ..CriticalInitParams::default() }, &input);
                let rb = b.measure_layer_rates(32, 128, &input);
                println!("up_count={up_count}: rate_reg_init lr={lr:<5}={:?}", pct(&rb));
            }
        }
    }
}
