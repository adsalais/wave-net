//! Baseline throughput benchmark for the `wave_net` engine.
//!
//! Reports waves/second on a 32×32 × 5 uniform feed-forward network. Calibration and the
//! operating-point guard run in setup (outside the measured region, added in later tasks); the
//! measured region runs random-noise L0 input through `Network::wave`.

use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use wave_net::wave_net::calibrate::{random_l0_input, CalibrateParams};
use wave_net::wave_net::config::{Config, LayerConfig};
use wave_net::wave_net::network::Network;
use wave_net::wave_net::synapse::TopologyLevel;

const SIZE: u32 = 32;
const LAYERS: usize = 5;
const SEED: u64 = 0xC0FFEE_1234_5678;
/// L0 injection density for the random-noise drive, Q16 (~30% of 65536). Same value for
/// calibration and measurement so the calibrated operating point transfers. A ~10% sparse drive
/// starves the hidden layers (the cue dies with depth in a 5-deep stack — calibration can't reach
/// 10% when a layer is input-starved), so we use the sustained density the training code calibrates
/// against (`calib_fraction_q16 = 20000`).
const NOISE_FRACTION_Q16: u32 = 20000;
/// Waves per measured iteration; also the `Throughput::Elements` count → criterion reports waves/s.
const WAVES_PER_ITER: u64 = 256;

/// The 32×32 × 5 uniform feed-forward config under test.
///
/// These are the engine's feed-forward `LayerConfig` values with two deliberate departures from the
/// literal `bench::rsnn::engine_config` defaults (up_count 16 / adapt_bump 20): the fan-out is raised
/// to the scaling study's **forward-drive threshold** `up_count = 32`, and adaptation is softened to
/// `adapt_bump = 5`. The literal defaults are sub-critical at depth 5 — the cue dies with depth
/// (rates ≈ 30, 3.7, 0.6, 0.6, 0.7 %), so calibration cannot reach a 10% operating point (that is the
/// documented "cue dies with depth" limitation; training's `rate_reg` revives it, calibration cannot).
/// At `up_count = 32, adapt_bump = 5` the uniform pure-FF stack genuinely propagates and calibrates to
/// ~10% through all layers, so the throughput number reflects a real spiking load.
fn build_config() -> Config {
    let layer = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
        leak: (3, 5),
        cooldown_base: 2,
        inhibitor_ratio: 0,
        threshold_jitter: 32,
        baseline_init: 6,
        adapt_bump: 5,
        adapt_decay: 6,
    };
    Config { seed: SEED, size: SIZE, layers: vec![layer; LAYERS] }
}

/// Build and calibrate the net to ~10% per-layer firing on the random-noise drive. Calibration is
/// setup, not measured. The same drive is used for measurement so the operating point transfers.
fn setup_net() -> Network {
    let mut net = Network::new(build_config());
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let params = CalibrateParams { target_permille: 100, ..CalibrateParams::default() };
    net.calibrate(&params, &input);
    net
}

/// Per-layer firing rate (fraction of neurons firing per wave) over a counted window, measured
/// through the public listener API. Warmup waves are discarded. Leaves the net in a warmed state.
fn measure_rates(
    net: &mut Network,
    warmup: usize,
    waves: usize,
    input: &impl Fn(usize) -> Vec<u32>,
) -> Vec<f64> {
    let l = net.layer_count();
    let counts = Arc::new(Mutex::new(vec![0u64; l]));
    for z in 0..l {
        let c = counts.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| {
            c.lock().unwrap()[z] += fired.len() as u64;
        }));
    }
    net.reset_state();
    for w in 0..warmup {
        net.wave(&input(w));
    }
    counts.lock().unwrap().iter_mut().for_each(|x| *x = 0); // discard warmup
    for w in 0..waves {
        net.wave(&input(warmup + w));
    }
    net.clear_listeners();
    let counts = std::mem::take(&mut *counts.lock().unwrap());
    let ls = (net.size() as u64) * (net.size() as u64);
    let denom = (ls * waves as u64) as f64;
    counts.iter().map(|&s| s as f64 / denom).collect()
}

/// Confirm the calibrated net is in a live, propagating regime before the timed run: print the
/// per-layer firing rates and assert every computational layer (1..L) is within a generous live
/// band around 10%. Fails loudly if calibration drifted (dead or saturated net).
fn assert_operating_point(net: &mut Network) {
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let rates = measure_rates(net, 32, 128, &input);
    let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
    println!("wave_net 32x32x5 FF per-layer firing rate (%): {pct:?}");
    for z in 1..net.layer_count() {
        assert!(
            (0.03..=0.25).contains(&rates[z]),
            "layer {z} firing rate {:.3} outside live band [0.03, 0.25] — calibration drifted",
            rates[z]
        );
    }
}

fn bench_throughput(c: &mut Criterion) {
    let mut net = setup_net();
    assert_operating_point(&mut net);
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    group.bench_function("ff_32x32x5", |b| {
        b.iter(|| {
            for w in 0..WAVES_PER_ITER as usize {
                net.wave(&input(w));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
