//! Throughput benchmark for the `wave_bitnet` engine — the ternary-native fork of `wave_net`.
//!
//! Mirrors `throughput.rs` (same 32×32 × 5 uniform feed-forward topology, same random-noise L0 drive,
//! same measured region: `WAVES_PER_ITER` waves through `Network::wave`) so the two `waves/second`
//! numbers are directly comparable. The difference under test: `wave_bitnet` scans a per-neuron
//! occupancy bitset and decodes targets arithmetically (no per-wave hashing), delivering 2-bit-packed
//! ±1/0 weights. `wave_bitnet` has no `critical_init` (fixed ±1 magnitude can't be scaled to σ≈1), so
//! the net runs at its self-regulated ALIF operating point; the per-layer firing rates are printed so
//! the load is transparent when comparing.

use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

// `random_l0_input` is an engine-agnostic L0 spike-site generator (just produces input locations),
// so the fork reuses wave_net's to keep the drive byte-identical to the baseline benchmark.
use wave_net::wave_net::critical_init::random_l0_input;

use wave_net::wave_bitnet::config::{Config, LayerConfig};
use wave_net::wave_bitnet::network::Network;
use wave_net::wave_bitnet::synapse::TopologyLevel;

const SIZE: u32 = 32;
const LAYERS: usize = 5;
const SEED: u64 = 0xC0FFEE_1234_5678;
const NOISE_FRACTION_Q16: u32 = 20000;
const WAVES_PER_ITER: u64 = 256;

/// The 32×32 × 5 uniform feed-forward config — identical topology to `throughput.rs` (r3/c32, adapt=5).
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

fn setup_net() -> Network {
    let mut net = Network::new(build_config());
    // Pure forward-throughput measurement: no training reads the eligibility, so skip accruing it.
    net.set_record_eligibility(false);
    net
}

/// Per-layer firing rate (fraction of neurons firing per wave) over a counted window. Warmup discarded.
fn measure_rates(net: &mut Network, warmup: usize, waves: usize, input: &impl Fn(usize) -> Vec<u32>) -> Vec<f64> {
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

/// Print the per-layer firing rates before the timed run (transparency for the comparison). Not
/// asserted: with fixed ±1 weights and no σ≈1 calibration, a uniform 5-deep stack is sub-critical
/// (the "cue dies with depth" regime — same topology as `throughput.rs`, but wave_net calibrates to
/// ~10% via critical_init while this fork runs at its natural self-regulated operating point).
fn report_operating_point(net: &mut Network) {
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let rates = measure_rates(net, 32, 128, &input);
    let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
    println!("wave_bitnet 32x32x5 FF per-layer firing rate (%): {pct:?}");
}

fn bench_throughput(c: &mut Criterion) {
    let mut net = setup_net();
    report_operating_point(&mut net);

    // Pre-generate the fixed noise ring OUTSIDE the timed loop.
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let noise: Vec<Vec<u32>> = (0..WAVES_PER_ITER as usize).map(&input).collect();

    let mut group = c.benchmark_group("throughput_bitnet");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    group.bench_function("ff_32x32x5", |b| {
        b.iter(|| {
            for v in &noise {
                net.wave(v);
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
