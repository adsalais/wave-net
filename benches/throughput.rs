//! Baseline throughput benchmark for the `wave_net` engine.
//!
//! Reports waves/second on a 32×32 × 5 uniform feed-forward network. Calibration and the
//! operating-point guard run in setup (outside the measured region, added in later tasks); the
//! measured region runs random-noise L0 input through `Network::wave`.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use wave_net::wave_net::calibrate::random_l0_input;
use wave_net::wave_net::config::{Config, LayerConfig};
use wave_net::wave_net::network::Network;
use wave_net::wave_net::synapse::TopologyLevel;

const SIZE: u32 = 32;
const LAYERS: usize = 5;
const SEED: u64 = 0xC0FFEE_1234_5678;
/// L0 injection density for the random-noise drive, Q16 (~10% of 65536). Same value for
/// calibration and measurement so the calibrated operating point transfers.
const NOISE_FRACTION_Q16: u32 = 6554;
/// Waves per measured iteration; also the `Throughput::Elements` count → criterion reports waves/s.
const WAVES_PER_ITER: u64 = 256;

/// The 32×32 × 5 uniform feed-forward config under test. Values are the current engine FF defaults,
/// inlined so the benchmark is self-documenting.
fn build_config() -> Config {
    let layer = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: 3, count: 16 }],
        leak: (3, 5),
        cooldown_base: 2,
        inhibitor_ratio: 0,
        threshold_jitter: 32,
        baseline_init: 6,
        adapt_bump: 20,
        adapt_decay: 6,
    };
    Config { seed: SEED, size: SIZE, layers: vec![layer; LAYERS] }
}

fn bench_throughput(c: &mut Criterion) {
    let net = Network::new(build_config());
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
