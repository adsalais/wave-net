//! Profiling harness (throwaway, not a test/bench): builds + calibrates the 32×32×5 feed-forward
//! net exactly like the throughput bench, then runs the forward pass in a tight loop so a sampling
//! profiler sees ~only the measured hot path. Prints wall-clock throughput as a sanity check.
//!
//!   CARGO_PROFILE_RELEASE_DEBUG=true RUSTFLAGS="-C force-frame-pointers=yes" \
//!     cargo build --release --example profile_wave
//!   perf record -g -- ./target/release/examples/profile_wave 3000000
//!   perf report --stdio | head

use wave_net::wave_net::calibrate::{random_l0_input, CalibrateParams};
use wave_net::wave_net::config::{Config, LayerConfig};
use wave_net::wave_net::network::Network;
use wave_net::wave_net::synapse::TopologyLevel;

const SIZE: u32 = 32;
const LAYERS: usize = 5;
const SEED: u64 = 0xC0FFEE_1234_5678;
const NOISE_FRACTION_Q16: u32 = 20000;
const RING: usize = 256;

fn main() {
    let waves: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3_000_000);

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
    let config = Config { seed: SEED, size: SIZE, layers: vec![layer; LAYERS] };
    let mut net = Network::new(config);
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    net.calibrate(&CalibrateParams { target_permille: 100, ..CalibrateParams::default() }, &input);
    net.set_record_eligibility(false); // pure forward pass, same as the throughput bench

    // Pre-generated noise ring so the loop measures the engine, not the input-hash RNG.
    let ring: Vec<Vec<u32>> = (0..RING).map(&input).collect();

    let t = std::time::Instant::now();
    let batches = waves / RING;
    for _ in 0..batches {
        for v in &ring {
            net.wave(v);
        }
    }
    let done = batches * RING;
    let el = t.elapsed().as_secs_f64();
    eprintln!("{done} waves in {el:.3}s -> {:.0} waves/s", done as f64 / el);
}
