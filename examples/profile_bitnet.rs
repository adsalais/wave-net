//! Profiling target: runs the wave_bitnet forward hot loop many times so a profiler's samples are
//! dominated by `Network::wave` / `process_layer`. Same 32×32×5 FF config as the throughput benchmark
//! (uncalibrated ±1, eligibility off). Run under perf/flamegraph after lowering perf_event_paranoid.
//!
//! Build: `cargo build --profile profiling --example profile_bitnet`
//! Run:   `./target/profiling/examples/profile_bitnet [n_waves]`

use wave_net::wave_bitnet::config::{Config, LayerConfig};
use wave_net::wave_bitnet::network::Network;
use wave_net::wave_bitnet::synapse::{random_l0_input, TopologyLevel};

fn main() {
    let n_waves: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let (size, seed) = (32u32, 0xC0FFEE_1234_5678u64);
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
    let mut net = Network::new(Config { seed, size, layers: vec![layer; 5] });
    net.set_record_eligibility(false);

    let input = random_l0_input(seed, size, 20000);
    let noise: Vec<Vec<u32>> = (0..256).map(&input).collect();

    for w in 0..64 {
        net.wave(&noise[w % noise.len()]);
    }
    for i in 0..n_waves {
        net.wave(&noise[i % noise.len()]);
    }
    // consume state so nothing is dead-code-eliminated
    let sink: i64 = net.with_layer(4, |l| l.potential.iter().map(|&p| p as i64).sum());
    println!("ran {n_waves} waves; sink={sink}");
}
