//! Profiling target for the wave_resonate forward (inference) loop — the dense BRF membrane update +
//! the firer-gated ternary delivery, training OFF. Same 32×32×5 / r3/c32 config as the throughput bench
//! (`benches/throughput_resonate.rs`), under random L0 drive. Build: `cargo build --profile profiling
//! --example profile_resonate`; run `./target/profiling/examples/profile_resonate [n_waves]`.

use wave_net::wave_resonate::config::{Config, LayerConfig};
use wave_net::wave_resonate::network::Network;
use wave_net::wave_resonate::synapse::{random_l0_input, TopologyLevel};

fn main() {
    let n_waves: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let (size, seed) = (32u32, 0xC0FFEE_1234_5678u64);
    let layer = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
        inhibitor_ratio: 0,
        omega_init: (5.0, 10.0),
        b_offset_init: (0.0, 0.2),
        tau_out: 20.0,
    };
    let mut net = Network::new(Config { seed, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: vec![layer; 5] });
    let input = random_l0_input(seed, size, 8000); // ~12% L0 drive
    let noise: Vec<Vec<u32>> = (0..256).map(&input).collect();
    for w in 0..64 {
        net.wave(&noise[w % noise.len()]);
    }
    for i in 0..n_waves {
        net.wave(&noise[i % noise.len()]);
    }
    let sink: f32 = net.with_layer(4, |l| l.x.iter().sum());
    println!("ran {n_waves} waves; sink={sink}");
}
