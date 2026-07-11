//! Profiling / throughput harness (throwaway, not a test/bench): builds + calibrates a 32×32×5
//! feed-forward net like the throughput bench, prints per-layer firing rates, then runs the forward
//! pass in a tight loop and reports wall-clock throughput. The per-neuron forward fan-out `up_count`
//! is overridable via the `WAVE_UP_COUNT` env var (for load-matched comparisons across variants).
//!
//!   CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release --example profile_wave
//!   WAVE_UP_COUNT=32 ./target/release/examples/profile_wave 900000

use std::sync::{Arc, Mutex};

use wave_net::bench::calibrate::{calibrate, CalibrateParams};
use wave_net::wave_net::critical_init::random_l0_input;
use wave_net::wave_net::config::{Config, LayerConfig};
use wave_net::wave_net::network::Network;
use wave_net::wave_net::synapse::TopologyLevel;

const SIZE: u32 = 32;
const LAYERS: usize = 5;
const SEED: u64 = 0xC0FFEE_1234_5678;
const NOISE_FRACTION_Q16: u32 = 20000;
const RING: usize = 256;

fn measure_rates(net: &mut Network, warmup: usize, waves: usize, input: &impl Fn(usize) -> Vec<u32>) -> Vec<f64> {
    let l = net.layer_count();
    let counts = Arc::new(Mutex::new(vec![0u64; l]));
    for z in 0..l {
        let c = counts.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| c.lock().unwrap()[z] += fired.len() as u64));
    }
    net.reset_state();
    for w in 0..warmup {
        net.wave(&input(w));
    }
    counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
    for w in 0..waves {
        net.wave(&input(warmup + w));
    }
    net.clear_listeners();
    let counts = std::mem::take(&mut *counts.lock().unwrap());
    let denom = ((net.size() * net.size()) as u64 * waves as u64) as f64;
    counts.iter().map(|&s| s as f64 / denom).collect()
}

fn main() {
    let waves: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(900_000);
    let up_count: u32 = std::env::var("WAVE_UP_COUNT").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

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
    let config = Config { seed: SEED, size: SIZE, layers: vec![layer; LAYERS] };
    let mut net = Network::new(config);
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    calibrate(&mut net, &CalibrateParams { target_permille: 100, ..CalibrateParams::default() }, &input);

    let rates = measure_rates(&mut net, 32, 128, &input);
    let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
    let total_fire: f64 = rates.iter().sum::<f64>() * (SIZE * SIZE) as f64; // spikes/wave summed over layers
    eprintln!("up_count={up_count} rates(%)={pct:?} spikes/wave≈{total_fire:.0}");

    net.set_record_eligibility(false);
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
