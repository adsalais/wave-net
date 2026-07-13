//! Throughput benchmark for the wave_driven event-driven engine on the procedural ±1 init. Sweeps L0
//! drive fraction (activity) and times both sparse and dense modes so the activity-scaling and the
//! sparse/dense crossover are visible. No training in Phase 1.

use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use wave_net::wave_driven::config::{Config, LayerConfig};
use wave_net::wave_driven::network::Network;
use wave_net::wave_driven::synapse::{random_l0_input, TopologyLevel};

const SIZE: u32 = 32;
const SEED: u64 = 0xC0FFEE_1234_5678;
const WAVES_PER_ITER: u64 = 256;

fn cfg() -> Config {
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
    Config { seed: SEED, size: SIZE, layers: vec![layer; 5] }
}

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
    counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
    for w in 0..waves {
        net.wave(&input(warmup + w));
    }
    net.clear_listeners();
    let counts = std::mem::take(&mut *counts.lock().unwrap());
    let denom = ((net.size() as u64) * (net.size() as u64) * waves as u64) as f64;
    counts.iter().map(|&s| s as f64 / denom).collect()
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_driven");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    for &frac in &[2000u32, 8000, 30000] {
        let input = random_l0_input(SEED, SIZE, frac);
        let noise: Vec<Vec<u32>> = (0..WAVES_PER_ITER as usize).map(&input).collect();

        let mut probe = Network::new(cfg());
        let rates = measure_rates(&mut probe, 32, 128, &input);
        let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
        println!("driven 32x32x5 drive_q16={frac} per-layer rate (%): {pct:?}");

        let mut sparse = Network::new(cfg());
        group.bench_function(format!("sparse_q16_{frac}"), |b| b.iter(|| {
            for v in &noise {
                sparse.wave(v);
            }
        }));
        let mut dense = Network::new_dense(cfg());
        group.bench_function(format!("dense_q16_{frac}"), |b| b.iter(|| {
            for v in &noise {
                dense.wave(v);
            }
        }));
    }
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
