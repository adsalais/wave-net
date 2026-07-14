//! Throughput benchmark for the wave_resonate (BRF + HYPR) engine on the procedural ±1 init. Mirrors
//! benches/throughput_driven.rs (same size / depth / fan-in and the L0 drive-fraction sweep) so the two
//! engines stand side by side. Unlike wave_driven there is **no sparse/dense split** — BRF resonators ring
//! continuously, so the membrane update is *always* dense; the drive sweep instead exposes how much cost
//! is the fixed dense membrane vs the firer-gated (activity-scaled) delivery. Two paths per fraction:
//! `infer` (dense membrane + delivery only) and `train` (adds the per-wave HYPR eligibility accrual +
//! one `dfa_update`/`reset_eligibility` per `TRIAL_WAVES` "trial"), so the training perf wall — the GPU
//! target — is visible next to raw inference throughput.

use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use wave_net::wave_resonate::config::{Config, LayerConfig};
use wave_net::wave_resonate::network::Network;
use wave_net::wave_resonate::synapse::{random_l0_input, TopologyLevel};
use wave_net::wave_resonate::training::{Edge, EligParams};

const SIZE: u32 = 32;
const SEED: u64 = 0xC0FFEE_1234_5678;
const WAVES_PER_ITER: u64 = 256;
const TRIAL_WAVES: usize = 32; // waves between dfa_update + reset_eligibility (mirrors a training "trial")
const RADIUS: u32 = 3;
const COUNT: u32 = 32;
const LAYERS: usize = 5;

fn cfg() -> Config {
    // Same shape as throughput_driven (size 32, 5 layers, r3/c32) at the resonate operating point used by
    // the experiments: dt 0.05, γ 0.9, θ_c 0.1 (DC-drive liveness), ω∈[5,10] (δ·ω_hi=0.5≤1), b′∈[0,0.2].
    let layer = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: RADIUS, count: COUNT }],
        inhibitor_ratio: 0,
        omega_init: (5.0, 10.0),
        b_offset_init: (0.0, 0.2),
        tau_out: 20.0,
    };
    Config { seed: SEED, size: SIZE, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: vec![layer; LAYERS] }
}

/// DFA edge wiring matching `cfg()`: a level-1 forward edge on every layer but the last (readout target).
fn entries() -> Vec<Vec<Edge>> {
    (0..LAYERS)
        .map(|z| if z == LAYERS - 1 { vec![] } else { vec![Edge { level: 1, count: COUNT as usize, radius: RADIUS }] })
        .collect()
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
    let mut group = c.benchmark_group("throughput_resonate");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    let entries = entries();
    // Constant DFA credit signal: `dfa_update` cost depends on the dirty-row count + fan-in, not on the
    // signal values, so a fixed vector is representative and keeps the bench self-contained.
    let ls = (SIZE as usize) * (SIZE as usize);
    let signal: Vec<Vec<f32>> = vec![vec![-1.0f32; ls]; LAYERS];

    for &frac in &[2000u32, 8000, 30000] {
        let input = random_l0_input(SEED, SIZE, frac);
        let noise: Vec<Vec<u32>> = (0..WAVES_PER_ITER as usize).map(&input).collect();

        let mut probe = Network::new(cfg());
        let rates = measure_rates(&mut probe, 32, 128, &input);
        let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
        println!("resonate 32x32x5 drive_q16={frac} per-layer rate (%): {pct:?}");

        // --- inference: dense membrane + firer-gated delivery, training off ---
        let mut infer = Network::new(cfg());
        group.bench_function(format!("infer_q16_{frac}"), |b| b.iter(|| {
            for v in &noise {
                infer.wave(v);
            }
        }));

        // --- training: per-wave HYPR eligibility accrual + one dfa_update per TRIAL_WAVES ("trial") ---
        let mut train = Network::new(cfg());
        train.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: false });
        train.enable_training();
        group.bench_function(format!("train_q16_{frac}"), |b| b.iter(|| {
            for (i, v) in noise.iter().enumerate() {
                train.wave(v);
                if (i + 1) % TRIAL_WAVES == 0 {
                    train.dfa_update(&entries, &signal, 2.0);
                    train.reset_eligibility();
                }
            }
        }));
    }
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
