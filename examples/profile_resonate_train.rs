//! Profiling target for the wave_resonate **training** path — the dense BRF forward + the per-wave HYPR
//! eligibility accrual (`Network::accrue_eligibility`, run inside `wave()` when training) + a periodic
//! `dfa_update`, which `profile_resonate.rs` (forward-only) never exercises. Runs the same FF r3/c32
//! workload as `benches/throughput_resonate.rs` (train path) in a tight loop so a sampler's samples are
//! dominated by the trial, not by Criterion's harness. Prints a coarse std::time phase split — wave-loop
//! (forward + eligibility accrual) vs dfa_update vs reset_eligibility — so you get the phase breakdown
//! with NO root; `perf`/`flamegraph` then drills into the dominant phase (expected: `accrue_eligibility`).
//!
//! Build: `cargo build --profile profiling --example profile_resonate_train`
//! Run:   `./target/profiling/examples/profile_resonate_train [size] [n_trials]`  (defaults: 32 600)

use std::time::{Duration, Instant};

use wave_net::wave_resonate::config::{Config, LayerConfig};
use wave_net::wave_resonate::network::Network;
use wave_net::wave_resonate::synapse::{random_l0_input, TopologyLevel};
use wave_net::wave_resonate::training::{Edge, EligParams};

const UC: u32 = 32; // forward count
const UR: u32 = 3; // forward radius
const LAYERS: usize = 5;
const TRIAL_WAVES: usize = 32; // waves between dfa_update + reset_eligibility (mirrors the bench "trial")
const SEED: u64 = 0xC0FFEE_1234_5678;

fn make_ff(seed: u64, size: u32) -> (Network, Vec<Vec<Edge>>) {
    let lc = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: UR, count: UC }],
        inhibitor_ratio: 0,
        omega_init: (5.0, 10.0),
        b_offset_init: (0.0, 0.2),
        tau_out: 20.0,
    };
    let mut net = Network::new(Config { seed, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: vec![lc; LAYERS] });
    net.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: false });
    net.enable_training();
    let entries = (0..LAYERS)
        .map(|z| if z == LAYERS - 1 { vec![] } else { vec![Edge { level: 1, count: UC as usize, radius: UR }] })
        .collect();
    (net, entries)
}

fn frac(part: Duration, whole: Duration) -> f64 {
    if whole.is_zero() { 0.0 } else { part.as_secs_f64() / whole.as_secs_f64() * 100.0 }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let size: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let n_trials: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(600);

    let (mut net, entries) = make_ff(SEED, size);
    let ls = (size * size) as usize;
    let signal: Vec<Vec<f32>> = vec![vec![-1.0f32; ls]; LAYERS]; // fixed synthetic credit (machinery cost)
    let input = random_l0_input(SEED, size, 8000); // ~12% L0 drive
    let noise: Vec<Vec<u32>> = (0..TRIAL_WAVES).map(&input).collect();

    // Warm up to the ringing operating point + leave a clean, bounded eligibility active set.
    for _ in 0..8 {
        for v in &noise {
            net.wave(v);
        }
        net.dfa_update(&entries, &signal, 2.0);
        net.reset_eligibility();
    }

    let (mut t_wave, mut t_update, mut t_reset) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    for _ in 0..n_trials {
        let s = Instant::now();
        for v in &noise {
            net.wave(v);
        }
        t_wave += s.elapsed();

        let s = Instant::now();
        net.dfa_update(&entries, &signal, 2.0);
        t_update += s.elapsed();

        let s = Instant::now();
        net.reset_eligibility();
        t_reset += s.elapsed();
    }

    let total = t_wave + t_update + t_reset;
    let waves = (n_trials * TRIAL_WAVES) as f64;
    let sink: f32 = net.with_layer(LAYERS - 1, |l| l.x.iter().sum()); // consume state (no dead-code elim)

    println!("wave_resonate training profile — FF r{UR}/c{UC}, size {size}");
    println!("  {n_trials} trials × {TRIAL_WAVES} waves = {} waves; sink={sink}", n_trials * TRIAL_WAVES);
    println!("  {:.0} waves/s, {:.3} ms/trial\n", waves / total.as_secs_f64(), total.as_secs_f64() * 1e3 / n_trials as f64);
    println!("  phase                        ms/trial      %");
    let row = |name: &str, d: Duration| {
        println!("  {name:<24} {:9.4}  {:5.1}%", d.as_secs_f64() * 1e3 / n_trials as f64, frac(d, total));
    };
    row("wave-loop (fwd+elig)", t_wave);
    row("dfa_update", t_update);
    row("reset_eligibility", t_reset);
    println!("  {:<24} {:9.4}  100.0%", "total", total.as_secs_f64() * 1e3 / n_trials as f64);
}
