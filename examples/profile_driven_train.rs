//! Profiling target for the wave_driven **training** path — the online activity-scaled learning
//! trial, which `examples/profile_driven.rs` (forward-only, training disabled) never exercises.
//!
//! Mirrors `driven::train_trial` from `benches/train_sidecar_compare.rs` (the criterion learning
//! benchmark) but runs it in a tight loop so a sampler's samples are dominated by the trial, not by
//! Criterion's measurement harness. It also prints a coarse `std::time` phase split — wave-loop
//! (forward + online eligibility accrual) vs build_signal vs dfa_update — so you get the phase
//! breakdown with NO root; `perf`/`flamegraph` then drills into the dominant phase.
//!
//! Same side-car parity-N=4 / rec-16 config as the criterion bench, size 32 by default.
//!
//! Build: `cargo build --profile profiling --example profile_driven_train`
//! Run:   `./target/profiling/examples/profile_driven_train [size] [n_trials]`   (defaults: 32 2000)

use std::time::{Duration, Instant};

// Engine-agnostic task drive (wave_bitnet's hash; wave_driven's `mix`/`key` are byte-identical copies).
use wave_net::wave_bitnet::synapse::{key, mix};
use wave_net::wave_driven::config::{Config, LayerConfig};
use wave_net::wave_driven::network::Network;
use wave_net::wave_driven::synapse::TopologyLevel;
use wave_net::wave_driven::training::{Edge, EligParams};

// side-car params (match benches/train_sidecar_compare.rs :: driven)
const UC: u32 = 32; // forward count
const UR: u32 = 3; // forward radius
const REC: u32 = 16; // recurrent count
const R: u32 = 4; // recurrent radius
const ADAPT_BUMP: i16 = 5;
const ADAPT_DECAY: u8 = 6;
const PRESENT: usize = 6;
const DELAY: usize = 8;
const READ: usize = 8;
const PARITY_N: usize = 4;
const CUE_P: u64 = 0xC0E;
const P_DFA: u64 = 61;
const RATE_REG: f32 = 5.0;
const RATE_TARGET: f32 = 0.1;
const HIDDEN_LR: f32 = 0.004;
const SEED: u64 = 0xE9_0B_0A17;

fn cue_sites(task_seed: u64, size: u32, class: usize) -> Vec<u32> {
    let ls = (size * size) as u32;
    (0..ls).filter(|&loc| mix(key(task_seed, loc, class as i32, 0, CUE_P)) & 3 == 0).collect()
}

fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
    if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
}

/// A fixed representative parity-N=4 trial drive: N cue bits presented (with delays) then a read window.
fn build_drive(task_seed: u64, size: u32) -> Vec<Vec<u32>> {
    let bits: Vec<usize> = (0..PARITY_N).map(|i| (mix(key(task_seed, 7, 0, i as u32, 51)) & 1) as usize).collect();
    let mut drive = Vec::new();
    for (pos, &class) in bits.iter().enumerate() {
        if pos > 0 {
            for _ in 0..DELAY {
                drive.push(Vec::new());
            }
        }
        for _ in 0..PRESENT {
            drive.push(cue_sites(task_seed, size, class));
        }
    }
    for _ in 0..READ {
        drive.push(Vec::new());
    }
    drive
}

fn make_sidecar(seed: u64, size: u32) -> (Network, Vec<Vec<Edge>>) {
    let mk = |topology| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump: ADAPT_BUMP, adapt_decay: ADAPT_DECAY };
    let layers = vec![
        mk(vec![TopologyLevel { level: 1, radius: UR, count: UC }]),
        mk(vec![TopologyLevel { level: 2, radius: UR, count: UC }]),
        mk(vec![TopologyLevel { level: 0, radius: R, count: REC }, TopologyLevel { level: 1, radius: R, count: REC }]),
        mk(vec![TopologyLevel { level: -1, radius: R, count: REC }, TopologyLevel { level: 1, radius: UR, count: UC }]),
        mk(vec![]),
    ];
    let mut net = Network::new(Config { seed, size, layers });
    net.set_elig_params(EligParams { rec_tau: 20.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 });
    net.enable_training();
    let entries = vec![
        vec![Edge { level: 1, count: UC as usize, radius: UR }],
        vec![Edge { level: 2, count: UC as usize, radius: UR }],
        vec![Edge { level: 0, count: REC as usize, radius: R }, Edge { level: 1, count: REC as usize, radius: R }],
        vec![Edge { level: -1, count: REC as usize, radius: R }, Edge { level: 1, count: UC as usize, radius: UR }],
        vec![],
    ];
    (net, entries)
}

fn build_signal(net: &Network, size: u32, ttot: usize) -> Vec<Vec<f32>> {
    let l = net.layer_count();
    let ls = (size * size) as usize;
    let top = l - 1;
    let denom = ttot.max(1) as f32;
    let err = [0.1f32, -0.1f32]; // fixed synthetic error (machinery cost, not accuracy)
    let mut signal = vec![vec![0f32; ls]; l];
    for tz in 1..l {
        let sc = net.layer_spike_count(tz);
        for j in 0..ls {
            let task_sig: f32 = (0..2).map(|c| { let b = if tz == top { 0.05 } else { dfa_weight(SEED, (tz * ls + j) as u32, c) }; b * err[c] }).sum();
            signal[tz][j] = task_sig + RATE_REG * (sc[j] as f32 / denom - RATE_TARGET);
        }
    }
    signal
}

fn frac(part: Duration, whole: Duration) -> f64 {
    if whole.is_zero() { 0.0 } else { part.as_secs_f64() / whole.as_secs_f64() * 100.0 }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let size: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let n_trials: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);

    let (mut net, entries) = make_sidecar(SEED, size);
    let drive = build_drive(SEED, size);
    let waves_per_trial = drive.len();

    // Warm up to the steady sub-critical operating point (adaptation + rate_reg regulate the rate).
    for _ in 0..16 {
        net.reset_state();
        for dv in &drive {
            net.wave(dv);
        }
        let signal = build_signal(&net, size, drive.len());
        net.dfa_update(&entries, &signal, HIDDEN_LR);
    }

    // Timed loop — coarse phase accumulators (reset / wave-loop / build_signal / dfa_update).
    let mut t_reset = Duration::ZERO;
    let mut t_wave = Duration::ZERO;
    let mut t_signal = Duration::ZERO;
    let mut t_update = Duration::ZERO;

    for _ in 0..n_trials {
        let s = Instant::now();
        net.reset_state();
        t_reset += s.elapsed();

        let s = Instant::now();
        for dv in &drive {
            net.wave(dv);
        }
        t_wave += s.elapsed();

        let s = Instant::now();
        let signal = build_signal(&net, size, drive.len());
        t_signal += s.elapsed();

        let s = Instant::now();
        net.dfa_update(&entries, &signal, HIDDEN_LR);
        t_update += s.elapsed();
    }

    let total = t_reset + t_wave + t_signal + t_update;
    let waves = (n_trials * waves_per_trial) as f64;
    // consume state so nothing is dead-code-eliminated
    let top = net.layer_count() - 1;
    let sink: i64 = net.with_layer(top, |l| l.potential.iter().map(|&p| p as i64).sum());

    println!("wave_driven training profile — side-car parity-N={PARITY_N} rec{REC}, size {size}");
    println!("  {n_trials} trials × {waves_per_trial} waves = {} waves; sink={sink}", n_trials * waves_per_trial);
    println!("  {:.0} waves/s, {:.3} ms/trial\n", waves / total.as_secs_f64(), total.as_secs_f64() * 1e3 / n_trials as f64);
    println!("  phase                        ms/trial      %");
    let row = |name: &str, d: Duration| {
        println!("  {name:<24} {:9.4}  {:5.1}%", d.as_secs_f64() * 1e3 / n_trials as f64, frac(d, total));
    };
    row("wave-loop (fwd+elig)", t_wave);
    row("build_signal", t_signal);
    row("dfa_update", t_update);
    row("reset_state", t_reset);
    println!("  {:<24} {:9.4}  100.0%", "total", total.as_secs_f64() * 1e3 / n_trials as f64);
}
