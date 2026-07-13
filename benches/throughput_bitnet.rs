//! Throughput benchmark for the `wave_bitnet` engine, measured on a **trained** backward-fed side-car
//! net (not the untrained ±1 init) so the forward-delivery load reflects real post-training weights
//! (pruned zeros, learned liveness).
//!
//! The trained model is cached to disk: on the first run (no cache) it builds the 5-layer side-car and
//! trains it on parity N=3 with **early-exit best-checkpoint** — the peak held-out model is written to
//! the `.wbm` cache as training progresses, so it is immune to the `rate_reg` over-training collapse —
//! then the benchmark loads that cached peak model. Subsequent runs load it instantly. Delete the cache
//! file (or change the config) to retrain. The training loop is duplicated from the `wave_bitnet_bench`
//! test harness (code duplication is intended for benchmarks).
//!
//! Timed region: `WAVES_PER_ITER` forward `Network::wave` calls under fixed random-noise L0 drive, with
//! eligibility recording off. The per-layer firing rates are printed first so the operating point is
//! transparent.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

// `random_l0_input` is an engine-agnostic L0 spike-site generator (input locations only).
use wave_net::wave_bitnet::config::{Config, LayerConfig};
use wave_net::wave_bitnet::multilayer_dfa::{multilayer_dfa_step, Edge, EligParams, TrialRecords, PSI_WIDTH};
use wave_net::wave_bitnet::network::Network;
use wave_net::wave_bitnet::synapse::{key, mix, random_l0_input, TopologyLevel};

const SIZE: u32 = 32;
const SEED: u64 = 0xC0FFEE_1234_5678;
const NOISE_FRACTION_Q16: u32 = 20000;
const WAVES_PER_ITER: u64 = 256;

// side-car topology (matches `bitnet_sidecar_parity`): forward r4/c48, recurrent r4/c24, ALIF bump 5.
const FWD_R: u32 = 4;
const FWD_C: u32 = 48;
const REC_R: u32 = 4;
const REC_C: u32 = 24;
const ADAPT_BUMP: i16 = 5;
const ADAPT_DECAY: u8 = 6;

// early-exit best-checkpoint training budget (one-time; cached).
const PARITY_N: usize = 3;
const EVAL_EVERY: usize = 300;
const PATIENCE: usize = 3;
const MAX_TRIALS: usize = 3000;
const HOLDOUT: usize = 300;

fn cache_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("wave_bitnet_sidecar_s32_par3.wbm")
}

// ---- side-car builder (backward-fed 5-layer side-car; duplicated from the bench harness) ----
// L0→L1(+1); L1→L3(+2 skip); L2 self(0)+→L3(+1); L3→L2(−1)+→L4(+1); L4 read. Forward/skip use (uc, ur);
// the recurrent side-car uses (n, r). `entries` mirror the built topology order so DFA credit lines up.
fn make_sidecar(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
    let mk = |topology| LayerConfig {
        topology,
        leak: (3, 5),
        cooldown_base: 2,
        inhibitor_ratio: 0,
        threshold_jitter: 32,
        baseline_init: 6,
        adapt_bump,
        adapt_decay,
    };
    let layers = vec![
        mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]),
        mk(vec![TopologyLevel { level: 2, radius: ur, count: uc }]),
        mk(vec![TopologyLevel { level: 0, radius: r, count: n }, TopologyLevel { level: 1, radius: r, count: n }]),
        mk(vec![TopologyLevel { level: -1, radius: r, count: n }, TopologyLevel { level: 1, radius: ur, count: uc }]),
        mk(vec![]),
    ];
    let net = Network::new(Config { seed, size, layers });
    let entries = vec![
        vec![Edge { level: 1, count: uc as usize, radius: ur }],
        vec![Edge { level: 2, count: uc as usize, radius: ur }],
        vec![Edge { level: 0, count: n as usize, radius: r }, Edge { level: 1, count: n as usize, radius: r }],
        vec![Edge { level: -1, count: n as usize, radius: r }, Edge { level: 1, count: uc as usize, radius: ur }],
        vec![],
    ];
    (net, entries)
}

// ---- training machinery (duplicated from `wave_bitnet_bench`) ----
const CUE_P: u64 = 0xC0E;
const P_DFA: u64 = 61;

fn cue_sites(task_seed: u64, size: u32, class: usize) -> Vec<u32> {
    let ls = (size * size) as u32;
    (0..ls).filter(|&loc| mix(key(task_seed, loc, class as i32, 0, CUE_P)) & 3 == 0).collect()
}

fn softmax2(z0: f32, z1: f32) -> (f32, f32) {
    let m = z0.max(z1);
    let (e0, e1) = ((z0 - m).exp(), (z1 - m).exp());
    let s = (e0 + e1).max(1e-30);
    (e0 / s, e1 / s)
}

fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
    if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
}

/// N-bit sequential parity: N deterministic cue bits, label = their XOR.
fn task_parity(seed: u64, t: usize, n: usize) -> (Vec<usize>, usize) {
    let bits: Vec<usize> = (0..n).map(|i| (mix(key(seed, t as u32, 0, i as u32, 51)) & 1) as usize).collect();
    let label = bits.iter().fold(0usize, |a, &b| a ^ b);
    (bits, label)
}

struct TaskCfg {
    size: u32,
    present: usize,
    delay: usize,
    read: usize,
    holdout: usize,
    readout_lr: f32,
    hidden_lr: f32,
    rate_reg: f32,
    rate_target: f32,
    elig: EligParams,
}

fn sidecar_cfg() -> TaskCfg {
    TaskCfg {
        size: SIZE,
        present: 6,
        delay: 8,
        read: 8,
        holdout: HOLDOUT,
        readout_lr: 0.02,
        hidden_lr: 0.004,
        rate_reg: 5.0,
        rate_target: 0.1,
        elig: EligParams { rec_tau: 20.0, elig_beta: 0.4, elig_psi_width: PSI_WIDTH, use_bump: true, adapt_decay: 6 },
    }
}

fn run_trial(net: &mut Network, size: u32, classes: &[usize], task_seed: u64, present: usize, delay: usize, read: usize) -> (Vec<f32>, TrialRecords) {
    let l = net.layer_count();
    let ls = (size * size) as usize;
    let top = l - 1;
    let spikes_acc: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 0..l {
        let acc = spikes_acc.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| acc.lock().unwrap()[z].push(fired.to_vec())));
    }
    net.reset_state();
    let mut pots: Vec<Vec<Vec<i16>>> = vec![Vec::new(); l];
    let mut effs: Vec<Vec<Vec<i32>>> = vec![Vec::new(); l];
    let snapshot = |net: &Network, pots: &mut Vec<Vec<Vec<i16>>>, effs: &mut Vec<Vec<Vec<i32>>>| {
        for z in 0..l {
            pots[z].push(net.layer_decide_potential(z));
            effs[z].push(net.layer_decide_effective_threshold(z));
        }
    };
    for (pos, &class) in classes.iter().enumerate() {
        if pos > 0 {
            for _ in 0..delay {
                net.wave(&[]);
                snapshot(net, &mut pots, &mut effs);
            }
        }
        for _ in 0..present {
            let sites = cue_sites(task_seed, size, class);
            net.wave(&sites);
            snapshot(net, &mut pots, &mut effs);
        }
    }
    let read_start = spikes_acc.lock().unwrap()[top].len();
    for _ in 0..read {
        net.wave(&[]);
        snapshot(net, &mut pots, &mut effs);
    }
    net.clear_listeners();
    let spikes = spikes_acc.lock().unwrap().clone();
    let mut act = vec![0f32; ls];
    for wv in spikes[top].iter().skip(read_start) {
        for &loc in wv {
            act[loc as usize] += 1.0;
        }
    }
    (act, TrialRecords { spikes, pots, effs })
}

fn build_signal(rec: &TrialRecords, w: &[Vec<f32>], err: &[f32], seed: u64, l: usize, ls: usize, top: usize, cfg: &TaskCfg) -> Vec<Vec<f32>> {
    let ttot = rec.spikes[top].len().max(1) as f32;
    let mut signal = vec![vec![0f32; ls]; l];
    for tz in 1..l {
        for j in 0..ls {
            let task_sig: f32 = (0..2)
                .map(|c| {
                    let b = if tz == top { w[c][j] } else { dfa_weight(seed, (tz * ls + j) as u32, c) };
                    b * err[c]
                })
                .sum();
            let fired_j = rec.spikes[tz].iter().filter(|wv| wv.contains(&(j as u32))).count() as f32;
            let rate = fired_j / ttot;
            signal[tz][j] = task_sig + cfg.rate_reg * (rate - cfg.rate_target);
        }
    }
    signal
}

/// Train the side-car with periodic held-out eval; **checkpoint the best model to `cache`** on every
/// improvement (so the file always holds the peak, immune to rate_reg over-training), stopping after
/// `PATIENCE` non-improving evals or `MAX_TRIALS`.
fn train_and_checkpoint(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, cache: &Path) {
    const EVAL_OFFSET: usize = 10_000_000;
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let top = l - 1;
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> (f32, f32) {
        (w[0].iter().zip(a).map(|(x, y)| x * y).sum(), w[1].iter().zip(a).map(|(x, y)| x * y).sum())
    };
    let (mut best, mut stale, mut t) = (0u64, 0usize, 0usize);
    while t < MAX_TRIALS {
        let stop = (t + EVAL_EVERY).min(MAX_TRIALS);
        while t < stop {
            let (classes, label) = task_parity(task_seed, t, PARITY_N);
            let (act, rec) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
            let (s0, s1) = score(&w, &act);
            let (p0, p1) = softmax2(s0, s1);
            let err = [p0 - if label == 0 { 1.0 } else { 0.0 }, p1 - if label == 1 { 1.0 } else { 0.0 }];
            for c in 0..2 {
                for j in 0..ls {
                    w[c][j] -= cfg.readout_lr * err[c] * act[j];
                }
            }
            if cfg.hidden_lr != 0.0 {
                let signal = build_signal(&rec, &w, &err, seed, l, ls, top, cfg);
                multilayer_dfa_step(net, entries, &rec, &signal, cfg.hidden_lr, &cfg.elig);
            }
            t += 1;
        }
        let mut correct = 0usize;
        for i in 0..cfg.holdout {
            let (classes, label) = task_parity(task_seed, EVAL_OFFSET + i, PARITY_N);
            let (act, _) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
            let (s0, s1) = score(&w, &act);
            if ((s1 > s0) as usize) == label {
                correct += 1;
            }
        }
        let acc = (correct as u64 * 1000) / cfg.holdout as u64;
        if acc > best {
            best = acc;
            stale = 0;
            net.save_model_path(cache).expect("checkpoint best side-car model");
            println!("  trial {t}: held-out {acc}permille (new best -> checkpointed)");
        } else {
            stale += 1;
            println!("  trial {t}: held-out {acc}permille (stale {stale}/{PATIENCE})");
            if stale >= PATIENCE {
                break;
            }
        }
    }
    // Defensive: guarantee the cache exists even in the pathological never-improved case.
    if !cache.exists() {
        net.save_model_path(cache).expect("checkpoint side-car model");
    }
    println!("side-car training done: best held-out {best}permille");
}

/// Load the cached trained side-car model, training + checkpointing it first if no cache exists.
fn ensure_model() -> Network {
    let path = cache_path();
    if path.exists() {
        println!("wave_bitnet side-car: loading cached model {}", path.display());
    } else {
        println!("wave_bitnet side-car: no cache at {} — training (early-exit best-checkpoint)...", path.display());
        let (mut net, entries) = make_sidecar(SEED, SIZE, FWD_C, FWD_R, REC_C, REC_R, ADAPT_BUMP, ADAPT_DECAY);
        train_and_checkpoint(&mut net, &entries, SEED, SEED, &sidecar_cfg(), &path);
    }
    Network::load_model_path(&path).expect("load cached side-car model")
}

fn setup_net() -> Network {
    let mut net = ensure_model();
    // Pure forward-throughput measurement: no training reads eligibility, so skip accruing it.
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

/// Print the loaded model's per-layer firing rates before the timed run (transparency for the load).
fn report_operating_point(net: &mut Network) {
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let rates = measure_rates(net, 32, 128, &input);
    let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
    println!("wave_bitnet trained side-car 32x32x5 per-layer firing rate (%): {pct:?}");
}

fn bench_throughput(c: &mut Criterion) {
    let mut net = setup_net();
    report_operating_point(&mut net);

    // Pre-generate the fixed noise ring OUTSIDE the timed loop.
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let noise: Vec<Vec<u32>> = (0..WAVES_PER_ITER as usize).map(&input).collect();

    let mut group = c.benchmark_group("throughput_bitnet");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    group.bench_function("sidecar_32x32x5_par3", |b| {
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
