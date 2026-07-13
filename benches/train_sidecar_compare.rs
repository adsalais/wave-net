//! Criterion benchmark comparing the **two engines' training cost** on the *same* side-car topology
//! (parity N=4, rec_count 16). It times **one full training trial** — drive the trial over its waves,
//! build the eligibility, and apply one weight update — for each engine:
//!
//! - `wave_bitnet` (the size-bound engine): records the whole trial (`TrialRecords` — per-wave
//!   spikes + decide-time pot/eff snapshots for every neuron) then computes eligibility in the offline
//!   `O(L·size²·T·count)` `temporal_eligibility` pass (bump-ψ ALIF εᵃ, its winning side-car config).
//! - `wave_driven` (the activity-bound engine): accrues the eligibility **online** during `wave()`
//!   (spike-ψ ALIF εᵃ), so the trial cost scales with activity, and `dfa_update` reads it.
//!
//! Both run the identical side-car graph and the identical fixed drive, at size 16 and 32, so the
//! ratio is a like-for-like measure of "how much cheaper is the online activity-scaled trainer on the
//! config that reached ceiling (parity N=4, size 32, rec 16)."
//!
//! Note: the timed closure applies the weight update, so the net trains slightly over criterion's
//! samples; at the sub-critical side-car operating point (adaptation + rate_reg regulate the rate) the
//! per-trial *activity* — and hence the cost — stays stable, so this measures throughput, not learning.
//! A fixed synthetic error drives the learning signal (we measure machinery cost, not accuracy).
//!
//! Observed (median, one trial): size 16 → bitnet 8.6 ms vs driven 4.6 ms (~1.85×); size 32 → 36.6 ms
//! vs 20.4 ms (~1.80×). The speedup is a **constant factor** (not growing with size) because the
//! side-car's firing rate is a fixed *fraction* of the layer, so activity ∝ size² for both engines —
//! wave_driven's win here is the constant "skip the quiescent neurons + accrue eligibility online"
//! factor, not an asymptotic one. The asymptotic activity-scaling win appears when activity is sparse
//! *relative* to size (much larger layers at fixed absolute activity / lower firing rates).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::{Arc, Mutex};

// Engine-agnostic task drive (wave_bitnet's hash; wave_driven's `mix`/`key` are byte-identical copies).
use wave_net::wave_bitnet::synapse::{key, mix};

// side-car params (match bench::wave_driven_bench::make_sidecar and benches/throughput_bitnet.rs)
const UC: u32 = 32; // forward count
const UR: u32 = 3; // forward radius
const REC: u32 = 16; // recurrent count (rec 16)
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

// ------------------------------------------------------------------------------------------------
// wave_bitnet: offline eligibility (record whole trial -> temporal_eligibility -> update)
// ------------------------------------------------------------------------------------------------
mod bitnet {
    use super::*;
    use wave_net::wave_bitnet::config::{Config, LayerConfig};
    use wave_net::wave_bitnet::multilayer_dfa::{multilayer_dfa_step, Edge, EligParams, TrialRecords, PSI_WIDTH};
    use wave_net::wave_bitnet::network::Network;
    use wave_net::wave_bitnet::synapse::TopologyLevel;

    pub fn make_sidecar(seed: u64, size: u32) -> (Network, Vec<Vec<Edge>>) {
        let mk = |topology| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump: ADAPT_BUMP, adapt_decay: ADAPT_DECAY };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: UR, count: UC }]),
            mk(vec![TopologyLevel { level: 2, radius: UR, count: UC }]),
            mk(vec![TopologyLevel { level: 0, radius: R, count: REC }, TopologyLevel { level: 1, radius: R, count: REC }]),
            mk(vec![TopologyLevel { level: -1, radius: R, count: REC }, TopologyLevel { level: 1, radius: UR, count: UC }]),
            mk(vec![]),
        ];
        let mut net = Network::new(Config { seed, size, layers });
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

    fn build_signal(rec: &TrialRecords, l: usize, ls: usize, top: usize) -> Vec<Vec<f32>> {
        let ttot = rec.spikes[top].len().max(1) as f32;
        let err = [0.1f32, -0.1f32]; // fixed synthetic error (machinery cost, not accuracy)
        let mut signal = vec![vec![0f32; ls]; l];
        for tz in 1..l {
            for j in 0..ls {
                let task_sig: f32 = (0..2).map(|c| { let b = if tz == top { 0.05 } else { dfa_weight(SEED, (tz * ls + j) as u32, c) }; b * err[c] }).sum();
                let fired_j = rec.spikes[tz].iter().filter(|wv| wv.contains(&(j as u32))).count() as f32;
                signal[tz][j] = task_sig + RATE_REG * (fired_j / ttot - RATE_TARGET);
            }
        }
        signal
    }

    /// One training trial: drive (recording spikes + decide-time pot/eff) -> offline eligibility -> update.
    pub fn train_trial(net: &mut Network, entries: &[Vec<Edge>], drive: &[Vec<u32>], size: u32) {
        let l = net.layer_count();
        let ls = (size * size) as usize;
        let top = l - 1;
        let spikes_acc: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
        for z in 0..l {
            let acc = spikes_acc.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| acc.lock().unwrap()[z].push(f.to_vec())));
        }
        net.reset_state();
        let mut pots: Vec<Vec<Vec<i16>>> = vec![Vec::new(); l];
        let mut effs: Vec<Vec<Vec<i32>>> = vec![Vec::new(); l];
        for dv in drive {
            net.wave(dv);
            for z in 0..l {
                pots[z].push(net.layer_decide_potential(z));
                effs[z].push(net.layer_decide_effective_threshold(z));
            }
        }
        net.clear_listeners();
        let spikes = spikes_acc.lock().unwrap().clone();
        let rec = TrialRecords { spikes, pots, effs };
        let signal = build_signal(&rec, l, ls, top);
        let p = EligParams { rec_tau: 20.0, elig_beta: 0.4, elig_psi_width: PSI_WIDTH, use_bump: true, adapt_decay: ADAPT_DECAY };
        multilayer_dfa_step(net, entries, &rec, &signal, HIDDEN_LR, &p);
    }
}

// ------------------------------------------------------------------------------------------------
// wave_driven: online eligibility (accrue during wave() -> update)
// ------------------------------------------------------------------------------------------------
mod driven {
    use super::*;
    use wave_net::wave_driven::config::{Config, LayerConfig};
    use wave_net::wave_driven::network::Network;
    use wave_net::wave_driven::synapse::TopologyLevel;
    use wave_net::wave_driven::training::{Edge, EligParams};

    pub fn make_sidecar(seed: u64, size: u32) -> (Network, Vec<Vec<Edge>>) {
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

    /// One training trial: drive (online accrual during wave) -> update.
    pub fn train_trial(net: &mut Network, entries: &[Vec<Edge>], drive: &[Vec<u32>], size: u32) {
        net.reset_state();
        for dv in drive {
            net.wave(dv);
        }
        let signal = build_signal(net, size, drive.len());
        net.dfa_update(entries, &signal, HIDDEN_LR);
    }
}

fn bench_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_sidecar_par4_rec16");
    for &size in &[16u32, 32u32] {
        let drive = build_drive(SEED, size);

        let (mut bn, bentries) = bitnet::make_sidecar(SEED, size);
        group.bench_with_input(BenchmarkId::new("wave_bitnet_offline", size), &size, |b, &size| {
            b.iter(|| bitnet::train_trial(&mut bn, &bentries, &drive, size))
        });

        let (mut dn, dentries) = driven::make_sidecar(SEED, size);
        group.bench_with_input(BenchmarkId::new("wave_driven_online", size), &size, |b, &size| {
            b.iter(|| driven::train_trial(&mut dn, &dentries, &drive, size))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_train);
criterion_main!(benches);
