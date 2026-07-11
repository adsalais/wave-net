//! `multilayer_dfa` — a self-contained temporal multi-topology multi-layer-DFA training engine, staged in
//! `bench` until proven, then lifted into `wave_net`. Depends ONLY on `wave_net` (no other bench file).
//! Engine-to-be (this module): temporal eligibility + the multi-layer update step over
//! `Network::eprop_update_synaptic`. Bench-owned (the `#[cfg(test)]` block): the trial, readout, DFA
//! signal, tasks, and loop. Spec: docs/superpowers/specs/2026-07-11-multilayer-dfa-engine-design.md.

use crate::wave_net::network::Network;
use crate::wave_net::synapse::target_of;

/// One topology edge of a source layer, in the SAME order as the built `LayerConfig` topology, so slot
/// indices align with `out_weights` (the invariant `rsnn::train_multilayer`'s `layer_entries` keeps by hand).
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub level: i32,
    pub count: usize,
    pub radius: u32,
}

/// Per-wave records for every layer over one trial (produced by the bench trial runner).
pub struct TrialRecords {
    pub spikes: Vec<Vec<Vec<u32>>>, // [z][wave] = fired local ids
    pub pots: Vec<Vec<Vec<i16>>>,   // [z][wave][local] = decide_potential
    pub effs: Vec<Vec<Vec<i32>>>,   // [z][wave][local] = decide_eff threshold
}

/// Temporal-eligibility knobs (the engine's own — NOT task/readout).
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub rec_tau: f32,        // presynaptic-trace decay time constant (waves)
    pub elig_beta: f32,      // ALIF adaptation coupling β (0 = membrane-only)
    pub elig_psi_width: f32, // bump-ψ half-width W
    pub use_bump: bool,      // bump-ψ (centered at decide_eff) vs raw spike ψ
    pub adapt_decay: u8,     // ALIF adaptation decay shift → ρ = 1 − 2^(−adapt_decay)
}

/// Sane default bump-ψ half-width in i16 potential units. (Copied from rsnn to keep this file free of
/// bench-file dependencies.)
pub const PSI_WIDTH: f32 = 16.0;

/// Dampening γ for the bump pseudo-derivative ψ = γ·max(0, 1−|v−eff|/W). (Copied from rsnn.)
const PSI_GAMMA: f32 = 0.3;

/// Σ_t of the ALIF eligibility trace e_ij(t) = ψ_j(t)·(εᵛ_i(t) − β·εᵃ_ij(t)), εᵃ recursed at ρ.
/// β = 0 reduces to the plain membrane trace Σ_t ψ·εᵛ. (Copied from rsnn — Bellec et al. 2020, Eq. 24–25.)
fn elig_adapt_sum(ttot: usize, beta: f32, rho: f32, psi: impl Fn(usize) -> f32, ev: impl Fn(usize) -> f32) -> f32 {
    let mut eps_a = 0.0f32;
    let mut e = 0.0f32;
    for tt in 0..ttot {
        let p = psi(tt);
        let v = ev(tt);
        e += p * (v - beta * eps_a);
        eps_a = p * v + (rho - beta * p) * eps_a;
    }
    e
}

/// Temporal per-synapse eligibility for every layer/edge from one trial's per-wave records.
/// Returns `e[z][entry_idx][i*count + k]`; off-stack / into-L0 targets are 0 (untrainable).
pub fn temporal_eligibility(net: &Network, entries: &[Vec<Edge>], rec: &TrialRecords, p: &EligParams) -> Vec<Vec<Vec<f32>>> {
    let seed = net.seed_val();
    let size = net.size();
    let l = net.layer_count();
    let ls = (size as usize) * (size as usize);
    let ttot = rec.spikes[l - 1].len();
    // fired[z][t][j] ∈ {0,1}
    let mut fired = vec![vec![vec![0f32; ls]; ttot]; l];
    for z in 0..l {
        for (t, wv) in rec.spikes[z].iter().enumerate() {
            for &loc in wv {
                fired[z][t][loc as usize] = 1.0;
            }
        }
    }
    // pretr[z][t][i]: decaying presynaptic trace
    let decay = 1.0 - 1.0 / p.rec_tau.max(1.0);
    let mut pretr = vec![vec![vec![0f32; ls]; ttot]; l];
    for z in 0..l {
        for i in 0..ls {
            let mut tr = 0.0f32;
            for t in 0..ttot {
                tr = tr * decay + fired[z][t][i];
                pretr[z][t][i] = tr;
            }
        }
    }
    let use_adapt = p.elig_beta != 0.0;
    let use_bump = p.use_bump || use_adapt;
    let rho = 1.0 - (2.0f32).powi(-(p.adapt_decay as i32));
    // ψ[z][t][j]: bump centered on decide_eff, else raw spike
    let mut psi = vec![vec![vec![0f32; ls]; ttot]; l];
    for z in 0..l {
        for t in 0..ttot {
            for j in 0..ls {
                psi[z][t][j] = if use_bump {
                    (PSI_GAMMA * (1.0 - (rec.pots[z][t][j] as f32 - rec.effs[z][t][j] as f32).abs() / p.elig_psi_width.max(1.0))).max(0.0)
                } else {
                    fired[z][t][j]
                };
            }
        }
    }
    // per (layer, edge): e_ij correlation
    let mut out: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l);
    for z in 0..l {
        let mut layer_out: Vec<Vec<f32>> = Vec::with_capacity(entries[z].len());
        for edge in &entries[z] {
            let count = edge.count;
            let mut e_entry = vec![0f32; ls * count];
            let tz_i = z as i32 + edge.level;
            if tz_i >= 1 && (tz_i as usize) < l {
                let tz = tz_i as usize;
                for i in 0..ls {
                    let sg = (z * ls + i) as u32;
                    for k in 0..count {
                        let j = target_of(seed, sg, i as u32, edge.level, k as u32, edge.radius, size) as usize;
                        e_entry[i * count + k] = if use_adapt {
                            elig_adapt_sum(ttot, p.elig_beta, rho, |t| psi[tz][t][j], |t| pretr[z][t][i])
                        } else {
                            let mut s = 0f32;
                            for t in 0..ttot {
                                s += pretr[z][t][i] * psi[tz][t][j];
                            }
                            s
                        };
                    }
                }
            }
            layer_out.push(e_entry);
        }
        out.push(layer_out);
    }
    out
}

/// One training step: build the temporal eligibility from `rec`, then update **every** trainable edge via
/// `Network::eprop_update_synaptic` with the caller-supplied per-target-layer `signal` (`signal[tz][j]`).
/// Edges whose target is off-stack or into L0 (`tz ∉ [1, L−1]`) are skipped (untrainable). Requantising the
/// source layer once per edge is equivalent to accumulating then requantising once.
pub fn multilayer_dfa_step(net: &mut Network, entries: &[Vec<Edge>], rec: &TrialRecords, signal: &[Vec<f32>], lr: f32, p: &EligParams) {
    let l = net.layer_count();
    let elig = temporal_eligibility(net, entries, rec, p);
    for z in 0..l {
        for (e_idx, edge) in entries[z].iter().enumerate() {
            let tz_i = z as i32 + edge.level;
            if tz_i < 1 || tz_i as usize >= l {
                continue;
            }
            let tz = tz_i as usize;
            net.eprop_update_synaptic(z, e_idx, &elig[z][e_idx], &signal[tz], lr);
        }
    }
}

/// Bench-owned test/benchmark harness for `multilayer_dfa` — the trial runner, readout, DFA signal, tasks,
/// net builders, and liveness reports. Shared by this file's unit tests and `multilayer_dfa_bench.rs`.
/// Test-only (`#[cfg(test)]`); it does NOT move to `wave_net` with the engine.
#[cfg(test)]
pub(crate) mod harness {
    use super::{multilayer_dfa_step, Edge, EligParams, TrialRecords, PSI_WIDTH};
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::critical_init::{forward_avalanche, layer_rates};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::{key, mix, TopologyLevel};
    use std::sync::{Arc, Mutex};

    pub(crate) const CUE_P: u64 = 0xC0E;
    pub(crate) const P_DFA: u64 = 61; // fixed random DFA feedback (copied from rsnn — no rsnn dep)

    /// Deterministic, class-distinct L0 spike pattern (~25% density), stable across waves.
    pub(crate) fn cue_sites(task_seed: u64, size: u32, class: usize) -> Vec<u32> {
        let ls = (size * size) as u32;
        (0..ls).filter(|&loc| mix(key(task_seed, loc, class as i32, 0, CUE_P)) & 3 == 0).collect()
    }

    pub(crate) fn softmax2(z0: f32, z1: f32) -> (f32, f32) {
        let m = z0.max(z1);
        let (e0, e1) = ((z0 - m).exp(), (z1 - m).exp());
        let s = (e0 + e1).max(1e-30);
        (e0 / s, e1 / s)
    }

    pub(crate) fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
        if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
    }

    /// Drive a cue sequence and record per-wave fired-sets + decide potential/eff for EVERY layer.
    /// Returns (top-layer read-window spike counts, records). Bench owns the trial.
    pub(crate) fn run_trial(net: &mut Network, size: u32, classes: &[usize], task_seed: u64, present: usize, delay: usize, read: usize) -> (Vec<f32>, TrialRecords) {
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

    pub(crate) struct TaskCfg {
        pub(crate) size: u32,
        pub(crate) present: usize,
        pub(crate) delay: usize,
        pub(crate) read: usize,
        pub(crate) trials: usize,
        pub(crate) holdout: usize,
        pub(crate) readout_lr: f32,
        pub(crate) hidden_lr: f32,
        pub(crate) rate_reg: f32,
        pub(crate) rate_target: f32,
        pub(crate) elig: EligParams,
    }

    /// Bench readout + DFA + rate_reg → `signal[tz][j]` (symmetric readout on top, random DFA deeper;
    /// per-neuron rate_reg on ALL layers — no rec_stab, per spec).
    pub(crate) fn build_signal(rec: &TrialRecords, w: &[Vec<f32>], err: &[f32], seed: u64, l: usize, ls: usize, top: usize, cfg: &TaskCfg) -> Vec<Vec<f32>> {
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

    /// Full training loop (bench-owned) over the engine step. Returns held-out accuracy permille.
    pub(crate) fn train_and_eval(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let mut w = vec![vec![0f32; ls]; 2];
        let score = |w: &[Vec<f32>], a: &[f32]| -> (f32, f32) {
            (w[0].iter().zip(a).map(|(x, y)| x * y).sum(), w[1].iter().zip(a).map(|(x, y)| x * y).sum())
        };
        for t in 0..cfg.trials {
            let (classes, label) = task(task_seed, t);
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
        }
        let mut correct = 0usize;
        for t in cfg.trials..cfg.trials + cfg.holdout {
            let (classes, label) = task(task_seed, t);
            let (act, _) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
            let (s0, s1) = score(&w, &act);
            if ((s1 > s0) as usize) == label {
                correct += 1;
            }
        }
        (correct as u64 * 1000) / cfg.holdout as u64
    }

    /// Like `train_and_eval` but with a **training-duration axis**: train continuously and evaluate held-out
    /// accuracy at each cumulative trial count in `checkpoints` (ascending), against a FIXED held-out set
    /// (disjoint from training, same across checkpoints). One training pass to `checkpoints.last()`; returns
    /// one accuracy permille per checkpoint. `cfg.trials` is ignored.
    pub(crate) fn train_and_eval_curve(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize), checkpoints: &[usize]) -> Vec<u64> {
        const EVAL_OFFSET: usize = 10_000_000; // held-out trials, disjoint from any training length
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let mut w = vec![vec![0f32; ls]; 2];
        let score = |w: &[Vec<f32>], a: &[f32]| -> (f32, f32) {
            (w[0].iter().zip(a).map(|(x, y)| x * y).sum(), w[1].iter().zip(a).map(|(x, y)| x * y).sum())
        };
        let mut accs = Vec::with_capacity(checkpoints.len());
        let mut trained = 0usize;
        for &ckpt in checkpoints {
            for t in trained..ckpt {
                let (classes, label) = task(task_seed, t);
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
            }
            trained = ckpt;
            let mut correct = 0usize;
            for i in 0..cfg.holdout {
                let (classes, label) = task(task_seed, EVAL_OFFSET + i);
                let (act, _) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                if ((s1 > s0) as usize) == label {
                    correct += 1;
                }
            }
            accs.push((correct as u64 * 1000) / cfg.holdout as u64);
        }
        accs
    }

    /// Feed-forward net of `layers` layers + matching `entries` (each layer but the top has one +1 edge).
    pub(crate) fn make_ff(seed: u64, size: u32, layers: usize, up_count: u32, up_radius: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay,
        };
        let net = Network::new(Config { seed, size, layers: vec![lc; layers] });
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: up_count as usize, radius: up_radius }] })
            .collect();
        (net, entries)
    }

    /// Multi-topology stack (3 layers) with a LEVEL-0 self-recurrence on the ACTIVE middle layer L1:
    /// L0→L1(+1); L1 self(0) + →L2(+1); L2 read. L1 sits on the forward path (driven by L0), so its
    /// recurrent edge reliably gets real eligibility.
    pub(crate) fn make_hidden_rec(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
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
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }, TopologyLevel { level: 0, radius: r, count: n }]),
            mk(vec![]),
        ];
        let net = Network::new(Config { seed, size, layers });
        let entries = vec![
            vec![Edge { level: 1, count: uc as usize, radius: ur }],
            vec![Edge { level: 1, count: uc as usize, radius: ur }, Edge { level: 0, count: n as usize, radius: r }],
            vec![],
        ];
        (net, entries)
    }

    /// Backward-fed side-car (5 layers), mirroring `rsnn::engine_config_sidecar`'s topology ORDER exactly so
    /// `entries` line up with `out_weights`: L0→L1(+1); L1→L3(+2 skip); L2 self(0)+ →L3(+1); L3 →L2(−1)+
    /// →L4(+1); L4 read. Forward/skip path uses (uc, ur); the recurrent side-car uses its own (n, r).
    pub(crate) fn make_sidecar(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
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

    /// Single-cue 2-class task: present class c, label = c. Immediately separable (fast learning check).
    pub(crate) fn single_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let c = (mix(key(seed, t as u32, 0, 0, 71)) & 1) as usize;
        (vec![c], c)
    }

    /// Temporal XOR: two cue bits (a, b), label = a XOR b.
    pub(crate) fn xor_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let a = (mix(key(seed, t as u32, 0, 0, 51)) & 1) as usize;
        let b = (mix(key(seed, t as u32, 0, 0, 53)) & 1) as usize;
        (vec![a, b], a ^ b)
    }

    /// N-bit sequential parity: N deterministic cue bits, label = their XOR (non-monotone; needs memory).
    pub(crate) fn task_parity(seed: u64, t: usize, n: usize) -> (Vec<usize>, usize) {
        let bits: Vec<usize> = (0..n).map(|i| (mix(key(seed, t as u32, 0, i as u32, 51)) & 1) as usize).collect();
        let label = bits.iter().fold(0usize, |a, &b| a ^ b);
        (bits, label)
    }

    /// Default training config (size overridden by the caller). ALIF on; `rate_reg` liveness on all layers.
    pub(crate) fn ff_cfg(trials: usize, hidden_lr: f32, elig_beta: f32) -> TaskCfg {
        TaskCfg {
            size: 8,
            present: 6,
            delay: 4,
            read: 6,
            trials,
            holdout: 200,
            readout_lr: 0.02,
            hidden_lr,
            rate_reg: 5.0,
            rate_target: 0.1,
            elig: EligParams { rec_tau: 6.0, elig_beta, elig_psi_width: PSI_WIDTH, use_bump: elig_beta != 0.0, adapt_decay: 6 },
        }
    }

    /// Per-layer firing rate (fraction of neurons firing per wave) under random L0 drive — the substrate's
    /// spiking-activity profile through depth. Mutates (drives) the net; weights are untouched.
    pub(crate) fn per_layer_rates(net: &mut Network, drive_seed: u64) -> Vec<f64> {
        layer_rates(net, drive_seed, 20000, 32, 96)
    }

    /// σ branching ratio: geometric per-hop growth of a single-spike avalanche through the computational
    /// layers (>1 super-critical/grows, <1 sub-critical/dies, ≈1 critical), from `forward_avalanche`'s
    /// per-layer footprint. Returns 0.0 if the avalanche is dead everywhere.
    pub(crate) fn sigma_ratio(net: &mut Network, drive_seed: u64) -> f64 {
        let fp = forward_avalanche(net, drive_seed, 20000, 16, 8, 1);
        let l = fp.len();
        let (mut logsum, mut n) = (0.0f64, 0usize);
        for z in 1..l.saturating_sub(1) {
            if fp[z] > 0.0 && fp[z + 1] > 0.0 {
                logsum += (fp[z + 1] / fp[z]).ln();
                n += 1;
            }
        }
        if n == 0 { 0.0 } else { (logsum / n as f64).exp() }
    }
}

#[cfg(test)]
mod tests {
    use super::harness::*;
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    // A 2-layer, radius-0, count-1 up net: target of source local i is local i above.
    fn tiny_net() -> Network {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 0, count: 1 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 6,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        Network::new(Config { seed: 1, size: 4, layers: vec![lc; 2] })
    }

    // Records where every neuron of both layers fires on each of `ttot` waves.
    fn dense_records(ls: usize, l: usize, ttot: usize) -> TrialRecords {
        let all: Vec<u32> = (0..ls as u32).collect();
        TrialRecords {
            spikes: vec![vec![all.clone(); ttot]; l],
            pots: vec![vec![vec![0i16; ls]; ttot]; l],
            effs: vec![vec![vec![1i32; ls]; ttot]; l],
        }
    }

    #[test]
    fn temporal_eligibility_membrane_matches_hand_computed() {
        let net = tiny_net();
        let ls = 16;
        let ttot = 3;
        let rec = dense_records(ls, 2, ttot);
        let entries = vec![vec![Edge { level: 1, count: 1, radius: 0 }], vec![]];
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let e = temporal_eligibility(&net, &entries, &rec, &p);
        // membrane, spike-ψ (fired=1 every wave): pretr = 1, 1.75, 2.3125 (decay = 1 - 1/4 = 0.75).
        // e = Σ_t pretr_i(t)·fired_j(t) = 1 + 1.75 + 2.3125 = 5.0625 for every synapse.
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].len(), 1); // one edge on layer 0
        assert_eq!(e[1].len(), 0); // top layer: no outgoing edge
        for &v in &e[0][0] {
            assert!((v - 5.0625).abs() < 1e-4, "e = {v}");
        }
    }

    #[test]
    fn temporal_eligibility_beta_changes_result_and_is_deterministic() {
        let net = tiny_net();
        let rec = dense_records(16, 2, 4);
        let entries = vec![vec![Edge { level: 1, count: 1, radius: 0 }], vec![]];
        let base = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let adapt = EligParams { elig_beta: 0.4, ..base };
        let e0a = temporal_eligibility(&net, &entries, &rec, &base);
        let e0b = temporal_eligibility(&net, &entries, &rec, &base);
        let ea = temporal_eligibility(&net, &entries, &rec, &adapt);
        assert_eq!(e0a, e0b, "eligibility must be deterministic");
        assert!(ea[0][0] != e0a[0][0], "β>0 (ALIF εᵃ) must change the eligibility");
    }

    #[test]
    fn multilayer_dfa_step_raises_weights_on_negative_signal() {
        let mut net = tiny_net();
        let ls = 16;
        let rec = dense_records(ls, 2, 3);
        let entries = vec![vec![Edge { level: 1, count: 1, radius: 0 }], vec![]];
        // signal into the top layer (tz = 1) is negative → weights should rise (fire more).
        let signal = vec![vec![0.0f32; ls], vec![-1.0f32; ls]];
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let before: f32 = net.with_layer(0, |lz| lz.out_shadow.iter().sum());
        multilayer_dfa_step(&mut net, &entries, &rec, &signal, 0.02, &p);
        let after: f32 = net.with_layer(0, |lz| lz.out_shadow.iter().sum());
        // Δ per synapse = -lr·signal·e = -0.02·(-1)·5.0625 > 0
        assert!(after > before + 1.0, "negative signal + positive eligibility must raise layer-0 weights: {before} -> {after}");
    }

    #[test]
    fn run_trial_records_are_shaped_and_deterministic() {
        let (mut net1, _e) = make_ff(7, 8, 3, 12, 3, 20, 6);
        let (mut net2, _e2) = make_ff(7, 8, 3, 12, 3, 20, 6);
        let (act1, rec1) = run_trial(&mut net1, 8, &[0, 1], 7, 4, 2, 4);
        let (act2, rec2) = run_trial(&mut net2, 8, &[0, 1], 7, 4, 2, 4);
        let l = 3;
        let ls = 64;
        // every layer recorded the same number of waves for spikes/pots/effs
        let ttot = rec1.spikes[l - 1].len();
        assert!(ttot > 0);
        for z in 0..l {
            assert_eq!(rec1.spikes[z].len(), ttot);
            assert_eq!(rec1.pots[z].len(), ttot);
            assert_eq!(rec1.effs[z].len(), ttot);
            assert_eq!(rec1.pots[z][0].len(), ls);
        }
        // determinism: same (seed, config, input) → identical records + activity
        assert_eq!(act1, act2);
        assert_eq!(rec1.spikes, rec2.spikes);
        assert_eq!(rec1.pots, rec2.pots);
        assert_eq!(rec1.effs, rec2.effs);
    }

    #[test]
    fn multilayer_dfa_learns_separable_2class_above_chance() {
        // Deep (4-layer) FF net, generous fan-in (size 16, up_count 32) so every layer stays alive; a
        // 2-class separable task trains to ceiling when every layer is trained (multi-layer DFA + rate_reg).
        // Empirical: 4 layers at size 8 is fan-in-starved (dead deep layers → chance); size 16 + up 32 is
        // the target regime. If this dips below ~600, raise up_count/size/trials — do NOT lower the bar
        // toward chance (500).
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 20, 6);
        let mut cfg = ff_cfg(400, 0.004, 0.0);
        cfg.size = 16;
        let acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, single_task);
        assert!(acc > 750, "2-class separable should train well above chance: {acc}");
    }

    #[test]
    fn training_is_deterministic_both_elig_flavors() {
        // A live config (L3, size 8, up 32) so both the membrane and ALIF-εᵃ paths actually train; the
        // point is bit-reproducibility of the whole loop.
        let seed = 0x1234_5678u64;
        let run = |beta: f32| {
            let (mut net, entries) = make_ff(seed, 8, 3, 32, 3, 20, 6);
            train_and_eval(&mut net, &entries, seed, seed, &ff_cfg(120, 0.004, beta), single_task)
        };
        assert_eq!(run(0.0), run(0.0), "membrane-eligibility training must be deterministic");
        assert_eq!(run(0.4), run(0.4), "ALIF-eligibility training must be deterministic");
    }

    #[test]
    #[ignore] // expensive; run manually in --release: the real temporal task
    fn multilayer_dfa_learns_temporal_xor() {
        // Temporal XOR (memory across a gap): FF-readout-only is ~chance; training every layer (ALIF +
        // rate_reg, generous fan-in) must clear it.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 20, 6);
        let mut cfg = ff_cfg(1500, 0.004, 0.4);
        cfg.size = 16;
        cfg.delay = 8;
        cfg.holdout = 400;
        let acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, xor_task);
        assert!(acc > 640, "temporal XOR should train above chance: {acc}");
    }

    #[test]
    fn multilayer_dfa_trains_recurrent_edge() {
        // Multi-topology: the LEVEL-0 self-recurrence on the active middle layer must be trained, not just the
        // forward path. Assert the recurrent-edge trainable accumulator (out_shadow) on L1 moves from its ±1
        // init — the shadow is the precise "did this edge receive updates" signal (the quantised i8 weight
        // only flips once the shadow crosses ±0.5, which the small DFA signal need not do in a short run).
        let seed = 0xE9_0B_0A17u64;
        let (uc, n) = (32u32, 12u32);
        let (mut net, entries) = make_hidden_rec(seed, 8, uc, 3, n, 3, 20, 6);
        // L1 out_shadow layout: total_slots = uc + n; the level-0 (recurrent) slots are [uc .. uc+n) per source.
        let rec_shadow = |net: &Network| -> Vec<f32> {
            net.with_layer(1, |lz| {
                let ts = lz.total_slots;
                let ls = lz.out_shadow.len() / ts;
                let mut v = Vec::new();
                for i in 0..ls {
                    for k in 0..(n as usize) {
                        v.push(lz.out_shadow[i * ts + uc as usize + k]);
                    }
                }
                v
            })
        };
        let before = rec_shadow(&net);
        // spike-ψ (beta 0): robust eligibility. (bump-ψ collapses to ~0 under strong drive when the potential
        // overshoots eff by more than PSI_WIDTH — see the PSI_WIDTH note; the ALIF-εᵃ path is covered by the
        // eligibility unit tests.)
        let mut cfg = ff_cfg(400, 0.004, 0.0);
        cfg.size = 8;
        let _acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, single_task);
        let after = rec_shadow(&net);
        assert_ne!(before, after, "the level-0 recurrent edge on the active layer must train (non-FF path)");
    }
}
