# Multi-layer temporal-DFA training engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, task-agnostic temporal multi-topology multi-layer-DFA training engine in a new `bench` file, and extend the engine's e-prop update primitive with a non-factored (per-synapse) variant so the rule runs on engine primitives.

**Architecture:** The only `wave_net` change is a new standalone `Network::eprop_update_synaptic`. The new file `bench/multilayer_dfa.rs` holds the *engine-to-be* (temporal eligibility + the multi-layer update step, `pub`) and, in `#[cfg(test)]`, the *bench-owned* task harness (trial runner, readout, DFA signal, tasks, loop, assertions). It depends **only** on `wave_net` — no other bench file — so it lifts into the engine later untouched. `bench/rsnn.rs` is never modified.

**Tech Stack:** Rust, edition 2024, std-only, `cargo test`.

## Global Constraints

- Rust edition 2024; **std-only** in `src/`; **no `unsafe`**; **warning-free build** (`cargo build`).
- **Determinism** is a hard requirement — every result a pure function of `(seed, config, input)`.
- `wave_state_machine` untouched; `bench/rsnn.rs` untouched; `eprop_update` left **literally unchanged**.
- TDD; **one commit per task**; conventional-commit messages; **NO `Co-Authored-By` trailer**.
- **Never push.** Work stays on branch `feat/multilayer-dfa-engine`.
- `rate_reg` is applied to **all** edge types; `rec_stab` is out of scope (see spec note).
- Spec: `docs/superpowers/specs/2026-07-11-multilayer-dfa-engine-design.md`.

---

### Task 1: Engine primitive `eprop_update_synaptic`

**Files:**
- Modify: `src/wave_net/eprop.rs` (add one method in the `impl Network` block, after `eprop_update`)
- Test: `src/wave_net/eprop.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `Network::seed_val`, `size`, `layer_count`, `with_layer`, `with_layer_mut`, `synapse::target_of`, `Layer.{topology,total_slots,out_shadow,out_weights}` (all already used by `eprop_update`).
- Produces: `pub fn eprop_update_synaptic(&mut self, source_z: usize, entry_idx: usize, elig: &[f32], signal: &[f32], lr: f32)` — `elig` indexed `[i*count + kk]`, `signal` indexed by target-local `j`.

- [ ] **Step 1: Write the failing test**

In `src/wave_net/eprop.rs`, inside `mod tests`, add:

```rust
#[test]
fn eprop_update_synaptic_applies_expected_delta() {
    // radius-0, count-1 up entry: target of source local i is local i in the layer above.
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
    let mut net = Network::new(Config { seed: 1, size: 4, layers: vec![lc; 2] });
    let ls = 16;
    let elig = vec![2.0f32; ls];   // [i*count + kk], count = 1 -> indexed by i
    let signal = vec![0.5f32; ls]; // per target-local j
    let before = net.with_layer(0, |l| l.out_shadow[0]);
    net.eprop_update_synaptic(0, 0, &elig, &signal, 0.1);
    let after = net.with_layer(0, |l| l.out_shadow[0]);
    // Δ = -lr·signal[0]·elig[0] = -0.1·0.5·2 = -0.1
    assert!((after - before + 0.1).abs() < 1e-4, "{before} -> {after}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p wave_state_machine --lib eprop_update_synaptic_applies_expected_delta 2>&1 | tail -20`
(If the crate name differs, use `cargo test eprop_update_synaptic_applies_expected_delta`.)
Expected: FAIL — `no method named eprop_update_synaptic`.

- [ ] **Step 3: Write minimal implementation**

In `src/wave_net/eprop.rs`, in the `impl Network` block right after `eprop_update`, add:

```rust
/// Apply one e-prop weight update to `entry_idx` of layer `source_z` from a caller-supplied
/// **per-synapse** eligibility `elig` (indexed `[i*count + kk]`) and per-target `signal` (indexed by
/// target local `j`): `out_shadow[i*total_slots + slot_base + kk] += -lr · signal[j] · elig[i*count+kk]`,
/// then requantise. `target_of` recovers each synapse's target `j` (no re-scatter). No-op if the entry's
/// target layer is off the stack. Standalone (does NOT share code with `eprop_update`) so the factored
/// path stays byte-identical — f32 multiplication is not associative.
pub fn eprop_update_synaptic(&mut self, source_z: usize, entry_idx: usize, elig: &[f32], signal: &[f32], lr: f32) {
    let seed = self.seed_val();
    let size = self.size();
    let l = self.layer_count();
    let ls = (size as usize) * (size as usize);
    let (level, radius, count, slot_base, total_slots) = self.with_layer(source_z, |lz| {
        let e = &lz.topology[entry_idx];
        let slot_base: usize = lz.topology[..entry_idx].iter().map(|t| t.count as usize).sum();
        (e.level, e.radius, e.count as usize, slot_base, lz.total_slots)
    });
    let tz = source_z as i32 + level;
    if tz < 0 || tz as usize >= l {
        return;
    }
    let base = source_z * ls;
    self.with_layer_mut(source_z, |lz| {
        for i in 0..ls {
            let sg = (base + i) as u32;
            for kk in 0..count {
                let j = target_of(seed, sg, i as u32, level, kk as u32, radius, size) as usize;
                lz.out_shadow[i * total_slots + slot_base + kk] += -lr * signal[j] * elig[i * count + kk];
            }
        }
        for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
            *wq = s.round().clamp(-127.0, 127.0) as i8;
        }
    });
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test eprop_update_synaptic_applies_expected_delta && cargo test eprop_update_applies_expected_delta train_ff_moves_weights_by_signal`
Expected: PASS for the new test, and the two existing `eprop_update`/`train_ff` tests still PASS (factored path untouched).

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/eprop.rs
git commit -m "feat(wave_net): add non-factored eprop_update_synaptic primitive"
```

---

### Task 2: New module + temporal eligibility builder

**Files:**
- Create: `src/bench/multilayer_dfa.rs`
- Modify: `src/bench/mod.rs` (add `pub mod multilayer_dfa;`)
- Test: `src/bench/multilayer_dfa.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `Network::{seed_val,size,layer_count}`, `synapse::target_of`.
- Produces: `struct Edge { level: i32, count: usize, radius: u32 }`; `struct TrialRecords { spikes: Vec<Vec<Vec<u32>>>, pots: Vec<Vec<Vec<i16>>>, effs: Vec<Vec<Vec<i32>>> }`; `struct EligParams { rec_tau: f32, elig_beta: f32, elig_psi_width: f32, use_bump: bool, adapt_decay: u8 }`; `pub const PSI_WIDTH: f32 = 16.0;`; `pub fn temporal_eligibility(net: &Network, entries: &[Vec<Edge>], rec: &TrialRecords, p: &EligParams) -> Vec<Vec<Vec<f32>>>` (indexed `[z][entry_idx][i*count + k]`).

- [ ] **Step 1: Create the module skeleton and register it**

Create `src/bench/multilayer_dfa.rs` with the header, imports, types, and copied helpers:

```rust
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
```

Then add to `src/bench/mod.rs` (alphabetical-ish, next to the others):

```rust
pub mod multilayer_dfa;
```

- [ ] **Step 2: Write the failing test**

Append to `src/bench/multilayer_dfa.rs`:

```rust
#[cfg(test)]
mod tests {
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
}
```

- [ ] **Step 3: Run tests to verify they fail, then pass**

Run: `cargo test -p wave_state_machine multilayer_dfa 2>&1 | tail -20` (or `cargo test multilayer_dfa`).
Because the implementation in Step 1 is already complete, both tests should PASS on first run. If `temporal_eligibility` were missing they would fail to compile — this confirms the module is wired.

- [ ] **Step 4: Verify warning-free build**

Run: `cargo build 2>&1 | grep -i warning; echo "exit: $?"`
Expected: no warnings printed.

- [ ] **Step 5: Commit**

```bash
git add src/bench/multilayer_dfa.rs src/bench/mod.rs
git commit -m "feat(bench): multilayer_dfa module + temporal eligibility builder"
```

---

### Task 3: The multi-layer update step

**Files:**
- Modify: `src/bench/multilayer_dfa.rs` (add `multilayer_dfa_step` after `temporal_eligibility`)
- Test: `src/bench/multilayer_dfa.rs` (`mod tests`)

**Interfaces:**
- Consumes: `temporal_eligibility` (Task 2), `Network::{layer_count, eprop_update_synaptic}` (Task 1).
- Produces: `pub fn multilayer_dfa_step(net: &mut Network, entries: &[Vec<Edge>], rec: &TrialRecords, signal: &[Vec<f32>], lr: f32, p: &EligParams)` — `signal` indexed `[layer][target-local j]`.

- [ ] **Step 1: Write the failing test**

In `mod tests`, add:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test multilayer_dfa_step_raises_weights_on_negative_signal 2>&1 | tail -20`
Expected: FAIL — `cannot find function multilayer_dfa_step`.

- [ ] **Step 3: Write minimal implementation**

In `src/bench/multilayer_dfa.rs`, after `temporal_eligibility`, add:

```rust
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test multilayer_dfa_step_raises_weights_on_negative_signal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bench/multilayer_dfa.rs
git commit -m "feat(bench): multilayer_dfa_step over eprop_update_synaptic"
```

---

### Task 4: Bench-owned task harness (trial runner, readout, signal, tasks)

**Files:**
- Modify: `src/bench/multilayer_dfa.rs` (add helpers inside `mod tests`)
- Test: same (`mod tests`)

**Interfaces:**
- Consumes: `Network::{on_layer,reset_state,wave,clear_listeners,layer_decide_potential,layer_decide_effective_threshold,layer_count,with_layer}`, `synapse::{key,mix}`, `config::{Config,LayerConfig}`, `multilayer_dfa_step`, `TrialRecords`, `Edge`, `EligParams`.
- Produces (test-only): `cue_sites`, `run_trial`, `softmax2`, `dfa_weight`, `TaskCfg`, `build_signal`, `train_and_eval`, `make_ff`, `xor_task`, `single_task`.

- [ ] **Step 1: Add the harness helpers**

Extend the `use` lines at the top of `mod tests` to:

```rust
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::{key, mix, TopologyLevel};
    use std::sync::{Arc, Mutex};
```

Then add, inside `mod tests`:

```rust
    const CUE_P: u64 = 0xC0E;
    const P_DFA: u64 = 61; // fixed random DFA feedback (copied from rsnn — this file has no rsnn dep)

    /// Deterministic, class-distinct L0 spike pattern (~25% density), stable across waves.
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

    /// Drive a cue sequence and record per-wave fired-sets + decide potential/eff for EVERY layer.
    /// Returns (top-layer read-window spike counts, records). Bench owns the trial.
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

    struct TaskCfg {
        size: u32,
        present: usize,
        delay: usize,
        read: usize,
        trials: usize,
        holdout: usize,
        readout_lr: f32,
        hidden_lr: f32,
        rate_reg: f32,
        rate_target: f32,
        elig: EligParams,
    }

    /// Bench readout + DFA + rate_reg → `signal[tz][j]` (symmetric readout on top, random DFA deeper;
    /// per-neuron rate_reg on ALL layers — no rec_stab, per spec).
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

    /// Full training loop (bench-owned) over the engine step. Returns held-out accuracy permille.
    fn train_and_eval(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
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

    /// Feed-forward net of `layers` layers + matching `entries` (each layer but the top has one +1 edge).
    fn make_ff(seed: u64, size: u32, layers: usize, up_count: u32, up_radius: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
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

    /// Single-cue 2-class task: present class c, label = c. Immediately separable (fast learning check).
    fn single_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let c = (mix(key(seed, t as u32, 0, 0, 71)) & 1) as usize;
        (vec![c], c)
    }

    /// Temporal XOR: two cue bits (a, b), label = a XOR b.
    fn xor_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let a = (mix(key(seed, t as u32, 0, 0, 51)) & 1) as usize;
        let b = (mix(key(seed, t as u32, 0, 0, 53)) & 1) as usize;
        (vec![a, b], a ^ b)
    }
```

- [ ] **Step 2: Write the failing test (trial runner shape + determinism)**

In `mod tests`, add:

```rust
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
```

- [ ] **Step 3: Run test to verify it fails, then passes**

Run: `cargo test run_trial_records_are_shaped_and_deterministic 2>&1 | tail -20`
Expected: after Step 1 the helpers exist, so this PASSES. (Before Step 1 it fails to compile — the helper additions are what make it build.)

- [ ] **Step 4: Verify warning-free build and full suite**

Run: `cargo test multilayer_dfa && cargo build 2>&1 | grep -i warning; echo done`
Expected: tests PASS; no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/bench/multilayer_dfa.rs
git commit -m "test(bench): multilayer_dfa task harness (trial runner, readout, signal, tasks)"
```

---

### Task 5: Learns-above-chance tests

**Files:**
- Modify: `src/bench/multilayer_dfa.rs` (`mod tests`)

**Interfaces:**
- Consumes: `train_and_eval`, `make_ff`, `single_task`, `xor_task`, `EligParams`, `TaskCfg`, `PSI_WIDTH` (Task 4).

- [ ] **Step 1: Write the fast learns test + determinism-of-training**

In `mod tests`, add:

```rust
    fn ff_cfg(trials: usize, hidden_lr: f32, elig_beta: f32) -> TaskCfg {
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

    #[test]
    fn multilayer_dfa_learns_separable_2class_above_chance() {
        // Deep (4-layer) FF net in the generous-fan-in regime; a 2-class separable task must train well
        // above chance when every layer is trained (multi-layer DFA + rate_reg). Empirical bar: if this
        // ever dips below ~600, raise up_count/trials (the generous-fan-in target regime) — do NOT lower
        // the bar toward chance (500).
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 8, 4, 16, 3, 20, 6);
        let acc = train_and_eval(&mut net, &entries, seed, seed, &ff_cfg(500, 0.004, 0.0), single_task);
        assert!(acc > 600, "2-class separable should train above chance: {acc}");
    }

    #[test]
    fn training_is_deterministic_both_elig_flavors() {
        let seed = 0x1234_5678u64;
        let run = |beta: f32| {
            let (mut net, entries) = make_ff(seed, 8, 4, 16, 3, 20, 6);
            train_and_eval(&mut net, &entries, seed, seed, &ff_cfg(120, 0.004, beta), single_task)
        };
        assert_eq!(run(0.0), run(0.0), "membrane-eligibility training must be deterministic");
        assert_eq!(run(0.4), run(0.4), "ALIF-eligibility training must be deterministic");
    }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test multilayer_dfa_learns_separable_2class_above_chance training_is_deterministic_both_elig_flavors 2>&1 | tail -20`
Expected: PASS. If the learns test is below 600, increase `up_count` (16→24/32) or `trials` in the test and re-run — do not lower the assertion.

- [ ] **Step 3: Add the expensive temporal-XOR demonstration (ignored)**

In `mod tests`, add:

```rust
    #[test]
    #[ignore] // expensive; run manually in --release: the real temporal task
    fn multilayer_dfa_learns_temporal_xor() {
        // Temporal XOR (memory across a gap): FF-readout-only is ~chance; training every layer must clear it.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 16, 3, 20, 6);
        let mut cfg = ff_cfg(1500, 0.004, 0.4);
        cfg.size = 16;
        cfg.delay = 8;
        cfg.holdout = 400;
        let acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, xor_task);
        assert!(acc > 640, "temporal XOR should train above chance: {acc}");
    }
```

- [ ] **Step 4: Run the ignored test once to confirm it works**

Run: `cargo test --release multilayer_dfa_learns_temporal_xor -- --ignored --nocapture 2>&1 | tail -20`
Expected: PASS (`acc > 640`). If it does not clear the bar, tune `up_count`/`delay`/`rate_reg`/`trials` in the test (mirror the working `rsnn::deep_ff_ratereg_robustness` config: size 16, up_count 16–32, delay 12–20, rate_reg 5) — the goal is to demonstrate learning, not to lower the bar.

- [ ] **Step 5: Commit**

```bash
git add src/bench/multilayer_dfa.rs
git commit -m "test(bench): multilayer_dfa learns (2-class fast + temporal-XOR ignored) + training determinism"
```

---

### Task 6: Multi-topology (side-car) test

**Files:**
- Modify: `src/bench/multilayer_dfa.rs` (`mod tests`)

**Interfaces:**
- Consumes: `train_and_eval`, `EligParams`, `TaskCfg`, `Edge`, `Config`, `LayerConfig`, `TopologyLevel`, `Network::with_layer`, `xor_task`.

- [ ] **Step 1: Write the failing test**

In `mod tests`, add:

```rust
    /// Backward-fed side-car (5 layers), mirroring rsnn::engine_config_sidecar's topology ORDER exactly so
    /// entries line up with out_weights: L0→L1(+1); L1→L3(+2 skip); L2 self(0)+ →L3(+1); L3 →L2(−1)+ →L4(+1);
    /// L4 read.
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

    #[test]
    fn multilayer_dfa_trains_sidecar_recurrent_edges() {
        // The side-car's non-FF edges (L2 self-loop level 0, L3→L2 backward level −1) must be trained, not
        // just the forward path. Assert the recurrent-layer (L2) stored weights change from their ±1 init.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_sidecar(seed, 8, 12, 3, 8, 3, 20, 6);
        let before = net.with_layer(2, |lz| lz.out_weights.clone());
        let cfg = TaskCfg {
            size: 8,
            present: 6,
            delay: 6,
            read: 6,
            trials: 300,
            holdout: 100,
            readout_lr: 0.02,
            hidden_lr: 0.004,
            rate_reg: 5.0,
            rate_target: 0.1,
            elig: EligParams { rec_tau: 20.0, elig_beta: 0.4, elig_psi_width: PSI_WIDTH, use_bump: true, adapt_decay: 6 },
        };
        let _acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, xor_task);
        let after = net.with_layer(2, |lz| lz.out_weights.clone());
        assert_ne!(before, after, "side-car recurrent-layer (L2) weights must train (level-0 + forward edges)");
    }
```

- [ ] **Step 2: Run test to verify it fails (compile), then passes**

Run: `cargo test multilayer_dfa_trains_sidecar_recurrent_edges 2>&1 | tail -20`
Expected: PASS — L2 carries a level-0 self-loop + a level-1 forward edge, both trained, so its `out_weights` move from the ±1 init after 300 trials.

- [ ] **Step 3: Verify the whole suite + warning-free build**

Run: `cargo test multilayer_dfa && cargo build 2>&1 | grep -i warning; echo done`
Expected: all `multilayer_dfa` tests PASS (the `#[ignore]`d XOR one is skipped); no warnings.

- [ ] **Step 4: Commit**

```bash
git add src/bench/multilayer_dfa.rs
git commit -m "test(bench): multilayer_dfa trains side-car recurrent/backward edges"
```

---

## Self-Review

**Spec coverage:**
- Non-factored primitive (spec A) → Task 1. ✓
- Types + copied helpers + `temporal_eligibility` (spec B) → Task 2. ✓
- `multilayer_dfa_step` + `tz ∈ [1,L−1]` guard (spec B) → Task 3. ✓
- Bench trial runner / readout / DFA-signal / tasks / loop (spec C) → Task 4. ✓
- Tests D1 (primitive delta) → Task 1; D2 (factored unchanged, baseline) → Task 1 Step 4; D3 (determinism, both flavors) → Task 2 + Task 5; D4 (learns>chance) → Task 5; D5 (multi-topology) → Task 6. ✓
- `rate_reg` on all edges, no `rec_stab` (spec seam note) → `build_signal` (Task 4). ✓
- `signal[layer][j]` seam, `elig[z][entry][i*count+k]` → Tasks 2–4. ✓
- `mod.rs` registration → Task 2. ✓
- Self-contained (only `wave_net` imports; `elig_adapt_sum`/`dfa_weight`/`PSI_*` copied) → Tasks 2, 4. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; the two empirical learning thresholds carry explicit tuning instructions (raise fan-in/trials, never lower the bar), not placeholders.

**Type consistency:** `Edge{level,count,radius}`, `TrialRecords{spikes,pots,effs}`, `EligParams{rec_tau,elig_beta,elig_psi_width,use_bump,adapt_decay}`, `temporal_eligibility(&Network,&[Vec<Edge>],&TrialRecords,&EligParams)->Vec<Vec<Vec<f32>>>`, `multilayer_dfa_step(&mut Network,&[Vec<Edge>],&TrialRecords,&[Vec<f32>],f32,&EligParams)`, `eprop_update_synaptic(usize,usize,&[f32],&[f32],f32)` — used identically across Tasks 1–6. `signal[tz]` (per-layer) matches `build_signal`'s `[layer][j]` output. `elig[z][e_idx]` (`[i*count+k]`) matches the primitive's `elig[i*count+kk]`. ✓

## Notes for the executor

- The crate is a library named `wave_state_machine` (per `src/lib.rs`); if `cargo test <name>` is ambiguous, scope with `-p wave_state_machine --lib`.
- Determinism uses fixed single-threaded f32 accumulation order — do not parallelize the loops.
- The two learning-outcome assertions (Task 5) are empirical. Run them; if a bar is missed, tune the *config* (fan-in, trials, delay, rate_reg) toward the generous-fan-in regime described in `AGENTS.md`. Never weaken an assertion below a clear margin over chance (500).
