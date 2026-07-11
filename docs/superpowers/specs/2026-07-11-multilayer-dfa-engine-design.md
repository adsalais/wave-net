# Multi-layer temporal-DFA training engine (bench-side, staged extraction)

**Date:** 2026-07-11
**Status:** approved design — pending TDD plan
**Branch:** `feat/multilayer-dfa-engine`

## Motivation

`bench/rsnn.rs` holds the multi-layer-DFA training rule twice: a **factored** feed-forward path
(`train_eprop{multi_layer}`, eligibility `e = pre_i·ψ_j` from the engine's integer accumulators) and a
**temporal** multi-topology path (`train_multilayer` / `train_recurrent`, eligibility
`e_ij = Σ_t pretr_i(t)·ψ_j(t)` optionally with the ALIF adaptation term εᵃ). The temporal path is the one
behind the side-car / hidden-rec winners (a strict improvement over FF on every benchmark, per `AGENTS.md`).

The last session moved the **factored** update primitive + a feed-forward driver into the engine
(`wave_net/eprop.rs`: `eprop_update`, `windowed_eligibility`, `train_ff`), establishing the boundary
*"the engine owns the update mechanism; the caller owns the learning signal."* The temporal path does not
yet fit that boundary because its eligibility is **not separable** into `pre_i·ψ_j`, so the existing
`eprop_update` cannot apply it.

This spec extracts the **temporal multi-topology multi-layer-DFA engine** into a **new self-contained bench
file**, and **extends the engine's update primitive** with a non-factored variant so the rule can be
expressed on engine primitives. The engine-to-be code lives in `bench` until it is proven, then it moves
into `wave_net` wholesale.

## Goals

- One reusable, **task-agnostic** temporal multi-topology training engine that trains **every trainable
  weight** (forward `+1`, lateral `0`, backward `−1/−2`, skip `+n`) via the temporal (optionally ALIF-εᵃ)
  eligibility × a caller-supplied DFA learning signal.
- Extend `wave_net`'s update primitive to consume a **per-synapse** (non-factored) eligibility.
- The new bench file **depends only on `wave_net`** — no dependency on any other `bench` file — so it lifts
  into the engine later without untangling.
- Its **own** test suite. `bench/rsnn.rs` is **never touched**; its trainers and benchmarks keep running
  identically.

## Non-goals

- No change to `bench/rsnn.rs` (no rerouting of `train_multilayer` / `train_recurrent` / `train_eprop`).
- No move into `wave_net` yet (only the update-primitive extension lands in the engine now).
- No new task/benchmark experiments beyond what the test suite needs to prove the engine works. The
  systematic scaling re-runs stay in `rsnn.rs` for now.
- The `subthreshold_psi` ramp ψ variant (a documented near-dead-end ablation) is intentionally **omitted**
  from the clean engine; it supports bump-ψ and spike-ψ only.

## Ownership boundary

| Owner | Holds | Where it lives now |
|---|---|---|
| **Engine** | `eprop_update_synaptic` (the extended, non-factored update primitive) | `wave_net/eprop.rs` |
| **Engine-to-be** (task-agnostic) | temporal eligibility builder + the multi-layer update-application step | top of `bench/multilayer_dfa.rs` → moves to `wave_net` later |
| **Bench** (task-specific) | trial runner, readout, DFA feedback + `rate_reg` (the signal), task closures, training loop, assertions | `#[cfg(test)]` at the bottom of `bench/multilayer_dfa.rs` |

The engine-to-be knows nothing about tasks, readout, or DFA. The seam between the two halves — which
*becomes* the future `wave_net` API — is plain data:

```
TrialRecords { spikes[z][t], pots[z][t], effs[z][t] }   // bench trial   → engine (eligibility input)
elig:   Vec<Vec<Vec<f32>>>  [z][entry_idx][i*count + k]  // engine-built (temporal_eligibility)
signal: Vec<Vec<f32>>       [layer][target-local j]      // bench-built (readout + DFA + rate_reg)
```

`signal` is indexed **per (target layer, target-local)**: the learning signal for neuron `j` in layer `tz`
is `task_sig(tz, j) + rate_reg·(rate[tz][j] − r_target)`, applied uniformly to **every** edge into `j`
regardless of edge type. `elig` stays per-edge because each edge's target geometry differs; `signal` is
shared across all edges into the same neuron, so the step passes `signal[tz]` to each edge's update.

> **Note — supersedes a current AGENTS.md "hard rule":** AGENTS.md states `rate_reg` belongs on the forward
> path *only* and *hurts* on recurrent weights, with the per-layer class-preserving `rec_stab` as the fix.
> Per the more recent conclusion (`rate_reg` is better across all layer types; `rec_stab` was not carrying
> its weight), this engine applies `rate_reg` to **every** edge type and keeps `rec_stab` **out of scope**
> (trivial to reinstate — one `signal`-builder branch — if a recurrent config later needs it). AGENTS.md and
> `docs/experiments_results.md` should be reconciled separately.

## Design

### A — `wave_net/eprop.rs`: the non-factored update primitive (only engine edit)

Add a sibling to `eprop_update`:

```rust
/// Apply one e-prop weight update to `entry_idx` of layer `source_z` using a caller-supplied
/// per-synapse eligibility `elig` (indexed [i*count + kk]) and per-target `signal` (indexed by
/// target local j): out_shadow[i*total_slots + slot_base + kk] += -lr · signal[j] · elig[i*count+kk],
/// then requantise. `target_of` recovers each synapse's target j (no re-scatter). No-op if the
/// entry's target layer is off the stack.
pub fn eprop_update_synaptic(&mut self, source_z: usize, entry_idx: usize,
                             elig: &[f32], signal: &[f32], lr: f32)
```

- Mirrors the exact `slot_base` / `total_slots` / `target_of` / requantize machinery of `eprop_update`, but
  as a **standalone method** — `eprop_update` is left **literally untouched** (its callers and tests do not
  move). Extracting a shared helper is **rejected**: f32 multiplication is not associative, so routing the
  factored term `-lr·signal·pre·psi` through a differently-grouped helper could shift last-bit rounding and
  break byte-identity. The ~12-line duplication is the safe trade.
- The per-synapse term is `out_shadow[i*total_slots + slot_base + kk] += -lr · signal[j] · elig[i*count+kk]`
  (`count` read from the entry's topology so `elig`'s `[i*count+kk]` indexing lines up).
- Target-layer range guard identical to `eprop_update` (`tz < 0 || tz >= l` → no-op). The stricter
  "L0 is not a trainable target" guard (`tz < 1`) lives in the caller's step (below), matching
  `train_multilayer`.
- Skip `i` when its whole eligibility row is zero (cheap early-out, optional).

### B — `bench/multilayer_dfa.rs`, engine-to-be section (depends only on `wave_net`)

Imports allowed: `wave_net::network::Network`, `wave_net::synapse::{key, mix, target_of, TopologyLevel}`.
Copies in (from `rsnn.rs`, ~15 lines, so the file is self-contained): `elig_adapt_sum`, `PSI_GAMMA = 0.3`,
`PSI_WIDTH = 16.0`.

**Types:**

```rust
/// One topology edge of a source layer, in the SAME order as the built LayerConfig topology, so slot
/// indices align with out_weights (the invariant train_multilayer's layer_entries maintains by hand).
pub struct Edge { pub level: i32, pub count: usize, pub radius: u32 }

/// Per-wave records for every layer over one trial (produced by the bench trial runner).
pub struct TrialRecords {
    pub spikes: Vec<Vec<Vec<u32>>>,  // [z][wave] = fired local ids
    pub pots:   Vec<Vec<Vec<i16>>>,  // [z][wave][local] = decide_potential
    pub effs:   Vec<Vec<Vec<i32>>>,  // [z][wave][local] = decide_eff threshold
}

/// Eligibility knobs (the engine's own; NOT task/readout).
pub struct EligParams {
    pub rec_tau: f32,          // presynaptic-trace decay time constant (waves)
    pub elig_beta: f32,        // ALIF adaptation coupling β (0 = membrane-only)
    pub elig_psi_width: f32,   // bump-ψ half-width W
    pub use_bump: bool,        // bump-ψ (centered at decide_eff) vs raw spike ψ
    pub adapt_decay: u8,       // ALIF adaptation decay shift → ρ = 1 − 2^(−adapt_decay)
}
```

**Eligibility builder** (pure given `net` seed/size, `entries`, `records`, `params`):

```rust
pub fn temporal_eligibility(net: &Network, entries: &[Vec<Edge>], rec: &TrialRecords,
                            p: &EligParams) -> Vec<Vec<Vec<f32>>>  // [z][entry_idx][i*count + k]
```

For each layer `z` and edge `(level, count, radius)` with target `tz = z + level`:
- `decay = 1 − 1/max(rec_tau, 1)`; per-layer presynaptic trace
  `pretr[z][t][i] = pretr[z][t−1][i]·decay + fired[z][t][i]`.
- Post factor `ψ[tz][t][j]`:
  - `use_bump` (implied whenever `elig_beta ≠ 0`): `max(0, PSI_GAMMA·(1 − |pots[tz][t][j] − effs[tz][t][j]| / max(elig_psi_width, 1)))`.
  - else: raw spike `fired[tz][t][j]`.
- `ρ = 1 − 2^(−adapt_decay)`, `β = elig_beta`, `use_adapt = β ≠ 0`.
- For each source `i`, slot `k`: `j = target_of(seed, z*ls + i, i, level, k, radius, size)`.
  - `use_adapt`: `e = elig_adapt_sum(ttot, β, ρ, |t| ψ[tz][t][j], |t| pretr[z][t][i])`.
  - else: `e = Σ_t pretr[z][t][i]·ψ[tz][t][j]`.
  - Off-stack / into-L0 targets (`tz < 1 || tz ≥ L`): `e = 0` (untrainable).
- Result indexed `[z][entry_idx][i*count + k]` to match `eprop_update_synaptic`.

**Update step:**

```rust
pub fn multilayer_dfa_step(net: &mut Network, entries: &[Vec<Edge>], rec: &TrialRecords,
                           signal: &[Vec<f32>], lr: f32, p: &EligParams)  // signal: [layer][j]
```

- `let elig = temporal_eligibility(net, entries, rec, p);`
- For each layer `z`, each edge index `e` with `tz = z + entries[z][e].level` in `[1, L−1]`:
  `net.eprop_update_synaptic(z, e, &elig[z][e], &signal[tz], lr);`
- Requantising the source layer once per edge is equivalent to accumulating then requantising once
  (`out_weights` is recomputed from `out_shadow` each call; the shadow accumulates), so no reordering needed.

### C — `bench/multilayer_dfa.rs`, test section (`#[cfg(test)]`, bench-owned)

Self-contained task harness (copies in only what it needs; no `rsnn.rs` import):
- **Trial runner** — reset, drive a cue sequence over `present`/`delay`/`read` waves, record per-wave
  fired-sets + `decide_potential` + `decide_eff` for **every** layer via the existing engine API
  (`on_layer`, `layer_decide_potential`, `layer_decide_effective_threshold`), returning `TrialRecords`
  and the top-layer read-window activity. This is a clean re-derivation of `sequence_trial_layers`.
- **Readout** — softmax + delta-rule linear readout on the top-layer activity; held-out accuracy.
- **DFA signal** — `dfa_weight` (copied, `P_DFA = 61`) for deeper layers, symmetric readout weights for the
  top; per-neuron `rate_reg·(rate[tz][j] − r_target)` folded in for **all** layers (no edge-type branch,
  no `rec_stab`); builds `signal[tz][j]` for every target layer `tz`.
- **Tasks** — minimal `xor_task` and `task_parity` closures (copied), plus small helpers `pick_ab` / `key`
  usage for deterministic bits.
- **Training loop** — per trial: run trial → readout error → build `signal` → `multilayer_dfa_step`.
- **Config builders** — `entries` + a matching `Network` for (a) a plain FF stack and (b) a side-car stack,
  built locally (mirroring `engine_config_sidecar`'s topology order).

### D — Tests

1. **primitive:** `eprop_update_synaptic` applies `Δ = −lr·signal[j]·elig[i*count+k]` on a radius-0/count-1
   entry (deterministic hand-computed delta, mirroring the existing `eprop_update` test).
2. **factored path unchanged:** the existing `eprop_update` / `train_ff` tests stay green — `eprop_update`
   is untouched (standalone new method), so this is a baseline check, not a new test.
3. **determinism:** the full training loop is a pure function of `(seed, config)` — identical result on
   repeat — for both `elig_beta = 0` (membrane) and `elig_beta > 0` (ALIF).
4. **learns > chance:** temporal XOR (or parity N=3) trains above chance held-out on a small net.
5. **multi-topology:** a side-car-style `entries` moves and trains its non-FF (`0` / `−1`) weights (assert
   the recurrent/backward `out_weights` change from init and held-out beats chance).

Plus `bench/mod.rs`: add `pub mod multilayer_dfa;`.

## Files touched

- `src/wave_net/eprop.rs` — add `eprop_update_synaptic` + shared private helper + unit test.
- `src/bench/multilayer_dfa.rs` — **new** (engine-to-be section + `#[cfg(test)]` test section).
- `src/bench/mod.rs` — one line: `pub mod multilayer_dfa;`.
- `docs/experiments_results.md` — a short note once the engine is proven (deferred to implementation).

## Constraints (from AGENTS.md)

- Rust edition 2024, **std-only** in `src/`, **no `unsafe`**, **warning-free build**.
- **Determinism** is a hard requirement — results a pure function of `(seed, config, input)`.
- `wave_state_machine` untouched; `bench/rsnn.rs` untouched.
- TDD, one commit per task, conventional-commit messages, **no `Co-Authored-By` trailer**.
- Never push.

## Invariant to watch

`entries[z]` must list layer `z`'s edges in the **same order** as its built `LayerConfig.topology`, so
`elig[z][e]`'s slot indexing matches `out_weights` (same requirement `train_multilayer`'s hand-maintained
`layer_entries` carries). The side-car test config is the guard: if the order is wrong, its non-FF weights
will not train and the test fails.

## Future (out of scope now)

- Move the engine-to-be section into `wave_net` once proven; at that point `temporal_eligibility` can read
  each layer's topology from the net directly and drop the `entries` parameter.
- Optionally reconcile `eprop_update` and `eprop_update_synaptic` into one primitive.
- Optionally add a ballpark-equivalence test against `rsnn::train_multilayer` (read-only) once both are
  stable — not bit-identical (update order differs), sanity only.
- Reinstate `rec_stab` (a per-layer class-preserving reg branch in the `signal` builder) *iff* a recurrent
  config shows `rate_reg` homogenizing its class signal; reconcile AGENTS.md / `experiments_results.md` with
  whichever conclusion holds.
