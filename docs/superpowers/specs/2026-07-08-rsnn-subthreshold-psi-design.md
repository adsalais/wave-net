# Sub-threshold ψ — design

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** replace the temporal eligibility's spike-time pseudo-derivative (`ψ = fired_j`) with a
**sub-threshold** one computed from the **decide-time membrane potential**, so credit flows to neurons that
are charged-but-not-firing during the silent gap. Test on the exact backward-recurrence config that just
nulled (temporal XOR, LIF, delay 20, size 16, depth 4) — an isolated A/B where only `ψ` changes. Small
engine addition (a decide-time potential snapshot); rest in `bench::rsnn`.

## Why

The backward-recurrence null (with topology + capacity controlled out) implicated `ψ = fired_j` as the
credit blocker: during a silent gap no neuron spikes, so `ψ = 0` everywhere and no signal reaches the
recurrent weights that would need to learn to *sustain* a memory. A **sub-threshold `ψ`** — nonzero when a
neuron's potential is near threshold even without a spike — lets that credit flow. This is the
evidence-backed bottleneck for both recurrence and (plausibly) the multi-layer DFA depth ceiling.

## The ψ (saturating ramp from the decide-time potential)

For each wave `t` and target neuron `j`:
`ψ_j(t) = clamp( decide_potential_j(t) / θ_j , 0, 1 )`
where `decide_potential_j(t)` is the membrane potential **at the decide step, before fire-reset and leak**,
and `θ_j` is the (baseline) threshold. At decide time a firing neuron has `v ≥ θ → ψ ≈ 1`, and a
sub-threshold neuron gets `v/θ` — so the decide-time potential **subsumes** the old `fired` term (no
`max(fired, …)` needed). Used in the temporal eligibility exactly as before:
`e_ij = Σ_t pre_trace_i(t) · ψ_j(t)`, then `Δw = −lr · L_j · e_ij`.

**Why decide-time, not post-wave:** reading the potential *after* `net.wave()` is post-leak (biased low) and
post-reset (fired neurons → 0) — a confound that would muddy a null. Snapshotting at decide removes it, so
an A/B on `ψ` is clean.

## Engine change (minimal)

- **`neurons.rs`:** `Layer.decide_potential: Vec<i16>` (init zeros).
- **`wave.rs`:** in the decide loop, snapshot `decide_potential[i] = potential[i]` for every neuron
  **before** the fire/reset check (the pre-fire value). No other behavior changes.
- **`network.rs`:** zero `decide_potential` in `reset_state`; add `layer_decide_potential(z) -> Vec<i16>`
  (bulk read, like `layer_thresholds`).

Deterministic, integer, forward-only — no change to firing dynamics.

## Bench change (`src/bench/rsnn.rs`)

- `RsnnConfig.subthreshold_psi: bool` (default `false` — spike-time `ψ`, preserving current behavior).
- `xor_trial_layers` also records per-wave decide-potentials per layer (`layer_decide_potential(z)` after
  each `net.wave`), returning `(read activity, spikes, decide_potentials)`.
- `train_recurrent` builds the postsynaptic factor per layer:
  `post[z][t][j] = if subthreshold_psi { clamp(decide_pot[z][t][j] / θ[z][j], 0, 1) } else { fired[z][t][j] }`
  (`θ[z] = layer_thresholds(z)`), and uses `post` in the eligibility in place of `fired`.

## Success criterion

Three-way, on the identical backward config (temporal XOR, LIF, delay 20, size 16, depth 4), held-out +
multi-seed:
- **FF** (baseline, ~chance),
- **backward + spike-`ψ`** (the confirmed null, ~chance),
- **backward + sub-threshold `ψ`** — does it lift above chance and above FF?

**Success = the sub-threshold variant clears chance+margin and beats FF/spike-ψ**, worst-seed, deterministic.

**Honesty gate:** if sub-threshold `ψ` *still* nulls, the confound is now removed, so the blocker is deeper
than "credit during gaps" — the *sustaining dynamics* themselves (the floored leak kills the trace before
recurrence can hold it) or genuinely needing surrogate-gradient BPTT. Report the null and that narrowed
conclusion; do not fudge or single-seed.

## Testing

- `decide_potential_snapshots_pre_reset` (engine) — a firing neuron's `decide_potential ≥ threshold` while
  its post-wave `potential == 0`; a sub-threshold neuron's `decide_potential` equals its charge.
- `subthreshold_psi_vs_spike_psi_on_temporal_xor` (bench, `#[ignore]`, release) — the three-way headline.
- `subthreshold_psi_is_deterministic`.
- Regression: whole suite (incl. `wave_state_machine`) green; `subthreshold_psi=false` path unchanged.

## Deferred

Sub-threshold `ψ` on the *feed-forward* multi-layer DFA path (to test the depth ceiling); surrogate-gradient
BPTT (if the dynamics, not credit, prove to be the wall); a slower recurrent-layer leak (if sustaining is
the residual blocker).
