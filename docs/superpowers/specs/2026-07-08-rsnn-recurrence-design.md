# Recurrence via trained level-0 weights + temporal e-prop — design

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** make `wave_net` a *recurrent* trained SNN. Add **trainable `level 0` lateral** weights and a
**temporal (per-wave) e-prop eligibility**, and demonstrate on **temporal XOR** — a task a feed-forward net
cannot do. Increment 1 verifies FF is ~chance; increment 2 shows recurrence lifts it. `level 0` only
(generalize to `−1`/multi-layer later); memory optimization deferred. `wave_state_machine` + its pinned
bench untouched; all work in `wave_net` + `bench::rsnn`.

## Why temporal XOR

The RSNN pivot (feed-forward e-prop) learns reliably, but the store-recall task is solvable feed-forward, so
it can't show recurrence earning its keep. **Temporal XOR needs both memory and nonlinearity:** present cue
`A` then (after a gap) cue `B`; the answer is `A XOR B`. A feed-forward net sees `A` and `B` at different
times, so its accumulated state is ~additive `f(A)+f(B)` — and XOR is *not* linearly separable in `(A,B)`,
so a linear readout on additive activity **can't** solve it. **Recurrence** lets held-`A` state *interact*
with `B` (a coincidence/multiplicative term), making XOR separable. So XOR cleanly isolates what recurrence
adds — provided the FF baseline really is ~chance, which we verify first.

## Benchmark — temporal XOR (increment 1)

Per trial: draw `A, B ∈ {0,1}` deterministically from `(task_seed, trial)`. Run:
`reset → present cue(A) for present_waves → delay → present cue(B) for present_waves → read_waves`.
`cue(x)` reuses `cue_realization(task_seed, size, class=x, …)`. **Label = A XOR B** (K=2). The readout reads
reservoir activity over the **read window (after B)** — where held-`A` ⊗ `B` interaction must appear; with no
memory, that window reflects only `B` → chance.

**Increment 1 deliverable:** the temporal-XOR task + a test that the current FF path (`train_eprop`,
recurrence off) is **near chance** (say < 600‰ held-out, multi-seed). If FF already solves it (adaptation
holds `A` enough), the task doesn't isolate recurrence — report and make it harder (longer delay / weaker
adaptation) before proceeding. Honest gate, not a forced result.

## Recurrence — trained level-0 weights (increment 2)

- **Topology:** add `TopologyLevel { level: 0, radius, count }` to the hidden layer (the engine already
  supports `level 0`; only the FF configs omitted it — **no engine change**). Its weights are the hidden
  layer's `out_weights` level-0 slots — already stored int8, already trainable.
- **Temporal eligibility (bench-side, from recorded spikes):** record per-wave fired-sets (listeners give
  these per wave). For a recurrent synapse `i→j`,
  `e_ij = Σ_t pre_trace_i(t) · fired_j(t)`,
  where `pre_trace_i(t)` is a decaying trace of `i`'s spikes (time-constant `tau` = the lingering that lets
  `A` reach `B`) and `fired_j(t)` is the spike-time pseudo-derivative. This is the *sum-of-per-wave-products*
  (not product-of-sums) that captures the A→B temporal correlation. Computed after the trial from the
  recorded activity — the "store per-wave activity, recompute" path (scaling-friendly; no per-synapse engine
  state).
- **Update:** `Δw_ij = −lr · L_j · e_ij`, `L_j` = the symmetric-feedback learning signal from the readout
  (as in the FF e-prop). Quantize the shadow → int8. Recurrent weights are the hidden layer's level-0 slots;
  regenerate `j = target_of(level 0, …)` to pair each stored weight with its post neuron.

**What's trained:** readout + FF (`level+1`) weights + the new recurrent (`level 0`) weights.

## Success criterion

- **Recurrence lifts temporal XOR:** held-out test accuracy with recurrence-on is clearly above the
  (verified ~chance) FF baseline, **multi-seed** (worst seed still above chance + margin). Report FF-baseline
  vs recurrence-on per seed.
- Determinism: pure function of `(seed, task_seed, config)`.

**Honesty gate:** temporal XOR is genuinely hard (nonlinear + temporal); recurrence + crude spike-timing
eligibility may not crack it. If it doesn't beat the FF baseline after reasonable tuning (`tau`, recurrent
`radius/count`, `lr`), that is the finding — report it (points to `level −1`, per-synapse temporal
eligibility with proper `ψ`, or surrogate-gradient BPTT). Never a single seed or prequential number.

## Module & engine

- **Engine:** none required (`level 0` topology already supported; per-wave spikes via existing listeners).
- **Bench (`src/bench/rsnn.rs`):** the temporal-XOR trial runner, per-wave spike recording, the temporal
  eligibility, the recurrent-weight update, and the held-out/multi-seed harness. `target_of` gains a `level`
  argument (or a level-0 variant).

## Determinism & constraints

- Engine stays integer + deterministic; `f32` shadow/eligibility live in the bench. Single-threaded.
- Reuse `cue_realization`/`pick_class` as engine-agnostic task helpers. Held-out + multi-seed from the start.

## Testing

- `temporal_xor_ff_is_near_chance` (increment 1) — the FF baseline fails, multi-seed.
- `recurrence_lifts_temporal_xor` (increment 2) — recurrence-on beats the FF baseline, worst-seed above
  chance+margin.
- `recurrence_is_deterministic`.
- Regression: whole suite (incl. `wave_state_machine`) stays green.

## Deferred

`level −1` backward recurrence; multi-layer recurrent credit; per-synapse / shared temporal eligibility and
weight sharing (memory optimization); surrogate-gradient BPTT as the heavier alternative if e-prop stalls.
