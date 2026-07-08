# Temporal XOR bench — design (Spec 2b of the training test bench)

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** one iteration — the **temporal XOR** task `y(t) = u(t) ⊕ u(t−τ)`, swept over `τ`, for
{recurrent, feed-forward} × {ALIF, LIF}, producing accuracy-vs-`τ` curves. Reuses the binned bit-stream
harness and `RidgeReadout` from Spec 2. **No** adding/copy (still deferred), **no** e-prop, **no** datasets.

## Why this

MC (Spec 2) measured *linear echo* and LIF won; the finding was **"adaptation trades linear echo for
nonlinear computation."** Temporal XOR is nonlinear, so it is the **direct test of the other half of that
tradeoff:** does ALIF's nonlinearity — the thing that cost it MC — *buy* nonlinear temporal computation?

- **ALIF > LIF on XOR** → confirms the tradeoff; both memory axes are now mapped (linear echo → LIF;
  nonlinear temporal → ALIF).
- **ALIF ≤ LIF** → adaptation's memory is even more narrowly *held-category* (store-recall) only. Also a
  real, informative result.

We do **not** assume the direction — we run it and report the truth (honesty gate).

## Success criterion

For each of the four runs, an accuracy-vs-`τ` curve (fraction of held-out timesteps where the thresholded
linear readout gets `u(t) ⊕ u(t−τ)` right; chance = 50%). Assertions, deterministic:

1. **Sanity (firm):** the reservoir solves XOR **above chance at small `τ`** (e.g. `τ = 1`, accuracy well
   over 500‰) — XOR is not linearly separable in the raw inputs, so this confirms the reservoir
   nonlinearly separates it and the readout works. If this fails the setup is broken.
2. **ALIF-vs-LIF (report the truth):** assert whatever the tuned curves show — the direction is the
   experiment's finding, not an assumption. If ALIF does not beat LIF, that is reported, not forced; the
   sanity assertion is never weakened to manufacture a win.
3. `temporal_xor_is_deterministic` — identical curves across two runs.

## The task

- **Stream & state:** identical to MC — a binary i.i.d. bit stream `u(t)` driven in bins of `B` waves
  (continuous, no reset), state `x(t)` = per-neuron spike counts over the bin (+ bias). Reuses the MC
  streaming verbatim.
- **Target:** `y(t) = u(t) ⊕ u(t−τ) ∈ {0,1}`. The only thing that differs from MC is the target column;
  the design matrix `X` (the states) is the **same for every `τ`**, so factor the ridge **once** and
  solve per `τ`.
- **Readout:** `RidgeReadout` fit to the `{0,1}` target, **thresholded at 0.5** → predicted class →
  accuracy on the test split. A linear classifier on the reservoir state.

## Architecture — a small shared-streaming refactor, then the XOR module

The bit-stream + binned state collection currently lives inside `memory_capacity.rs`. Extract it so both
tasks share it (DRY; MC behavior unchanged):

```
src/bench/
  stream.rs          # NEW: bit(), stream_pattern(), collect_states(), engine_config()  [moved from
                     #      memory_capacity, made pub(crate)]
  memory_capacity.rs # refactored to call stream::* (behavior identical)
  temporal_xor.rs    # NEW: XorConfig, XorCurve, temporal_xor(), tests
  readout.rs         # RidgeReadout (reused, unchanged)
  linalg.rs          # reused, unchanged
```

`stream.rs` holds the shared streaming primitives and the engine-config builder (recurrent adds
`level 0/−1`, feed-forward is `level+1`-only; both use the dense drive the floored leak requires).

```rust
// bench/temporal_xor.rs
pub struct XorConfig { /* seed, size, layers, baseline_init, adapt_bump, adapt_decay, bit_seed,
                         stream_density_q16, bin_waves, warmup_bins, collect_bins, taus: Vec<usize>,
                         lambda, train_frac_permille, calib, calib_fraction_q16 */ }
pub struct XorCurve { pub taus: Vec<usize>, pub accuracy_permille: Vec<u64> }
/// Build+calibrate one variant, stream, and fit a thresholded ridge classifier per τ for u(t)⊕u(t−τ).
pub fn temporal_xor(cfg: &XorConfig, adapt_bump: i16, recurrent: bool) -> XorCurve;
```

`temporal_xor`: build+calibrate the variant, `stream::collect_states`, then with rows `[τ_max, split)`
train / `[split, n)` test (same `X` across `τ`), fit `RidgeReadout` once; per `τ` build the XOR target,
solve weights, predict, threshold at 0.5, accuracy → permille.

## Testing (inline `#[cfg(test)]`)

- `xor_target_is_correct` — `u(t) ⊕ u(t−τ)` matches a hand-computed small example (independent of engine).
- `xor_solvable_above_chance_at_small_tau` — recurrent reservoir solves `τ=1` XOR well above chance (both
  variants, or at least the stronger one) — the sanity gate.
- `temporal_xor_alif_vs_lif` — run all four; print the accuracy-vs-`τ` curves; assert the *observed*
  relationship (tuned; honesty gate — report a null/LIF-favoring result rather than forcing an ALIF win).
- `temporal_xor_is_deterministic` — identical curves across two runs.

## Determinism & constraints

- Std-only, no new deps. Engine (`src/wave_net/`) untouched — bench uses the public API only.
- `f64` in the bench (readout/metric); single-threaded, fixed reduction order → deterministic.
- Everything a pure function of `(seed, config, params)`.

## Parameters (starting points, tunable)

`size 8`, `4 layers`, `B = 3`, warmup `~150` bins, `T ≈ 1500` bins (~70/30 split), `τ ∈ {1,2,4,8,16}`,
`λ = 1.0`. Same dense drive as MC.

## Scope guard (YAGNI) — explicitly out

- adding / copy — still deferred.
- e-prop / internal training — Spec 3.
- external datasets — Tier 2.
- Graded input, sweeping `B`/λ as the deliverable — not needed to answer "does ALIF's nonlinearity buy
  nonlinear temporal computation."
