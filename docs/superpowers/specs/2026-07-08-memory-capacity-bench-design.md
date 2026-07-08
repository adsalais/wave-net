# Memory Capacity bench — design (Spec 2 of the training test bench)

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** one iteration — Memory Capacity (MC), the quantitative ALIF-vs-LIF memory metric, plus the
reusable `f64` ridge readout + `linalg` it requires. Measured in **four** runs: {recurrent,
feed-forward} × {ALIF, LIF}. **No** temporal-XOR / adding / copy (next spec), **no** e-prop (Spec 3),
**no** external datasets.

## Why this

Store-recall (Spec 1) gave a binary "ALIF holds a cue LIF forgets." MC gives the *continuous* picture:
a memory curve `r²_k` for each lag `k`, whose sum `MC = Σ_k r²_k` is the single most informative
ALIF-vs-LIF number — it shows exactly how far back the reservoir remembers and how much ALIF extends the
tail. It also introduces the `f64` **ridge/least-squares readout**, the reusable regression foundation the
remaining Tier-1 tasks (XOR/adding/copy) will consume.

Per the hybrid float ruling: the engine stays integer; the bench's readout/metrics use `f64` (still
std-only, single-threaded, fixed reduction order → deterministic).

## Success criterion

For each of the four runs, produce the memory curve `r²_k` (`k = 1..K`) and `MC = Σ_k r²_k`. The
automated test asserts, deterministically:

1. **Feed-forward — the de-risked headline:** `MC_ALIF(ff) > MC_LIF(ff)` by a margin, and ALIF's `r²_k`
   stays above LIF's at large `k` (LIF's tail collapses to ~0 past the feed-forward pipeline depth, since
   with the floored leak nothing but adaptation persists).
2. **Recurrent — the canonical number:** both variants have substantial `MC > 0` (the reservoir has
   memory), and `MC_ALIF(rec) ≥ MC_LIF(rec)` (adaptation does not *reduce* memory).
3. `memory_capacity_is_deterministic` — two runs give identical curves.

Constants are tuned once against the chosen config (like the calibration/store-recall constants); the run
is a pure function of `(seed, config, params)`.

**Honesty gate:** if recurrent MC shows `ALIF ≈ LIF`, that is a real finding (recurrence supplies the
memory; adaptation adds little on top) — report it, do not force a gap. Feed-forward is where ALIF must
clearly win; if it does not even there, that is a substrate finding to surface (not a test to fudge).

## The task — binned streaming MC

- **Stream:** a binary i.i.d. bit stream `u(t) ∈ {0,1}`, deterministic from seed (`u(t)` = one hashed
  bit per timestep).
- **Drive:** timestep `t` is a bin of `B` waves. If `u(t) = 1`, inject a **fixed** L0 pattern (hash-derived,
  same every "1") for all `B` waves of the bin; if `u(t) = 0`, inject nothing. The stream runs
  **continuously with no reset between bins**, so the reservoir integrates it; a warmup prefix of bins is
  discarded before collection.
- **State `x(t)`:** per-neuron spike counts over the bin, via the existing `readout::record_response`
  (drives the bin input, returns counts, does not reset). Features = computational layers `1..L`
  concatenated, dim `d = (L-1)·size·size`, plus a bias column of 1s for the readout intercept.

## Readout — `f64` ridge (the reusable deliverable)

Fit a linear readout per lag `k` to reconstruct `u(t−k)` from `x(t)`:

- Design matrix `X` (`n × (d+1)`, bias column), targets `y_k(t) = u(t−k)`.
- Ridge normal equations `(XᵀX + λI) w_k = Xᵀ y_k`. `A = XᵀX + λI` (`(d+1)×(d+1)`, symmetric, PD for
  `λ > 0`) is the **same for every lag**, so factor it **once** and back-substitute per lag.
- Solver: **Gaussian elimination with partial pivoting** (LU factor once, solve for each `Xᵀy_k`).

`RidgeReadout` lives in `readout.rs` beside `NearestCentroid`; `linalg.rs` holds the low-level `f64` ops.

```rust
// bench/linalg.rs
/// LU factorization (partial pivoting) of a square matrix; reusable across right-hand sides.
pub struct Lu { /* factors + pivots */ }
impl Lu {
    pub fn factor(a: Vec<Vec<f64>>) -> Lu;          // panics if singular
    pub fn solve(&self, b: &[f64]) -> Vec<f64>;      // solve A x = b
}
/// `Xᵀ X` (square, dim = cols of x) and `Xᵀ y`.
pub fn xt_x(x: &[Vec<f64>]) -> Vec<Vec<f64>>;
pub fn xt_y(x: &[Vec<f64>], y: &[f64]) -> Vec<f64>;

// bench/readout.rs
/// Ridge-regression linear readout: one weight vector per target column, sharing one LU factor.
pub struct RidgeReadout { lu: Lu }
impl RidgeReadout {
    /// Factor (XᵀX + λI) from the training design matrix (bias column already included).
    pub fn fit(x_train: &[Vec<f64>], lambda: f64) -> RidgeReadout;
    /// Weights reconstructing one target column from x.
    pub fn weights(&self, x_train: &[Vec<f64>], y_train: &[f64]) -> Vec<f64>;
    /// Prediction X·w.
    pub fn predict(x: &[Vec<f64>], w: &[f64]) -> Vec<f64>;
}
```

## Metric

`r²_k` = squared Pearson correlation between the ridge prediction and the target `u(t−k)` on a held-out
**test** split (train ridge on the first part of the collected stream, evaluate on the rest), clamped to
`[0, 1]`. `MC = Σ_{k=1}^{K} r²_k`. All `f64`.

```rust
// bench/memory_capacity.rs
pub struct McCurve { pub r2: Vec<f64>, pub total: f64 } // r2[k-1] for k=1..K; total = sum
pub fn memory_capacity(cfg: &McConfig, adapt_bump: i16, recurrent: bool) -> McCurve;
```

## Configs

Same engine `Config` with only `adapt_bump` (ALIF > 0 vs LIF 0) and topology (recurrent vs feed-forward)
differing; each variant **calibrated separately** to the same target rate.

- **Recurrent:** the dense drive the floored leak now requires — `level+1` (radius 3, count 16),
  `level 0` (count 3), `level −1` (count 3), matching the re-tuned calibration fixture.
- **Feed-forward:** `level+1` (radius 3, count 16) only — no `level 0/−1`, so nothing but adaptation
  survives a lag.

## Testing (inline `#[cfg(test)]`)

- `lu_solves_known_system` — `Lu::factor`/`solve` on a small system with a known solution (independent of
  the engine).
- `ridge_recovers_planted_linear_map` — generate `X`, `y = X·w_true + tiny`, assert `RidgeReadout` weights
  ≈ `w_true` and prediction `r²` ≈ 1.
- `bit_stream_is_deterministic_and_balanced` — `u(t)` reproducible across runs and ~50% ones.
- `memory_capacity_feedforward_alif_beats_lif` — the headline: `MC_ALIF(ff) > MC_LIF(ff)` by a margin and
  ALIF's tail `r²_k` exceeds LIF's at a large `k`.
- `memory_capacity_recurrent_has_memory` — both variants `MC > 0` substantially; `MC_ALIF(rec) ≥
  MC_LIF(rec) − ε`.
- `memory_capacity_is_deterministic` — identical curves across two runs.

## Determinism & constraints

- Std-only, no new dependencies. Engine (`src/wave_net/`) untouched — bench uses the public API only.
- `f64` allowed in the bench (readout/metric/linalg); single-threaded, fixed reduction order → deterministic.
- Everything a pure function of `(seed, config, params)`.

## Parameters (starting points, tunable)

`size 8`, `4 layers`, `B = 3`, warmup `~150` bins, collected `T ≈ 1500` bins (train/test split ~70/30),
`K ≈ 30` lags, `λ = 1.0`. Feature dim `d ≈ 3·64 = 192 < n_train`.

## Scope guard (YAGNI) — explicitly out

- temporal-XOR / adding / copy — next spec (they reuse `RidgeReadout`).
- e-prop / internal training — Spec 3.
- external datasets (SHD/SSC/psMNIST) — Tier 2.
- Graded/rate input encoding, sweeping `K`/`B` as the deliverable — not needed to answer "how far back,
  and does ALIF extend it."
