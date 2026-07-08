# Reservoir-regime diagnostic — design

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** one iteration — a *diagnostic* (not a fix). Measure properties of the calibrated-but-untrained
reservoir and find which one predicts learnability (for **both** V1 and V2b), and whether **topology couples
to the other knobs**. The output is a findings doc that scopes the actual brittleness fix.

## Why this

The one-at-a-time sweep showed learning collapses to chance outside a narrow config pocket, and firing-rate
calibration (all we have) doesn't prevent it — a layer can fire at 10% while the reservoir fails to
*separate* the classes. Before building a regime-targeting calibration we must know **which** regime
property to target. So: measure candidate metrics on the untrained reservoir, correlate with learnability,
and let the data pick the target. The user's hypothesis — topology dominates and shifts the other params'
optima — is tested directly with a 2D interaction grid (the OAT sweep can't see interactions).

## Metrics (on the calibrated, untrained reservoir)

The computational reservoir (`comp_layer` × `layers`) is **shared** by V1 and V2b, so each metric is one
number per config, correlated against both learners.

1. **Separation ceiling** *(primary hypothesis)* — the reservoir's intrinsic class-separability. Collect
   each trial's state (the per-neuron spike counts `trial_eligibility` returns, flattened over the
   computational layers), fit a **held-out** `NearestCentroid` on `(state → class)`, report test accuracy
   ‰. Hypothesis: this tracks learned accuracy, and dead configs sit at chance here.
2. **Effective dimensionality** — participation ratio `PR = (tr C)² / tr(C²)` of the state covariance `C`
   (trace-based; no eigensolver). Code richness: too sparse → low PR, saturated → low PR, good regime →
   high.
3. **Layer-gain profile** — mean firing fraction per computational layer (from the same states). Shows
   whether activity dies before the top or saturates.

## Experiments (reported, `#[ignore]` reproducible runs)

- **(a) Which metric predicts learnability?** Over the sweep's working + dead configs (`up_count ∈ {8,16,24}`,
  `up_radius ∈ {2,3,4}`, `layers ∈ {2,3,4}`, plus the baseline), print a table: `separation_ceiling`,
  `effective_dim`, top-layer gain, **V1 learned**, **V2b learned**. The metric that cleanly splits
  learns-vs-collapses (for both) becomes the fix's calibration target.
- **(b) Topology interaction (the hypothesis).** A 2D grid of `separation_ceiling` over `up_count ×
  adapt_bump` and `up_count × layers`. A **diagonal** good-region (not axis-aligned) confirms topology shifts
  the other optima → the fix must co-tune, not tune one knob at a time.

## Module & reuse

New `src/bench/regime.rs`:
- `separation_ceiling(cfg: &EpropConfig, trials: usize) -> u64` — build+calibrate (no training), collect
  per-class states via `trial_eligibility`, fit `NearestCentroid` on the first half of trials, test on the
  second half, return accuracy ‰.
- `effective_dim(states: &[Vec<f64>]) -> f64` — participation ratio of the state covariance.
- `layer_gain(cfg: &EpropConfig, trials: usize) -> Vec<f64>` — mean firing fraction per computational layer.

Reuses `EpropConfig`/`comp_layer` (make the eligibility/state helpers `pub(crate)` as needed),
`NearestCentroid` (`readout.rs`), and `bench::eprop::train` for the learned-accuracy column. No new linalg
(PR is trace-based). No engine change.

## Success criterion

- `separation_ceiling` **discriminates**: on the *known* working baseline it is clearly above chance, and on
  a *known* dead config (e.g. `up_count = 8`) it is at/near chance — a unit test asserts working > dead and
  working > chance.
- `effective_dim` matches hand-computed PR on synthetic matrices (rank-1 → ~1, isotropic `D`-dim → ~`D`).
- The experiments run deterministically and produce the two tables for the findings doc.

**Honesty gate:** if *no* metric cleanly predicts learnability (the split is muddy), that is the finding —
report it; it means the regime is not captured by these three and the fix needs a different handle (e.g.
a learnability proxy directly). Don't force a clean story.

## Determinism & constraints

- Std-only, no deps. Integer engine untouched; metrics are `f64` in the bench, single-threaded → deterministic.
- Pure function of `(seed, config)`. `NearestCentroid` and the trace-based PR are order-fixed.

## Testing (inline `#[cfg(test)]`)

- `separation_ceiling_discriminates_working_from_dead` — working baseline > chance+margin > dead config.
- `effective_dim_matches_known_participation_ratio` — rank-1 ≈ 1, isotropic ≈ D (synthetic states).
- `regime_metrics_are_deterministic`.
- `_regime_vs_learnability` and `_topology_interaction_grid` — `#[ignore]` reporting runs.
- Regression: whole suite stays green.

## Deliverable & next step

A findings doc (`docs/experiments_results.md` section): which metric predicts learnability (rule-independent
or not), and the topology-interaction map. That determines the **fix spec** — a regime-targeting
calibration that tunes the reservoir (thresholds, and possibly a co-tuned structural knob) to the predictive
metric so learning stops depending on landing in the pocket by hand.

## Deferred

- The fix itself (regime-targeting calibration) — its own spec, scoped by these findings.
- Perturbation/branching-ratio σ and kernel-rank/generalization-rank metrics — added only if the three here
  don't predict.
