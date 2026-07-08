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
number per config, correlated against both learners. Grouped by axis; all built on a shared `collect_states`
primitive (per-trial flattened computational-layer spike counts + class label).

**Separation (task-aligned):**
1. **Separation ceiling** *(primary hypothesis)* — held-out `NearestCentroid` on `(state → class)`, test
   accuracy ‰. Fit on the first half of trials, test on the second. Dead configs should sit at chance.
2. **Fisher discriminant ratio** — continuous `S_B / S_W` = between-class scatter over within-class scatter
   (trace form). Non-saturating companion to #1 (accuracy caps at 100%; Fisher keeps climbing → better
   correlation dynamic range). Nearly free — reuses the collected states.
3. **Forgetting curve** — the separation ceiling as `delay ∈ {0,2,4,8,16}`. Does class info *survive* the
   hold? Explains the depth/adaptation brittleness of a held-category task.

**Dimensionality / rank:**
4. **Effective dimensionality** — participation ratio `PR = (tr C)² / tr(C²)` of the state covariance
   (trace-based; no eigensolver). Code richness: sparse → low, saturated → low, good regime → high.
5. **Kernel − generalization rank** *(Legenstein–Maass)* — PR of states across **distinct** inputs (signal
   capacity) minus PR across **noisy copies of one** input (noise sensitivity). Captures the
   separation/generalization tradeoff that *is* the density sweet spot (sparse loses kernel rank, dense
   gains noise rank). PR is the soft/effective-rank proxy for the classic rank (no SVD) — documented.

**Dynamical regime:**
6. **Perturbation spread (σ / edge-of-chaos)** — flip one L0 injection site; per computational layer, the
   Hamming divergence (neurons whose spike count differs) between base and perturbed states; `σ` = the
   geometric growth of divergence layer-to-layer. `<1` ordered, `>1` chaotic, `≈1` critical. A distinct
   axis from separation — tests whether the working pocket is literally criticality.
7. **Layer-gain profile** — mean firing fraction per computational layer. Activity dies before the top, or
   saturates.

**Degeneracy (why a regime is bad):**
8. **Dead / saturated fraction + synchrony** — fraction of neurons that never fire or fire every wave, and
   mean pairwise spike-count correlation over a neuron-pair sample (lockstep → low-rank redundant code).

## Experiments (reported, `#[ignore]` reproducible runs)

- **(a) Which metric predicts learnability?** Over the sweep's working + dead configs (`up_count ∈ {8,16,24}`,
  `up_radius ∈ {2,3,4}`, `layers ∈ {2,3,4}`, plus the baseline), print a table: `separation_ceiling`,
  `effective_dim`, top-layer gain, **V1 learned**, **V2b learned**. The metric that cleanly splits
  learns-vs-collapses (for both) becomes the fix's calibration target.
- **(b) Topology interaction (the hypothesis).** A 2D grid of `separation_ceiling` over `up_count ×
  adapt_bump` and `up_count × layers`. A **diagonal** good-region (not axis-aligned) confirms topology shifts
  the other optima → the fix must co-tune, not tune one knob at a time.

## Module & reuse

New `src/bench/regime.rs`, built on one primitive:
- `collect_states(cfg, trials) -> (Vec<Vec<u32>>, Vec<usize>)` — build+calibrate (no training), run `trials`
  trials via `trial_eligibility`, return per-trial flattened computational-layer spike counts + class labels.

Then (each consumes states unless noted):
- `separation_ceiling(cfg, trials) -> u64` — held-out `NearestCentroid`, accuracy ‰.
- `fisher_ratio(states, labels, k) -> f64` — `S_B / S_W`.
- `effective_dim(states) -> f64` — participation ratio.
- `kernel_minus_gen_rank(cfg) -> f64` — PR over distinct-input states minus PR over noisy-same-input states
  (its own two collection passes).
- `perturbation_spread(cfg) -> f64` — per-layer Hamming divergence growth from a one-site input flip (a
  paired base/perturbed collection).
- `layer_gain(cfg, trials) -> Vec<f64>` — mean firing fraction per computational layer.
- `degeneracy(states) -> (f64, f64, f64)` — dead fraction, saturated fraction, sampled pairwise synchrony.

Reuses `EpropConfig`/`comp_layer`, `NearestCentroid` (`readout.rs`), `trial_eligibility` +
`cue_realization`/`probe_pattern` (make `pub(crate)` as needed), and `bench::eprop::train` for the
learned-accuracy columns. No new linalg (PR/Fisher are trace-based). **No engine change.**

## Success criterion

- `separation_ceiling` and `fisher_ratio` **discriminate**: on the *known* working baseline both are clearly
  above the *known* dead config (e.g. `up_count = 8`), and the ceiling is above chance — unit tests assert
  working > dead.
- `effective_dim` matches hand-computed PR on synthetic matrices (rank-1 → ~1, isotropic `D`-dim → ~`D`).
- `perturbation_spread` behaves at the extremes (a starved/subcritical config → `σ < 1`; a saturated config
  → larger spread) — a monotonicity unit test, not an absolute value.
- The experiments run deterministically and produce the metric-vs-learnability table + the topology grid.

**Honesty gate:** with eight metrics, expect several to *not* predict learnability — report which do and
which don't (that itself is informative: e.g. "criticality doesn't matter here but separation does"). If
*none* cleanly splits learns-vs-collapses, report that too — the regime needs a different handle. Don't
cherry-pick or force a clean story.

## Determinism & constraints

- Std-only, no deps. Integer engine untouched; metrics are `f64` in the bench, single-threaded → deterministic.
- Pure function of `(seed, config)`. `NearestCentroid` and the trace-based PR are order-fixed.

## Testing (inline `#[cfg(test)]`)

- `separation_ceiling_discriminates_working_from_dead` — working baseline > chance+margin > dead config.
- `fisher_ratio_discriminates_working_from_dead` — working > dead.
- `effective_dim_matches_known_participation_ratio` — rank-1 ≈ 1, isotropic ≈ D (synthetic states).
- `perturbation_spread_orders_regimes` — subcritical (starved) config `σ` < a denser config's.
- `degeneracy_flags_dead_and_saturated` — synthetic states with known dead/saturated neurons.
- `regime_metrics_are_deterministic`.
- `_regime_vs_learnability` and `_topology_interaction_grid` — `#[ignore]` reporting runs (all eight metrics
  + V1/V2b learned accuracy).
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
