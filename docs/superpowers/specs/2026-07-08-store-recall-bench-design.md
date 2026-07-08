# Store-recall bench — design (Spec 1 of the training test bench)

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** one iteration — a reusable bench module plus the **Tier-0 store-recall** experiment that
validates the ALIF substrate against a plain-LIF control, using an integer nearest-centroid readout.
**No** trained linear readout, **no** Memory Capacity / floats (Spec 2), **no** e-prop (Spec 3), **no**
external datasets (Tier 2).

## Why this first

We built ALIF (adaptive threshold + fixed-point memory) but never showed it *does* anything a static
threshold can't — the recurring "validate by use" gap. Store-recall is the canonical ALIF task and the
cheapest, most diagnostic check: present a cue, wait a delay `N` ≫ the leak timescale, then test whether
the cue is still decodable. Plain LIF forgets within ~`1/leak` waves; ALIF's slow adaptation should hold
it. If ALIF doesn't beat LIF here, the mechanism is broken — so this is the fastest signal, and it is the
baseline that e-prop (Spec 3) must later beat. It also builds the readout/feature/metric harness that
Specs 2–3 reuse.

The whole bench obeys the engine's **multi-wave rule**: a cue is driven over several waves and the
response is read over a multi-wave window — never a single wave.

## Success criterion

The experiment produces a **memory-horizon curve**: decode accuracy vs delay `N`, for two variants of the
*same* network — **ALIF** (`adapt_bump > 0`) and **LIF** (`adapt_bump = 0`). The automated test asserts,
deterministically:

1. **Encodable:** at small `N`, both variants decode well above chance (the cue reaches the readout at
   all).
2. **ALIF holds, LIF forgets:** at a large `N` (chosen `> 1/leak`), ALIF accuracy exceeds LIF accuracy by
   a margin, ALIF stays above chance, and LIF has collapsed toward chance (`1/K`).

Exact thresholds are config-specific and tuned once (like the calibration tests); the run is a pure
function of `(seed, config, task params)`, so the assertion is stable, not flaky.

## The mechanism being probed (why the probe matters)

After a long silent delay a calibrated reservoir may decay to near-silence, so a *silent* read would
measure nothing. The task therefore uses a **delayed-match** structure with a fixed **probe** at read
time:

- **present** the cue for `W` waves → **delay** `N` waves (silent) → **probe + read** for `R` waves,
  injecting a *fixed neutral probe* (identical for every cue) while recording spike counts.
- **ALIF:** the cue leaves a residual **adaptation footprint** (neurons that fired during presentation
  are still less excitable). That footprint *gates the probe response* — different cues suppress
  different neurons → different probe responses → decodable.
- **LIF:** no adaptation; potentials are long gone, so the probe response is cue-independent → chance.

The probe converts the slow, sub-threshold adaptation memory into a readable spike pattern; without it
the test would depend on fragile self-sustained activity.

## Architecture

New module tree (engine untouched):

```
src/
  lib.rs                 # add: pub mod bench;
  bench/
    mod.rs               # module declarations + shared bench types
    readout.rs           # spike-count feature extraction + integer NearestCentroid classifier
    store_recall.rs      # store-recall task, the memory-horizon experiment, inline tests
```

`src/bench/` is a plain public module (a research deliverable that Specs 2–3 reuse), integer-only, no
floats, no new dependencies. Experiments are inline `#[cfg(test)]` tests that assert the horizon gap.

### `bench/readout.rs`

**Feature vector.** Per-neuron **spike counts** over a read window, concatenated across the *computational*
layers (`z = 1..L`); L0 (the transducer) is excluded. Length = `(L-1) · size · size`.

**Recorder.** Runs the network over the read window with a given per-wave input and returns the feature
vector. Installs counting listeners on layers `1..L` (reusing the `on_layer` plumbing), runs the waves,
sums `fired.len()`-per-neuron into the feature vector, restores the caller's listeners. Does **not** reset
state (the trial's present/delay already set it up).

```rust
/// Run `waves` waves feeding `input(w)` each wave, returning per-neuron spike counts over layers 1..L
/// concatenated (layer 0 excluded). Saves/restores the caller's listeners; does not reset state.
pub fn record_response(net: &mut Network, waves: usize, input: impl Fn(usize) -> Vec<u32>) -> Vec<u32>;
```

**Decoder — integer NearestCentroid.** Fit = per-class mean feature vector over training samples; predict
= argmin squared-L2 distance to a centroid. All integer (`i64` accumulators; class mean via integer
division; distance in `i64`). Deterministic; no iterative training.

```rust
pub struct NearestCentroid { centroids: Vec<Vec<i64>>, k: usize }
impl NearestCentroid {
    /// Fit K class centroids from labelled feature vectors (labels in 0..k).
    pub fn fit(features: &[Vec<u32>], labels: &[usize], k: usize) -> NearestCentroid;
    /// Class with the nearest centroid (squared L2, i64).
    pub fn predict(&self, feature: &[u32]) -> usize;
}
```

### `bench/store_recall.rs`

**Cue encoding.** `K` classes (start `K = 4`). Each class has a base pattern of L0 sites, derived
deterministically from the seed (each L0 site belongs to class `c` with the base pattern selected by a
hash of `(seed, c, site)`). Distinct patterns per class.

**Within-class variability (required).** A fully deterministic task is degenerate — every trial of a class
would be identical, so nearest-centroid is trivially perfect for both variants and measures nothing. Each
trial therefore injects a **per-trial noisy realization** of its class pattern: for trial `t` of class
`c`, each wave of the presentation injects the base sites with per-site dropout plus a few random extra
sites, governed by a hash of `(seed, c, t, wave, site)`. This gives distinct-but-related patterns within a
class, so the decoder must extract the persistent cue signal from noisy responses — which is exactly where
ALIF's longer-lived memory should win at large `N`.

**Probe.** A single fixed pattern of L0 sites (derived from `hash(seed, "probe")`), identical for every
cue and trial, injected each wave of the read window.

**Trial.** `reset_state` → present (`W` waves of noisy cue injection) → delay (`N` waves, no injection) →
`record_response` over `R` waves of probe injection → feature vector + true label `c`.

**Experiment (`memory_horizon`).**
1. Build the net from a bench config; **calibrate once** (random input) so baselines sit at the target
   rate. `reset_state` between trials preserves the calibrated baselines.
2. For each variant (ALIF `adapt_bump > 0`, LIF `adapt_bump = 0` — calibrated **separately** to the *same*
   target rate, so they share an activity level and differ only in adaptation memory):
   - For each `N` in the sweep: run `T` trials (balanced across the `K` classes, deterministic per-trial
     seeds), split deterministically into train/test, `fit` centroids on train, measure accuracy on test.
3. Return both accuracy-vs-`N` curves; a helper formats them for human inspection (the memory-horizon
   plot in text).

```rust
pub struct BenchConfig { /* engine Config + K, W, R, T, n_sweep, target rate, calibration params */ }
pub struct HorizonCurve { pub delays: Vec<usize>, pub accuracy_permille: Vec<u64> }
/// Run the store-recall sweep for one variant; `adapt_bump_override` selects ALIF vs LIF (Some(0) = LIF).
pub fn memory_horizon(cfg: &BenchConfig, adapt_bump_override: i16) -> HorizonCurve;
```

*(Accuracy is a ratio of integer counts; represented as permille `u64` to stay float-free, e.g. 950 = 95%.
No `f64` anywhere in Spec 1.)*

## Fair comparison

LIF and ALIF are the **same engine config** with only `adapt_bump` differing, each **calibrated
separately** to the same target firing rate before the task. This isolates the one variable under test —
adaptation memory — from any difference in baseline excitability.

## Timing / parameter guidance (starting points, all tunable)

- Layers/size: reuse a small recurrent config with downward `level: -1` coupling (like the calibrated
  `test_config`) — size 16, ~4–6 layers.
- `W` (present) `≥` propagation depth (`~L` waves) so the cue reaches the top layer.
- `adapt_decay` chosen so ALIF's memory τ ≈ `2^adapt_decay` waves clearly exceeds the leak horizon
  (~`1/leak`, roughly 15–20 waves for leak `(3,5)`): e.g. `adapt_decay = 6` (τ ≈ 64).
- `N` sweep spanning both sides of the leak horizon, e.g. `{0, 4, 8, 16, 32, 64}`.
- `R` (probe/read) ~ `W`; `T` ~ 40 trials/class; `K = 4` (chance = 25%).

The assertion targets a large-`N` point (e.g. `N = 32`) where ALIF should still decode and LIF should be
near chance. These constants are tuned once against the chosen config, exactly as the calibration tests
were.

## Testing (inline `#[cfg(test)]`)

- `nearest_centroid_separates_clusters` — unit test of the decoder on hand-made separable/overlapping
  feature clusters (fit/predict correctness), independent of the engine.
- `record_response_counts_spikes` — inject a known L0 pattern, confirm the feature vector reflects spikes
  on the expected layers and is zero on a silent run.
- `cue_encoding_is_deterministic_and_distinct` — class patterns are reproducible across runs and differ
  between classes.
- `store_recall_alif_beats_lif_at_long_delay` — the headline test: run `memory_horizon` for both variants;
  assert (1) both decode well at small `N`, (2) at the large-`N` point ALIF accuracy > LIF accuracy by a
  margin and ALIF > chance while LIF ≈ chance.
- `memory_horizon_is_deterministic` — two runs of the same config yield identical curves.

## Determinism & constraints

- Std-only, no new dependencies, **integer-only in Spec 1** (nearest-centroid; accuracy as permille).
- Everything is a pure function of `(seed, config, task params)`; single-threaded.
- The engine (`src/wave_net/`) is untouched — the bench only *uses* the public API (`wave`, `reset_state`,
  `calibrate`, `on_layer`, per-neuron accessors). If a small public accessor is missing for feature
  extraction, add it minimally to the engine rather than reaching into internals.

## Scope guard (YAGNI) — explicitly out of this spec

- Trained linear readout / ridge / Memory Capacity and any `f64` — Spec 2.
- e-prop / any internal per-neuron training — Spec 3.
- External datasets (SHD/SSC/psMNIST) and preprocessing — Tier 2, later specs.
- Sweeping `K`, perceptron/LMS decoders, multiple task families — not needed to answer "does ALIF beat
  LIF on store-recall."
