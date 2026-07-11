# e-prop as an official `wave_net` init + training method — design

**Date:** 2026-07-11
**Status:** approved design
**Scope:** promote the criticality-init (`critical_init`, rate-free σ≈1) and the core e-prop machinery
from `bench/` into the `wave_net/` engine as **official** initialization + training, downgrade the old
firing-rate calibration to a `bench/` tool, and make `critical_init` the recommended default init for
feed-forward configs. **Feed-forward only** — recurrent (side-car loop-gain) criticality is deferred.

## Motivation

`docs/experiments_results.md` (Criticality-init section) established that the rate-free σ-eprop init
(`sigma_eprop_init`) is the robust replacement for the brittle firing-rate `calibrate`: it revives
sub-critical stacks calibration can't, targets the real order parameter σ (not the rate proxy), and
produces a computationally better substrate (multi-seed × two tasks). It's currently bench-side
experimental code. This task makes it (and the e-prop rule it's built on) a first-class engine feature.

This **inverts** the current documented architecture (engine = mechanism/hooks; `bench` = the learning
rules). That inversion is intentional and is reflected in the AGENTS.md rewrite below.

## Non-goals

- **Recurrent criticality.** The side-car loop gain is a separate quantity (`sigma_probe(Some(z))`), the
  greedy-FF structure doesn't map to the cyclic L2↔L3 topology, and `critical_init` stays FF-only.
  Recurrent configs keep using the (downgraded) calibration until a later recurrent-σ extension.
- **A general training driver.** The engine driver covers the **feed-forward** e-prop loop only; the
  side-car and multi-cue sequence trainers stay in `bench/` and call the engine's `eprop_update`
  primitive in their own loops.
- Changing `wave_state_machine` (frozen reference) — untouched.

## Module moves

**Into `wave_net/`:**
- `wave_net/eprop.rs` (new) — the generic e-prop **update primitive**, the **FF training driver**, and
  the windowed-eligibility helper.
- `wave_net/critical_init.rs` (new) — the σ diagnostic (`forward_avalanche`) + `critical_init`.
- `target_of` moves `bench/rsnn.rs` → `wave_net/synapse.rs` (pure procedural addressing; `rsnn.rs`
  re-imports it as `wave_net::wave_net::synapse::target_of`).

**Out of `wave_net/`:**
- `wave_net/calibrate.rs` → `bench/calibrate.rs` (downgraded): `Network::calibrate` becomes a free
  `fn calibrate(net: &mut Network, params: &CalibrateParams, input: &impl Fn(usize) -> Vec<u32>)` using
  the crate-visible `measure_layer_rates` / `with_layer_mut`; `CalibrateParams` moves with it.
  `random_l0_input` **stays public in the engine** — it moves into `wave_net/critical_init.rs` (used by
  `critical_init` / `forward_avalanche`) and the downgraded bench `calibrate` re-imports it from there.
  `Layer::calibrate_step` stays a `Layer` method (used by bench calibrate).

**Stays in `bench/`:** tasks, DFA / rate / σ learning-signal *computation*, side-car & sequence training
loops (now calling `eprop_update`), the criticality *experiments* (ignored tests, calling the engine
API), readouts (`readout.rs`, `linalg.rs`).

## Engine API

```rust
// Generic per-layer e-prop weight update from stored eligibility. `entry_idx` selects the topology
// entry (level/radius/count) whose weights are updated — so FF (the up entry) and, later, side-car
// edges both reuse it. `use_psi` gates the ψ factor (task e-prop = true; σ-init scaling = false).
pub fn Network::eprop_update(&mut self, source_z: usize, entry_idx: usize, signal: &[f32], lr: f32, use_psi: bool);

// σ diagnostic: per-hop forward damage-spreading footprint (footprint[z+1]/footprint[z] = σ_hop).
pub fn forward_avalanche(net: &mut Network, drive_seed: u64, frac_q16: u32, warmup: usize, n_perturb: usize, burst: usize) -> Vec<f64>;

// The rate-free σ≈1 init (greedy bottom-up; built on the two above). The recommended default init.
pub fn Network::critical_init(&mut self, drive_seed: u64, frac_q16: u32, params: &CriticalInitParams);

// Feed-forward e-prop training driver. `drive(trial, wave) -> L0 sites`; `signal(&net, trial) ->
// per-computational-layer learning signal`. The driver runs trials, accumulates eligibility, and
// applies `eprop_update` to each trained layer; returns per-trial metrics.
pub fn Network::train_ff(&mut self, trials: usize, present: usize,
    drive: impl Fn(usize, usize) -> Vec<u32>,
    signal: impl Fn(&Network, usize) -> Vec<Vec<f32>>) -> Vec<TrialMetric>;
```

`CriticalInitParams` and the `SigmaEprop`-style knobs consolidate into the engine-side
`CriticalInitParams`. The bench-side rate-init variants (`rate_reg_init`, the gain-scaling controller)
are **not** promoted — they were stepping stones; only the σ-eprop init (the keeper) becomes official.
Their experiment code may remain in `bench/` (ignored tests) for the record or be dropped — see the
plan.

## Default / fallback story

- "Default" = the **recommended** init the FF pipeline calls: `net.critical_init(...)` replaces
  `net.calibrate(...)` in the FF `train_*` paths. The engine does **not** auto-init in `new()`
  (the init needs a drive and is iterative).
- **Recurrent / side-car configs** call the `bench::calibrate` free function as the fallback, until the
  recurrent-σ extension. AGENTS.md documents: `critical_init` = default for FF; `calibrate` = bench
  fallback for recurrent.

## Validation gate (hard prerequisite before the FF default flips)

An `#[ignore]`d experiment that trains a **feed-forward task end-to-end** (e-prop + trained readout)
with `critical_init` vs the downgraded `calibrate`, comparing **held-out trained accuracy** — because
σ-init is so far validated as a *substrate*, not inside the actual train loop. It must stress the regime
where the init matters (a shallow/1-layer net is trivially solved by calibration):

- **Depth ≥ 4 computational layers** (use the standard 5-layer FF stack).
- **Width 32** (`size 32` → 1024 neurons/layer).
- **Several density settings** — a small sweep over `up_count` and/or `up_radius` (e.g. `up_count ∈
  {16, 32}`, `up_radius ∈ {3}` at minimum; more if cheap), since density sets σ and calibration's
  brittleness is density/depth-dependent.
- **Multi-seed** (≥3) so the comparison isn't a single-seed artifact.
- Task: a FF-learnable temporal task from the existing suite (temporal-XOR or parity) run through the
  new `train_ff` driver.

**Gate:** `critical_init ≥ calibration` on trained accuracy across the settings, and it should **win
where calibration is brittle** (deep + low `up_count`, where calibration lets the cue die with depth).
If `critical_init` fails to beat calibration in that regime, the default does **not** flip and the
finding is recorded instead.

## Constraints

- **Standard library only** in `src/` (both engine and bench already are; the σ diagnostic uses
  `std::collections::HashSet` + `std::sync::{Arc, Mutex}`, all already used in the engine). No `unsafe`.
- **Determinism** — pure function of `(seed, config, input)`; the init and driver are deterministic.
- **Warning-free build; all existing tests stay green** — the moves are mechanical (import-path
  updates); the ~10 `net.calibrate(...)` call sites in `rsnn.rs` become `calibrate(net, ...)`.
- **AGENTS.md rewrite** — the "three modules" and architecture-map sections updated so `wave_net` owns
  e-prop (update primitive + FF driver) + `critical_init` (default) + the σ diagnostic, and `bench`
  owns tasks/DFA/side-car/sequence + calibration (fallback).

## File / structure summary

```
wave_net/
  synapse.rs        # + target_of (moved from rsnn)
  eprop.rs          # NEW: eprop_update, train_ff, windowed_eligibility
  critical_init.rs  # NEW: forward_avalanche (σ diagnostic), critical_init (default init), CriticalInitParams,
                    #   + random_l0_input (drive helper, moved here from the removed calibrate.rs)
  calibrate.rs      # REMOVED
  network.rs        # + eprop_update / critical_init / train_ff methods (thin wrappers or here)
bench/
  calibrate.rs      # NEW: calibrate() free fn + CalibrateParams (downgraded from the engine)
  rsnn.rs           # calls engine eprop_update / critical_init; side-car & sequence loops stay
  critical_init.rs  # experiments only (ignored tests) calling the engine API
```
