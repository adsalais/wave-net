# Firing-rate calibration (wave_net) — design

**Date:** 2026-07-07
**Status:** approved (brainstorming complete)
**Scope:** Make the silent-by-default `wave_net` non-inert by lowering per-layer thresholds until
each layer fires near a target rate on a driven input. Single-threaded, deterministic. **Not** the
criticality/homeostasis program (`2026-07-06-criticality-homeostasis-design.md`) — no σ targeting,
no per-neuron field, no memory-capacity evaluation. Builds on the saturation-free engine.

## Problem shape

The engine starts silent (thresholds near `i16::MAX`); only injected L0 neurons fire. Calibration
must **lower** per-layer thresholds until layers 1..L−1 fire at a target rate — the inverse of the
legacy calibrator (which *raised* thresholds to cool a hot net), same iterate-to-target shape.

Two engine facts drive the design:
- **Deferred one-hop propagation** — signal climbs one layer per wave, so a measurement must
  discard a warmup while the front climbs, and a layer's realistic input only exists once the layer
  below fires. → tune **bottom-up**.
- **Recurrence exists** — the demo topology has `level 0` (lateral) and `level −1` (downward)
  synapses, so a freshly-tuned layer `z` feeds `z−1` and nudges its rate after it was frozen. →
  add **global-refine** passes after bottom-up (the "hybrid").

## Ownership: calibration lives at the layer level

Each `Layer` already carries everything that describes it — `topology`, `leak`, `cooldown_base`,
`inhibitor_ratio`, and its tuned `threshold` vector — so a `Layer` is a **self-contained,
persistable unit** and a `Network` is just `seed + size + layers`. Therefore:

- **The `Layer` owns tuning its own thresholds** (the calibratable state is layer-local).
- **The `Network` only orchestrates** what genuinely needs the whole stack — running waves and
  measuring per-layer firing rates — then hands each layer its measured rate and lets the layer tune
  itself.

**Persistence is structured-for now, built later.** Because each `Layer` owns its full descriptive
state and exposes threshold get/set, saving a calibrated net later is "snapshot each layer's params
+ thresholds" — a clean separate task. No serialization code (and no `serde`; std-only) in this
spec.

## `Layer` — new methods (`src/wave_net/neurons.rs`)

```rust
/// Subtract `delta` from every threshold (delta>0 lowers), clamped to [1, i16::MAX].
/// Uniform shift, so per-neuron jitter is preserved.
pub fn shift_threshold(&mut self, delta: i32);

/// One measure-informed tuning step toward `target` (both as fractions in 0..1). Returns whether
/// it adjusted. The layer owns the step policy: geometric `max_threshold >> step_shift`, lower when
/// too cold, raise when too hot, no-op inside the tolerance band.
pub fn calibrate_step(&mut self, rate: f64, target: f64, tol: f64, step_shift: u32) -> bool;

pub fn thresholds(&self) -> &[i16];          // introspection / persistence
pub fn set_thresholds(&mut self, t: Vec<i16>); // restore (persistence / tests)
// max_threshold(&self) -> i16 already exists.
```

## `Network` — new methods (`src/wave_net/network.rs`)

```rust
/// Locked mutable access to one layer (crate-internal; how calibration reaches Layer methods).
pub(crate) fn with_layer_mut<R>(&self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R;

/// Reset, run `warmup` waves (discarded), then `waves` waves counting spikes per layer; return
/// per-layer firing rate = spikes / (layer_size * waves). Single-threaded, deterministic.
/// SAVES the caller's registered listeners, installs temporary counting listeners, and RESTORES
/// the saved listeners before returning — calibration never clobbers user listeners.
pub(crate) fn measure_layer_rates(&mut self, warmup: usize, waves: usize,
                                  input: &impl Fn(usize) -> Vec<u32>) -> Vec<f64>;

/// A copy of a layer's per-neuron thresholds (public introspection / determinism tests).
pub fn layer_thresholds(&self, z: usize) -> Vec<i16>;
```

Listener save/restore mechanism (boxed `Fn` closures aren't `Clone`, so we **move**, not copy):
`std::mem::replace(&mut self.listeners, fresh_none_slots)` yields the user's listeners; install
counters into the fresh slots; run; then `self.listeners = saved` drops the counters and restores
the user's. Because it saves/restores every call, `measure_layer_rates` is self-contained and safe
to call repeatedly.

## Public entry (`src/wave_net/calibrate.rs`)

```rust
pub struct CalibrateParams {
    pub target_permille: u64, // desired per-layer firing rate, e.g. 100 = 10%
    pub tol_permille: u64,    // stop a layer when |rate - target| <= tol
    pub warmup: usize,        // waves discarded per measurement (signal climb + settle)
    pub waves: usize,         // waves counted per measurement
    pub max_steps: usize,     // max adjust steps per layer in the bottom-up phase
    pub refine_passes: usize, // global all-layers passes after bottom-up
    pub step_shift: u32,      // geometric step = max_threshold >> step_shift
}
// Default: target 100, tol 20, warmup 32, waves 128, max_steps 48, refine_passes 4, step_shift 2
// (max_steps gives descent headroom from i16::MAX to the low thresholds sparse ±1 layers need;
//  the geometric step auto-shrinks near target, so 48 does not hurt precision. All are tunable.)

impl Network {
    /// Lower per-layer thresholds (layers 1..L; L0 is the input surface, left as-is) so each fires
    /// near target on `input`. Mutates in place; preserves the caller's listeners.
    pub fn calibrate(&mut self, params: &CalibrateParams, input: &impl Fn(usize) -> Vec<u32>);
}

/// A deterministic per-wave input: injects each L0 local with probability `fraction_q16 / 2^16`.
pub fn random_l0_input(seed: u64, size: u32, fraction_q16: u32) -> impl Fn(usize) -> Vec<u32>;
```

`calibrate` is a second `impl Network` block living in `calibrate.rs`; it uses the `pub(crate)`
`measure_layer_rates` + `with_layer_mut` and delegates every adjustment to `Layer::calibrate_step`.

## Algorithm (hybrid: bottom-up then global refine)

```
target = target_permille/1000; tol = tol_permille/1000
// Phase 1 — bottom-up: fix each layer before moving up (its feeder is now firing)
for z in 1..L:
    for _ in 0..max_steps:
        rates = self.measure_layer_rates(warmup, waves, input)   // re-measure after each adjust
        adjusted = self.with_layer_mut(z, |l| l.calibrate_step(rates[z], target, tol, step_shift))
        if !adjusted: break
// Phase 2 — global refine: absorb the downward (level 0/-1) coupling
for _ in 0..refine_passes:
    rates = self.measure_layer_rates(warmup, waves, input)       // one snapshot per pass
    moved = false
    for z in 1..L:
        moved |= self.with_layer_mut(z, |l| l.calibrate_step(rates[z], target, tol, step_shift))
    if !moved: break
```

`Layer::calibrate_step`'s geometric `step = max_threshold >> 2` self-damps: from `i16::MAX` it
descends ~×0.75/step and shrinks as the threshold shrinks, converging into the tolerance band from
any starting scale. L0 is never tuned — its rate is just the input injection fraction.

## Determinism

`reset_state` before each measurement + a deterministic seed-based `input` closure ⇒ the resulting
thresholds are a pure function of `(net's seed/config, params, input)`. `measure_layer_rates` is
single-threaded. No `Date`/`rand`/hash-map iteration.

## Known limitation

Pure bottom-up ignores the downward feedback; the `refine_passes` phase corrects most of it but is
capped, so with very strong recurrence a residual per-layer rate error can remain. Acceptable for
v1. No σ / temporal structure is targeted — only firing rate.

## Testing (inline `#[cfg(test)]`, test-first)

Test config: `size = 8`, `L = 4`, forward+lateral topology (`level +1` r2 c6, `level 0` r1 c2),
`inhibitor_ratio = 0` (all excitatory, predictable), small `threshold_jitter`.

1. **`measure_layer_rates` sanity** — with `random_l0_input(seed, size, frac)`,
   `rates[0] ≈ frac/65536` (L0 fires exactly when injected; its high threshold blocks recurrent
   firing). Loose band.
2. **calibration warms a silent net** — before: top-layer rate ≈ 0; after `calibrate`: top-layer
   rate in a band around target (e.g. `(target/2, target*2)`, `> 0`), and `layer_thresholds(top)`'s
   max `< i16::MAX` (it dropped).
3. **thresholds lowered per layer** — every calibrated layer `1..L` has a max threshold `< i16::MAX`.
4. **determinism** — two fresh nets, identical config/params/input → identical `layer_thresholds(z)`
   for all `z`.
5. **listeners preserved** — register a layer‑0 listener, run `calibrate`, then one wave: the
   listener still fires (save/restore worked).
6. **`Layer::shift_threshold` clamp** (unit test in `neurons.rs`) — subtracting past 1 floors at 1;
   adding past `i16::MAX` caps; a uniform shift preserves the jitter spread.

`cargo build` warning-free; no `unsafe`; standard library only.

## Files touched

- **`src/wave_net/neurons.rs`** — `Layer::shift_threshold`, `calibrate_step`, `thresholds`,
  `set_thresholds` (+ a `shift_threshold` unit test). Layer stays the self-contained persistable unit.
- **`src/wave_net/network.rs`** — `with_layer_mut`, `measure_layer_rates` (with listener
  save/restore), `layer_thresholds`.
- **`src/wave_net/calibrate.rs`** — replace stub: `CalibrateParams`, `random_l0_input`, and
  `impl Network { pub fn calibrate }`, plus integration tests.
- **`src/wave_net/synapse.rs`** — add `pub const P_INPUT: u64 = 5;`.
- `mod.rs` unchanged.

## Non-goals

No σ / criticality targeting, no per-neuron homeostatic field, no memory-capacity / XOR evaluation,
no threading, and **no serialization code yet** — persistence is structured-for (each `Layer` is a
self-contained unit with threshold get/set) and built as a separate task when needed.
