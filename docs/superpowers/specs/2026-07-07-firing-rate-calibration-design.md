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

Thresholds are already "calibration-owned mutable state" (engine design), so calibration **mutates
the live `Network` in place** rather than rebuilding from a config. Saturation no longer exists, so
it plays no part.

## Public API — `src/wave_net/calibrate.rs`

```rust
pub struct CalibrateParams {
    pub target_permille: u64, // desired per-layer firing rate, e.g. 100 = 10%
    pub tol_permille: u64,    // stop a layer when |rate - target| <= tol
    pub warmup: usize,        // waves discarded per measurement (signal climb + settle)
    pub waves: usize,         // waves counted per measurement
    pub max_steps: usize,     // max adjust steps per layer in the bottom-up phase
    pub refine_passes: usize, // global all-layers passes after bottom-up
    pub step_shift: u32,      // geometric step = max_layer_threshold >> step_shift
}
// Default: target 100, tol 20, warmup 32, waves 128, max_steps 24, refine_passes 4, step_shift 2

/// Lower per-layer thresholds (layers 1..L; L0 is the input surface, left as-is) so each fires
/// near `target` on `input`. Mutates `net` in place. Uses (and clears) the layer listeners.
pub fn calibrate(net: &mut Network, params: &CalibrateParams, input: &impl Fn(usize) -> Vec<u32>);

/// A deterministic per-wave input: injects each L0 local with probability `fraction_q16 / 2^16`.
pub fn random_l0_input(seed: u64, size: u32, fraction_q16: u32) -> impl Fn(usize) -> Vec<u32>;
```

## `Network` additions — `src/wave_net/network.rs`

```rust
/// Subtract `delta` from every threshold in `layer` (delta>0 lowers), clamped to [1, i16::MAX].
/// Uniform shift, so per-neuron jitter is preserved.
pub fn shift_layer_threshold(&self, layer: usize, delta: i32);

/// The layer's current maximum threshold (sizes the geometric step).
pub fn max_layer_threshold(&self, layer: usize) -> i16;

/// A copy of the layer's per-neuron thresholds (introspection / determinism tests).
pub fn layer_thresholds(&self, layer: usize) -> Vec<i16>;
```

All three are `&self` (interior mutability via the per-layer `Mutex`).

## `synapse.rs` addition

```rust
pub const P_INPUT: u64 = 5; // hash purpose tag for the calibration input generator
```

## Algorithm (hybrid: bottom-up then global refine)

Two shared free helpers in `calibrate.rs`:

```rust
// Reset, run `warmup` waves (discarded), then `waves` waves counting spikes per layer.
// Returns per-layer firing rate = spikes / (layer_size * waves). Single-threaded, deterministic.
fn measure(net: &mut Network, warmup, waves, input) -> Vec<f64> {
    attach a counting listener on every layer (Arc<Mutex<Vec<u64>>>);
    net.reset_state();
    for w in 0..warmup            { net.wave(&input(w)); }
    zero the counters;                        // discard warmup
    for w in 0..waves             { net.wave(&input(warmup + w)); }
    net.clear_listeners();
    counts[z] / (layer_size * waves) for each z
}

// One measure-informed nudge of layer z toward target. Returns whether it adjusted.
fn step_layer(net: &Network, z, rate: f64, target: f64, tol: f64, step_shift) -> bool {
    if (rate - target).abs() <= tol { return false; }
    let step = ((net.max_layer_threshold(z) as i32) >> step_shift).max(1);
    let delta = if rate < target { step } else { -step };  // too cold -> lower (delta>0)
    net.shift_layer_threshold(z, delta);
    true
}
```

`calibrate`:

```
target = target_permille/1000; tol = tol_permille/1000
// Phase 1 — bottom-up: fix each layer before moving up (its feeder is now firing)
for z in 1..L:
    for _ in 0..max_steps:
        rates = measure(net, warmup, waves, input)       // re-measure after each adjust
        if !step_layer(net, z, rates[z], target, tol, step_shift): break
// Phase 2 — global refine: absorb the downward (level 0/-1) coupling
for _ in 0..refine_passes:
    rates = measure(net, warmup, waves, input)           // one snapshot per pass
    moved = false
    for z in 1..L: moved |= step_layer(net, z, rates[z], target, tol, step_shift)
    if !moved: break
```

Geometric `step = max_threshold >> 2` self-damps: from `i16::MAX` it descends ~×0.75/step
(~15 steps to reach the low hundreds), and shrinks as the threshold shrinks, so it converges into
the tolerance band from any starting scale. L0 is never tuned — its rate is just the input
injection fraction.

## Determinism

`reset_state` before each measurement + a deterministic seed-based `input` closure ⇒ the resulting
thresholds are a pure function of `(net's seed/config, params, input)`. `measure` is single-threaded
(`net.wave` loop). No `Date`/`rand`/hash-map iteration.

## Known limitation

Pure bottom-up ignores the downward feedback; the `refine_passes` phase corrects most of it but is
itself capped, so with very strong recurrence a residual per-layer rate error can remain. Reported
acceptable for v1 (the ablation-grade tuning is the separate criticality program). No σ / temporal
structure is targeted — only firing rate.

## Testing (inline `#[cfg(test)]`, test-first)

Test config: `size = 8`, `L = 4`, forward+lateral topology (`level +1` r2 c6, `level 0` r1 c2),
`inhibitor_ratio = 0` (all excitatory, predictable), small `threshold_jitter`.

1. **`measure` sanity** — with `random_l0_input(seed, size, frac)`, `rates[0] ≈ frac/65536`
   (L0 fires exactly when injected; its high threshold blocks any recurrent firing). Loose band.
2. **calibration warms a silent net** — before: top-layer rate ≈ 0; after `calibrate`: top-layer
   rate is in a band around target (e.g. `(target/2, target*2)` and `> 0`), and
   `max_layer_threshold(top) < i16::MAX` (it dropped).
3. **thresholds lowered per layer** — every calibrated layer `1..L` has `max_layer_threshold < i16::MAX`.
4. **determinism** — two fresh nets, identical config/params/input → identical `layer_thresholds(z)`
   for all `z`.
5. **`shift_layer_threshold` / clamp** — subtracting past 1 floors at 1; adding past `i16::MAX`
   caps; jitter spread preserved by a uniform shift (unit test on a small layer).

`cargo build` warning-free; no `unsafe`; standard library only.

## Files touched

- **`src/wave_net/calibrate.rs`** — replace stub: `CalibrateParams`, `calibrate`, `random_l0_input`,
  `measure`/`step_layer` helpers, tests.
- **`src/wave_net/network.rs`** — add `shift_layer_threshold`, `max_layer_threshold`,
  `layer_thresholds`.
- **`src/wave_net/synapse.rs`** — add `P_INPUT`.
- `mod.rs` unchanged (`calibrate` already declared).

## Non-goals

No σ / criticality targeting, no per-neuron homeostatic field, no memory-capacity / XOR evaluation,
no threading, no persistence of the calibrated thresholds (they live in the mutated `Network`;
rebuilding from `Config` resets them — a later export/import can add persistence if needed).
