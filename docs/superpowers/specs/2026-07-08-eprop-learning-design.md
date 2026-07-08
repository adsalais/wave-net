# e-prop-like threshold learning — design (Spec 3, v1)

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** one iteration — a first, gradient-free, e-prop-like **learning rule** that trains per-neuron
baseline thresholds from a reward signal, demonstrated on a held-category task with spiking output
neurons. **v1**: global-reward × eligibility, graded reward, spiking trainable outputs. **Deferred**:
non-spiking potential readout (v2), per-wave credit (v3), broadcast-error alignment, training adaptation
params, surrogate-gradient e-prop.

## Why this

The bench established *what* memory ALIF provides (held-category, across a delay). Spec 3 tests whether the
per-neuron threshold — the one trainable parameter in this weightless engine — can be **learned from
reward** via the e-prop signature (a per-neuron **eligibility trace** × a modulatory signal), rather than
only calibrated. This is the smallest thing that is genuinely "learning" here, and the prerequisite for
any real training layer.

## Success criterion

Run the same task+seed **with learning** (`lr > 0`) and as a **frozen-threshold control** (`lr = 0`).
Assert, deterministically:

1. **Learning happens:** the learning run's windowed accuracy climbs meaningfully above chance (50% for
   `K=2`) over training.
2. **It's the plasticity:** the learning run's final accuracy **exceeds the frozen control's** by a clear
   margin (the control stays near its starting/chance level — same reservoir, no threshold updates).
3. `eprop_is_deterministic` — two learning runs produce identical accuracy trajectories.

The frozen control is what makes "learning" unambiguous (not just a lucky reservoir).

**Honesty gate:** global-reward × eligibility is the *crudest* credit assignment (it reinforces all active
neurons on reward). It may learn a `K=2` discrimination or it may not. If it does not after reasonable
tuning, that is the finding — reported, pointing to the deferred upgrades (potential readout / broadcast
alignment) — never a faked learning curve. Tuning levers include shortening the delay (toward 0) to
isolate the *learning* mechanism from the *memory* demand.

## The task

Held-category, reusing store-recall's cue/probe encoding (lifted to `pub(crate)`):

- `K = 2` classes. Each trial: `reset` → **present** the class cue for `W` waves → **delay** `N` waves
  (silent) → **probe + read** for `R` waves (fixed probe, as in Spec 1).
- **Output neurons:** the first `K` locals of the top layer — ordinary spiking neurons whose thresholds
  are trained like all others. **Prediction** = the output neuron with the most spikes over the trial.
- **Graded reward:** `R = count[correct_output] − max(count[other_outputs])` — a signed margin (positive
  when the right output wins, and by how much; negative when it loses).

## The learning rule (v1)

Per neuron `j` (all computational layers `1..L`, output neurons included):

- **Eligibility** `eⱼ` = j's spike count over the whole trial, accumulated via `on_layer` listeners. A
  per-neuron, constant-memory trace, reset each trial. (The trial-level accumulation is what does the
  temporal credit assignment — the deferred one-hop delay washes out over the window.)
- **Modulatory signal** `s = R − R̄`, where `R̄` is a running mean of `R` (reward-prediction-error).
  Centering means a constantly-active neuron (same `eⱼ` every trial) drifts to zero net update, while a
  class-selective neuron gets correlated updates → builds selectivity.
- **Update:** `Δθⱼ = − lr · s · eⱼ` (a better-than-expected trial *lowers* active neurons' thresholds →
  reinforces firing in that context; worse-than-expected raises them).

**Integer thresholds + tiny updates → f64 shadow.** Thresholds are `i16`, but `lr · s · eⱼ` is small and
would round to 0 (the same dead-zone hazard we hit with adaptation). The trainer therefore keeps a **f64
shadow** of every threshold (initialized from the calibrated integer thresholds), accumulates the fractional
updates there, and writes the engine's integer threshold each trial as `shadow.round().clamp(1, i16::MAX)`.
The bench may use `f64` (hybrid policy); the engine stays integer.

## Training loop

1. Build a dense-ALIF net (held memory needs dense fan-out); **calibrate** once; snapshot the calibrated
   thresholds into the f64 shadow.
2. For `T` trials: pick a class (deterministic per-trial hash) → run the trial accumulating per-neuron
   eligibility → read output counts → compute `R`, update `R̄` → if `lr > 0`, update shadows and write
   engine thresholds. Record correct/incorrect.
3. Return a windowed accuracy trajectory (e.g. accuracy over trailing `M` trials).

```rust
// bench/eprop.rs
pub struct EpropConfig { /* net (size/layers/dense ALIF), task (K, W, N, R, cues), calib, lr, trials,
                           acc_window, seeds */ }
pub struct LearnCurve { pub accuracy_permille: Vec<u64> } // windowed accuracy over training
pub fn train(cfg: &EpropConfig, lr: f64) -> LearnCurve;   // lr = 0.0 → frozen control
```

## Engine access — no engine change

- **Read spikes:** `on_layer` listeners accumulate per-neuron spike counts per trial (eligibility + output
  counts). Already public.
- **Write thresholds:** the in-crate `pub(crate) Network::with_layer_mut` + the public `Layer.threshold`
  field let the trainer set per-neuron thresholds each trial. No new engine API required (a minimal
  accessor may be added only if it reads cleaner).
- **Reuse:** `store_recall::{cue_realization, probe_pattern}` become `pub(crate)` and are shared.

## Module

New `src/bench/eprop.rs`: `EpropConfig`, `LearnCurve`, `train`, the three-factor update, and tests. Reuses
`store_recall` cue/probe and the calibration/listener plumbing.

## Testing (inline `#[cfg(test)]`)

- `reward_prediction_error_centers` — unit test the `R̄` running-mean update and that a constant `R` yields
  `s → 0` (no drift), independent of the engine.
- `shadow_write_roundtrips_thresholds` — updating the f64 shadow and writing back changes the engine's
  per-neuron thresholds as expected (small-update accumulation crosses integer boundaries).
- `eprop_learns_and_beats_frozen_control` — the headline: `train(cfg, lr)` final accuracy > chance and >
  `train(cfg, 0.0)` final accuracy by a margin; print both trajectories.
- `eprop_is_deterministic` — two `train(cfg, lr)` runs give identical `accuracy_permille`.

## Determinism & constraints

- Std-only, no new deps. Engine untouched; bench uses the public/in-crate API. `f64` allowed in the bench
  (shadow, reward); single-threaded, fixed reduction order → deterministic.
- Everything a pure function of `(seed, config, params)`.

## Parameters (starting points, tunable)

`size 8–16` dense (`level+1` count 16), `3–4` layers, `K=2`, `W=6`, `N≈8` (reducible toward 0 to isolate
learning), `R=6`, `T≈1000` trials, `acc_window≈100`, `lr` small (tuned so the shadow moves ~1 threshold
unit over tens of trials). Delay/`lr`/cue-overlap are the main tuning levers.

## Scope guard (YAGNI) — explicitly out

- Non-spiking potential readout (v2); per-wave / TD credit (v3).
- Broadcast-error alignment (per-output feedback weights); training adaptation params (`adapt_bump`/decay);
  surrogate-gradient e-prop with a differentiable shadow.
- `K > 2`, curriculum, multiple tasks — not needed to answer "does the three-factor rule learn at all?"
