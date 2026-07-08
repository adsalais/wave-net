# Broadcast-error alignment — design (Spec 3, V2b)

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** one iteration — make the non-spiking potential readout (V2a) actually **learn** by replacing the
global scalar reward with a **per-output broadcast error** carried to internal neurons via fixed random
feedback weights (feedback-alignment). Only the *third factor* of the rule changes. No engine change.

## Why this

V2a proved a global *scalar* reward can't drive all-internal (feedback-alignment) learning — the fixed ±1
readout projection doesn't separate classes, so `(R − R̄) → 0`. Each internal neuron needs **class-specific
credit**. Broadcast-error alignment provides it: project the per-output error back to internal neurons
through fixed random weights. This is the standard e-prop learning-signal shape, and the piece the roadmap
said the readout must pair with.

## The change — only the learning signal

Everything from V2a stays: the non-spiking readout layer (`Layer.readout`, `new_with_readout`), potential
population **scores**, spike-count **eligibility** `eⱼ`, the f64 shadow, the trial mechanics. The rule's
third factor changes from a global scalar to a per-neuron broadcast error:

- **Softmax** the `K` readout scores: `pᵢ = softmax((scoreᵢ − max) / T)` (f64; subtract-max + temperature
  `T` avoid overflow — scores are large potential sums).
- **Per-output error** `errᵢ = targetᵢ − pᵢ`, `target` = one-hot(correct class). Already centered
  (`Σ errᵢ = 0`), so **no `R̄`**.
- **Fixed random feedback weights** `B(j, i) ∈ {−1, +1}`, hash-derived from the neuron's global id and
  output index via `synapse::{key, mix}` (deterministic, stored-free — the engine's ethos). Feedback
  *alignment*: `B` is random and fixed, not the forward readout weights.
- **Per-neuron learning signal** `Lⱼ = Σᵢ B(j, i) · errᵢ`.
- **Update** `Δθⱼ = − lr · Lⱼ · eⱼ`, over all computational neurons `1..L−1` (the readout has `eⱼ = 0` and
  is auto-excluded). Sign: a neuron aligned (`B=+1`) with the *under-firing* correct output (`err>0`) gets
  `Lⱼ>0` → `Δθ<0` → fires more. (Sign is a tuning check — flip `lr` if it learns the wrong way.)

So V2b = V2a with `(R − R̄)` replaced by `Lⱼ`. `RewardTracker` is unused in this path.

## Success criterion

- **V2b learns:** `train(cfg, lr)` with `broadcast = true` late-half accuracy > chance and clearly beats
  the frozen control (`lr = 0`), deterministically.
- **Comparison printed:** V1 (~770‰, spiking trainable output + global reward), V2a (null, readout + global
  reward), V2b (readout + broadcast) — so we see whether broadcast credit rescues the readout.
- `eprop_broadcast_is_deterministic`.

**Honesty gate:** feedback-alignment with a crude *spike-count* eligibility (which can't wake silent
neurons) may still be weak. If V2b can't beat frozen after reasonable tuning (`lr`, temperature `T`, the
update sign), that is the finding — reported, pointing to **symmetric feedback** (use the actual readout ±1
weights) or **potential-based internal eligibility**. Never a faked curve.

## Reuse & module

Extend `src/bench/eprop.rs`. Add `EpropConfig { broadcast: bool, softmax_temp: f64, .. }`; `train` branches
the *update* on `cfg.broadcast` (broadcast error vs the existing global reward), sharing scoring, eligibility,
shadow, and loop. V1 and V2a paths + tests stay intact. New helpers: `softmax`, `feedback_weight`.

- `feedback_weight(seed, global_id: u32, output: usize) -> f64`: `±1` from `mix(key(seed, global_id,
  output as i32, 0, P_FEEDBACK))`. Global id for computational neuron `(layer z, local i)` = `z·ls + i`.

**No engine change** — V2a's readout layer suffices.

## Testing (inline `#[cfg(test)]`)

- `feedback_weights_are_deterministic_and_signed` — `feedback_weight` reproducible, values in `{−1, +1}`,
  and different `(neuron, output)` decorrelate.
- `softmax_is_a_distribution` — sums to 1, monotone in the input, overflow-safe on large scores.
- `eprop_broadcast_learns_and_beats_frozen` — the headline (+ V1/V2a/V2b print); tuned.
- `eprop_broadcast_is_deterministic`.
- Regression: V1 + V2a `eprop` tests and the whole suite stay green.

## Determinism & constraints

- Std-only, no new deps. `B` hash-derived; `softmax` single-threaded f64 → deterministic. Engine untouched.
- `f64` only in the bench. Pure function of `(seed, config, params)`.

## Parameters (starting points, tunable)

Reuse the V2a demo (`size 8`, 3 computational + 1 readout, `K=2`, dense `level+1` count 16, `trials ≈
2400`, `block 200`). New: `softmax_temp` ≈ the score scale (start ~100), `lr` re-tuned for the bounded
`err ∈ [−1,1]` × spike-count magnitude (so a *larger* `lr` than V2a — the signal is now O(1), not
hundreds).

## Deferred (updated roadmap)

- **Symmetric feedback:** replace random `B` with the actual readout ±1 projection (hash-derivable) — proper
  e-prop credit, if feedback-alignment underperforms.
- **Potential-based internal eligibility:** let sub-threshold potential contribute to `eⱼ` so silent
  neurons can be recruited.
- High-drive float-free trainer, per-wave/TD credit, `K > 2`, training adaptation params — later rungs.
