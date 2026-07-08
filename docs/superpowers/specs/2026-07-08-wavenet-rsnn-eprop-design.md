# wave_net → trained SNN via e-prop on stored int8 weights — design

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** turn `wave_net` from a fixed procedural reservoir into a **trained** spiking net by making synapse
**weights stored (int8) and plastic**, trained with **e-prop**. **First version = feed-forward only** (level
`+1` weights + a trained readout); recurrence (`level 0/−1`) is a later increment. Held-out + multi-seed
evaluation from the start. `wave_state_machine` stays the frozen procedural reference; the `bench` stays
pinned to it — this work targets `wave_net` and its own new training module.

## Why

The verification arc proved that training only per-neuron **thresholds** over frozen random weights is
structurally unreliable (a lucky reservoir×task lottery; width/hash/richer-weights all failed to fix it).
Thresholds can only *gate* a fixed projection; they cannot *shape* it. **Trainable weights shape the
projection — real feature learning** — which is the standard, reliable thing SNNs do. Per Knight & Nowotny,
we keep the **addresses procedural** (the expensive part, regenerated from the hash — free) and store only
the **plastic weights** (int8), the memory-minimal way to make a procedural net learnable. e-prop (Bellec
et al. 2020) is the forward, constant-memory, biologically-plausible approximation to BPTT — a fit for the
integer streaming engine and a reuse of the eligibility/broadcast/shadow machinery we already built.

## Substrate — stored int8 weights (increment 1, no learning yet)

- **`Layer.out_weights: Vec<i8>`** indexed by `(source local, slot)`, where `slot` runs over all
  `(topology level, k)` in generation order — i.e. `ls × total_slots` entries (`total_slots = Σℓ countℓ`).
- **`Layer.out_shadow: Vec<f32>`** (same shape) — the higher-precision training accumulator (the shadow
  trick, as with thresholds). Forward pass reads `out_weights` (int8); training accumulates in `out_shadow`
  and quantizes back (`round().clamp(-127, 127)`).
- **`generate_into`** looks up `out_weights[local·total_slots + slot]` and attaches it to the emitted
  `Synapse` (address still hash-generated) instead of hashing a ±1 sign. `Synapse.weight` stays `i16` (the
  delivered value); the drain already sums it.
- **Init:** the current hash-`±1` sign, written into int8 — so **increment 1 is behaviour-identical** to
  today (weights merely *stored* rather than regenerated); `wave_net`'s module tests pass unchanged. Training
  (increment 2) then moves them into the full int8 range.

## Learning — e-prop, feed-forward (increment 2)

**Factored eligibility (the key simplification).** For feed-forward synapses the e-prop eligibility
factorises into **per-neuron** terms, so we store O(neurons) eligibility state, not O(synapses):
- **Pre-trace** `pre_i` — a low-pass / trial-sum of presynaptic neuron `i`'s spikes.
- **Post-factor** `psi_j` — the trial-accumulated **pseudo-derivative** (surrogate gradient) of neuron `j`:
  large when `j`'s potential sits near its effective threshold, ~0 far away (a triangle/box around
  threshold). The engine accumulates `pre_i`, `psi_j` per wave (cheap, per-neuron; reset per trial).
- Synapse eligibility is the product on demand: `e_ij = pre_i · psi_j`.

**Learning signal.** `L_j = Σ_c B_jc · err_c`, with `err = softmax(readout_scores) − onehot(class)`.
**Symmetric feedback:** `B` = the (trained) readout weights — stronger than random alignment, and we have
them. (Random/broadcast alignment is a later ablation.)

**Update.** Loop over stored weights; for each `w_ij` (source `i`, target `j` via the *regenerated*
procedural address): `out_shadow_ij += −lr · L_j · pre_i · psi_j`, then quantise to `out_weights_ij` (int8).
O(synapses) only at update time; per-neuron during the wave.

**Trained readout.** A **non-spiking readout layer** (the V2a readout infra): output = readout-layer
potentials, produced by **stored + trained readout weights** (`K × N_output`, int8 + shadow, same e-prop
update with `err` as its learning signal). Graded output feeds the softmax/error cleanly.

## Recurrence (increment 3, deferred)

Enable `level 0/−1` topology and train those weights with the same rule → the full RSNN. Deferred so we
first isolate *does weight-e-prop learn* from *does recurrent credit assignment work*.

## Module & engine changes

- **Engine (`wave_net` only):** `Layer.out_weights`/`out_shadow`, per-neuron `elig_pre`/`elig_post`
  accumulators (updated in `process_layer`, reset by `reset_state`), `generate_into` reads stored weights,
  and small accessors to read/write weights + read eligibility (mirroring `layer_thresholds`/`with_layer_mut`).
- **Training (new `src/bench/rsnn.rs`, targeting `wave_net`):** the train loop, the e-prop update, the
  trained readout, the cue/probe task (reuse the *patterns*, rebuilt against `wave_net`), and the held-out +
  multi-seed evaluation. Does **not** touch the `wave_state_machine`-pinned bench.

## Success criterion

- **Reliable learning:** held-out test accuracy **above chance across multiple seeds** (not a single lucky
  one) on the K=2 store-recall task — the exact bar threshold-learning failed. Report mean ± spread over
  ≥3 seeds and the held-out (not prequential) number.
- Determinism: pure function of `(seed, task_seed, config)`.

**Honesty gate:** if weight-e-prop *also* fails the multi-seed held-out bar, that is a real (and important)
finding — report it; do not fall back to prequential accuracy or a single seed. But the prior is strong that
training weights (vs thresholds) crosses the reliability threshold.

## Determinism & constraints

- Engine stays integer + deterministic; the **f32 shadow** lives in the training path (bench), as with
  thresholds. Single-threaded, fixed reduction order.
- Std-only in the engine; the `blake3`/`random_weights` features remain test-only and untouched.

## Testing

- Engine: `stored_weights_roundtrip` (write shadow → quantise → read); `identity_init_matches_procedural`
  (increment 1 behaviour-identical to ±1); `eligibility_accumulates_and_resets`.
- Training: `rsnn_learns_and_generalizes` (held-out > chance) and `rsnn_is_seed_robust` (≥3 seeds, worst
  seed still > chance) — the headline; `rsnn_is_deterministic`.
- Regression: the whole existing suite (incl. `wave_state_machine` + bench) stays green.

## Deferred

- Recurrence (`level 0/−1` trained). Weight sharing / compression (convolutional per-level kernels,
  hashing-trick shared table) once learning works. Lower precision (int4 / ternary). Random-feedback
  ablation vs symmetric. Surrogate-gradient BPTT as a stronger-but-heavier alternative.
