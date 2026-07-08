# Multi-layer credit via DFA — design

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** extend `wave_net`'s feed-forward e-prop to train **every** layer's weights (the input projection
`L0→L1` through `L(top−1)→top`), not just the last hidden layer. Deeper layers get their learning signal by
**Direct Feedback Alignment** (fixed random per-layer feedback of the output error); the top layer keeps
symmetric readout feedback. Demonstrated on a **deep net** where separation erodes with depth — multi-layer
should learn reliably there, single-layer should struggle. All in `bench::rsnn`; no engine change.

## Why

Task 4 trained only the last hidden projection and reached ~890–1000‰ reliably — but on a *shallow* net
(one trainable layer sufficed). The regime diagnostic showed **separation decays with depth**: fixed
intermediate layers wash the class signal out. So on a *deep* net, training only the last layer should
weaken (its input is already degraded), while **training all layers** lets each intermediate layer *preserve
and transform* the signal. Multi-layer credit is the capability that makes depth usable — and DFA is the
forward, local, scalable way to route credit to deep layers without backprop.

## The learning signal per layer (DFA + symmetric top)

For a trainable feed-forward layer `z` (its `out_weights` are the `level+1` synapses into layer `z+1`), the
weight update needs a per-neuron learning signal `L_j` for the **target** neurons `j` in layer `z+1`:

- **`z+1 == top`** (feeds the readout): **symmetric feedback** — `L_j = Σ_c w[c][j]·err_c` (readout
  weights), exactly as Task 4.
- **`z+1 < top`** (a deeper layer): **DFA** — a fixed random per-layer feedback matrix broadcasts the output
  error directly: `L_j = Σ_c B_{z+1}(j, c)·err_c`, with `B_{z+1}(j,c) ∈ {−1,+1}` hash-derived
  (`mix(key(seed, z+1 global id of j, c, 0, P_DFA))`, stored-free). No backprop; each deep layer sees the
  output error through its own random projection.

## The update (factored eligibility, per layer)

For each trained layer `z`, for each stored weight `out_weights[z][i·total_slots + slot]` (source `i` in `z`,
target `j = target_of(level 1, …)` in `z+1`):
`out_shadow[z] += −lr · L_j · elig_pre[z][i] · elig_post[z+1][j]`, then quantize → int8.
The factored per-neuron eligibility (`elig_pre` of the source layer × `elig_post` of the target layer) is
already accumulated by the engine (feed-forward form). Read both via `with_layer_mut`.

- **Single-layer (baseline):** train only `z = top−1`.
- **Multi-layer:** train `z ∈ {0, …, top−1}` — every projection, *including the input projection* `L0→L1`
  (L0 is the transducer; its `out_weights` are the learnable input encoding).

Readout reads the **top** layer (as Task 4), so the comparison is "can the trained net make the top layer
separable at depth."

## Success criterion

- **Multi-layer beats single-layer at depth, reliably:** on a deep net (layers 4–5), multi-layer held-out
  test accuracy is clearly above single-layer's, worst-seed above chance+margin, multi-seed. Print
  single-vs-multi per seed and per depth.
- Determinism preserved.

**Honesty gate:** if single-layer already handles depth (multi-layer only "still works"), report that —
it means one trainable layer + the fixed reservoir suffices even deep, and DFA credit is a capability we
have but don't yet need. Also honest if DFA is *unreliable* (random feedback can be noisy): report the
multi-seed spread, don't cherry-pick.

## Module & engine

- **Engine:** none (`elig_pre`/`elig_post` per layer already exist; `level+1` topology already used).
- **Bench (`src/bench/rsnn.rs`):** `RsnnConfig.multi_layer: bool`; a `dfa_weight` helper (`P_DFA` purpose);
  `train_eprop` generalized to loop over trained layers with per-layer feedback; a deep-net config; the
  single-vs-multi × depth test harness.

## Testing

- `dfa_weights_are_deterministic_and_signed` — unit on `dfa_weight`.
- `multilayer_beats_single_layer_at_depth` — the headline (deep net, single vs multi, multi-seed, held-out).
- `multilayer_is_deterministic`.
- Regression: whole suite (incl. `wave_state_machine`) stays green.

## Deferred

DFA-everywhere (drop symmetric top); training `level 0/−1` recurrent weights with multi-layer credit
(combine with the recurrence track); weight sharing / int8 readout (memory); per-layer learning rates.
