# wave_bitnet — optional `TrainState` (training data behind a toggle)

**Date:** 2026-07-13
**Status:** design approved, ready for implementation plan

## Motivation

Every `Layer` today permanently allocates its training state alongside the inference state, so a model
loaded purely to run forward passes still pays for weights it never reads. The dominant cost is the
`f32` **shadow** — 4 bytes/synapse, **16×** the 2-bit `codes` the forward pass actually uses — plus
four per-neuron scratch arrays. At size 32 that is ~155 KB/layer of which only ~21 KB is needed for
inference.

The forward hot path already reads only `codes` and already gates eligibility writes behind a flag, so
inference is *computationally* train-free; it is only paying the **allocation**. This spec moves all
training data into an `Option<TrainState>` that exists **only while training is enabled**, toggleable
on and off within one process, so inference (and untrained-forward profiling) pays zero for it.

**In scope:** relocating the engine's training state (`shadow`, `decide_potential`, `decide_eff`) into
an optional per-layer `TrainState`; a `Network` enable/disable toggle; deleting the dead `elig_pre` /
`elig_post`; the resulting `.wbr` format change; updating tests, harnesses, and `AGENTS.md`.

**Out of scope:** the bench-side training bookkeeping (readout weights, DFA feedback, trial records) —
that already lives in the `#[cfg(test)]` harness, not the engine. The forward algorithm, topology
materialization, weight quantization, and the `.wbm` model format are unchanged.

## Locked decisions

1. **Approach A — `Layer` owns `Option<TrainState>`.** `shadow` stays adjacent to `codes`, so
   `repack_row` (the shadow→codes requantize) remains a self-contained `Layer` method. The coupling
   stays local; no per-neuron plumbing across the `Layer`/`Network` boundary.

2. **`TrainState` holds exactly the training data:**
   ```rust
   pub struct TrainState {
       pub shadow: Vec<f32>,           // ls * total_slots
       pub decide_potential: Vec<i16>, // ls  — decide-time membrane snapshot
       pub decide_eff: Vec<i32>,       // ls  — decide-time effective threshold snapshot
   }
   ```
   `Layer` gains `pub train: Option<TrainState>` and loses `shadow`, `elig_pre`, `elig_post`,
   `decide_potential`, `decide_eff` as direct fields.

3. **Lean by default; explicit enable.** `Network::new`, `new_with_readout`, and `load_model` all
   produce `train: None`. Fresh and loaded nets behave identically — the rule is uniform: *training
   state exists iff you asked for it.* This is the most faithful reading of "initialised only when we
   train," at the cost of an explicit `enable_training()` at each training site.

4. **`elig_pre` / `elig_post` are deleted, not relocated.** Nothing reads them: the multi-layer-DFA
   path builds eligibility from `TrialRecords` (spikes/pots/effs), and `eprop_update_synaptic` takes an
   external `elig` slice. Their `(A2)` pre-trace bump loop is removed from `process_layer` — a small
   forward-pass win as a bonus.

5. **`record_eligibility` flag is removed.** Recording the decide snapshot is now driven purely by
   `train.is_some()`. One concept ("am I training") replaces two. `set_record_eligibility` and its two
   now-redundant harness call sites are deleted.

6. **`.wbm` unchanged; `.wbr` slimmed.** The model format never carried training state. The runtime
   overlay currently serializes `elig_pre/elig_post/decide_potential/decide_eff` — all now deleted or
   transient — so the new overlay is just the genuinely-resumable runtime: `potential, cooldown, adapt,
   pending` (+ `wave_id` + fingerprint). The shared `VERSION` splits into `MODEL_VERSION = 1`
   (unchanged) and `RUNTIME_VERSION = 2`; existing `.wbm` files stay valid, old `.wbr` overlays
   (ephemeral caches) are rejected loud.

7. **`disable_training` is lossy for in-flight sub-threshold shadow — by design, and consistent with
   existing semantics.** Re-enabling reconstructs `shadow = decode(codes)`, snapping fractional
   accumulation back to the quantized ±1/0. This is **identical to a `.wbm` save/load round-trip**
   (codes are the cross-checkpoint master; shadow never was). Documented on `disable_training`.

## API

On `Network`:

- `enable_training(&mut self)` — for every layer, allocate `TrainState { shadow: decode(codes),
  decide_potential: zeroed, decide_eff: zeroed }`. **Idempotent**: if a layer is already training, leave
  its `TrainState` untouched (never clobber accumulated shadow).
- `disable_training(&mut self)` — set every `layer.train = None`, freeing the memory. No-op if already
  lean. Doc-comment the lossy-round-trip semantic (decision 7).
- `is_training(&self) -> bool`.

On `Layer`:

- `train: Option<TrainState>` field (public, matching the existing all-public field convention).
- `enable_training(&mut self)` / `disable_training(&mut self)` — per-layer helpers the `Network`
  methods map over. `enable` allocates `shadow` as the per-slot decode of `codes` (reusing the same
  decode `from_parts` performs today) and zeroes the two decide arrays.
- `repack_row` and any shadow access panic with a clear message if `train` is `None` (calling a
  training operation without enabling training is a programmer error).

## Behavioral changes by file

- **`neurons.rs`** — add `TrainState`; add `Layer.train`; remove the five relocated/deleted fields from
  `Layer` and from both `new` and `from_parts` constructors. `Layer::new` seeds `codes` from the
  procedural ±1 sign (via a transient shadow it does **not** retain) and ends `train: None`.
  `from_parts` **stops** reconstructing shadow; ends `train: None`. `repack_row` reads/writes
  `train.shadow`. Add `enable_training`/`disable_training` and a synapse-count helper
  (`ls * total_slots`) for footprint reporting.

- **`wave.rs`** — `process_layer` replaces the `record_elig: bool` parameter with access to the
  optional decide scratch (via `&mut Layer`, reading `layer.train`). `(A0)` writes
  `decide_potential`/`decide_eff` **iff `train` is `Some`**. `(A2)` elig-pre bump loop is **deleted**.
  Forward generate/delivery path is unchanged.

- **`network.rs`** — remove the `record_eligibility` field, `set_record_eligibility`, and the
  `record_elig` argument threading in `wave`. Add `enable_training`/`disable_training`/`is_training`.
  `reset_state` zeroes the decide scratch only when `train` is `Some` (and no longer touches the deleted
  elig arrays; shadow is weights and is **not** reset). `build`/`from_layers` set `train: None`.
  `layer_decide_potential` / `layer_decide_effective_threshold` read from `train` and **panic with a
  clear message when lean** (only meaningful mid-training — fail loud, per the AGENTS ethos).
  `eprop_update_synaptic` reads/writes `train.shadow`.

- **`persist.rs`** — split `VERSION` into `MODEL_VERSION`/`RUNTIME_VERSION`. `save_runtime` /
  `apply_runtime` drop the four elig/decide vectors; the staged tuple and length checks shrink to
  `potential, cooldown, adapt, pending`. `save_model`/`load_model` unchanged. Update the module doc
  comment. The `model_roundtrip_inference_equivalent` and `runtime_resume_equivalent` tests compare
  `decide_potential` on **lean** loaded nets today; switch those inference-equivalence checks to compare
  `potential` (and/or fired spikes), which lean inference actually produces, since the decide accessors
  now panic when lean.

- **`multilayer_dfa.rs`** — no logic change (it already sources from `TrialRecords`); update tests that
  read `l.shadow` to go through `l.train`.

- **`bench/wave_bitnet_bench.rs`** — `make_ff` / `make_sidecar` (and any other builder) call
  `net.enable_training()` after construction. `weight_sparsity` counts via `ls * total_slots` instead of
  `shadow.len()`.

- **`examples/profile_bitnet.rs`, `benches/throughput_bitnet.rs`** — delete the now-redundant
  `set_record_eligibility(false)` calls (both nets are `train: None`).

- **`AGENTS.md`** — update the `neurons.rs` architecture-map line and the "f32 training shadow" /
  "elig/decide state" mentions to describe the optional `TrainState` and the enable/disable toggle.

## Testing

TDD, inline `#[cfg(test)]` per module, matching the repo convention. New/updated coverage:

- **Toggle allocates/frees** — a fresh net is `!is_training()`; after `enable_training()` it is, and
  every layer's `train` is `Some` with `shadow.len() == ls * total_slots`; after `disable_training()`
  every `train` is `None`.
- **Enable is idempotent / preserves shadow** — mutate a shadow value, call `enable_training()` again,
  assert the value is unchanged.
- **Enable reconstructs decode(codes)** — right after `enable_training()` on a fresh net,
  `shadow[s] == weight_at(s) as f32` for all `s` (the init identity).
- **Lean inference matches trained-net inference** — a loaded lean net and the same net with training
  enabled produce identical `wave`/decode sequences (forward reads only codes).
- **Training still trains** — the existing `multilayer_dfa` "raises weights on negative signal" test
  passes after `enable_training()`.
- **`.wbr` round-trip on the slimmed format** — `runtime_resume_equivalent` updated to compare only
  `potential/cooldown/adapt/pending`; add a check that an old-version runtime header is rejected.
- **Panic-on-lean** — `repack_row` / decide accessors panic when `train` is `None` (documented
  programmer error).

## Payoff

At size 32 a trained layer's heap drops from ~155 KB to ~21 KB (**~86% freed**, dominated by `shadow`
at 16× the codes); the ratio holds at every size. Loaded-for-serving models and untrained-forward
profiling pay **zero** for training state, and the toggle makes `train → checkpoint → disable → serve
lean` a first-class in-process workflow.
