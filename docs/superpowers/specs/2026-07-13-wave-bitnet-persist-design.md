# wave_bitnet — save/load (static model + runtime overlay)

**Date:** 2026-07-13
**Status:** design approved, ready for implementation plan

## Motivation

A trained `wave_bitnet` net currently lives only in memory. Two capabilities are wanted, *independently*:

1. **Persist a trained model** so inference can be re-run later (or shipped) without re-training.
2. **Snapshot the running state** mid-inference so a wave sequence can be paused and resumed exactly.

AGENTS.md already frames the target: *"a `Layer` is a self-contained, persistable unit (owns its
structure, thresholds, and stored weights) — serialization itself is not yet built."* This builds it.

**Out of scope (deferred):** the training checkpoint (readout weights, DFA feedback, trial counter,
best-checkpoint bookkeeping). Those live in the `#[cfg(test)]` bench harness, not in the engine, and the
engine itself holds no persistent training state beyond the weights. This spec covers only the two
engine-level saves.

## Locked decisions

1. **Two independent formats, layered.** A **static model** (`.wbm`) is self-contained and inference-ready.
   A **runtime overlay** (`.wbr`) carries *only* the mutable per-neuron state; it is **not standalone** —
   the caller `load_model`s first, then `apply_runtime` onto that network.
2. **Structure — self-contained (stored, not regenerated).** The materialized occupancy bitset (`occ`)
   and per-neuron `threshold` are written to disk. Load does **not** depend on `(seed, Config)` or the
   procedural hash; the file survives future changes to topology-generation code. (The cheap derived
   LUTs — `offsets`, `off_flat`, `neigh`, `occ_wpn`, `slot_bases`, `total_slots` — are pure functions of
   `topology`+`size` and are **rebuilt on load**, never stored.)
3. **Weights — 2-bit codes only.** The static model stores the packed ±1/0 `codes` (~0.25 B/synapse) plus
   `ternary_threshold`, not the `f32` shadow. Enough to run inference identically. The loaded model's
   `shadow` is reconstructed as the *decode* of the codes (codes are authoritative for the forward pass);
   it is **not** the original training master, so re-training from a code-only model is not meaningful.
   This is consistent with the deferred-training-checkpoint decision.
4. **Std-only, hand-rolled little-endian binary.** No serde / bincode (AGENTS.md: standard library only in
   `src/`). A small internal byte-I/O layer over `impl Write` / `impl Read`.
5. **Integrity + binding via a fingerprint.** An FNV-1a 64-bit fingerprint over a network's
   structural+weight bytes is written into the model header (verified on load) and into the runtime header
   (verified on `apply_runtime`, so an overlay can only be applied to the model it came from). Fail loud.
6. **Fail loud on any mismatch.** Bad magic, unknown version, length mismatch, or fingerprint mismatch →
   `io::Error(InvalidData)` with a descriptive message.

## Module layout

New file **`src/wave_bitnet/persist.rs`** (added to `mod.rs`), keeping `neurons.rs` / `network.rs`
focused. Three internal units:

- **Byte primitives** — `write_*` / `read_*` for LE scalars (`u8/u16/u32/u64/i8/i16/i32/f32`) and
  length-prefixed `Vec`s (`u64` length), generic over `impl Write` / `impl Read`, returning `io::Result`.
- **Fingerprint** — `Network::model_fingerprint(&self) -> u64`, FNV-1a over the structural+weight state in
  a fixed field order (dims, per-layer topology + scalars + `threshold` + `occ` + `codes`). Deterministic;
  independent of runtime state.
- **(de)serializers** for the two formats.

### New construction path (required)

Loading a self-contained model bypasses the seed-based `Layer::new`. Add:

- `Layer::from_parts(...)` — builds a `Layer` from stored fields (`topology`, scalars, `threshold`,
  `occ`, `codes`), **rebuilds** the derived LUTs from `topology`+`size`, reconstructs `shadow` as the
  per-slot decode of `codes`, and zeroes all runtime arrays. Validates `occ` / `codes` / `threshold`
  lengths against the rebuilt layout (`count ≤ (2r+1)²`, `occ[level].len() == ls·occ_wpn[level]`,
  `codes.len() == ceil(ls·total_slots/32)`, `threshold.len() == ls`).
- `Network::from_layers(size, layers)` — assembles a `Network` (`wave_id=0`, zeroed `scratch.deliv`,
  `record_eligibility` default-on, no `listeners`).

## Public API (on `Network`)

```rust
fn save_model(&self, w: impl Write) -> io::Result<()>;
fn load_model(r: impl Read) -> io::Result<Network>;          // associated fn
fn save_runtime(&self, w: impl Write) -> io::Result<()>;
fn apply_runtime(&mut self, r: impl Read) -> io::Result<()>; // in-place overlay
// path conveniences, wrapping File + BufWriter/BufReader:
fn save_model_path(&self, path: impl AsRef<Path>) -> io::Result<()>;
fn load_model_path(path: impl AsRef<Path>) -> io::Result<Network>;
fn save_runtime_path(&self, path: impl AsRef<Path>) -> io::Result<()>;
fn apply_runtime_path(&mut self, path: impl AsRef<Path>) -> io::Result<()>;
```

## Model format (`.wbm`)

```
magic:     4 bytes  b"WBNM"
version:   u16
size:      u32
n_layers:  u32
per layer:
  topology:          len-prefixed Vec of { level:i32, radius:u32, count:u32 }
  leak:              (u8, u8)
  cooldown_base:     u8
  adapt_bump:        i16
  adapt_decay:       u8
  readout:           u8   (bool)
  ternary_threshold: f32
  threshold:         len-prefixed Vec<i16>   (len == ls)
  occ:               n_levels × len-prefixed Vec<u64>   (occ[level], len == ls·occ_wpn[level])
  codes:             len-prefixed Vec<u64>   (len == ceil(ls·total_slots/32))
fingerprint: u64   (FNV-1a over all bytes above)
```

**Load:** read header + layers via `Layer::from_parts`; recompute the fingerprint over the payload and
verify it equals the trailing value (integrity); assemble with `Network::from_layers`. `shadow` is the
decode of `codes`; all runtime arrays zero; `wave_id = 0`.

## Runtime format (`.wbr`)

```
magic:     4 bytes  b"WBNR"
version:   u16
model_fp:  u64   (fingerprint of the model this overlay belongs to)
size:      u32
n_layers:  u32
wave_id:   u64
per layer (each len-prefixed):
  potential:i16 · cooldown:u8 · adapt:i32 · pending:i32
  elig_pre:i32 · elig_post:i32 · decide_potential:i16 · decide_eff:i32
fingerprint: u64   (FNV-1a over all bytes above)
```

**Apply (`apply_runtime`, in place):** check magic/version and the trailing fingerprint; check
`model_fp == self.model_fingerprint()` and `size`/`n_layers`/per-layer lengths match `self`; then
overwrite the eight arrays + `wave_id`.

**Why these fields and no others.** Between waves `scratch.deliv` is provably all-zero — `Network::wave`
drains each layer's incoming accumulator into `pending` (step 1) and swaps `pending`↔`deliv` at wave-end,
so the carried inter-wave signal lives entirely in `pending`. Saves/applies happen between `wave()` calls
(a wave is atomic; there is no mid-wave API), so `deliv` is never stored. `listeners` are closures
(unsaveable) and `record_eligibility` is a caller-owned mode; neither is persisted. The eight arrays are
exactly the mutable per-neuron runtime state (the set `reset_state` clears, plus `decide_eff`).

## Error handling

All fallible ops return `io::Result`. Validation failures use `io::Error::new(ErrorKind::InvalidData, msg)`
with a descriptive message (which check failed, expected vs actual). No custom error type, no new deps.

## Testing (TDD)

Inline `#[cfg(test)]` in `persist.rs`:

- **Model round-trip (fields)** — build a small net (and mutate a few `codes`/`threshold` to be
  non-trivial), `save_model` to a `Vec<u8>`, `load_model`, assert every persisted Layer field **and** the
  rebuilt derived LUTs equal the original.
- **Model round-trip (inference-equivalence)** — run N waves on the original and on the loaded net from
  identical input; assert identical spikes / `decide_potential` throughout.
- **Runtime round-trip** — drive a net with several waves (dirtying runtime), `save_runtime`; on a **fresh**
  `load_model` of the same model, `apply_runtime`; assert the eight per-neuron arrays + `wave_id` match,
  **and** that subsequent waves produce identical outputs (true resume).
- **Fingerprint binding** — `apply_runtime` of an overlay onto a *different* model (different weights or
  dims) returns `InvalidData`.
- **Malformed input** — bad magic, truncated stream, and unknown version each return `InvalidData`.
- **Byte-stability / determinism** — saving the same net twice yields identical bytes;
  `model_fingerprint` is stable across save→load.

## Deferred (explicitly out of this cut)

- **Training checkpoint** — readout weights, DFA feedback, trial counter, best-checkpoint. Lives in the
  bench harness; the runtime overlay already carries the engine-side state a future training-resume needs.
- **f32 shadow persistence / training-resume from a saved model** — follows from the codes-only decision;
  add a shadow section to `.wbm` (or a separate format) if/when training-resume is built.
- **Cross-version migration** — the `version` field reserves the space; no migration path is built yet
  (an unknown version fails loud).
