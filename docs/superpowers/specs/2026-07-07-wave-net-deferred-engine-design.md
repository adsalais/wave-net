# wave_net engine — deferred one-hop propagation (v1) — design

**Date:** 2026-07-07
**Status:** approved (brainstorming complete)
**Scope of this spec:** v1 = a *technically functioning*, deterministic, single-threaded
engine. **No threads, no training, no calibration logic, no demo/example.** Not
byte-identical to `wave_reservoir` — it is a deliberately different model.

## Relationship to prior work

This **supersedes** the deleted `readable-wave-net-engine-design.md` (a shelved attempt at a
bit-identical readable twin of `wave_reservoir::LayerNet`). It also **diverges on purpose** from
the within-wave propagation model that `AGENTS.md` currently documents as canonical — see
"Divergences from `wave_reservoir`". `wave_reservoir` stays frozen as the reference base;
`legacy_net` (the earlier fork) stays as read-only reference. `AGENTS.md`'s "engine model"
section should get a follow-up edit once this lands (flagged, not done here).

The one inherited principle (from `AGENTS.md`, non-negotiable): **synapses are never stored —
they are recomputed on demand from a hash** as a pure function of `(seed, source, config)`.
Determinism flows from `(seed, config)`. We train per-neuron parameters later, never a stored
synapse matrix.

## Resolved design decisions (the forks)

1. **Potential is `i16`, signed.** Rest = 0. Inhibitory deliveries (`−1`) hyperpolarize below 0.
   One saturation clamp per wave.
2. **Thresholds start near `i16::MAX` (silent).** `threshold = i16::MAX − rand(0..jitter)`, stored
   `i16`, per neuron. Consequence: with `±1` deliveries, **no neuron fires from integration** — only
   forced input (L0) fires — until calibration lowers thresholds. v1 is *functioning but inert above
   L0*, by choice. There is no low-threshold "base" knob; the single jitter mechanism replaces the
   legacy `spread_log2` (dropped as redundant).
3. **Propagation is deferred, one hop per wave.** A firing neuron's deliveries land in the target
   layers' inboxes for the **next** wave. Signal advances exactly one layer per wave. This makes
   layer processing order-independent (and trivially parallelizable later).
4. **Saturation is per-layer and calibration-owned** (`i16`). v1 default = `i16::MAX` (pure
   overflow/runaway backstop). Invariant `saturation ≥ max threshold in that layer`, asserted at
   construction and to be maintained by calibration. Rationale: a neuron whose `threshold >
   saturation` can never fire (potential is clamped below the bar), so this invariant is a
   well-formedness condition, not just an injection detail.
5. **Input is a sparse `Vec<u32>` of L0 local addresses** = spike injection, not graded current.
   Injecting a neuron forces it to fire this wave.
6. **Synapse generation returns deliveries grouped by relative level** (`Vec<SynapseGroup>`); the
   `Network` does all absolute-layer math, bounds-checking, and routing. Generation is
   geometry-agnostic (knows only relative levels).
7. **Single-threaded.** `Vec<Mutex<Layer>>` is kept to signal future threading intent, but v1 locks
   sequentially.

## Data structures

### `synapse.rs` — wiring types + hash primitives + generation

```rust
pub struct TopologyLevel { pub level: i32, pub radius: u32, pub count: u32 }

pub struct Synapse { pub target: u32, pub inhibitory: bool }   // target = LOCAL index in dest layer

pub struct SynapseGroup { pub level: i32, pub synapses: Vec<Synapse> }
```

Hash primitives already present (`mix`, `key`, `map_range`, `map_range24`) carry over from
`wave_reservoir::hash` unchanged. Generation (new):

```rust
// Append one firing neuron's synapses into `groups`. Caller pre-sizes `groups` to one entry per
// topology level (in topology order), sets each `.level`, and clears `.synapses` once per wave;
// every firer in the layer appends into the matching group → aggregation with no per-firer alloc.
pub fn generate_into(
    seed: u64,
    source_global: u32,      // z*size*size + local — hash uniqueness key
    src_local: u32,          // (x,y) offset base within the layer
    size: u32,               // layer side (power of two); toroidal wrap = & (size-1)
    topology: &[TopologyLevel],
    inhibitor_ratio: u32,    // Q16: inhibitory iff (hash & 0xFFFF) < inhibitor_ratio
    groups: &mut [SynapseGroup],
)
```

Per topology entry, for `k in 0..count`: one hash → `dx,dy` in `[-radius, radius]` (via
`map_range24`), toroidal-wrapped to `(tx,ty)`; `inhibitory` from the low 16 bits vs
`inhibitor_ratio`; push `Synapse { target: ty*size+tx, inhibitory }` into that entry's group. This is
the same hash layout as `wave_reservoir::wiring::for_each_target`, emitting **relative** levels
instead of absolute targets.

A tiny square-grid index helper (local ↔ `(x,y)`, wrap by `& (size-1)`) lives in `synapse.rs`
(replaces legacy `Dims`; `size` is a single network-wide power-of-two side, so all math is
shifts/masks).

### `neurons.rs` — `Layer` owns its state + per-layer config

```rust
pub struct Layer {
    // wave-mutable hot state
    potential: Vec<i16>,     // rest 0, ± hyperpolarize/depolarize, clamped ±saturation
    cooldown:  Vec<u8>,      // refractory counter, fires at 0
    inbox:     Vec<Synapse>, // deliveries to drain THIS wave (filled last wave)
    outbox:    Vec<Synapse>, // deliveries being filled for NEXT wave; swapped with inbox at wave end

    // tunable params (constant during a wave; rewritten between phases by calibration/training)
    threshold: Vec<i16>,     // per neuron; i16::MAX - rand(0..jitter)
    saturation: i16,         // per layer; default i16::MAX

    // fixed structure
    leak: (u8, u8),          // leak_a, leak_b shift amounts (≥1)
    cooldown_base: u8,       // refractory reload on fire (≥1)
    topology: Vec<TopologyLevel>,
    inhibitor_ratio: u32,    // Q16 inhibitory fraction
}
```

Everything a layer needs lives in `Layer` (the sketch's choice). The deferred model makes this safe
even for future threading: `process_layer` only ever borrows its **own** layer (it generates into a
neutral scratch, never into sibling layers), so the borrow conflict that forced `wave_reservoir` to
split `LayerCfg`/`Layer` does not arise here. Two changes vs. the sketch: **`potential`/`threshold`
are `i16`** (was `u16`), and **`update_buffer` becomes the `inbox`/`outbox` pair** (double-buffering
is what makes propagation truly one-hop). `listeners` move out of `Layer` onto `Network` (they are
not per-neuron state; keeping them off the mutex mirrors the reference engine).

Fixes rolled in: `treshold`→`threshold`, `inibitor_synapse_ratio`→`inhibitor_ratio`, `spread_log2`
dropped.

### `network.rs` — owns layers, does all routing

```rust
pub struct Network {
    seed: u64,
    size: u32,                 // square side, power of two; shared by all layers
    layers: Vec<Mutex<Layer>>,
    wave_id: AtomicUsize,      // monotonic, for listener callbacks
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,   // per layer
}
```

### `config.rs` — construction input (new module)

```rust
pub const THRESHOLD_JITTER_DEFAULT: u16 = 128;

pub struct LayerConfig {
    pub topology: Vec<TopologyLevel>,
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub inhibitor_ratio: u32,
    pub threshold_jitter: u16,   // rand span subtracted from i16::MAX
    pub saturation: i16,         // default i16::MAX in v1
}

pub struct Config {
    pub seed: u64,
    pub size: u32,               // power of two
    pub layers: Vec<LayerConfig>,
}
// Config::demo() -> a small valid network for tests.
```

## The wave step — `wave.rs::process_layer`

Processes **one** layer for one wave. Mutates only that layer; needs no other layer. Standard
integrate → fire → decay LIF ordering:

```
process_layer(layer, layer_index, seed, size, input, out, fired):
  1. cooldown -= 1                              (saturating)
  2. drain inbox:   for s in inbox: acc[s.target] += (s.inhibitory ? -1 : +1)   (i32 accumulate)
                    potential[i] = (potential[i] as i32 + acc[i]) saturated into i16
                    inbox.clear()
  3. inject (L0 only): for a in input: potential[a] = saturation; cooldown[a] = 0
  4. clamp:         potential[i] = potential[i].clamp(-saturation, +saturation)   ← the one sat clamp
  5. decide:        for i in 0..layer_size:
                        if cooldown[i]==0 && potential[i] >= threshold[i]:
                            potential[i] = 0; cooldown[i] = cooldown_base; fired.push(i)
  6. generate:      for each firer, generate_into(...) → aggregate into `out` (grouped by level)
  7. leak:          potential[i] -= (potential[i]>>leak.0) + (potential[i]>>leak.1)   (survivors)
```

Notes:
- **Exactly one saturation clamp per wave** (step 4), applied after the full inbox is summed in
  `i32` → order-independent. Drain's `i16` store (step 2) is only type-safety narrowing, never the
  saturation bound, and never triggers under any physical fan-in.
- `acc` in step 2 is a reused `Vec<i32>` scratch of length `layer_size` (Network- or Layer-owned),
  used purely to sum a wave's deliveries per target before the single narrowing/clamp — this is what
  makes accumulation order-independent rather than relying on per-delivery saturating adds.
- **Injection fires reliably** because `saturation ≥ threshold` (invariant): `potential = saturation`
  clears any threshold. Placed after drain, before decide, with no intervening leak. It resets
  `cooldown` to 0, so forced input is immune to refractory (by design).
- **Leak is last**, so it can't pull an injected `saturation` below threshold; it only decays
  survivors into the next wave. Fired neurons are `0`, and `leak(0)=0`, so leak is a no-op on them.
- `out`/`fired` are reused scratch buffers owned by `Network` (zero per-wave allocation). `out` has
  one group per topology level, `.level` preset, `.synapses` cleared before the layer runs.

## Network orchestration — `Network::wave`

```
wave(&self, input: &[u32]):
  w = wave_id.fetch_add(1)
  for z in 0..L:
      inp = (z == 0) ? input : &[]
      lock layer[z]; process_layer(layer[z], z, seed, size, inp, &mut out, &mut fired); unlock
      for g in &out:                          # route: Network owns absolute-layer math
          tl = z as i32 + g.level
          if 0 <= tl < L: lock layer[tl]; layer[tl].outbox.extend(&g.synapses); unlock
      if let Some(f) = &listeners[z]: f(w, &fired)
  for layer in layers: swap(layer.inbox, layer.outbox)   # outbox (filled this wave) → next inbox
```

Because scatter targets each layer's **outbox** (not the inbox being drained), a lower layer feeding
a higher one that is processed later in the *same* sweep does not deliver early — it's genuinely one
hop/wave, and the per-layer result is independent of sweep order. `wave_id` is monotonic for listener
stream ordering.

## Public API (v1)

```rust
impl Network {
    pub fn new(config: Config) -> Network;        // build layers, hash thresholds, assert invariants
    pub fn wave(&self, input: &[u32]);            // one deferred wave; input = L0 forced-fire locals
    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>);
    pub fn clear_listeners(&mut self);
    pub fn reset_state(&self);                     // zero potential/cooldown/inbox/outbox
    pub fn potential(&self, layer: usize, local: usize) -> i16;   // readout
    pub fn size(&self) -> u32;
    pub fn layer_count(&self) -> usize;
    pub fn n_total(&self) -> usize;
}
```

`calibrate.rs` stays a **no-op stub** in v1 (documented placeholder; real per-layer
threshold/saturation tuning is a later phase).

## Construction & invariants (asserted in `Network::new`)

- `size` is a power of two, `size ≥ 1`.
- `layers.len() = L ≥ 1`.
- Per layer: `leak.0 ≥ 1`, `leak.1 ≥ 1`, `cooldown_base ≥ 1`, `saturation ≥ 1`.
- Threshold init: `threshold[global] = i16::MAX − map_range(mix(key(seed, global, 0, 0, P_THRESHOLD)),
  jitter)`, deterministic from `(seed, config)`.
- **`saturation ≥ max threshold in the layer`** — holds for free at v1 defaults
  (`saturation = i16::MAX ≥ threshold`), asserted so future calibration fails loudly if it violates
  it.

## Divergences from `wave_reservoir` (explicit)

| Aspect | `wave_reservoir` / `AGENTS.md` | this engine |
|---|---|---|
| Propagation | within-wave, forward as the front climbs | deferred, one hop/wave |
| Input | dense `drive: &[i16]` (graded current) | sparse `Vec<u32>` L0 spikes |
| Thresholds | low (~4), fire from integration | near `i16::MAX`, silent until calibration |
| Saturation | one global constant | per-layer, calibration-owned |
| Delivery | applied immediately into target potentials | buffered (inbox/outbox), summed next wave |
| Threading | wavefront pipeline, band locks | single-threaded (v1) |
| Determinism basis | order-independent `Σ±1` + entry-ordered pipeline | order-independent `Σ±1` + layer-independent deferral |

## Determinism argument

- Wiring is a pure function of `(seed, source, config)` — no stored graph.
- Deliveries are `±1` summed in `i32` before a single clamp → order-independent.
- Deferred model: a layer reads only the inbox filled *last* wave and writes only outboxes for
  *next* wave, so no layer observes another's same-wave writes → per-wave result is independent of
  layer sweep order (and safe to parallelize over layers later).
- Firer lists are ascending local index (decide scans `0..layer_size`).

## Verification (v1 "functioning")

The default config is inert above L0, so dynamics are tested two ways:

**Unit tests on `process_layer` / generation (thresholds hand-set low to exercise dynamics):**
- integrate → fire → reset (`potential=0`, `cooldown=cooldown_base`) when `potential ≥ threshold`;
- refractory blocks re-fire while `cooldown > 0`, re-fires after `cooldown_base` waves;
- leak decays a sub-threshold potential toward 0 (and works on negatives);
- inhibition drives potential negative; single clamp bounds to `±saturation`;
- `generate_into` produces the expected per-level synapse counts and toroidal targets;
- deferred routing: deliveries land in `outbox`, become readable only after the swap.

**Integration tests on assembled `Network` (default near-max thresholds):**
- `new` builds a valid config and asserts invariants; invalid configs panic (non-pow2 `size`, empty
  `layers`, `saturation < max threshold`);
- injecting a set of L0 locals fires exactly those (only forced neurons fire);
- **one-hop deferral:** an injected L0 neuron's deliveries change L1 potentials at wave *t+1*, not
  *t*;
- determinism: identical `(seed, config, input sequence)` → identical potentials and fire sets across
  two independent runs;
- `reset_state` zeros everything.

All inline `#[cfg(test)]` per module (repo convention), test-first where practical. `cargo build`
stays warning-free; no `unsafe`; standard library only.

## Files touched (all under `src/wave_net/`, nothing else changes)

- **`config.rs`** (new): `Config`, `LayerConfig`, `Config::demo`, `THRESHOLD_JITTER_DEFAULT`.
- **`synapse.rs`**: keep hash primitives; add `TopologyLevel`/`Synapse`/`SynapseGroup`,
  `generate_into`, grid index helper.
- **`neurons.rs`**: `Layer` as above (i16, inbox/outbox, typo fixes, listeners removed).
- **`wave.rs`**: `process_layer`.
- **`network.rs`**: `Network`, `new`, `wave`, `on_layer`, `clear_listeners`, `reset_state`, readout.
- **`calibrate.rs`**: no-op stub with a doc comment.
- **`mod.rs`**: declare `config`; keep the rest.

## Non-goals (v1)

- No threading, no training, no calibration logic (stub only).
- No demo/example (`cargo run` smoke path is out of scope for this effort).
- Not byte-identical to `wave_reservoir`; the two engines coexist.
- No graded/analog input, no per-neuron saturation, no ragged (per-layer) sizes.

## Future (noted, not built here)

Calibration lowers per-layer thresholds toward a firing-rate target and sets each layer's saturation
to a margin above its threshold band (maintaining `saturation ≥ max threshold`), which is what makes
higher layers fire and the net non-inert. Threading can parallelize over layers within a wave (the
deferred model already guarantees the needed independence). Training (reward-modulated per-neuron
thresholds) builds on top — see `reward-modulated-threshold-training-design.md`.
