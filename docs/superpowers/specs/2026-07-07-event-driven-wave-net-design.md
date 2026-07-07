# Event-driven wave_net engine — design

**Date:** 2026-07-07
**Status:** approved (brainstorming complete)

## Goal

`wave_net` diverges from `wave_reservoir` by getting its own engine: event-driven
spike propagation with lazily-updated neuron state, so per-wave cost scales with
activity (deliveries) instead of network size (full scans). `wave_reservoir` stays
frozen and serves as the bit-exact reference oracle. The deliverable includes a
rate-sweep benchmark that measures where event-driven actually beats dense — that
measurement is the point of the exercise.

Honest expectation, stated up front: at 5% firing with fan-in ≈ 11 the touched
fraction per wave is ≈ `1 − e^(−0.55)` ≈ 42%, so the win there is modest (~1.5–2×
at best). The win scales as `1/(rate × fan_in)`; the order-of-magnitude regime is
< 1% activity or large sheets with localized activity. The benchmark quantifies
the crossover for our topologies.

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Trainable per-neuron field | Becomes a **threshold offset** (`set_threshold_offsets`), not a drive bias. Dense drive fields stay possible on `wave_reservoir`. |
| Correctness contract | **Bit-identical spike streams** to `wave_reservoir` (given the threshold ≥ 8 invariant); the old engine is a differential-testing oracle. Semantic divergence comes later as deliberate, tested steps. |
| Threading | **Sequential first.** The threaded pipeline is a follow-up, validated against the sequential event engine. The design keeps it cheap (lock band shrinks to `[z−back, z]`). |
| Toolkit scope | **Engine + minimal harness** (input adapter for the differential tests). `calibrate`/`depth`/`readout` migrate later. |
| Approach | Event-driven lazy catch-up (option A), over fused-dense-scan (B) and active-block bitmask (C). |

## Architecture

New files, both in `src/wave_net/`:

- **`layer.rs`** — physics: `Layer` (all per-neuron state + the layer's event
  buffers), `Delivery`, `WakeupRing`, and the neuron-level operations
  (`advance_neuron`, `finalize_potential`, `leak_step`).
- **`engine.rs`** — orchestration: `EventNet` construction/validation, the wave
  loop (`process_layer` and its six named steps), housekeeping, input injection,
  listeners, public API, differential tests.

Reused from `wave_reservoir` (frozen, shared substrate): `config::IntConfig`,
`hash`, `index::Dims`, `wiring::for_each_layered`. Same config + same hash-derived
wiring is what makes bit-identical differential testing possible; the *engine* is
what diverges.

## Data structures

```rust
/// One layer: every per-neuron field, plus the event buffers the layer owns.
struct Layer {
    // per-neuron state (SoA)
    potential:    Vec<i16>,
    cooldown:     Vec<u8>,
    last_touched: Vec<u8>,   // epoch (wave % 256) this neuron was last brought current
    threshold:    Vec<i16>,  // hash-jittered base + trainable offset; min 8 enforced

    // event buffers
    inbox:   Vec<Delivery>,  // level ≤ 0 deliveries from the previous wave
    wakeups: WakeupRing,     // per-wave slots of neurons whose cooldown expires then
    touched: Vec<u32>,       // scratch: this wave's decide set
}

/// A wave climbing the stack: its id plus the forward deliveries it carries.
struct Wave {
    id: usize,                    // epoch = id % 256
    forward: Vec<Vec<Delivery>>,  // ring, one slot per forward level
}

/// Packed (local, sign): u32 with the sign in the top bit (4 bytes on the hot
/// path); named accessors (.local(), .sign()) keep call sites readable.
struct Delivery(u32);
```

`WakeupRing` is `refractory + 1` lists (refractory is a per-layer constant): firing
at wave `t` schedules a decide-check at `t + refractory`. In `Drop` mode wakeups
are unnecessary (potential is held at 0 while cooling) and may be skipped.

**Routing rule:** a spike's synapses targeting layers *above* the source ride
inside the `Wave` (consumed later in the same wave — the wavefront); synapses
targeting the source's layer *or below* go to the target layer's `inbox`
(consumed by the *next* wave). This matches the dense engine, where backward and
lateral deliveries land after the target layer's decide.

Memory: 6 bytes/neuron fixed (dense uses 5; `last_touched` is the +1), plus
O(events) buffers.

## The wave loop

```rust
fn process_layer(&self, z: usize, wave: &mut Wave, layer: &mut Layer) {
    self.absorb_inbox(layer, wave.id);       // 1. previous wave's backward deliveries
    self.land_wavefront(layer, wave, z);     // 2. this wave's forward deliveries
    self.inject_input(layer, wave, z);       // 3. external sparse input events
    self.wake_refractory(layer, wave.id);    // 4. cooldown expiries due this wave
    self.decide_and_fire(layer, wave.id, z); // 5. threshold test over the touched set
    self.scatter_spikes(layer, wave, z);     // 6. firers → wave.forward / target inboxes
}
```

### `advance_neuron(layer, n, to_wave)` — the single lazy primitive

Contract: bring neuron `n` current through "mid-wave `to_wave`, pre-decide".
`last_touched = w` means: `potential` includes wave `w`'s leak and all wave-`w`
deliveries seen so far; wave `w`'s finalize (clamp + Drop-zero) is **deferred** to
the next advance. The replay per missed wave is: `finalize_potential` for the
last-touched wave (first iteration only — later ones are provably identity for an
untouched neuron), then one `leak_step` and one cooldown decrement per wave.
**Early-exit at the leak fixed point**: when `leak(p) == p` and `cooldown == 0`
(and not Drop-cooling), the remaining replays are identity — this bounds replay at
~60 steps worst case and is where the savings over dense live. The first advance
of a neuron in a wave pushes it onto `layer.touched`.

### Inbox ordering (the exactness-critical subtlety)

Inbox deliveries belong to the **previous** wave. `absorb_inbox` is two passes:

1. For each delivery: `advance_neuron(n, t−1)`, then `potential += sign`
   (unclamped — dense accumulates all of a wave's backward deliveries, then
   clamps once at finalize). In `Drop` mode, skip the add if the neuron is
   cooling (dense zeroes it at finalize anyway).
2. For each delivery: `advance_neuron(n, t)`; if it actually advanced, push onto
   `touched` (idempotence makes the collection unique). This applies the deferred
   finalize (the single clamp) and wave `t`'s leak *after* the deliveries — the
   same order the dense engine produces.

Getting this wrong (leaking before adding the previous wave's backward
deliveries) diverges from the oracle only near saturation — exactly what the
differential tests exist to catch.

### Decide completeness (why touched-only decide is exact, not approximate)

A neuron can fire at wave `t` only if:
- it received a forward delivery or input this wave → in `touched` (step 2/3);
- backward deliveries raised it last wave → `absorb_inbox` put it in `touched`;
- its cooldown expired at `t` with potential still ≥ threshold (CarryOver) →
  the wakeup ring scheduled it at fire time.

Any other neuron has only leaked since a decide it already failed. The shift-leak
`p − (p>>a) − (p>>b)` has fixed points: positive `p ≤ 7` stalls, negative `p`
climbs to +1 and stalls — so leak can never raise a potential past 7.
**Invariant: every effective threshold ≥ 8** (base + jitter + offset), checked at
construction and in `set_threshold_offsets`. With it, the equivalence is a proof.
Duplicate entries in `touched` are harmless (decide is idempotent: a firer resets
and enters cooldown), but the two-pass inbox keeps the list unique anyway.

### Housekeeping wave

Epoch is `wave % 256`; every 255 waves a maintenance wave walks the stack like
any other wave and `advance_neuron`s every neuron, so `(epoch_now −
last_touched) mod 256` can never alias. It fires nothing (advancing without an
event cannot cross a threshold), costs O(N) once per 255 waves, and needs no
special synchronization.

### Threading (follow-up, designed-for now)

Everything a wave mutates is wave-private (`forward`) or owned by a layer within
`[z−back, z]` (`inbox` reaches down at most `back` layers). Leak is lazy, so the
dense engine's leak-front/leading-edge machinery — and its forward locks —
disappear. The future pipelined variant uses band `[z−back, z]`, giving *more*
waves in flight than `LayerNet`.

## Public API

```rust
impl EventNet {
    fn new(cfg: IntConfig) -> EventNet;   // validates all thresholds ≥ 8 (incl. jitter)
    fn run_stream(&mut self, waves: usize, input_fn: impl FnMut(usize, &mut Vec<InputEvent>));
    fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32])>);
    fn set_threshold_offsets(&mut self, layer: usize, offsets: &[i16]); // trainable field
    fn reset_state(&mut self);
    fn potential_global(&mut self, idx: usize) -> i16; // advances the neuron, then reads
}
```

- `InputEvent = (layer, local, amount: i16)` — sparse by construction. Minimal
  harness: a `BipolarInput → InputEvent` adapter in `stream.rs` (sites are
  already a sparse list).
- Firer lists are emitted in **touched order, not index order** (documented;
  readout/depth consume them as sets; tests sort before comparing).
- The trainable field is a per-layer `Vec<i16>` threshold offset, set between
  hill-climb evaluations (which already `reset_state`). Offsets that would push
  any effective threshold below 8 are rejected.

## Testing

1. **Spike-stream equivalence (differential oracle):** same config, listeners on
   every layer of `EventNet` and `LayerNet`, identical bit-stream input; assert
   sorted firer lists identical per (wave, layer). Configs: demo topology,
   backward-heavy, level-+2 skips, both refractory modes, near-saturation drive.
2. **Potential equivalence:** after a run, `potential_global` matches the dense
   engine for every neuron (catches deferred-finalize bugs spike streams miss).
3. **Laziness units:** `advance_neuron` replay vs step-by-step reference over
   random (state, gap) pairs; epoch wraparound across the 255-wave housekeeping
   boundary; wakeup-ring firing with no same-wave input; fixed-point early-exit
   exactness.
4. **Validation:** configs that could yield an effective threshold < 8 (base +
   negative jitter or offset) are rejected at construction / setter time.

## Benchmark — the study deliverable

`examples/event_vs_dense.rs`: one config family calibrated to target rates
(~10%, 5%, 2%, 1%, 0.5%, 0.1%); run both engines on the same stream; report
waves/sec, speedup, mean touched fraction, mean replay length — and assert
spike-count equality, so the benchmark doubles as a large differential test.
Output: a table answering *where is the crossover for these topologies*.

## Error handling

Existing house style: `validate() → Result` at construction; asserts with
messages on API misuse (wrong lengths, out-of-range layer); no panics reachable
from a valid config.

## Out of scope (explicitly)

- Threaded pipelined event engine (follow-up; design keeps it cheap).
- Migrating `calibrate`/`depth`/`readout` to `EventNet` (after the engine settles).
- Dense per-wave drive fields on `EventNet` (use `wave_reservoir` for those).
- Any semantic change vs the reference engine (comes later, deliberately).
