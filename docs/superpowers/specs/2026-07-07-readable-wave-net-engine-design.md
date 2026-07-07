# Readable wave_net engine (sequential + threaded) — design

**Date:** 2026-07-07
**Status:** approved (brainstorming complete)

## Goal

Give `wave_net` its own **readable, procedural** rendering of the dense wave engine
— both the sequential `wave()` and the threaded `run_stream()` paths of
`wave_reservoir::pipeline::LayerNet`. Same dynamics, **bit-identical** results
(including across thread counts), reorganized so a `Layer` owns all its per-neuron
fields and the per-neuron update reads as named procedural steps.
`wave_reservoir` stays frozen and is the reference oracle.

Hard constraints:
- **Bit-identical** to `LayerNet` — sequential vs `LayerNet::wave()`, and threaded
  vs the sequential engine at every thread count. Verified by differential tests,
  not by eye.
- **No performance traded for readability.** Every loop, fusion, and the entire
  concurrency discipline (band locking, entry-ordering, leak-front) are preserved
  1:1. Readability comes from naming and struct organization only — never from
  splitting a fused pass, adding a hot-path allocation, or weakening the locking.

## Non-goals (explicit)

- No change to the toolkit in this effort: `calibrate`/`depth`/`readout`/
  `field_training` keep running on `LayerNet`. This engine is a standalone readable
  twin. (Migrating the toolkit onto it can follow once it's proven.)
- No semantic change of any kind vs the reference engine.
- No event-driven / lazy-state machinery (that effort is shelved).

## Reuse (shared substrate — unchanged)

From `wave_reservoir`: `config::{IntConfig, IntLayer, IntLevel, RefractoryMode}`,
`hash::{key, mix, P_THRESHOLD}`, `index::Dims`, `wiring::for_each_layered`.
Identical config + identical hash-derived wiring is what lets the differential test
demand bit-identity.

## Data structures

### `Layer` — owns everything about a layer's neurons (the mutex-guarded state)

Decision (option A): all per-neuron fields **and** the per-layer dynamics scalars
live in `Layer`, so the step methods are self-contained. Only the synapse-wiring
params are pulled out (see below), because scatter forces it.

```rust
struct Layer {
    // per-neuron state
    potential: Vec<i16>,
    cooldown:  Vec<u8>,
    threshold: Vec<i16>,   // hash-jittered base, computed once at construction

    // per-layer dynamics scalars (read while mutating this same layer, one layer at a time)
    leak_a: u8,
    leak_b: u8,
    refractory: u8,
}
```

`threshold` is computed exactly as in `LayerNet::new`
(`mix(key(seed, global, 0, 0, P_THRESHOLD))`, masked jitter around `threshold_base`),
moved into `Layer` with the identical formula so values match bit-for-bit.

### `Wiring` — the read-only synapse-projection params (lock-free sidecar)

```rust
struct Wiring {
    topology:  Vec<IntLevel>,
    p_inh_q16: u32,
}
```

**Why this must stay out of `Layer`:** during scatter a worker mutably borrows every
band layer's `potential` at once (`pots: Vec<&mut [i16]>`) while reading the *source*
layer's `topology`/`p_inh`, and the source layer is always inside its own band
`[s-back, s+fwd]`. If wiring lived in `Layer`, that's a simultaneous `&mut
layers[s].potential` + `&layers[s].topology` — a borrow-checker conflict. Keeping
`Wiring` on the net and reading it via `&self` (never borrowed by `pots`) resolves
it. This is the same reason `LayerNet` split `LayerCfg` from `Layer`; we keep only
the *irreducible* part of that split.

### `Net`

```rust
pub struct Net {
    layers: Vec<Mutex<Layer>>,   // mutable state, one guard per layer
    wiring: Vec<Wiring>,         // read-only, lock-free via &self
    dims: Dims,
    seed: u64,
    ls: usize,                   // layer size = w*h
    l: usize,                    // layer count
    sat: i16,                    // saturation clamp
    drop: bool,                  // RefractoryMode::Drop
    fwd: usize,                  // global forward reach (leak-front lead)
    capacity: usize,             // max non-overlapping bands in flight = ceil(l/band_width)
    band: Vec<(usize, usize)>,           // per source: [s-back, s+fwd] clamped
    clamp_after: Vec<Vec<usize>>,        // layers finalized after each source
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}
```

`band`, `clamp_after`, `fwd`, `capacity`, `listeners` are constructed and behave
exactly as in `LayerNet` (same `level_range`, `last_source`, `div_ceil` math). The
concurrency-relevant fields keep `LayerNet`'s semantics verbatim.

### The read-only-under-mutex tradeoff (accepted)

`Mutex<Layer>` now guards read-only `threshold`/`leak_a`/`leak_b`/`refractory`,
relaxing `LayerNet`'s "a mutex should only guard what mutates." This is **cost-free
at runtime**: those fields are read only by the worker holding that layer's guard,
and bands never overlap, so there is no added contention and no extra
synchronization — the reads already happen under a guard the worker holds. The
relaxation buys self-contained, readable step methods. Documented in the module doc
so the choice is explicit.

## The per-neuron steps (each = exactly one loop from `LayerNet`)

Every method owns one loop that already exists in `leak_decay` / decide /
`finalize`. No pass added; no fused pass split.

| Method | Body (one pass) | Source in `pipeline.rs` |
|---|---|---|
| `Layer::refractory_decay(&mut self)` | `for c in &mut cooldown { *c = c.saturating_sub(1) }` | `leak_decay` pass 1 (`:177`) |
| `Layer::leak_potential(&mut self, drive: &[i16])` | `p = (p - (p>>leak_a) - (p>>leak_b)).saturating_add(drive[i])` — **leak + drive integration stay one fused pass** (comment states two passes would double memory traffic) | `leak_decay` pass 2 (`:182`) |
| `Layer::fire(&mut self, firers: &mut Vec<u32>)` | `if cooldown[i]==0 && potential[i]>=threshold[i] { potential[i]=0; cooldown[i]=refractory; firers.push(i) }` (disjoint-field borrow of potential/cooldown vs threshold) | decide (`:225`) |
| `scatter_from(pots, lo, wiring, seed, dims, firers)` (free fn) | `for_each_layered(src, seed, &wiring.topology, wiring.p_inh_q16, dims, |tl,tloc,inh| pots[tl-lo][tloc] += if inh {-1} else {1})` | apply (`:246`) |
| `Layer::settle(&mut self, sat, drop)` | `p = p.clamp(-sat, sat); if drop && cooldown[i]>0 { p = 0 }` | `finalize` (`:189`) |

One-line methods are `#[inline]` so codegen matches the inlined source loops.

## Shared orchestrator — the dense wave window, structure unchanged

The subtle, correctness-critical sequence (leak the leading edge once, decide,
scatter, finalize the trailing edge) keeps `LayerNet::process_source`'s exact
structure and the leak-once cursor; only names and comments improve. It operates on
held guards, so it is shared verbatim by the sequential and threaded paths.

```rust
fn process_source(&self, s, wave_id, lo,
                  guards: &mut [MutexGuard<Layer>], leaked_through: &mut usize,
                  drive: &[i16], firers: &mut Vec<u32>) {
    // 1. leak+drive each layer as it first enters this source's window (each leaked once/wave)
    let lead = (s + self.fwd).min(self.l - 1);
    while *leaked_through <= lead {
        let g = &mut guards[*leaked_through - lo];
        g.refractory_decay();
        g.leak_potential(&drive[*leaked_through*self.ls .. *leaked_through*self.ls + self.ls]);
        *leaked_through += 1;
    }
    // 2. decide layer s on its leaked + already-delivered snapshot
    firers.clear();
    guards[s - lo].fire(firers);
    if let Some(listener) = &self.listeners[s] { listener(wave_id, firers); }
    // 3. scatter s's spikes across the band (wiring read via &self; pots borrows the guards)
    let mut pots: Vec<&mut [i16]> = guards.iter_mut().map(|g| g.potential.as_mut_slice()).collect();
    scatter_from(&mut pots, lo, &self.wiring[s], self.seed, &self.dims, firers);
    // 4. finalize layers whose last writer is s
    for &j in &self.clamp_after[s] { guards[j - lo].settle(self.sat, self.drop); }
}
```

Why the leak-once cursor stays: source-layer bands overlap, so "leak my whole band
each source" would leak a layer several times per wave — wrong result *and* wasted
work. `leaked_through` preserves "leak each layer exactly once as the front passes."

## Sequential and threaded entry points

Both mirror `LayerNet` and share `process_source`, so there is a single
orchestration path.

- **`wave(&self, drive)`** — locks each band (uncontended) bottom-to-top and calls
  `process_source` with `wave_id = 0`. Same uncontended-lock structure as
  `LayerNet::wave()`, so no regression relative to the reference sequential path.
- **`run_stream(&self, waves, threads, drive_fn)`** — the threaded pipeline,
  preserved verbatim from `LayerNet`: `threads.min(capacity)`; per-worker wave-id
  pull via `AtomicUsize`; the `entry` `AtomicUsize` serializing first-band
  acquisition in wave order; hand-over-hand guard acquisition (increasing layer
  order → deadlock-free) extending to `hi` then releasing below `lo`; per-worker
  reused `drive_buf` re-zeroed each wave. Every comment explaining *why* each
  synchronization step exists is carried over.
- **`run(&self, drive, waves, threads)`** — constant-drive wrapper over
  `run_stream`.

Listeners: `on_layer(&mut self, layer, Box<dyn Fn(usize,&[u32]) + Send + Sync>)`,
emitted at decide under the layer lock (serialized, wave-ordered) — identical to
`LayerNet`, which is what makes the listener stream deterministic across thread
counts. Firer lists are ascending local index (the `fire` loop scans `0..ls`), so
they match `LayerNet` exactly with no sorting.

## Public API (mirrors `LayerNet`)

```rust
impl Net {
    pub fn new(cfg: IntConfig) -> Net;
    pub fn wave(&self, drive: &[i16]);
    pub fn run(&self, drive: &[i16], waves: usize, threads: usize);
    pub fn run_stream(&self, waves: usize, threads: usize, drive_fn: impl Fn(usize, &mut Vec<i16>) + Sync);
    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>);
    pub fn clear_listeners(&mut self);
    pub fn reset_state(&self);
    pub fn potential_global(&self, idx: usize) -> i16;
    pub fn n_total(&self) -> usize;
    pub fn pipeline_capacity(&self) -> usize;
}
```

## Verification — bit-identity is the acceptance criterion

Port `LayerNet`'s own test suite and add cross-engine differential checks:

1. **Sequential potential equivalence.** For several configs (demo; backward-heavy
   topology; level-+2 skips; both `RefractoryMode`s; near-saturation drive) run
   `Net::wave` and `LayerNet::wave` on identical drives for N waves; assert
   `potential_global(i)` equal for every neuron after the run.
2. **Threaded == sequential (this engine).** `run` at threads `[1,2,4,8,16]`
   matches `Net`'s own sequential `wave()` golden bit-for-bit — the port of
   `threaded_matches_sequential_all_thread_counts`, plus the deep-stack stress
   (`threaded_deterministic_under_stress`) and oversubscribed-thread clamp.
3. **Firer-stream equivalence + determinism.** Listener streams identical to
   `LayerNet`'s on the same run, and identical across thread counts (port of
   `listener_stream_deterministic_across_threads`).
4. **Golden trajectory.** Reproduce the demo anchor: top-layer per-wave counts
   `[138, 77, 158, 85]`, checksum `60071` (`top_layer_trajectory_golden`).
5. **Unit tests** ported: refractory blocks refiring; leak decays potential;
   forward delivery reaches the next layer; `reset_state` zeros everything;
   `run_stream` drive-buffer-zeroed-each-wave; wrong-length / zero-thread panics;
   `pipeline_capacity` counts.

## Performance argument

Preserved from `LayerNet`, so nothing to regress:
- same passes per wave, leak+drive fused, leak-once `leaked_through` cursor;
- `firers` and the `pots` band-slice transient built exactly as today;
- the full concurrency discipline (locking, entry-order, capacity clamp) verbatim;
- read-only fields folded into `Layer` are read only under an already-held guard
  (zero added contention).

A timing sanity check (`Net::run` vs `LayerNet::run` on the demo at a large `l`)
can confirm parity, but no regression is expected by construction.

## Error handling

`IntConfig::validate()` at construction (reuse `wave_reservoir`'s validator);
`assert!` with messages on API misuse (wrong drive length, `threads >= 1`). No panic
reachable from a valid config + valid drive.

## Files touched

- **New:** `src/wave_net/engine.rs` — `Net` + `Layer` + `Wiring` + `scatter_from` + tests.
- **Edit:** `src/wave_net/mod.rs` — add `pub mod engine;` and a module-doc line.
- Nothing in `wave_reservoir` changes.
