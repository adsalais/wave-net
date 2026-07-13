# wave_driven — event-driven active-set engine (Phase 1: inference)

- **Date:** 2026-07-13
- **Status:** design approved; ready for an implementation plan
- **Scope:** Phase 1 only — an event-driven inference engine in a new, independent `wave_driven`
  module. Training, persistence, procedural (un-materialized) topology, and GPU kernels are explicit
  non-goals of this phase.

## Motivation

`wave_bitnet`'s wave step costs `O(size²)` per layer per wave: the drain, decide, leak, and
adapt-decay passes sweep **every** neuron, whether or not it is doing anything. Only the *generate*
step is already activity-scaled (only firers scatter). In the network's normal operating regime —
sub-critical deep stacks, sparse recurrence, side-car scratchpads — most neurons are at rest most of
the time, so the size-bound sweeps dominate and waste work. This is the blocker recorded in the
project notes: *the single-threaded integer engine is too slow to run scaling sweeps multi-seed at
size ≥ 64.*

**Goal:** an engine whose per-wave cost scales with **activity** (spikes + deliveries) instead of
**size**, so large layers with sparse activity become affordable — and whose structure is a clean
fit for later GPU parallelization.

## The one idea

> Keep a **frontier** (worklist) of non-quiescent neurons per layer. A wave processes only the
> frontier, scatters deliveries into a **sparse accumulator**, and rebuilds the next frontier from
> (delivery targets ∪ neurons with residual state ∪ injections). Adaptation is never in the
> frontier test — it is decayed **lazily, in closed form, anchored to the last fire**. Cost scales
> with activity, not size.

A neuron is **quiescent** (safely skippable) when `potential == 0 && cooldown == 0` and it received
no delivery this wave. Each of those is a short-lived, absorbing-at-zero state: a zero potential
leaks to nothing, a zero cooldown stays zero, and no delivery means no new drive. Processing such a
neuron is a provable no-op — which is exactly why the sparse engine can equal a dense one bit-for-bit
(see Validation).

## Redefined dynamics — lazy, fire-anchored adaptation

This is the **one** dynamics change from `wave_bitnet` (permitted: `wave_driven` is a free
redefinition, not a bit-exact twin), and it is what makes "cost scales with activity" actually hold.

`wave_bitnet` decays adaptation every wave with `adapt -= adapt >> adapt_decay`. Its time constant is
up to thousands of waves, so if "adaptation still decaying" kept a neuron in the frontier, every
neuron that ever fired would stay awake for thousands of waves and the whole optimization would
evaporate. The escape: adaptation only affects behavior at **decide** time, so it need not be updated
until the neuron is next about to decide. We store adaptation as **its value at the last fire** and
reconstruct the current value on demand.

**State (per neuron):** `adapt_ref: i32` (Q`ADAPT_SHIFT`, the adaptation value at the last fire) and
`fire_wave: u32` (the wave that fire happened on). Fresh/never-fired: `adapt_ref = 0`, `fire_wave = 0`.

**Decay ratio:** `ρ = 1 − 2^(−adapt_decay)` per wave (same knob as `wave_bitnet`, now a true
geometric multiply rather than a right-shift — so it reaches exactly 0 at the horizon, with no
shift-based freeze floor).

**Closed-form current value at wave `w`:**

```
adapt(w) = ( adapt_ref · POW[min(w − fire_wave, HORIZON)] ) >> FRAC          // i64 intermediate
POW[k]   = round( ρ^k · 2^FRAC ),   POW[0] = 2^FRAC,   POW[k ≥ HORIZON] = 0
```

**On fire at wave `w`:** `a = adapt(w); adapt_ref = min(a + (adapt_bump << ADAPT_SHIFT), ADAPT_MAX);
fire_wave = w`. **Effective threshold at decide:** `eff = threshold + (adapt(w) >> ADAPT_SHIFT)`.

**Why this is exact and lazy at once — path independence.** `adapt(w)` is a pure closed-form
function of `(adapt_ref, fire_wave)`, and those change *only on fire*. Between fires the value is a
single jump from the last fire, so catching up a gap of 15 in one step equals catching it up as
5 then 10. A neuron may nap for 10,000 waves and wake with the correct adaptation in **one multiply**
— and, crucially, a dense engine that recomputes `adapt(w)` every wave and a sparse engine that
recomputes it only on wake produce the **identical** value, because both evaluate the same single-jump
formula from the same anchor. Adaptation therefore appears in **no** quiescence predicate.

**Constants (recommended):** `ADAPT_SHIFT = 12`, `ADAPT_MAX = i16::MAX << 12`, `FRAC = 30` (so
`adapt_ref ≤ 2^27` times `POW ≤ 2^30` stays within `i64`, then `>> 30`). `HORIZON` is computed per
`adapt_decay` at layer build as the least `k` with `ρ^k · ADAPT_MAX < 1`, capped at `1 << 16` to
bound the table; beyond the cap `adapt(w) = 0` (negligible for the usual `adapt_decay ≤ 8`). The
`POW` table is built once per distinct `adapt_decay`. *(Alternative, noted for large τ / GPU:
exponentiation-by-squaring of `ρ` in fixed point computes `ρ^gap` in `O(log gap)` with no table.)*

`Config::validate` drops `wave_bitnet`'s `adapt_decay ≤ ADAPT_SHIFT` dead-zone bound — the geometric
multiply has no dead zone. All other integer ops (drain clamp, decide compare, floored-shift leak,
generate word-scan) are **copied unchanged** from `wave_bitnet`.

## Data structures

**Per-layer neuron state (Structure-of-Arrays, dense — `O(size²)` *memory*; never swept):**

```
potential : i16                  // rest 0; drain folds in i32 then clamps to i16 (sole overflow guard)
cooldown  : u8                   // refractory counter
threshold : i16                  // per-neuron baseline
adapt_ref : i32  (Q ADAPT_SHIFT) // adaptation at last fire
fire_wave : u32                  // wave of last fire
```

**Topology (copied verbatim from `wave_bitnet` — already activity-friendly):** per-neuron occupancy
bitset (`occ`), the per-level offset LUTs (`offsets`, `off_flat`), and the 2-bit packed weight codes
(`codes`, `0b00`→0 / `0b01`→+1 / `0b11`→−1). Weights are the procedural `±1/0` init (a fresh
seed-derived random projection); Phase 1 has no training shadow.

**Sparse work structures (per layer):**

```
Frontier { list: Vec<u32>, mark: bitset[size²/64] }   // double-buffered: current + next
acc_val  : i32[size²]                                 // incoming deliveries for next wave
```

`Frontier` splits two jobs: `list` gives ordered, cache-friendly iteration; `mark` gives O(1)
"already queued?" so an insert is a **test-and-set** and no neuron is ever queued twice:

```
push(t):  word = t>>6; bit = t&63
          if mark[word] & (1<<bit) == 0 { mark[word] |= 1<<bit; list.push(t) }
```

The `mark` bitset is what prevents a double-drain when two firers hit the same target, or a target is
also a carryover. It is cleared by **walking `list`** (O(activity)), never by zeroing `size²`. This
same primitive is the GPU unique-frontier append (`atomicOr` on `mark` + `atomicAdd` on a counter),
which is why it is chosen over a `HashSet` (whose iteration order is also non-deterministic — a
non-starter here).

`acc_val` is stored dense but only ever touched through the frontier: it is filled by generate and
**drained-and-zeroed inline** when its neuron is processed. Because generate adds *every* delivery
target to the next frontier, the frontier is a superset of the touched targets, so every nonzero
`acc_val` entry is guaranteed to be cleared during drain — no separate clear pass.

## The wave algorithm

**Per neuron `i` in `frontier[z]` (normal computational layer `1..L−1`):**

```
drain    v = potential[i] as i32 + acc_val[i];  potential[i] = clamp_i16(v);  acc_val[i] = 0
decide   a   = adapt(w, adapt_ref[i], fire_wave[i])          // lazy closed form
         c   = cooldown[i].saturating_sub(1)
         eff = threshold[i] + (a >> ADAPT_SHIFT)
         if c == 0 && potential[i] >= eff:                   // FIRE
             potential[i] = 0;  cooldown[i] = cooldown_base
             adapt_ref[i] = min(a + (adapt_bump << ADAPT_SHIFT), ADAPT_MAX);  fire_wave[i] = w
             record i in fired[z]
         else:
             cooldown[i] = c
leak     d = (potential[i]>>la) + (potential[i]>>lb)
         potential[i] -= if potential[i] > 0 { d.max(1) } else { d }         // copied from wave_bitnet
carry    if potential[i] != 0 || cooldown[i] != 0 { frontier_next[z].push(i) }   // NOT adapt
```

**Generate (firers only)** — the exact `wave_bitnet` word-scan (the single documented `unsafe`
loop): for each firer, scan its occupancy bitset (`trailing_zeros`), decode each wired cell to its
target `t` in layer `tz` (interior fast path / wrapping edge path), and for each target:

```
acc_val_next[tz][t] += weight ;  frontier_next[tz].push(t)
```

Deliveries are deferred **one hop**: they land in the *next* wave's accumulator (same semantics as
`wave_bitnet`, on which the multi-wave read/train rule depends).

**Wave orchestration — `Network::wave(input)`:**

1. `w = wave_id; wave_id += 1`.
2. **Inject (L0):** for each `a in input`, `potential[a] = i16::MAX; cooldown[a] = 0;
   frontier[0].push(a)` — so injected sites are decided this wave and fire.
3. For `z in 0..L`: run `process_layer_sparse(z)` over `frontier[z]`, draining `acc_val[z]`, writing
   `acc_val_next[·]` and `frontier_next[·]`.
4. **Wave end**, per layer: empty the consumed `frontier[z]` (walk its list, clear its marks), swap
   `frontier ↔ frontier_next`, and swap `acc_val ↔ acc_val_next`. The swapped-in `acc_val` buffer is
   already all-zero (inline drain guarantees it), so it is clean for reuse.

Every step is `O(|frontier| + |deliveries| + |firers|·fan-in)` — i.e. `O(activity)`. No loop runs
over `0..size²`.

## Layer types and quiescence

| Layer | Behavior | In the frontier (non-quiescent) when |
|---|---|---|
| **normal (1..L−1)** | drain → decide/fire → leak; lazy adapt | `potential ≠ 0` ‖ `cooldown ≠ 0` ‖ received a delivery |
| **L0 transducer** | `threshold = i16::MAX`, `adapt_bump = 0`; fires only on injection | injected this wave ‖ `potential ≠ 0` (level −1 feedback residue leaking back down) |
| **readout (last)** | drain-only integrator; never fires, never leaks | received a delivery (only) — never carries itself over |

The readout case is why quiescence is *per layer type*: a readout neuron holds a nonzero integrated
potential forever, but its state does not change without input, so it is quiescent whenever it has no
delivery. It is added to the frontier only on the wave a delivery arrives.

## Determinism

Results are a pure function of `(seed, config, input)`. The frontier is iterated in insertion order;
insertion order is itself deterministic (carryover in frontier-iteration order, delivery targets in
firer × wired-cell order), and the scatter-add into `acc_val` is integer and commutative. Two builds
run identically. Recorded `fired` lists are in frontier order; a sorted view is available where a
canonical order is wanted. (On a future GPU the neuron *states* stay deterministic — per-neuron
independent updates plus commutative integer adds — while frontier *list order* is not; sort if
ordered records are ever required.)

## GPU-friendliness (honored as a design constraint; no kernels this phase)

The structure maps 1:1 onto the standard frontier/BFS GPU pattern: one thread per frontier neuron;
`acc_val += weight` is an `atomicAdd`; `frontier_next.push` is `atomicOr(mark) + atomicAdd(counter)`
(stream compaction); adaptation is a single per-thread multiply with no global sweep and no per-wave
reduction. SoA layout and branch-light per-neuron kernels keep occupancy scaling with the active set.
Nothing in the CPU design blocks this — that is the point of choosing the bitset+worklist form over a
hash set and of the fire-anchored (sweep-free) adaptation.

## Module layout and public API

```
src/wave_driven/                 (independent; copies from wave_bitnet, no dependency on it)
  mod.rs        wiring
  config.rs     Config / LayerConfig / validate            (copied; adapt_decay dead-zone bound dropped)
  synapse.rs    hash + topology helpers                     (copied verbatim)
  neurons.rs    Layer SoA state + occupancy bitset + codes + adapt_ref/fire_wave + POW decay table
  frontier.rs   Frontier (Vec + mark bitset) + sparse delivery accumulator
  wave.rs       process_layer_sparse (frontier)  +  process_layer_dense (the oracle)
  network.rs    Network: layer stack, wave(), injection-into-frontier, layer-type dispatch, double buffers
```

**API** (mirrors `wave_bitnet` so the harness feels familiar): `Network::new(Config)` /
`new_with_readout(Config)` → `wave(&[u32])` → introspection (`with_layer`, per-layer fired /
potential). Build-from-seed only; weights are the procedural `±1/0` init.

## Validation plan

The engine ships only when all of the following hold:

1. **`sparse ≡ dense` (primary oracle, self-contained).** `process_layer_dense` runs the *identical*
   per-neuron update over *all* neurons every wave. Over a random-L0-drive sequence, assert full
   state-array equality (`potential`, `cooldown`, `adapt_ref`, `fire_wave`, `codes`) each wave, and
   equal fired sets. This proves the frontier never wrongly drops a neuron whose processing would not
   have been a no-op. Fire-anchored adaptation makes dense and sparse compute bit-identical values.
2. **`adapt_bump == 0` ⟹ `wave_driven ≡ wave_bitnet`, bit-for-bit.** With adaptation off, the one
   redefined dynamic vanishes, so a `wave_driven` net and a `wave_bitnet` net built from the same
   `Config` must produce identical potentials and identical fired sets over the same input. A free
   cross-engine check of drain / decide / leak / generate and of the copied topology + weight init.
3. **Determinism.** Two builds from the same `(seed, config, input)` are identical.
4. **Throughput bench + profiling harness** (the payoff), mirroring `benches/throughput_bitnet.rs`
   and `examples/profile_bitnet.rs`: waves/s as a function of L0 drive fraction and of `size`,
   demonstrating that cost tracks activity, plus the sparse-vs-dense **crossover** (the activity level
   above which the dense sweep wins). Sweep multiple sizes and seeds; report worst + mean.

Tests are inline `#[cfg(test)]` per module (TDD where practical); the throughput/crossover experiments
are `#[ignore]`d (run in `--release`). `cargo build` stays warning-free; `cargo test` stays green.

## Phasing

- **Phase 1 (this spec):** event-driven inference engine + the four validation items above.
- **Phase 2 (separate spec, after Phase 1 is proven):** activity-scaled training — port the f32
  shadow + multi-layer-DFA credit, but with **online eligibility maintained on the frontier** instead
  of the current offline `O(size²·waves)` post-hoc pass, so training also scales with activity.
- **Phase 3 (future, no implementation):** GPU kernels. Honored now only as a structural constraint.

## Non-goals (Phase 1)

Training / eligibility / f32 shadow / decide snapshots; persistence (`.wbm`/`.wbr`); procedural
(un-materialized) topology or weights; any GPU kernel. Memory stays `O(size²)` for state and
`O(size²·fan-in)` for materialized topology — this is a **compute**-scaling change, not a
sparse-memory rewrite. (Un-materializing structure/untrained-weights into the seed, with only trained
weight *deltas* stored, is a real and compatible future axis — but out of scope here.)

## Risks and open questions

- **Activity crossover.** When activity approaches saturation, the frontier ≈ all neurons and the
  bitset/worklist bookkeeping is pure overhead versus the dense sweep. Validation item 4 measures the
  crossover; the engine is expected to win in the sparse regime that is this network's norm, and to
  lose slightly near saturation. Acceptable — and worth stating in the results.
- **`HORIZON` cap vs. very slow adaptation.** A `1<<16` table cap truncates adaptation to 0 beyond
  ~65k waves; for the usual `adapt_decay ≤ 8` the true horizon is far below that. If a config wants a
  very long τ, switch that layer to the exponentiation-by-squaring path (no cap).
- **Memory of dense state at very large size.** `O(size²)` state arrays + `O(size²·fan-in)` topology
  are unchanged from `wave_bitnet`; genuinely huge single layers still hit that wall. That is the
  un-materialization axis above, deliberately deferred.

## Appendix — copied vs. new

- **Copied ~verbatim:** `synapse.rs` (all helpers); `config.rs` (`Config`/`LayerConfig`/`validate`,
  minus the `adapt_decay` dead-zone bound); the topology/layout from `neurons.rs` (`DerivedLayout`,
  `occ`, `offsets`, `off_flat`, `codes`, `for_wired`, `decode`, `weight_at`); the drain / decide /
  leak integer ops and the generate word-scan (with its single documented `unsafe`) from `wave.rs`.
- **New:** `frontier.rs` (the `Frontier` + sparse accumulator); fire-anchored lazy adaptation
  (`adapt_ref` / `fire_wave` / `POW`); `process_layer_sparse` and `process_layer_dense`; the
  `Network` orchestration with double-buffered frontier/accumulator and injection-into-frontier.
- **Dropped for Phase 1:** `TrainState` / f32 shadow / decide snapshots; `persist.rs`;
  `multilayer_dfa.rs`; `eprop_update_synaptic`.
