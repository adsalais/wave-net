# wave_bitnet — a memory-lean, ternary-native engine fork

**Date:** 2026-07-12
**Status:** design approved, ready for implementation plan

## Motivation

This session established that **pure ±1/0 ternary weights train as well as (and, at depth, better
than) int8** in the multi-layer-DFA engine, once ALIF is dialed to `adapt_bump=5`: pure ternary does
both feed-forward and recurrence, and with a half-count +2 skip it reaches depth 24+. It is time to
"bank" that result with an engine that stores weights the way BitNet wants them — a few bits each —
instead of the current int8 + f32-shadow layout, and that drops the per-wave procedural target
generation in favour of a materialized topology.

`wave_net` is the active R&D engine and stays a **frozen reference** from here on (like
`wave_state_machine`). `wave_bitnet` is a **clean fork** — code duplication is acceptable and
intended.

## Locked decisions

1. **Scope — trains, keeps the f32 shadow.** `wave_bitnet` trains ternary nets with the existing
   DFA / e-prop rule and keeps a per-synapse `f32` shadow during training (proven update rule, no new
   math to validate). The compact bitsets are the stored/delivered form; **the memory win lands at
   rest / inference** (drop the shadow → ship just the bitsets). Training still pays the 4-byte shadow.
2. **Weight scheme — pure ±1/0 only.** No scaled ±g, no int8. A weight is 2 bits: a `nonzero` mask bit
   and a `sign` bit. The tunable prune threshold `t` carries over (default 0.5, sweet spot ~0.7).
3. **First cut — core engine + one smoke benchmark.** Port the five engine files plus the training
   rule; validate with unit tests and a single FF depth-8 pure-ternary benchmark that confirms it
   trains AND measures the memory/speed win. The full harness/benchmark suite is ported later.
4. **Topology representation — Approach A (neighborhood occupancy bitset).** Per source neuron per
   topology level, a `(2r+1)²`-bit occupancy bitset; at startup draw `count` **distinct** cells and set
   them. Weights are rank-parallel packed bits + an `f32` shadow. Forward and update iterate the bitset
   — **no hashing after startup.**
5. **`Config::validate` enforces `count ≤ (2r+1)²`** — a startup check that returns a descriptive
   `Err` (via `Network::new`) before any layer is built, since a per-cell bitset caps fan-in at the
   neighborhood size.
6. **Change-bitset — out of the first cut.** Deferred (see Deferred, below).

## Module layout

`src/wave_bitnet/` — a fourth top-level module (peer of `wave_net` / `wave_state_machine`), fully
self-contained. Files mirror `wave_net`:

- `mod.rs`
- `config.rs` — `Config`, `LayerConfig` (topology, leak, cooldown_base, inhibitor_ratio,
  threshold_jitter, baseline_init, `adapt_bump` default **5**, adapt_decay). **No `WeightQuant`** — pure
  ternary is baked in. `validate()` enforces `count ≤ (2r+1)²` per level.
- `synapse.rs` — hash helpers copied verbatim (`mix`, `key`, `map_range`, `local_of`, `xy_of`, `wrap`);
  a `sample_distinct_cells` routine (startup fill); a `decode_cell` routine (cell index → target local).
  `TopologyLevel { level, radius, count }` and `Synapse { target, weight: i8 }` unchanged.
- `neurons.rs` — `Layer` with the bitset representation (below) + neuron/eligibility state.
- `network.rs` — `Network::new` / `new_with_readout` (build layers, fill topology at startup, force L0
  transducer, validate), `wave()`, and the update primitive.
- `wave.rs` — `process_layer` (forward pass; bitset scan instead of `generate_into`).
- `multilayer_dfa.rs` — the ported training engine (`Edge`, `TrialRecords`, `EligParams`,
  `temporal_eligibility`, `multilayer_dfa_step`), adapted to decode targets from the occupancy bitset.

`wave_net` is not touched.

## Core data structures

**Key invariant:** because we sample exactly `count` *distinct* cells per level, the number of wired
synapses per neuron is still `total_slots = Σ count` — **identical to `wave_net`**. Only the *encoding*
(2-bit packed vs i8) and the *indexing* (spatial cell-rank vs hash-slot) change; the topology bitset is
purely additive.

Per `Layer`:

```
// neuron state (same as wave_net): potential, cooldown, adapt, threshold, inbox
//   + eligibility state: decide_potential, decide_eff, elig_pre, elig_post

// TOPOLOGY (static, filled at startup) — per level ℓ (radius rℓ, Nℓ = (2rℓ+1)² cells):
occupancy[ℓ]: bitset          // ls · Nℓ bits; neuron i's Nℓ-bit slice = which cells are wired
                              //   popcount of each slice == countℓ (distinct sample guarantees it)

// WEIGHTS (plastic) — length ls · total_slots, indexed [i·total_slots + slot_base(ℓ) + rank]:
w_nonzero: bitset             // 1 = ±1, 0 = pruned-to-0
w_sign:    bitset             // 1 = +, 0 = − (meaningful only where nonzero)
shadow:    Vec<f32>           // training accumulator; DROPPED for an at-rest / inference model

// config: leak, cooldown_base, topology (Vec<TopologyLevel>), total_slots, adapt_bump,
//   adapt_decay, readout, ternary_threshold (default 0.5)
```

`slot_base(ℓ) = Σ_{ℓ'<ℓ} countℓ'` (same cumulative-count layout as `wave_net`). The delivered weight
for the r-th wired synapse of level ℓ is `nonzero ? (sign ? +1 : −1) : 0` — **2 bits/synapse**; the
`f32` shadow is the only 4-byte cost, and it is training-only.

A minimal internal `bitset` helper (get/set/iterate-set-bits over a `Vec<u64>` or `Vec<u8>`) lives in
`neurons.rs` or a small `bits.rs`; std-only, no external crates.

## Startup fill & target decode (replaces procedural generation)

At `Network::new`, after `validate()`:

- **Fill topology.** For each neuron `i`, level `ℓ`: draw `countℓ` **distinct** cells from `0..Nℓ` via a
  **partial Fisher-Yates shuffle** of the `Nℓ` cell indices, seeded by the hash stream (`key`/`mix` over
  `i, ℓ`, one draw per swap), taking the first `countℓ`, and set them in `occupancy[ℓ]`. Fisher-Yates
  (not rejection sampling) is chosen because `count` can be a large fraction of `Nℓ` (e.g. c48 of N81),
  where rejection degrades; it is O(count), deterministic, and gives exactly `count` distinct cells with
  no collision loss — a deliberate, clean difference from `wave_net`'s collisions-allowed hashing.
- **Init weights.** Seed `shadow` (±1 sign from `inhibitor_ratio`, as `wave_net`) → pack into
  `w_nonzero` / `w_sign` via the repack routine.
- **L0 transducer.** Force L0 to `threshold = i16::MAX`, `adapt_bump = 0` (the input transducer;
  giving it adaptation collapses the net — established this session).

**Cell decode** (no hash, pure arithmetic), for a set cell `c` at level with radius `r`, source at
`(sx, sy)`:

```
dx = c % (2r+1) − r ;  dy = c / (2r+1) − r
target = local_of( wrap(sx, dx, size), wrap(sy, dy, size), size )
```

This is the entire replacement for `target_of` / `generate_into`'s per-wave hashing — done once at
startup, a bitset scan thereafter.

## Forward path (`wave.rs::process_layer`)

Neuron dynamics identical to `wave_net` (drain inbox → sum delivered ±1 into potential; inject L0;
decide/fire/ALIF/leak). **Only the generate step changes:**

```
for each firing neuron i, each level ℓ:
    for the r-th set bit (cell c) in occupancy[ℓ]'s slice for i:
        target = decode_cell(c, i, ℓ)                  // arithmetic, no hash
        w      = weight_at(i, slot_base(ℓ) + r)        // nonzero ? (sign ? +1 : -1) : 0
        if w != 0 { deliveries[target_layer].push(Synapse{target, w}) }   // pruned skipped
```

Pruned synapses (w=0) stay wired in `occupancy` (so pruning stays reversible, per this session's
recovery finding) but deliver nothing. Out-of-range target layers are dropped (same toroidal-in-depth
boundary). The `Synapse` still carries the ±1 weight; drain-and-sum at the target is unchanged.

## Training / update path (`multilayer_dfa.rs`, ported)

Keep the proven DFA / e-prop rule; port three pieces onto the bitset engine:

- **`temporal_eligibility`** — identical math (`e = Σₜ pretrace_i · ψ_target`, optional ALIF εᵃ),
  except the synapse's `target` is **decoded from `occupancy`** instead of `target_of`-hashed. Training
  loses the hash too.
- **`build_signal`** (readout + DFA + `rate_reg`) — unchanged (bench-owned harness).
- **update** (`eprop_update_synaptic`-equivalent) — `shadow[base+r] += −lr · signal[target] ·
  elig[base+r]` per wired synapse, then **repack the touched rows**: `γ = mean(|shadow|)` over the
  neuron's `total_slots`, and for each synapse write `nonzero = (|shadow|/γ ≥ t)`, `sign = shadow > 0`
  straight into `w_nonzero` / `w_sign`. This is `wave_net`'s `requantize_row` retargeted at bitsets.

The update stays shadow-based and therefore *behaviorally equivalent* to `multilayer_dfa`, so trained
accuracy should reproduce.

## Memory & speed (the payoff)

Per synapse (level with radius r, count c, Nℓ=(2r+1)²; **r4/c48** shown):

| | weight | shadow | topology (amortized) | total |
|---|---|---|---|---|
| wave_net | i8 = 1 B | f32 = 4 B | 0 (recomputed) | **5 B** |
| wave_bitnet, training | 2 bits | 4 B | N/c bits ≈ 1.7 b | **≈ 4.5 B** |
| wave_bitnet, at rest (no shadow) | 2 bits | — | ≈ 1.7 b | **≈ 0.46 B** |

Smoke net (FF depth-8, size 32, r4/c48 ≈ 344k synapses): wave_net ≈ 1.7 MB → training ≈ 1.5 MB →
**at-rest ≈ 0.16 MB (~11×)**.

The three real payoffs (memory during training is only ~11% less — the shadow dominates, accepted):

1. **At-rest / inference ~11× smaller** (drop the shadow, ship the bitsets).
2. **No per-wave hashing** — the speed win, directly relevant to the "single-threaded integer engine
   too slow at size ≥ 64" blocker.
3. **Clean ternary-native engine** — no `WeightQuant` branching.

## Validation plan

**Unit tests** (`wave_bitnet` `#[cfg(test)]`):

- `validate` rejects `count > (2r+1)²`; accepts valid configs.
- Startup fill sets exactly `count` bits per (neuron, level); deterministic (same seed → same bits).
- `decode_cell` maps cells to the correct toroidal targets; round-trips.
- Shadow → repack → unpack delivers the right ±1/0 (pack/unpack correctness).
- Forward determinism: same (seed, config, input) → identical spikes.
- One training step raises a pruned synapse's shadow / flips it on (mirrors the multilayer_dfa unit
  test).
- A small FF net trains a separable 2-class task above chance.

**Smoke benchmark** (`#[ignore]`d, `--release`): FF depth-8 pure ternary, r4/c48, adapt=5, 3 seeds →
must reach **~1000** (parity with wave_net's finding). Prints: per-synapse bytes (training + at-rest),
total footprint, and per-wave forward throughput vs `wave_net`.

**Success criteria:**

- Trains FF depth-8 to ~1000 (accuracy parity with `wave_net`).
- At-rest footprint ~10× smaller (quantified).
- Per-wave forward throughput ≥ `wave_net` (no-hash win; measured).

This is **accuracy** parity, not bit-identical output: the fork samples distinct cells (vs collisions)
and indexes weights by spatial rank, so it is a legitimately different net that should reach the same
result.

## Deferred (explicitly out of the first cut)

- **Change-bitset update transmission.** Because delivery reads `w_nonzero`/`w_sign` live each wave,
  there is no cache to invalidate, so a per-row weight-churn mask buys nothing yet. Revisit only if a
  delivery-side cache or churn logging/transmission is added.
- **Scaled ±g ternary, int8** — dropped by design.
- **Full harness/benchmark-suite port** (side-car, XOR, threshold sweeps, per-layer sparsity, etc.) —
  after the smoke benchmark proves the engine.
- **On-disk serialization of the at-rest bitset form** — the ~11× win is realized in RAM; a compact
  file format is a later concern.
- **A shared engine trait** over `wave_net` + `wave_bitnet` — the fork intentionally duplicates instead.

## Constraints (carried from AGENTS.md)

std-only, warning-free build, determinism (`pure function of (seed, config, input)`), one commit per
task, no `Co-Authored-By` trailer, branch first for non-trivial work.
