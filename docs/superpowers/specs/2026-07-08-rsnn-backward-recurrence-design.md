# Backward recurrence (level −1/−2) + width — design

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** try recurrence via **backward** connections (`level −1`, `level −2`) plus **wider layers**,
trained by the temporal eligibility + multi-layer DFA credit, on **temporal XOR** (the FF-fails variant).
Dual purpose: (a) does backward recurrence + width beat feed-forward where level-0 lateral couldn't; and
(b) a **diagnostic** — if this stronger topology + capacity *still* nulls, it isolates the spike-time
pseudo-derivative `ψ` as the true credit blocker (the next lever). All in `bench::rsnn`; no engine change.

## Why this (and what it tests)

Level-0 lateral recurrence nulled on temporal XOR. Two suspects remained: **(1)** the topology/capacity
(level-0 within-layer loops are weak; 64 neurons too few), and **(2)** the credit rule (`ψ = fired_j` gives
no signal during silent gaps). This experiment maximizes **(1)**'s side — cross-layer backward loops
(`Lz → Lz−1 → Lz`, and `Lz → Lz−2`) are a stronger recurrent structure than lateral, and width (size 16,
which fixed multi-layer reliability) gives ample capacity. **If it works**, recurrence earns its keep. **If
it still nulls**, we've controlled out topology and capacity, leaving `ψ` as the indicated blocker — which
is precisely the point of running it.

## Topology & size

- Each computational layer's topology: `level +1` (feed-forward, `up_count`/`up_radius`) **plus**, when
  `back_count > 0`, `level −1` and `level −2` (`back_count`/`back_radius`). Backward synapses create
  cross-layer loops. Weights are the layer's `out_weights` slots — already stored int8, already trainable
  (the engine already supports arbitrary topology levels — **no engine change**).
- **Width:** `size = 16`. **Depth:** ≥ 4 computational layers so `−1`/`−2` land on real computational
  layers (backward into the L0 transducer is inert — forced-fire — and simply wasted, not harmful).

## Learning — one temporal eligibility over all levels

The recurrent loops make timing matter for every synapse, so **all** trained synapses use the temporal
eligibility (not the factored feed-forward form): from recorded per-wave fired-sets, for a synapse
`z:i → (z+level):j`,
`e_ij = Σ_t pre_trace_i^z(t) · fired_j^{z+level}(t)` (decaying pre-trace = the temporal constant).
Iterate over each layer's topology **entries** (`level`, `count`, `radius`, slot offset); for entry slot `k`,
the target layer is `z+level` and target neuron `j = target_of(seed, …, level, k, radius, size)`. Update
`out_shadow[z][i·total_slots + slot] += −lr · L_j · e_ij`, quantize → int8. `total_slots = Σ counts`.

**Credit `L_j`** for target neuron `j` in layer `tz = z+level`: **symmetric** (readout weights) if `tz` is
the top layer, else **DFA** (fixed random hash-derived feedback) — the multi-layer credit already built.
(Backward targets are always below the top → DFA.) Feed-forward and backward weights train together.

## Benchmark & comparison

Temporal XOR, **LIF** (`adapt_bump = 0`), **delay 20** — the verified FF-fails variant. Readout on the top
layer's read-window activity. Compare, held-out + multi-seed:
- **FF-only** (`back_count = 0`): level+1 only, multi-layer trained. Expected ~chance (established).
- **+backward** (`back_count > 0`): level+1 + level−1/−2, all trained.

## Success criterion

- **Backward recurrence + width beats FF:** worst-seed held-out clearly above the FF baseline and above
  chance+margin. Print FF vs +backward per seed.
- Determinism preserved.

**Honesty gate (the diagnostic):** if +backward does **not** beat FF after reasonable tuning (`back_count`,
`back_radius`, `rec_tau`, `hidden_lr`, depth/width), **report the null as the finding** — with topology and
capacity controlled out, it points at `ψ` (spike-time-only) as the credit blocker, motivating the
sub-threshold-`ψ` lever next. Multi-seed, never a lucky seed. No fudged threshold.

## Module & engine

- **Engine:** none (arbitrary topology levels + per-layer per-wave spikes via listeners already exist).
- **Bench (`src/bench/rsnn.rs`):** `RsnnConfig.{back_count, back_radius}`; a recurrent XOR engine config with
  backward levels; a general temporal-eligibility training loop over topology entries (feed-forward +
  backward), reusing `dfa_weight`/`target_of`/`softmax` and the per-wave spike recording; the FF-vs-backward
  multi-seed harness.

## Testing

- `backward_recurrence_config_builds` — the topology has the expected `+1/−1/−2` entries and slot count.
- `backward_recurrence_vs_ff_on_temporal_xor` — the headline (FF vs +backward, multi-seed, held-out); the
  assertion encodes whichever outcome holds (beats FF, or documents the null → `ψ` blocker).
- `backward_recurrence_is_deterministic`.
- Regression: whole suite (incl. `wave_state_machine`) stays green.

## Deferred

Sub-threshold `ψ` (the indicated next lever if this nulls); combining backward + lateral (`level 0`);
memory optimization (per-synapse eligibility → recompute/share); symmetric feedback for deep credit.
