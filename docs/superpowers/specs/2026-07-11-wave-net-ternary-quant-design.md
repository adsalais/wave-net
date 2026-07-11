# Ternary (BitNet) weight-quantization mode in `wave_net` — design

**Date:** 2026-07-11
**Status:** approved design → plan → implement (user waived spec review)
**Branch:** `feat/wave-net-ternary-quant`

## Motivation

Research question: **can BitNet-style low-precision (ternary) weights still train the network?** `wave_net`
already stores plastic weights as `i8` (`Layer.out_weights`) trained through an `f32` shadow
(`out_shadow`) with a requantize step. BitNet-style ternary is mechanically a **different quantizer** on the
same shadow→weight path — so this is a **quantization MODE of `wave_net`, not a new engine**. (A full
`wave_bitnet` fork was rejected: it would duplicate the whole engine for a one-function change and leave a
third active engine to keep in sync — the reason `wave_state_machine` is frozen.)

## Locked decisions

- **Mode of `wave_net`**, not a fork.
- **Ternary, pure ±1/0 delivery** (first step): quantize to `{−1, 0, +1}`; the delivered contribution is
  exactly ±1 or 0 (no float scale on the fire-time delivery path — the engine is integer). Training reshapes
  **signs + sparsity** (which synapses prune to 0); magnitude stays at the ±1 init scale. (Option 2, an
  integer per-layer/​row *gain* on delivery, is the follow-up if pure-ternary underperforms.)
- **Per-row γ** (per *source* neuron): γ decides which of a source's weights round to 0, computed over that
  source neuron's outgoing slots — so only *firing* neurons' rows change per update and cost tracks spiking
  activity, not layer size (not per-layer γ, which is a full-layer reduction, and not a global fixed
  threshold).
- **`Config` untouched.** The mode lives on `Layer` (default `Int8`) and is flipped via
  `Network::set_weight_quant` — avoids churning every `Config { … }` literal in the tree.
- **Benchmarks in a NEW file** `bench/multilayer_dfa_bitnet_bench.rs`.
- The experiment runs through the `multilayer_dfa` engine path only; `rsnn.rs` and `eprop_update` (the
  factored FF path) are **not touched**.

## Design

### A — `WeightQuant` + `Layer` (`wave_net/neurons.rs`)

```rust
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WeightQuant { Int8, Ternary }
```

`Layer` gains `pub weight_quant: WeightQuant`, set to `WeightQuant::Int8` in `Layer::new` (so a fresh net and
every existing test/benchmark are byte-identical). New method — **per-row** requantize:

```rust
// requantize source neuron i's row = out_{weights,shadow}[i*total_slots .. +total_slots]
pub fn requantize_row(&mut self, i: usize) {
    let ts = self.total_slots;
    if ts == 0 { return; }                 // readout / no-outgoing layers
    let base = i * ts;
    match self.weight_quant {
        WeightQuant::Int8 => {
            for s in 0..ts { self.out_weights[base+s] = self.out_shadow[base+s].round().clamp(-127.0, 127.0) as i8; }
        }
        WeightQuant::Ternary => {
            let mut sum = 0.0f32;
            for s in 0..ts { sum += self.out_shadow[base+s].abs(); }
            let gamma = sum / ts as f32;    // per-row absmean
            for s in 0..ts {
                self.out_weights[base+s] = if gamma <= 0.0 { 0 } else { (self.out_shadow[base+s] / gamma).round().clamp(-1.0, 1.0) as i8 };
            }
        }
    }
}
```

At init (`out_shadow = ±1`) a ternary row has `γ = 1` → weights `±1` — identical to int8 init, so a fresh
ternary net equals the reference until training diverges.

### B — The one hot-path change: `eprop_update_synaptic` (`wave_net/eprop.rs`)

Replace the trailing **full-layer** requantize pass with **per-touched-row** requantize:

```rust
self.with_layer_mut(source_z, |lz| {
    for i in 0..ls {
        let sg = (base + i) as u32;
        let mut touched = false;
        for kk in 0..count {
            let e = elig[i * count + kk];
            if e == 0.0 { continue; }       // skip: shadow += ∓0.0 is a no-op (byte-identical)
            touched = true;
            let j = target_of(seed, sg, i as u32, level, kk as u32, radius, size) as usize;
            lz.out_shadow[i * total_slots + slot_base + kk] += -lr * signal[j] * e;
        }
        if touched { lz.requantize_row(i); }
    }
});
```

**Byte-identical for `Int8`:** unchanged rows (all-zero eligibility) keep their already-correct weights (their
shadow didn't change since the last requantize; init `±1` == `round(±1)`); touched rows are requantized to the
same values a full pass would produce (int8 quantization is per-weight, order-independent). The existing
`multilayer_dfa` tests (determinism, learns, recurrent-edge, step) are the guard. **`eprop_update` (factored)
is left untouched** (still int8 full-pass) — not on the experiment's path.

### C — Flipping a net (`wave_net/network.rs`)

```rust
pub fn set_weight_quant(&mut self, q: WeightQuant) {
    for layer in self.layers.iter_mut() {
        layer.weight_quant = q;
        if layer.total_slots == 0 { continue; }
        let rows = layer.out_shadow.len() / layer.total_slots;
        for i in 0..rows { layer.requantize_row(i); }
    }
}
```

One-time at setup; on a fresh `±1` net → `Ternary` it's a no-op (γ=1 → ±1). The A/B builds a net with the
`multilayer_dfa` harness (default int8) and flips one copy to ternary — **no builder churn**.

### D — Experiment (`bench/multilayer_dfa_bitnet_bench.rs`, NEW)

Per the AGENTS.md benchmark convention (sweep every axis + several seeds; read top spiking layer; report
density / σ / per-layer spiking / accuracy). An **A/B** of the *same* config under `Int8` vs `Ternary`:

- Tasks: temporal XOR (4 layers — the discriminating task; single_task as a fast sanity), size 32, generous
  fan-in.
- Axes: fan-in (radius/count) × training duration × seeds.
- Per config, report: **int8** acc curve (mean/worst over seeds) vs **ternary** acc curve, the **ternary
  weight-sparsity** (fraction of computational-layer `out_weights == 0`), and per-layer spiking (seed 0).
  Compare at the **peak** of the duration curve (the `rate_reg` over-training collapse is documented; don't
  read the final point).
- Success question: *does per-row pure-ternary reach the int8 baseline, and at what sparsity?*

Harness helper (`multilayer_dfa::harness`): `weight_sparsity(net) -> f64` = fraction of zero `out_weights`
over computational layers `1..L`.

### E — Tests + determinism

Per-row γ is a fixed-order sum → deterministic; `round` deterministic.

1. `requantize_row` Ternary: an all-`±1` row → `±1` (γ=1); a row with values `[2, 2, 0.1, 0.1]` (γ=1.05,
   0.5γ≈0.525) → `[1,1,0,0]` (hand-computed).
2. `eprop_update_synaptic` under `Ternary` leaves every `out_weight` in `{−1,0,+1}`.
3. Int8 path unchanged: the existing `multilayer_dfa` unit tests (determinism, learns-2class, recurrent-edge,
   step) stay green after the per-row refactor — the byte-identity guard.
4. `set_weight_quant(Ternary)` on a fresh net is a no-op on `out_weights` (still `±1`); flips `weight_quant`.
5. A small "ternary net trains above chance" test (size 8–16, single_task, few hundred trials).

## Files touched

- `src/wave_net/neurons.rs` — `WeightQuant` enum, `Layer.weight_quant` field (+ `Layer::new` default),
  `Layer::requantize_row`.
- `src/wave_net/network.rs` — `Network::set_weight_quant`.
- `src/wave_net/eprop.rs` — `eprop_update_synaptic` per-touched-row requantize (byte-identical int8).
- `src/wave_net/mod.rs` — re-export `WeightQuant` if convenient (else import from `neurons`).
- `src/bench/multilayer_dfa.rs` — `harness::weight_sparsity` helper (+ ternary unit tests in `mod tests`).
- `src/bench/multilayer_dfa_bitnet_bench.rs` — **new** A/B benchmark.
- `src/bench/mod.rs` — `pub mod multilayer_dfa_bitnet_bench;`.
- **Untouched:** `Config`, `eprop_update`, `rsnn.rs`, `wave_state_machine`.

## Constraints (AGENTS.md)

- Rust 2024, std-only, no `unsafe`, warning-free `cargo build`.
- Determinism a hard requirement (pure function of seed/config/input).
- TDD, one commit per task, conventional commits, **no `Co-Authored-By`**, never push.
- Benchmarks: sweep every axis + several seeds (worst+mean), read the top spiking layer, report
  density/σ/per-layer-spiking/accuracy; compare at the duration peak (over-training caveat).

## Honest expectation

Pure ternary fixes magnitude at ±1 — training reshapes signs + sparsity but cannot grow strong synapses
(int8 can reach ±127). It may **not** match int8; that gap is the finding. If so, the follow-up is Option 2
(an integer per-row *gain* on delivery, `w = ±g·sign`, restoring per-row magnitude) — deliberately deferred
to keep this first probe cheap.
