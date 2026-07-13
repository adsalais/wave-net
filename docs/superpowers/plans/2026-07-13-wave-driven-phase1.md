# wave_driven Phase 1 (event-driven inference) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (this repo mandates **inline** execution — never subagent-driven; see AGENTS.md "Plan execution is inline and autonomous"). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `wave_driven` — a new, independent spiking-inference engine whose per-wave cost scales with **activity** (spikes + deliveries) instead of layer **size**, via a per-layer frontier worklist, a sparse delivery accumulator, and lazy fire-anchored adaptation.

**Architecture:** A stack of square layers (copied topology substrate from `wave_bitnet`: occupancy-bitset connectivity + 2-bit ±1/0 codes). Each wave processes only a **frontier** of non-quiescent neurons per layer, scatters deliveries into a next-wave accumulator, and rebuilds the next frontier from (delivery targets ∪ residual-state neurons ∪ injections). Adaptation is stored as its value at the last fire and reconstructed in closed form on demand, so it never keeps a neuron awake. A **dense** mode (process all neurons every wave) is the correctness oracle: sparse must equal dense bit-for-bit.

**Tech Stack:** Rust edition 2024, standard library only, single-threaded, deterministic. Inline `#[cfg(test)]` tests; `#[ignore]`d throughput experiments run in `--release`.

## Global Constraints

- **Standard library only** in `src/`; **warning-free** `cargo build`; `cargo test` stays green. (verbatim from AGENTS.md)
- **Determinism is a hard requirement** — output is a pure function of `(seed, config, input)`; single-threaded. (verbatim from AGENTS.md)
- **No `unsafe`** except the one documented `wave_bitnet` generate-loop pattern, if copied verbatim with its `SAFETY:` comments. Prefer the safe reslice pattern; only carry over `unsafe` with its justification intact.
- **NEVER add a `Co-Authored-By` trailer** to commit messages. Conventional-commit style (`feat:`/`refactor:`/`docs:`/`test:`). **One commit per task.** **NEVER push.**
- `wave_driven` must **not depend on `wave_bitnet`** — copy code as needed (duplication is allowed for this module).
- `size` is a power of two; local index is `y*size + x`; global neuron id is `layer*size*size + local`; per-layer state is struct-of-arrays.
- Branch: `feat/wave-driven` (already checked out).

---

### Task 1: Module scaffold + copied `synapse` helpers

**Files:**
- Create: `src/wave_driven/mod.rs`
- Create: `src/wave_driven/synapse.rs`
- Modify: `src/lib.rs` (add `pub mod wave_driven;`)

**Interfaces:**
- Produces: `wave_driven::synapse::{TopologyLevel, key, mix, map_range, map_range24, local_of, xy_of, wrap, neigh_size, decode_cell, sample_distinct_cells, random_l0_input, P_TARGET, P_THRESHOLD, P_INPUT}` — identical signatures to the `wave_bitnet` versions.

- [ ] **Step 1: Copy `synapse.rs` verbatim.** Copy `src/wave_bitnet/synapse.rs` to `src/wave_driven/synapse.rs` **unchanged** (all helpers + its `#[cfg(test)]` module). It is engine-agnostic pure arithmetic; keep the `Synapse` struct too. Do not edit anything.

- [ ] **Step 2: Create `src/wave_driven/mod.rs`** with the module declaration and a doc comment:

```rust
//! `wave_driven` — an event-driven, active-set spiking **inference** engine (Phase 1). Per-wave cost
//! scales with activity, not layer size: a per-layer frontier of non-quiescent neurons, a sparse
//! delivery accumulator, and lazy fire-anchored adaptation. Independent of `wave_bitnet` (topology
//! substrate is copied). Spec: docs/superpowers/specs/2026-07-13-wave-driven-event-active-set-design.md.

pub mod synapse;
```

- [ ] **Step 3: Wire into `src/lib.rs`.** Add after the `wave_bitnet` line:

```rust
pub mod wave_driven; // event-driven active-set inference engine (Phase 1)
```

- [ ] **Step 4: Build + run the copied tests.**

Run: `cargo test wave_driven::synapse -- --nocapture`
Expected: PASS (`decode_center_is_self_and_corners_wrap`, `sample_is_distinct_bounded_and_deterministic`). `cargo build` warning-free.

- [ ] **Step 5: Commit.**

```bash
git add src/lib.rs src/wave_driven/mod.rs src/wave_driven/synapse.rs
git commit -m "feat(wave_driven): scaffold module + copy synapse helpers"
```

---

### Task 2: Copied `config` (dead-zone bound dropped)

**Files:**
- Create: `src/wave_driven/config.rs`
- Modify: `src/wave_driven/mod.rs` (add `pub mod config;`)

**Interfaces:**
- Produces: `wave_driven::config::{Config, LayerConfig, THRESHOLD_JITTER_DEFAULT}` with the same fields as `wave_bitnet`. `Config::validate` accepts `adapt_decay` in `1..=MAX_ADAPT_DECAY` (no `<= ADAPT_SHIFT` cap).

- [ ] **Step 1: Copy `config.rs`** from `src/wave_bitnet/config.rs` to `src/wave_driven/config.rs`, then change two things:
  1. In the `use` line, point at `crate::wave_driven::synapse::TopologyLevel` and (in `validate`) `crate::wave_driven::synapse::neigh_size`.
  2. Replace the `adapt_decay` validation. `wave_bitnet` had:

```rust
if lc.adapt_decay == 0 || lc.adapt_decay as u32 > crate::wave_bitnet::neurons::ADAPT_SHIFT {
    return Err(format!("layer {z}: adapt_decay must be in 1..={} (ADAPT_SHIFT; ...)",
        crate::wave_bitnet::neurons::ADAPT_SHIFT));
}
```

  Replace it with (the geometric multiply has no dead zone; the only bound is `< FRAC=30`, so the `ρ` fixed-point shift is valid — cap at 24 for headroom):

```rust
if lc.adapt_decay == 0 || lc.adapt_decay > crate::wave_driven::neurons::MAX_ADAPT_DECAY {
    return Err(format!(
        "layer {z}: adapt_decay must be in 1..={} (defines the geometric decay ratio ρ = 1 − 2^−decay)",
        crate::wave_driven::neurons::MAX_ADAPT_DECAY
    ));
}
```

  (`MAX_ADAPT_DECAY` is defined in Task 3. Until then this won't compile — that's expected; Task 2 is committed after Task 3 wires the constant. To keep Task 2 self-contained, temporarily inline the literal `24` and a `// TODO` — **no**: instead, define `MAX_ADAPT_DECAY` here is wrong. Simplest: hardcode `24u8` in Task 2 and replace with the named constant in Task 3. Use `24u8` now.)

  Concretely for Task 2, write:

```rust
if lc.adapt_decay == 0 || lc.adapt_decay > 24 {
    return Err(format!(
        "layer {z}: adapt_decay must be in 1..=24 (defines the geometric decay ratio ρ = 1 − 2^−decay)"
    ));
}
```

- [ ] **Step 2: Replace the `rejects_zero_adapt_decay` test** and add an acceptance test for a large decay. In the copied `#[cfg(test)]` module, keep `demo_is_valid`, `rejects_non_power_of_two_size`, `rejects_empty_layers`, `validate_accepts_fan_in_within_neighborhood`, `validate_rejects_fan_in_over_neighborhood`. Replace the zero-adapt test with:

```rust
#[test]
fn rejects_zero_adapt_decay() {
    let mut c = Config::demo();
    c.layers[0].adapt_decay = 0;
    assert!(c.validate().is_err());
}

#[test]
fn accepts_large_adapt_decay() {
    // wave_driven has no ADAPT_SHIFT dead-zone bound; a slow decay (τ large) is valid.
    let mut c = Config::demo();
    c.layers[0].adapt_decay = 20;
    assert!(c.validate().is_ok(), "adapt_decay 20 must be accepted (no dead-zone cap)");
}
```

- [ ] **Step 3: Add `pub mod config;`** to `src/wave_driven/mod.rs`.

- [ ] **Step 4: Run tests.**

Run: `cargo test wave_driven::config`
Expected: PASS all config tests.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/config.rs src/wave_driven/mod.rs
git commit -m "feat(wave_driven): copy config, drop the adapt_decay dead-zone bound"
```

---

### Task 3: `neurons` — Layer state, topology layout, procedural init, accessors

**Files:**
- Create: `src/wave_driven/neurons.rs`
- Modify: `src/wave_driven/mod.rs` (add `pub mod neurons;`)

**Interfaces:**
- Produces:
  - Constants: `ADAPT_SHIFT: u32 = 12`, `ADAPT_MAX: i32`, `FRAC: u32 = 30`, `MAX_ADAPT_DECAY: u8 = 24`, `HORIZON_CAP: usize = 1<<16`, `WCODE: [i8;4]`.
  - `struct Layer` with public SoA fields: `potential: Vec<i16>`, `cooldown: Vec<u8>`, `threshold: Vec<i16>`, `adapt_ref: Vec<i32>`, `fire_wave: Vec<u32>`, `pending: Vec<i32>`, plus config (`leak`, `cooldown_base`, `topology`, `adapt_bump`, `adapt_decay`, `readout`, `ternary_threshold`), derived layout (`total_slots`, `slot_bases`, `neigh`, `occ_wpn`, `occ`, `offsets`, `off_flat`, `codes`), and `pow_decay: Vec<i64>`.
  - `Layer::new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer` — procedural init: occupancy bitset, thresholds, ±1 codes (init to the procedural sign), and the `pow_decay` table.
  - Methods: `slot_base`, `weight_at`, `synapse_count`, `for_wired`, `decode`, `decayed_adapt`, and `build_pow_decay` (free fn).
- Consumes: `wave_driven::synapse::*`, `wave_driven::config::LayerConfig`.

- [ ] **Step 1: Write failing tests** in `src/wave_driven/neurons.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_driven::config::LayerConfig;
    use crate::wave_driven::synapse::TopologyLevel;

    fn lc(topology: Vec<TopologyLevel>) -> LayerConfig {
        LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 }
    }

    #[test]
    fn new_wires_exactly_count_distinct_cells_deterministically() {
        let size = 8u32;
        let ls = (size * size) as usize;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 16 }]);
        let a = Layer::new(&cfg, 7, 0, size);
        let b = Layer::new(&cfg, 7, 0, size);
        assert_eq!(a.total_slots, 16);
        for i in 0..ls {
            let mut cnt = 0usize;
            let mut cells = Vec::new();
            a.for_wired(0, i, |_r, c| { cnt += 1; cells.push(c); });
            assert_eq!(cnt, 16);
            assert!(cells.windows(2).all(|w| w[0] < w[1]));
        }
        assert_eq!(a.occ, b.occ, "deterministic occupancy");
        assert_eq!(a.codes, b.codes, "deterministic ±1 codes");
    }

    #[test]
    fn weight_at_decodes_pm1_from_codes() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]);
        let l = Layer::new(&cfg, 3, 0, size);
        // fresh net: every wired synapse is ±1 (procedural sign), never 0 (inhibitor_ratio 0 => all +1).
        for s in 0..l.synapse_count() {
            assert!(matches!(l.weight_at(s), 1 | -1), "fresh code is ±1, got {}", l.weight_at(s));
        }
    }

    #[test]
    fn decode_center_is_self() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let l = Layer::new(&cfg, 7, 0, size);
        let src = crate::wave_driven::synapse::local_of(3, 4, size);
        assert_eq!(l.decode(0, src, 12, size), src, "center cell (idx 12, span 5) maps to self");
    }
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::neurons`
Expected: FAIL (`Layer` not defined).

- [ ] **Step 3: Implement `neurons.rs`.** Reuse the `wave_bitnet` layout machinery (copy `DerivedLayout` + `derive_layout` verbatim), drop the `TrainState`/`shadow`/`repack`/`from_parts` (Phase 1 needs no shadow), and add the adaptation table. Full listing:

```rust
//! `neurons` — a `Layer`'s per-neuron SoA state and its bitset topology substrate (copied from
//! wave_bitnet) plus the lazy fire-anchored adaptation state and its geometric decay table.

use crate::wave_driven::config::LayerConfig;
use crate::wave_driven::synapse::{key, local_of, map_range, mix, neigh_size, sample_distinct_cells, wrap, xy_of, TopologyLevel, P_TARGET, P_THRESHOLD};

/// Fixed-point scale for the adaptation contribution to the effective threshold (`adapt >> ADAPT_SHIFT`).
pub const ADAPT_SHIFT: u32 = 12;
/// Ceiling for the reconstructed adaptation (so its threshold contribution never exceeds i16::MAX).
pub const ADAPT_MAX: i32 = (i16::MAX as i32) << ADAPT_SHIFT;
/// Fixed-point fraction bits of the geometric decay table `pow_decay` (i64 math: adapt_ref≤2^27 · POW≤2^30 ⊂ i64).
pub const FRAC: u32 = 30;
/// Upper bound on `adapt_decay` (must stay < FRAC so ρ's fixed-point shift is valid; 24 leaves headroom).
pub const MAX_ADAPT_DECAY: u8 = 24;
/// Cap on the decay-table length (waves). Beyond it, reconstructed adaptation is 0.
pub const HORIZON_CAP: usize = 1 << 16;
/// 2-bit weight code decode LUT: 0b00→0, 0b01→+1, 0b11→−1.
pub(crate) const WCODE: [i8; 4] = [0, 1, 0, -1];

pub(crate) struct DerivedLayout {
    pub total_slots: usize,
    pub slot_bases: Vec<usize>,
    pub neigh: Vec<usize>,
    pub occ_wpn: Vec<usize>,
    pub offsets: Vec<Vec<(i8, i8)>>,
    pub off_flat: Vec<Vec<i32>>,
}

pub(crate) fn derive_layout(topology: &[TopologyLevel], size: u32) -> DerivedLayout {
    let n_levels = topology.len();
    let mut slot_bases = Vec::with_capacity(n_levels);
    let mut neigh = Vec::with_capacity(n_levels);
    let mut occ_wpn = Vec::with_capacity(n_levels);
    let mut offsets: Vec<Vec<(i8, i8)>> = Vec::with_capacity(n_levels);
    let mut off_flat: Vec<Vec<i32>> = Vec::with_capacity(n_levels);
    let mut total_slots = 0usize;
    for t in topology {
        slot_bases.push(total_slots);
        let n = neigh_size(t.radius);
        neigh.push(n);
        occ_wpn.push((n + 63) / 64);
        let span = 2 * t.radius + 1;
        let r = t.radius as i32;
        offsets.push((0..n).map(|c| (((c as u32 % span) as i32 - r) as i8, ((c as u32 / span) as i32 - r) as i8)).collect());
        off_flat.push((0..n).map(|c| { let dx = (c as u32 % span) as i32 - r; let dy = (c as u32 / span) as i32 - r; dy * size as i32 + dx }).collect());
        total_slots += t.count as usize;
    }
    DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat }
}

/// `pow_decay[k] = round(ρ^k · 2^FRAC)`, ρ = 1 − 2^−adapt_decay, `pow_decay[0] = 2^FRAC`. Grows until
/// even `ADAPT_MAX` decays to 0 through it, capped at `HORIZON_CAP`. Reconstructs adaptation exactly and
/// path-independently (a single jump from the last fire), so dense and sparse agree bit-for-bit.
pub fn build_pow_decay(adapt_decay: u8) -> Vec<i64> {
    let one = 1i64 << FRAC;
    let rho = one - (1i64 << (FRAC - adapt_decay as u32)); // ρ in fixed point
    let mut table = vec![one];
    let mut cur = one;
    while table.len() < HORIZON_CAP {
        // round(cur · ρ / 2^FRAC)
        let next = ((cur as i128 * rho as i128 + (1i128 << (FRAC - 1))) >> FRAC) as i64;
        if next <= 0 || ((ADAPT_MAX as i128 * next as i128) >> FRAC) == 0 {
            break; // even the largest possible adapt now reconstructs to 0 → horizon reached
        }
        table.push(next);
        cur = next;
    }
    table
}

pub struct Layer {
    // neuron state
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub threshold: Vec<i16>,
    pub adapt_ref: Vec<i32>, // adaptation value at the last fire (Q ADAPT_SHIFT)
    pub fire_wave: Vec<u32>, // wave index of the last fire
    pub pending: Vec<i32>,   // per-target incoming accumulator, drained (folded) each wave
    // config
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<TopologyLevel>,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub readout: bool,
    pub ternary_threshold: f32,
    // derived layout
    pub total_slots: usize,
    pub slot_bases: Vec<usize>,
    pub neigh: Vec<usize>,
    pub occ_wpn: Vec<usize>,
    pub occ: Vec<Vec<u64>>,
    pub offsets: Vec<Vec<(i8, i8)>>,
    pub off_flat: Vec<Vec<i32>>,
    pub codes: Vec<u64>, // 2-bit ±1/0 codes, 32 per u64
    // lazy adaptation decay table (per adapt_decay)
    pub pow_decay: Vec<i64>,
}

impl Layer {
    #[inline]
    pub fn slot_base(&self, level_idx: usize) -> usize { self.slot_bases[level_idx] }

    #[inline]
    pub fn weight_at(&self, widx: usize) -> i8 {
        WCODE[((self.codes[widx >> 5] >> ((widx & 31) * 2)) & 0b11) as usize]
    }

    #[inline]
    pub fn synapse_count(&self) -> usize { self.total_slots * self.threshold.len() }

    /// Iterate the wired cells of neuron `i` at level `lvl` in ascending cell order, calling `f(rank, cell)`.
    #[inline]
    pub fn for_wired(&self, lvl: usize, i: usize, mut f: impl FnMut(usize, usize)) {
        let wpn = self.occ_wpn[lvl];
        let words = &self.occ[lvl][i * wpn..i * wpn + wpn];
        let mut rank = 0usize;
        for (wi, &w0) in words.iter().enumerate() {
            let mut word = w0;
            let cbase = wi * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                f(rank, cbase + bit);
                rank += 1;
                word &= word - 1;
            }
        }
    }

    /// Decode neighborhood `cell` of a source at `src_local` to its target local index (offset LUT + wrap).
    #[inline]
    pub fn decode(&self, lvl: usize, src_local: u32, cell: usize, size: u32) -> u32 {
        let (sx, sy) = xy_of(src_local, size);
        let (dx, dy) = self.offsets[lvl][cell];
        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size)
    }

    /// Reconstruct neuron `i`'s adaptation at wave `w`: `(adapt_ref · ρ^(w − fire_wave)) >> FRAC`, or 0
    /// beyond the decay horizon. Pure function of the stored anchor — path-independent.
    #[inline]
    pub fn decayed_adapt(&self, i: usize, w: u32) -> i32 {
        let gap = w.wrapping_sub(self.fire_wave[i]) as usize;
        if gap >= self.pow_decay.len() {
            0
        } else {
            ((self.adapt_ref[i] as i64 * self.pow_decay[gap]) >> FRAC) as i32
        }
    }

    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;
        let DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat } = derive_layout(&cfg.topology, size);

        // thresholds: baseline_init + rand(0..threshold_jitter), clamp(1, i16::MAX)
        let mut threshold = vec![0i16; ls];
        for (local, th) in threshold.iter_mut().enumerate() {
            let global = (base + local) as u32;
            let h = mix(key(seed, global, 0, 0, P_THRESHOLD));
            let jitter = map_range(h as u32, cfg.threshold_jitter as u32) as i32;
            *th = (cfg.baseline_init as i32 + jitter).clamp(1, i16::MAX as i32) as i16;
        }

        // occupancy: `count` distinct cells per neuron per level, word-aligned
        let mut occ: Vec<Vec<u64>> = occ_wpn.iter().map(|&wpn| vec![0u64; ls * wpn]).collect();
        for (li, t) in cfg.topology.iter().enumerate() {
            let wpn = occ_wpn[li];
            for i in 0..ls {
                let sg = (base + i) as u32;
                for &cell in &sample_distinct_cells(seed, sg, t.level, t.radius, t.count) {
                    let c = cell as usize;
                    occ[li][i * wpn + c / 64] |= 1u64 << (c % 64);
                }
            }
        }

        // codes: init each wired synapse to the procedural ±1 sign (rank-indexed, wired-rank order)
        let mut codes = vec![0u64; (ls * total_slots + 31) / 32];
        for i in 0..ls {
            let sg = (base + i) as u32;
            for (li, t) in cfg.topology.iter().enumerate() {
                for r in 0..(t.count as usize) {
                    let h = mix(key(seed, sg, t.level, r as u32, P_TARGET));
                    let sign_code: u64 = if ((h & 0xFFFF) as u32) < cfg.inhibitor_ratio { 0b11 } else { 0b01 };
                    let idx = i * total_slots + slot_bases[li] + r;
                    let wshift = (idx & 31) * 2;
                    codes[idx >> 5] |= sign_code << wshift;
                }
            }
        }

        Layer {
            potential: vec![0i16; ls],
            cooldown: vec![0u8; ls],
            threshold,
            adapt_ref: vec![0i32; ls],
            fire_wave: vec![0u32; ls],
            pending: vec![0i32; ls],
            leak: cfg.leak,
            cooldown_base: cfg.cooldown_base,
            topology: cfg.topology.clone(),
            adapt_bump: cfg.adapt_bump,
            adapt_decay: cfg.adapt_decay,
            readout: false,
            ternary_threshold: 0.5,
            total_slots,
            slot_bases,
            neigh,
            occ_wpn,
            occ,
            offsets,
            off_flat,
            codes,
            pow_decay: build_pow_decay(cfg.adapt_decay),
        }
    }
}
```

- [ ] **Step 4: Add `pub mod neurons;`** to `src/wave_driven/mod.rs`. Then update Task 2's hardcoded `24` to the named constant: change `lc.adapt_decay > 24` in `config.rs` `validate` to `lc.adapt_decay > crate::wave_driven::neurons::MAX_ADAPT_DECAY`, and the error text `1..=24` to `1..={}` with `crate::wave_driven::neurons::MAX_ADAPT_DECAY`.

- [ ] **Step 5: Run tests.**

Run: `cargo test wave_driven::neurons wave_driven::config`
Expected: PASS. `cargo build` warning-free.

- [ ] **Step 6: Commit.**

```bash
git add src/wave_driven/neurons.rs src/wave_driven/mod.rs src/wave_driven/config.rs
git commit -m "feat(wave_driven): Layer state, topology substrate, procedural init, decay table"
```

---

### Task 4: `neurons` — lazy adaptation behavior tests

**Files:**
- Modify: `src/wave_driven/neurons.rs` (extend the `#[cfg(test)]` module)

**Interfaces:**
- Consumes: `Layer::new`, `Layer::decayed_adapt`, `build_pow_decay`, `ADAPT_SHIFT`, `ADAPT_MAX`, `FRAC`.

- [ ] **Step 1: Write failing tests** appended to the `tests` module in `neurons.rs`:

```rust
#[test]
fn decayed_adapt_at_gap_zero_is_adapt_ref() {
    let size = 8u32;
    let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
    let mut l = Layer::new(&cfg, 7, 0, size);
    l.adapt_ref[0] = 5 << ADAPT_SHIFT;
    l.fire_wave[0] = 100;
    // gap 0 → POW[0] = 2^FRAC → returns adapt_ref exactly
    assert_eq!(l.decayed_adapt(0, 100), 5 << ADAPT_SHIFT);
}

#[test]
fn decayed_adapt_is_monotonic_nonincreasing_and_hits_zero() {
    let size = 8u32;
    let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
    let mut l = Layer::new(&cfg, 7, 0, size); // adapt_decay 6
    l.adapt_ref[0] = ADAPT_MAX;
    l.fire_wave[0] = 0;
    let mut prev = i32::MAX;
    for w in 0..l.pow_decay.len() as u32 {
        let a = l.decayed_adapt(0, w);
        assert!(a <= prev, "non-increasing at w={w}: {a} > {prev}");
        assert!(a >= 0);
        prev = a;
    }
    // beyond the horizon it is exactly 0
    assert_eq!(l.decayed_adapt(0, l.pow_decay.len() as u32 + 1), 0);
}

#[test]
fn decayed_adapt_path_independent_across_reads() {
    // Reading at intermediate waves must not change the value at a later wave (pure fn of anchor+w):
    // this is the property that makes dense (reads every wave) == sparse (reads only on wake).
    let size = 8u32;
    let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
    let mut l = Layer::new(&cfg, 7, 0, size);
    l.adapt_ref[0] = 3 << ADAPT_SHIFT;
    l.fire_wave[0] = 0;
    let one_jump = l.decayed_adapt(0, 15);
    let mut acc = 0;
    for w in 0..=15 { acc = l.decayed_adapt(0, w); } // reads at every wave; state never mutated
    assert_eq!(one_jump, acc, "value at w=15 is independent of intermediate reads");
}

#[test]
fn pow_decay_matches_geometric_within_rounding() {
    let table = build_pow_decay(6);
    let rho = 1.0f64 - 2f64.powi(-6);
    for (k, &p) in table.iter().enumerate().take(200) {
        let want = (rho.powi(k as i32) * (1i64 << FRAC) as f64).round() as i64;
        assert!((p - want).abs() <= 2, "POW[{k}] {p} vs geometric {want}");
    }
}
```

- [ ] **Step 2: Run to verify they pass** (the implementation from Task 3 already satisfies these — these tests *characterize* the adaptation model; no new code expected).

Run: `cargo test wave_driven::neurons`
Expected: PASS. If any fails, fix `decayed_adapt`/`build_pow_decay` in Task 3's code, not the test.

- [ ] **Step 3: Commit.**

```bash
git add src/wave_driven/neurons.rs
git commit -m "test(wave_driven): characterize lazy fire-anchored adaptation"
```

---

### Task 5: `frontier` — worklist + dedup mark bitset

**Files:**
- Create: `src/wave_driven/frontier.rs`
- Modify: `src/wave_driven/mod.rs` (add `pub mod frontier;`)

**Interfaces:**
- Produces: `wave_driven::frontier::Frontier { list: Vec<u32>, mark: Vec<u64> }` with `new(ls: usize) -> Frontier`, `push(&mut self, t: u32) -> bool` (test-and-set; true if newly inserted), `contains(&self, t: u32) -> bool`, `clear(&mut self)` (walk `list`, clear its marks, empty `list`), `len(&self) -> usize`, `is_empty(&self) -> bool`.

- [ ] **Step 1: Write failing tests** in `src/wave_driven/frontier.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_dedups() {
        let mut f = Frontier::new(64);
        assert!(f.push(5));
        assert!(!f.push(5), "second push of 5 is a no-op");
        assert!(f.push(9));
        assert_eq!(f.list, vec![5, 9], "each neuron appears once, in insertion order");
        assert!(f.contains(5) && f.contains(9) && !f.contains(7));
    }

    #[test]
    fn clear_empties_list_and_resets_marks() {
        let mut f = Frontier::new(64);
        f.push(1); f.push(2); f.push(63);
        f.clear();
        assert!(f.is_empty());
        assert!(!f.contains(1) && !f.contains(2) && !f.contains(63));
        // reusable after clear
        assert!(f.push(1));
        assert_eq!(f.list, vec![1]);
    }

    #[test]
    fn handles_bit_boundaries() {
        let mut f = Frontier::new(128);
        for t in [0u32, 63, 64, 127] { assert!(f.push(t)); }
        for t in [0u32, 63, 64, 127] { assert!(f.contains(t)); }
        assert_eq!(f.len(), 4);
    }
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::frontier`
Expected: FAIL (`Frontier` not defined).

- [ ] **Step 3: Implement `frontier.rs`:**

```rust
//! `frontier` — the per-layer active-set worklist. A `Vec` gives ordered, cache-friendly iteration; a
//! 1-bit-per-neuron `mark` bitset makes insertion a test-and-set so no neuron is ever queued twice
//! (two firers hitting one target, or a target that is also a carryover). Cleared by walking `list`
//! (O(activity)), never by zeroing size². This is exactly the GPU unique-frontier-append primitive.

pub struct Frontier {
    pub list: Vec<u32>,
    pub mark: Vec<u64>, // ceil(ls / 64) words
}

impl Frontier {
    pub fn new(ls: usize) -> Frontier {
        Frontier { list: Vec::new(), mark: vec![0u64; (ls + 63) / 64] }
    }

    /// Test-and-set insert. Returns true iff `t` was newly added (was not already queued).
    #[inline]
    pub fn push(&mut self, t: u32) -> bool {
        let w = (t >> 6) as usize;
        let bit = 1u64 << (t & 63);
        if self.mark[w] & bit == 0 {
            self.mark[w] |= bit;
            self.list.push(t);
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn contains(&self, t: u32) -> bool {
        self.mark[(t >> 6) as usize] & (1u64 << (t & 63)) != 0
    }

    /// Empty the worklist and reset its marks by walking `list` (O(activity)).
    #[inline]
    pub fn clear(&mut self) {
        for &t in &self.list {
            self.mark[(t >> 6) as usize] &= !(1u64 << (t & 63));
        }
        self.list.clear();
    }

    #[inline]
    pub fn len(&self) -> usize { self.list.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.list.is_empty() }
}
```

- [ ] **Step 4: Add `pub mod frontier;`** to `mod.rs`. Run tests.

Run: `cargo test wave_driven::frontier`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/frontier.rs src/wave_driven/mod.rs
git commit -m "feat(wave_driven): frontier worklist + dedup mark bitset"
```

---

### Task 6: `wave` — the per-layer step (dense + sparse via one shared path)

**Files:**
- Create: `src/wave_driven/wave.rs`
- Modify: `src/wave_driven/mod.rs` (add `pub mod wave;`)

**Interfaces:**
- Produces:
  - `enum Work<'a> { Dense, Sparse { cur: &'a [u32], frontier_next: &'a mut [Frontier] } }`
  - `fn process_layer(layer: &mut Layer, layer_index: u32, size: u32, input: &[u32], w: u32, work: Work, deliv: &mut [Vec<i32>], fired: &mut Vec<u32>)`
- Consumes: `neurons::{Layer, ADAPT_SHIFT, ADAPT_MAX, FRAC, WCODE}`, `frontier::Frontier`, `synapse::{local_of, wrap, xy_of}`.

**Design note.** One function, two modes. `Dense` iterates `0..ls` and maintains no frontier; `Sparse` iterates `cur` and, per surviving neuron, carries it into `frontier_next[layer_index]`, and per delivery pushes the target into `frontier_next[tz]`. The per-neuron arithmetic is a **single code path** so dense and sparse can only differ in *which neurons are visited* — exactly the frontier-completeness property the oracle (Task 8) checks. Injection (L0) overrides potential after drain, matching the drain→inject→decide order.

- [ ] **Step 1: Write a failing unit test** in `wave.rs` — a firing neuron scatters its decoded ±1 weights into the next layer's accumulator (mirrors the `wave_bitnet` generate test), exercised through `Work::Dense`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_driven::config::LayerConfig;
    use crate::wave_driven::neurons::Layer;
    use crate::wave_driven::synapse::TopologyLevel;

    fn one_up(size: u32, count: u32) -> Layer {
        let cfg = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        Layer::new(&cfg, 5, 0, size)
    }

    #[test]
    fn dense_firer_scatters_decoded_weights() {
        let size = 4u32;
        let ls = (size * size) as usize;
        let mut l = one_up(size, 3);
        l.threshold.iter_mut().for_each(|t| *t = 1);
        l.cooldown.iter_mut().for_each(|c| *c = 0);
        l.potential[0] = 100;
        // expected: sum decoded nonzero weights per target for neuron 0
        let base = l.slot_base(0);
        let mut expect = vec![0i32; ls];
        l.for_wired(0, 0, |r, cell| {
            let wt = l.weight_at(0 * l.total_slots + base + r);
            if wt != 0 { expect[l.decode(0, 0, cell, size) as usize] += wt as i32; }
        });
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; ls]; 2];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, size, &[], 0, Work::Dense, &mut deliv, &mut fired);
        assert_eq!(fired, vec![0], "only neuron 0 fires (others at rest below threshold)");
        assert_eq!(deliv[1], expect, "scatter-adds decoded weights into layer 1's accumulator");
    }

    #[test]
    fn sparse_matches_dense_on_one_step() {
        // Same primed layer, processed dense vs sparse (cur = {0}); the visited neuron's post-state
        // and the deliveries must match.
        let size = 4u32;
        let ls = (size * size) as usize;
        let mut ld = one_up(size, 3);
        let mut lspx = one_up(size, 3);
        for l in [&mut ld, &mut lspx] {
            l.threshold.iter_mut().for_each(|t| *t = 1);
            l.potential[0] = 100;
        }
        let mut dd = vec![vec![0i32; ls]; 2];
        let mut fd = Vec::new();
        process_layer(&mut ld, 0, size, &[], 0, Work::Dense, &mut dd, &mut fd);
        let mut ds = vec![vec![0i32; ls]; 2];
        let mut fs = Vec::new();
        let mut fnext = vec![Frontier::new(ls), Frontier::new(ls)];
        let cur = vec![0u32];
        process_layer(&mut lspx, 0, size, &[], 0, Work::Sparse { cur: &cur, frontier_next: &mut fnext }, &mut ds, &mut fs);
        assert_eq!(fd, fs);
        assert_eq!(dd, ds);
        assert_eq!(ld.potential[0], lspx.potential[0]);
        assert_eq!(ld.cooldown[0], lspx.cooldown[0]);
    }
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::wave`
Expected: FAIL (`process_layer` not defined).

- [ ] **Step 3: Implement `wave.rs`:**

```rust
//! `wave` — one layer's per-wave step, event-driven. Drain the sparse accumulator, (L0) inject,
//! decide/fire/leak with lazy fire-anchored adaptation, then scatter deliveries (the wave_bitnet
//! word-scan). `Work::Sparse` visits only a frontier and rebuilds the next one; `Work::Dense` visits
//! all neurons and is the equivalence oracle. Both share the per-neuron arithmetic.

use crate::wave_driven::frontier::Frontier;
use crate::wave_driven::neurons::{Layer, ADAPT_MAX, ADAPT_SHIFT, FRAC, WCODE};
use crate::wave_driven::synapse::{local_of, wrap, xy_of};

pub enum Work<'a> {
    Dense,
    Sparse { cur: &'a [u32], frontier_next: &'a mut [Frontier] },
}

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    size: u32,
    input: &[u32],
    w: u32,
    mut work: Work,
    deliv: &mut [Vec<i32>],
    fired: &mut Vec<u32>,
) {
    let ls = (size as usize) * (size as usize);
    fired.clear();

    // The visited set: all neurons (dense) or the current frontier (sparse). Materialize a small slice
    // reference; dense uses a 0..ls range handled inline to avoid allocating.
    // --- 1. drain: fold pending into potential (i32), clamp to i16, clear pending ---
    match &work {
        Work::Dense => {
            for i in 0..ls {
                let v = layer.potential[i] as i32 + layer.pending[i];
                layer.potential[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                layer.pending[i] = 0;
            }
        }
        Work::Sparse { cur, .. } => {
            for &iu in cur.iter() {
                let i = iu as usize;
                let v = layer.potential[i] as i32 + layer.pending[i];
                layer.potential[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                layer.pending[i] = 0;
            }
        }
    }

    // --- 2. inject (L0 only): override drained potential to forced-fire, clear cooldown ---
    for &a in input {
        layer.potential[a as usize] = i16::MAX;
        layer.cooldown[a as usize] = 0;
    }

    // --- readout: drain-only integrator; no decide/leak/generate/carry ---
    if layer.readout {
        return;
    }

    // --- 3. decide / fire / leak / carry (single arithmetic path; iteration set differs by mode) ---
    let (la, lb) = layer.leak;
    let cb = layer.cooldown_base;
    let bump = (layer.adapt_bump as i32) << ADAPT_SHIFT;
    let powlen = layer.pow_decay.len();

    // A closure would need to borrow layer mutably + immutably; instead inline via a macro-like helper.
    macro_rules! step {
        ($i:expr, $iu:expr, $carry:expr) => {{
            let i = $i;
            let gap = w.wrapping_sub(layer.fire_wave[i]) as usize;
            let a = if gap >= powlen { 0 } else { ((layer.adapt_ref[i] as i64 * layer.pow_decay[gap]) >> FRAC) as i32 };
            let c = layer.cooldown[i].saturating_sub(1);
            let eff = layer.threshold[i] as i32 + (a >> ADAPT_SHIFT);
            if c == 0 && layer.potential[i] as i32 >= eff {
                layer.potential[i] = 0;
                layer.cooldown[i] = cb;
                layer.adapt_ref[i] = (a + bump).min(ADAPT_MAX);
                layer.fire_wave[i] = w;
                fired.push($iu);
            } else {
                layer.cooldown[i] = c;
            }
            let pot = layer.potential[i];
            let d = (pot >> la) + (pot >> lb);
            layer.potential[i] = pot - if pot > 0 { d.max(1) } else { d };
            if $carry && (layer.potential[i] != 0 || layer.cooldown[i] != 0) {
                // carry this neuron into the next frontier (sparse only)
                if let Work::Sparse { frontier_next, .. } = &mut work {
                    frontier_next[layer_index as usize].push($iu);
                }
            }
        }};
    }

    match &work {
        Work::Dense => {
            for i in 0..ls {
                step!(i, i as u32, false);
            }
        }
        Work::Sparse { cur, .. } => {
            // `cur` is an immutable borrow of frontier_next-sibling data; copy the indices out first so
            // the macro can borrow `work` mutably for the carry. (cur points into a *different* Frontier
            // than frontier_next entries we push to, but the borrow checker needs them disjoint.)
            let idxs: &[u32] = cur;
            // SAFETY-free: iterate by value.
            let cur_vec: Vec<u32> = idxs.to_vec();
            for &iu in &cur_vec {
                step!(iu as usize, iu, true);
            }
        }
    }

    // --- 4. generate: word-scan each firer's occupancy, decode, scatter weight into target accumulator ---
    let layer_count = deliv.len() as i32;
    for &local in fired.iter() {
        let li = local as usize;
        let (sx, sy) = xy_of(local, size);
        for (lvl, entry) in layer.topology.iter().enumerate() {
            let tl = layer_index as i32 + entry.level;
            if tl < 0 || tl >= layer_count {
                continue;
            }
            let tz = tl as usize;
            let wpn = layer.occ_wpn[lvl];
            let words = &layer.occ[lvl][li * wpn..li * wpn + wpn];
            let wbase = li * layer.total_slots + layer.slot_bases[lvl];
            let lut = &layer.offsets[lvl];
            let flat = &layer.off_flat[lvl];
            let r = entry.radius;
            let hi = size.saturating_sub(r);
            let interior = sx >= r && sx < hi && sy >= r && sy < hi;
            let li_i = li as i32;
            let mut rank = 0usize;
            for (wi, &w0) in words.iter().enumerate() {
                let mut word = w0;
                let cbase = wi * 64;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    let cell = cbase + bit;
                    let widx = wbase + rank;
                    let code = (layer.codes[widx >> 5] >> ((widx & 31) * 2)) & 0b11;
                    let wt = WCODE[code as usize] as i32;
                    let target = if interior {
                        (li_i + flat[cell]) as usize
                    } else {
                        let (dx, dy) = lut[cell];
                        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize
                    };
                    deliv[tz][target] += wt;
                    if let Work::Sparse { frontier_next, .. } = &mut work {
                        frontier_next[tz].push(target as u32);
                    }
                    rank += 1;
                    word &= word - 1;
                }
            }
        }
    }
}
```

> Note on the `cur_vec` copy: it keeps the borrow checker happy (the macro borrows `layer` and `work` mutably while iterating). It is `O(activity)` and only in the sparse path. A later optimization can replace it with a split-borrow of `layer` fields (as `wave_bitnet` does) once the engine is proven — do **not** micro-optimize now. This plan prioritizes a correct, readable first cut (see the "don't dismiss on weak implementation / converge first" project note).

- [ ] **Step 4: Run the tests.**

Run: `cargo test wave_driven::wave`
Expected: PASS (`dense_firer_scatters_decoded_weights`, `sparse_matches_dense_on_one_step`). `cargo build` warning-free (the `macro_rules!` is used; if the `#[allow]` is needed for an unused-mut on `work` in a branch, add `#[allow(unused_mut)]` narrowly — but prefer no warnings).

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/wave.rs src/wave_driven/mod.rs
git commit -m "feat(wave_driven): per-layer wave step (dense + sparse, shared arithmetic)"
```

---

### Task 7: `network` — orchestration, both modes, injection, swaps, introspection

**Files:**
- Create: `src/wave_driven/network.rs`
- Modify: `src/wave_driven/mod.rs` (add `pub mod network;`)

**Interfaces:**
- Produces: `wave_driven::network::Network` with:
  - `new(Config) -> Network` (sparse), `new_with_readout(Config) -> Network` (sparse, last layer readout), `new_dense(Config) -> Network` (dense oracle, no readout).
  - `wave(&mut self, input: &[u32])`, `reset_state(&mut self)`, `size() -> u32`, `layer_count() -> usize`, `with_layer<R>(&self, z, f) -> R`, `on_layer(&mut self, z, Box<dyn Fn(usize,&[u32])+Send+Sync>)`, `clear_listeners(&mut self)`.
- Consumes: `config::Config`, `neurons::Layer`, `frontier::Frontier`, `wave::{process_layer, Work}`.

- [ ] **Step 1: Write failing tests** in `network.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::synapse::TopologyLevel;

    fn two_layer(size: u32) -> Config {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Config { seed: 9, size, layers: vec![up, top] }
    }

    #[test]
    fn l0_is_forced_transducer() {
        let net = Network::new(two_layer(8));
        net.with_layer(0, |l| {
            assert!(l.threshold.iter().all(|&t| t == i16::MAX), "L0 threshold forced to i16::MAX");
            assert_eq!(l.adapt_bump, 0, "L0 does not adapt");
        });
    }

    #[test]
    fn wave_is_deterministic() {
        let mut a = Network::new(two_layer(8));
        let mut b = Network::new(two_layer(8));
        let inputs: [&[u32]; 6] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[], &[1]];
        for inp in inputs { a.wave(inp); b.wave(inp); }
        a.with_layer(1, |la| b.with_layer(1, |lb| {
            assert_eq!(la.potential, lb.potential);
            assert_eq!(la.adapt_ref, lb.adapt_ref);
            assert_eq!(la.fire_wave, lb.fire_wave);
        }));
    }

    #[test]
    fn readout_integrates_without_firing() {
        // Last layer is a drain-only readout: it accumulates potential and never fires.
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        let cfg = Config { seed: 4, size: 8, layers: vec![up.clone(), LayerConfig { topology: vec![], ..up }] };
        let mut net = Network::new_with_readout(cfg);
        let fired_top = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let ft = fired_top.clone();
        net.on_layer(1, Box::new(move |_w, fired| *ft.lock().unwrap() += fired.len()));
        for _ in 0..12 { net.wave(&[0, 1, 2, 8, 9, 10]); }
        assert_eq!(*fired_top.lock().unwrap(), 0, "readout never fires");
        let any_pot = net.with_layer(1, |l| l.potential.iter().any(|&p| p != 0));
        assert!(any_pot, "readout integrated some potential");
    }
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::network`
Expected: FAIL (`Network` not defined).

- [ ] **Step 3: Implement `network.rs`:**

```rust
//! `network` — owns the layer stack and drives each wave. Sparse mode processes only per-layer
//! frontiers and rebuilds them; dense mode processes all neurons (the equivalence oracle). Deliveries
//! are deferred one hop: generated into `deliv`, swapped into each layer's `pending` at wave end.

use crate::wave_driven::config::Config;
use crate::wave_driven::frontier::Frontier;
use crate::wave_driven::neurons::Layer;
use crate::wave_driven::wave::{process_layer, Work};

#[derive(Clone, Copy, PartialEq)]
enum Mode { Sparse, Dense }

pub struct Network {
    size: u32,
    layers: Vec<Layer>,
    wave_id: u32,
    mode: Mode,
    fired: Vec<u32>,
    deliv: Vec<Vec<i32>>,        // per layer: NEXT wave's incoming accumulator
    frontier: Vec<Frontier>,     // per layer: current worklist (sparse only)
    frontier_next: Vec<Frontier>,// per layer: worklist being built (sparse only)
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

impl Network {
    pub fn new(config: Config) -> Network { Network::build(config, false, Mode::Sparse) }
    pub fn new_with_readout(config: Config) -> Network { Network::build(config, true, Mode::Sparse) }
    /// Dense oracle build (processes all neurons every wave; no readout). For equivalence testing.
    pub fn new_dense(config: Config) -> Network { Network::build(config, false, Mode::Dense) }

    fn build(config: Config, readout_last: bool, mode: Mode) -> Network {
        config.validate().expect("invalid config");
        let size = config.size;
        let l = config.layers.len();
        let ls = (size as usize) * (size as usize);
        let mut layers = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let mut layer = Layer::new(lc, config.seed, z as u32, size);
            if z == 0 {
                // L0 transducer: fires only on injection (baseline i16::MAX), never adapts.
                layer.threshold.iter_mut().for_each(|t| *t = i16::MAX);
                layer.adapt_bump = 0;
            }
            if readout_last && z == l - 1 {
                layer.readout = true;
            }
            layers.push(layer);
        }
        Network {
            size,
            layers,
            wave_id: 0,
            mode,
            fired: Vec::new(),
            deliv: (0..l).map(|_| vec![0i32; ls]).collect(),
            frontier: (0..l).map(|_| Frontier::new(ls)).collect(),
            frontier_next: (0..l).map(|_| Frontier::new(ls)).collect(),
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&mut self, input: &[u32]) {
        let w = self.wave_id;
        self.wave_id = self.wave_id.wrapping_add(1);
        let l = self.layers.len();
        let size = self.size;
        match self.mode {
            Mode::Dense => {
                let Self { layers, deliv, fired, listeners, .. } = self;
                for z in 0..l {
                    let inp: &[u32] = if z == 0 { input } else { &[] };
                    process_layer(&mut layers[z], z as u32, size, inp, w, Work::Dense, deliv, fired);
                    if let Some(cb) = &listeners[z] { cb(w as usize, fired); }
                }
                for z in 0..l { std::mem::swap(&mut layers[z].pending, &mut deliv[z]); }
            }
            Mode::Sparse => {
                // seed L0's current frontier with the injection sites so they are visited this wave
                for &a in input { self.frontier[0].push(a); }
                let Self { layers, deliv, fired, frontier, frontier_next, listeners, .. } = self;
                for z in 0..l {
                    let inp: &[u32] = if z == 0 { input } else { &[] };
                    let cur = &frontier[z].list;
                    process_layer(&mut layers[z], z as u32, size, inp, w, Work::Sparse { cur, frontier_next }, deliv, fired);
                    if let Some(cb) = &listeners[z] { cb(w as usize, fired); }
                }
                // deferred one hop: this wave's deliveries become next wave's pending
                for z in 0..l { std::mem::swap(&mut layers[z].pending, &mut deliv[z]); }
                // install the freshly built worklists as current; empty the consumed ones for reuse
                std::mem::swap(frontier, frontier_next);
                for f in frontier_next.iter_mut() { f.clear(); }
            }
        }
    }

    pub fn reset_state(&mut self) {
        for g in self.layers.iter_mut() {
            g.potential.iter_mut().for_each(|p| *p = 0);
            g.cooldown.iter_mut().for_each(|c| *c = 0);
            g.adapt_ref.iter_mut().for_each(|a| *a = 0);
            g.fire_wave.iter_mut().for_each(|f| *f = 0);
            g.pending.iter_mut().for_each(|p| *p = 0);
        }
        for d in self.deliv.iter_mut() { d.iter_mut().for_each(|x| *x = 0); }
        for f in self.frontier.iter_mut() { f.clear(); }
        for f in self.frontier_next.iter_mut() { f.clear(); }
        self.wave_id = 0;
    }

    pub fn size(&self) -> u32 { self.size }
    pub fn layer_count(&self) -> usize { self.layers.len() }
    pub fn with_layer<R>(&self, z: usize, f: impl FnOnce(&Layer) -> R) -> R { f(&self.layers[z]) }
    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>) {
        self.listeners[layer] = Some(listener);
    }
    pub fn clear_listeners(&mut self) { for l in self.listeners.iter_mut() { *l = None; } }
}
```

> **Reset ordering caveat (important for `wave_id`).** `reset_state` sets `wave_id = 0` and clears `fire_wave` to 0, so `gap = wave_id − fire_wave = 0` at the first post-reset wave, which yields `POW[0]` (adapt_ref intact). Since `reset_state` also zeroes `adapt_ref`, the reconstructed adaptation is 0 regardless — correct (a reset net has no adaptation history). Good.

- [ ] **Step 4: Add `pub mod network;`** to `mod.rs`. Run tests.

Run: `cargo test wave_driven::network`
Expected: PASS (`l0_is_forced_transducer`, `wave_is_deterministic`, `readout_integrates_without_firing`). `cargo build` warning-free.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/network.rs src/wave_driven/mod.rs
git commit -m "feat(wave_driven): network orchestration (sparse + dense), injection, deferred swap"
```

---

### Task 8: Equivalence oracles — sparse ≡ dense, and `adapt_bump==0` ≡ `wave_bitnet`

**Files:**
- Create: `src/wave_driven/equivalence_tests.rs` (a test-only module: `#[cfg(test)] mod equivalence_tests;` — put it behind cfg so it never ships in a normal build)
- Modify: `src/wave_driven/mod.rs` (add `#[cfg(test)] mod equivalence_tests;`)

**Interfaces:**
- Consumes: `wave_driven::{config, network, synapse}` and (for the cross-check) `crate::wave_bitnet::{config as bcfg, network as bnet, synapse as bsyn}`.

**Design note.** The sparse≡dense test builds two `Network`s from the identical `Config` (`Network::new` vs `Network::new_dense`), drives both with the same random-L0 sequence, and asserts full per-layer state equality every wave. The `wave_bitnet` cross-check builds a `wave_driven` net and a `wave_bitnet` net from field-identical configs with `adapt_bump = 0` (adaptation off ⇒ the one redefined dynamic vanishes) and asserts identical potentials and fired sets each wave.

- [ ] **Step 1: Write the failing tests** in `src/wave_driven/equivalence_tests.rs`:

```rust
//! Equivalence oracles for the event-driven engine (test-only).

use crate::wave_driven::config::{Config, LayerConfig};
use crate::wave_driven::network::Network;
use crate::wave_driven::synapse::{random_l0_input, TopologyLevel};

fn deep_cfg(size: u32, adapt_bump: i16) -> Config {
    let mk = |topology| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump, adapt_decay: 6 };
    // 4 computational layers + empty top; forward +1, with a level-0 recurrent edge on L2 and a
    // level -1 feedback edge on L3 to exercise cross-layer waking and L0 feedback.
    let layers = vec![
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: 0, radius: 1, count: 3 }]),
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: -1, radius: 1, count: 2 }]),
        mk(vec![]),
    ];
    Config { seed: 0xABCD_1234, size, layers }
}

fn assert_layers_eq(a: &Network, b: &Network, wave: usize) {
    for z in 0..a.layer_count() {
        a.with_layer(z, |la| b.with_layer(z, |lb| {
            assert_eq!(la.potential, lb.potential, "wave {wave} layer {z} potential");
            assert_eq!(la.cooldown, lb.cooldown, "wave {wave} layer {z} cooldown");
            assert_eq!(la.adapt_ref, lb.adapt_ref, "wave {wave} layer {z} adapt_ref");
            assert_eq!(la.fire_wave, lb.fire_wave, "wave {wave} layer {z} fire_wave");
        }));
    }
}

#[test]
fn sparse_equals_dense_over_random_drive() {
    let size = 16u32;
    let mut sparse = Network::new(deep_cfg(size, 5));
    let mut dense = Network::new_dense(deep_cfg(size, 5));
    let input = random_l0_input(0xABCD_1234, size, 12000);
    for wv in 0..300 {
        let inp = input(wv);
        sparse.wave(&inp);
        dense.wave(&inp);
        assert_layers_eq(&sparse, &dense, wv);
    }
}

#[test]
fn sparse_equals_dense_with_bursty_gaps() {
    // Long input-free gaps stress lazy adaptation catch-up (large `gap` on wake).
    let size = 16u32;
    let mut sparse = Network::new(deep_cfg(size, 5));
    let mut dense = Network::new_dense(deep_cfg(size, 5));
    let input = random_l0_input(0x5151, size, 30000);
    for wv in 0..400 {
        let inp = if wv % 40 < 4 { input(wv) } else { Vec::new() }; // bursts then silence
        sparse.wave(&inp);
        dense.wave(&inp);
        assert_layers_eq(&sparse, &dense, wv);
    }
}
```

- [ ] **Step 2: Add the `wave_bitnet` cross-check** in the same file. It builds equivalent configs on both engines (`adapt_bump = 0`) and compares potentials + fired sets via listeners:

```rust
#[test]
fn adapt_bump_zero_matches_wave_bitnet() {
    use crate::wave_bitnet::config::{Config as BConfig, LayerConfig as BLayerConfig};
    use crate::wave_bitnet::network::Network as BNetwork;
    use crate::wave_bitnet::synapse::{random_l0_input as brandom, TopologyLevel as BTop};
    use std::sync::{Arc, Mutex};

    let size = 16u32;
    let seed = 0xFEED_BEEFu64;
    // field-identical layer configs, adapt_bump 0 (LIF; adaptation off)
    let dtop = |lvl, r, c| TopologyLevel { level: lvl, radius: r, count: c };
    let btop = |lvl, r, c| BTop { level: lvl, radius: r, count: c };
    let d_layers = vec![
        LayerConfig { topology: vec![dtop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        LayerConfig { topology: vec![dtop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        LayerConfig { topology: vec![], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
    ];
    let b_layers = vec![
        BLayerConfig { topology: vec![btop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        BLayerConfig { topology: vec![btop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        BLayerConfig { topology: vec![], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
    ];
    let mut dn = Network::new(Config { seed, size, layers: d_layers });
    let mut bn = BNetwork::new(BConfig { seed, size, layers: b_layers });

    let dfire: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); 3]));
    let bfire: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); 3]));
    for z in 0..3 {
        let d = dfire.clone(); dn.on_layer(z, Box::new(move |_w, f| d.lock().unwrap()[z] = f.to_vec()));
        let b = bfire.clone(); bn.on_layer(z, Box::new(move |_w, f| b.lock().unwrap()[z] = f.to_vec()));
    }

    // both use their own random_l0_input; the generators are copies, so identical sites for equal args.
    let dinput = random_l0_input(seed, size, 12000);
    let binput = brandom(seed, size, 12000);
    for wv in 0..200 {
        let di = dinput(wv);
        let bi = binput(wv);
        assert_eq!(di, bi, "L0 drive generators agree at wave {wv}");
        dn.wave(&di);
        bn.wave(&bi);
        // fired sets per layer (as sorted sets, since frontier vs dense may differ in order)
        for z in 0..3 {
            let mut df = dfire.lock().unwrap()[z].clone(); df.sort_unstable();
            let mut bf = bfire.lock().unwrap()[z].clone(); bf.sort_unstable();
            assert_eq!(df, bf, "wave {wv} layer {z} fired set");
        }
        // potentials per computational layer
        for z in 0..3 {
            let dp = dn.with_layer(z, |l| l.potential.clone());
            let bp = bn.with_layer(z, |l| l.potential.clone());
            assert_eq!(dp, bp, "wave {wv} layer {z} potential");
        }
    }
}
```

- [ ] **Step 3: Wire the module** — add to `src/wave_driven/mod.rs`:

```rust
#[cfg(test)]
mod equivalence_tests;
```

- [ ] **Step 4: Run to verify failure first, then pass.** The tests may pass immediately if Tasks 3–7 are correct; run and fix real bugs (in the engine, never by weakening the assert):

Run: `cargo test wave_driven::equivalence_tests -- --nocapture`
Expected: initially may FAIL if a frontier-completeness or drain/generate bug exists → fix in `wave.rs`/`network.rs`; then PASS all three. If `adapt_bump_zero_matches_wave_bitnet` reveals a drift, the likely causes are code-init order or the leak expression — compare against `wave_bitnet::wave` line-for-line.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/equivalence_tests.rs src/wave_driven/mod.rs
git commit -m "test(wave_driven): sparse==dense oracle + adapt_bump==0 wave_bitnet cross-check"
```

---

### Task 9: Throughput bench + profiling example (the payoff)

**Files:**
- Create: `benches/throughput_driven.rs`
- Create: `examples/profile_driven.rs`
- Modify: `Cargo.toml` (register the bench)

**Interfaces:**
- Consumes: `wave_driven::{config, network, synapse}` public API only.

**Design note.** Unlike the `wave_bitnet` bench (which trains + caches a model), Phase 1 has no training, so throughput is measured on the **procedural ±1 init** at a chosen L0 drive fraction. The bench sweeps activity (drive fraction) and prints per-layer firing rates + waves/s so the activity-scaling is visible; it also times the **dense** mode at the same operating point to expose the sparse/dense crossover.

- [ ] **Step 1: Create `examples/profile_driven.rs`** (perf/flamegraph target — a tight sparse wave loop):

```rust
//! Profiling target for the wave_driven sparse forward loop. Same 32×32×5 forward config as the
//! wave_bitnet profiler, under random L0 drive. Build: `cargo build --profile profiling --example
//! profile_driven`; run `./target/profiling/examples/profile_driven [n_waves]`.

use wave_net::wave_driven::config::{Config, LayerConfig};
use wave_net::wave_driven::network::Network;
use wave_net::wave_driven::synapse::{random_l0_input, TopologyLevel};

fn main() {
    let n_waves: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let (size, seed) = (32u32, 0xC0FFEE_1234_5678u64);
    let layer = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
        leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32,
        baseline_init: 6, adapt_bump: 5, adapt_decay: 6,
    };
    let mut net = Network::new(Config { seed, size, layers: vec![layer; 5] });
    let input = random_l0_input(seed, size, 8000); // ~12% L0 drive
    let noise: Vec<Vec<u32>> = (0..256).map(&input).collect();
    for w in 0..64 { net.wave(&noise[w % noise.len()]); }
    for i in 0..n_waves { net.wave(&noise[i % noise.len()]); }
    let sink: i64 = net.with_layer(4, |l| l.potential.iter().map(|&p| p as i64).sum());
    println!("ran {n_waves} waves; sink={sink}");
}
```

- [ ] **Step 2: Create `benches/throughput_driven.rs`:**

```rust
//! Throughput benchmark for the wave_driven event-driven engine on the procedural ±1 init. Sweeps L0
//! drive fraction (activity) and times both sparse and dense modes so the activity-scaling and the
//! sparse/dense crossover are visible. No training in Phase 1.

use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use wave_net::wave_driven::config::{Config, LayerConfig};
use wave_net::wave_driven::network::Network;
use wave_net::wave_driven::synapse::{random_l0_input, TopologyLevel};

const SIZE: u32 = 32;
const SEED: u64 = 0xC0FFEE_1234_5678;
const WAVES_PER_ITER: u64 = 256;

fn cfg() -> Config {
    let layer = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
        leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32,
        baseline_init: 6, adapt_bump: 5, adapt_decay: 6,
    };
    Config { seed: SEED, size: SIZE, layers: vec![layer; 5] }
}

fn measure_rates(net: &mut Network, warmup: usize, waves: usize, input: &impl Fn(usize) -> Vec<u32>) -> Vec<f64> {
    let l = net.layer_count();
    let counts = Arc::new(Mutex::new(vec![0u64; l]));
    for z in 0..l {
        let c = counts.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| { c.lock().unwrap()[z] += fired.len() as u64; }));
    }
    net.reset_state();
    for w in 0..warmup { net.wave(&input(w)); }
    counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
    for w in 0..waves { net.wave(&input(warmup + w)); }
    net.clear_listeners();
    let counts = std::mem::take(&mut *counts.lock().unwrap());
    let denom = ((net.size() as u64) * (net.size() as u64) * waves as u64) as f64;
    counts.iter().map(|&s| s as f64 / denom).collect()
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_driven");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    for &frac in &[2000u32, 8000, 30000] {
        let input = random_l0_input(SEED, SIZE, frac);
        let noise: Vec<Vec<u32>> = (0..WAVES_PER_ITER as usize).map(&input).collect();

        let mut probe = Network::new(cfg());
        let rates = measure_rates(&mut probe, 32, 128, &input);
        let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
        println!("driven 32x32x5 drive_q16={frac} per-layer rate (%): {pct:?}");

        let mut sparse = Network::new(cfg());
        group.bench_function(format!("sparse_q16_{frac}"), |b| {
            b.iter(|| { for v in &noise { sparse.wave(v); } })
        });
        let mut dense = Network::new_dense(cfg());
        group.bench_function(format!("dense_q16_{frac}"), |b| {
            b.iter(|| { for v in &noise { dense.wave(v); } })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
```

- [ ] **Step 3: Register the bench** in `Cargo.toml` after the existing `[[bench]]` block:

```toml
[[bench]]
name = "throughput_driven"
harness = false
```

- [ ] **Step 4: Verify both compile and the example runs a smoke workload.**

Run: `cargo build --example profile_driven && cargo run --release --example profile_driven 20000`
Expected: prints `ran 20000 waves; sink=...`. Then `cargo build --benches` compiles `throughput_driven` (do not run the full criterion suite here).

- [ ] **Step 5: Commit.**

```bash
git add benches/throughput_driven.rs examples/profile_driven.rs Cargo.toml
git commit -m "feat(wave_driven): throughput bench + profiling example (activity vs size)"
```

---

### Task 10: Documentation — architecture map

**Files:**
- Modify: `AGENTS.md` (architecture map + a one-paragraph pointer)
- Modify: `src/lib.rs` (crate doc mentions the second engine)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `src/lib.rs` crate doc.** After the existing `wave_bitnet` sentence, add:

```rust
//! A second, independent engine [`wave_driven`] explores **event-driven active-set inference**: each
//! wave processes only a per-layer frontier of non-quiescent neurons, with lazy fire-anchored
//! adaptation, so cost scales with activity rather than layer size (Phase 1: inference only).
```

And add the module line already present from Task 1 (`pub mod wave_driven;`).

- [ ] **Step 2: Update `AGENTS.md`** — add a `wave_driven/` block to the Architecture map (after the `wave_bitnet/` block, before `bench/`), matching the existing style:

```
  wave_driven/           # NEW engine (Phase 1): event-driven active-set INFERENCE, independent of wave_bitnet
    synapse.rs           # copied hash/topology helpers
    config.rs            # copied Config/LayerConfig (adapt_decay now sets ρ = 1 − 2^−decay; no dead-zone bound)
    neurons.rs           # Layer SoA state + occupancy bitset + 2-bit codes + fire-anchored adapt (adapt_ref/fire_wave) + geometric decay table
    frontier.rs          # Frontier: worklist Vec + dedup mark bitset (GPU unique-append primitive)
    wave.rs              # process_layer(Work::Sparse|Dense) — frontier step + the dense equivalence oracle
    network.rs           # Network: sparse/dense orchestration, injection-into-frontier, deferred one-hop swap
    equivalence_tests.rs # (test-only) sparse==dense oracle + adapt_bump==0 wave_bitnet cross-check
```

Add one sentence near the top "The two modules" section noting there is now a second, **inference-only** engine `wave_driven` whose cost scales with activity (spec: `docs/superpowers/specs/2026-07-13-wave-driven-event-active-set-design.md`), and that `wave_bitnet` remains the trainable engine.

- [ ] **Step 3: Verify build + full test suite still green.**

Run: `cargo build && cargo test`
Expected: warning-free build; all tests (both engines) pass.

- [ ] **Step 4: Commit.**

```bash
git add AGENTS.md src/lib.rs
git commit -m "docs(wave_driven): add engine to the architecture map"
```

---

## Self-Review

**Spec coverage:**
- Frontier + mark bitset → Task 5. Sparse accumulator (dense storage, inline-clear via frontier coverage) → Task 6 (drain clears `pending[i]`) + Task 7 (deferred swap). Lazy fire-anchored adaptation → Tasks 3–4. Per-layer-type quiescence (normal/L0/readout) → Task 6 (readout early return; carry predicate excludes adapt) + Task 7 (L0 transducer). Deferred one-hop propagation → Task 7 swap. Determinism → Task 7 test. GPU-friendly structure (bitset worklist, atomic-shaped push, SoA) → Task 5/6 by construction. Validation items 1–4 → Tasks 8 (oracles) and 9 (throughput/crossover). Module layout + API → Tasks 1–7. Non-goals (no training/persist/procedural/GPU) → honored (none built). **All spec sections map to a task.**

**Placeholder scan:** No "TBD/TODO" remain except the deliberately-resolved note in Task 2 (hardcode `24`, then swap to `MAX_ADAPT_DECAY` in Task 3 Step 4) — that is a concrete, ordered instruction, not a placeholder. Every code step shows full code.

**Type consistency:** `Frontier::push -> bool`, `.list`, `.clear()` used consistently (Tasks 5–7). `process_layer(layer, layer_index, size, input, w, Work, deliv, fired)` signature identical across Tasks 6–7. `Work::Sparse { cur, frontier_next }` fields match between definition (Task 6) and construction (Task 7). `Network::{new,new_with_readout,new_dense}` used consistently (Tasks 7–9). `decayed_adapt(i, w)`, `adapt_ref`, `fire_wave`, `pow_decay`, `ADAPT_SHIFT/ADAPT_MAX/FRAC/MAX_ADAPT_DECAY` names consistent (Tasks 3–8). `wave_id: u32` matches `fire_wave: u32` and the `w: u32` threaded into `process_layer`.

**Known follow-ups (out of Phase 1 scope, do not implement now):** the `cur_vec` copy in the sparse path (Task 6) is a readability-first choice; replacing it with a split-borrow is a proven-engine optimization. The dense `0..ls` iteration allocates nothing but is `O(size²)` by definition (oracle only).
