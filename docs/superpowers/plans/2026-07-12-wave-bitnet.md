# wave_bitnet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fourth top-level engine module, `src/wave_bitnet/`, that trains pure ±1/0 ternary nets with the multi-layer-DFA rule while storing topology as a per-neuron neighborhood occupancy bitset (no per-wave hashing) and weights as 2-bit packed values plus an f32 training shadow.

**Architecture:** Clean fork of `wave_net` (duplication intended; `wave_net` stays frozen). Neuron dynamics, hash helpers, and the Network wave loop are copied verbatim; the topology/weight *storage* and the synapse *generation* are replaced. Targets are materialized once at startup (distinct-cell Fisher-Yates fill into a bitset) and decoded arithmetically thereafter. Training keeps the proven shadow-based update; only its indexing changes.

**Tech Stack:** Rust, std-only, no external crates. Deterministic integer engine.

## Global Constraints

- std-only; no new dependencies.
- Warning-free `cargo build` and `cargo build --tests`.
- Determinism: all output is a pure function of `(seed, config, input)`.
- One commit per task; no `Co-Authored-By` trailer.
- Work on branch `feat/wave-bitnet` (already created; spec committed there).
- Pure ternary only: no `WeightQuant` enum, no scaled ±g, no int8.
- `adapt_bump` default for computational layers is **5**; L0 is forced to a non-adapting transducer (`threshold = i16::MAX`, `adapt_bump = 0`).
- Weight = 2 bits: `nonzero` mask + `sign`. Prune threshold `t` default **0.5**.
- Per-neuron-per-level invariant: exactly `count` distinct wired cells; `total_slots = Σ count`; weight index for the r-th wired synapse of level ℓ is `i*total_slots + slot_base(ℓ) + r`.
- Validation: `count ≤ (2*radius+1)²` per level, enforced in `Config::validate()` and called by `Network::new`.

**Reference files to copy from (verbatim unless noted):** `src/wave_net/synapse.rs`, `src/wave_net/config.rs`, `src/wave_net/neurons.rs`, `src/wave_net/wave.rs`, `src/wave_net/network.rs`, `src/wave_net/multilayer_dfa.rs`, `src/bench/multilayer_dfa.rs` (harness), `src/bench/multilayer_dfa_bitnet_bench.rs` (benchmark shape).

---

### Task 1: Scaffold module + `bits.rs` (minimal bitset)

**Files:**
- Modify: `src/lib.rs` (add module declaration)
- Create: `src/wave_bitnet/mod.rs`
- Create: `src/wave_bitnet/bits.rs`

**Interfaces:**
- Produces: `struct BitSet` with `BitSet::zeros(n_bits: usize) -> BitSet`, `fn set(&mut self, i: usize)`, `fn get(&self, i: usize) -> bool`, `fn count_ones(&self) -> usize`, `fn iter_set_in(&self, start: usize, len: usize) -> impl Iterator<Item = usize> + '_` (yields set-bit offsets **relative to `start`**, ascending).

- [ ] **Step 1: Add the module to lib.rs and create empty mod.rs**

In `src/lib.rs`, after the `pub mod wave_state_machine;` line, add:
```rust
pub mod wave_bitnet; // memory-lean, ternary-native fork of wave_net (bitset topology + 2-bit weights)
```

Create `src/wave_bitnet/mod.rs`:
```rust
//! `wave_bitnet` — a memory-lean, ternary-native fork of `wave_net`. Pure ±1/0 weights stored as
//! 2-bit packed values; topology materialized once at startup into a per-neuron neighborhood
//! occupancy bitset (no per-wave hashing). `wave_net` is the frozen reference; duplication is intended.
//! Spec: docs/superpowers/specs/2026-07-12-wave-bitnet-design.md.

pub mod bits;
```

- [ ] **Step 2: Write the failing test for BitSet**

Create `src/wave_bitnet/bits.rs` with a test module:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_count_and_iter() {
        let mut b = BitSet::zeros(200);
        assert_eq!(b.count_ones(), 0);
        assert!(!b.get(5));
        for &i in &[5usize, 63, 64, 130, 199] {
            b.set(i);
        }
        assert!(b.get(64) && b.get(199) && !b.get(0));
        assert_eq!(b.count_ones(), 5);
        // iterate the neighborhood slice [64, 64+80): global 64 and 130 -> offsets 0 and 66.
        let got: Vec<usize> = b.iter_set_in(64, 80).collect();
        assert_eq!(got, vec![0, 66]);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib wave_bitnet::bits`
Expected: FAIL — `BitSet` not found.

- [ ] **Step 4: Implement BitSet**

Prepend to `src/wave_bitnet/bits.rs` (above the test module):
```rust
//! Minimal fixed-size bitset over `Vec<u64>` (std-only).

#[derive(Clone, Debug)]
pub struct BitSet {
    words: Vec<u64>,
    n_bits: usize,
}

impl BitSet {
    pub fn zeros(n_bits: usize) -> BitSet {
        BitSet { words: vec![0u64; (n_bits + 63) / 64], n_bits }
    }
    #[inline]
    pub fn set(&mut self, i: usize) {
        debug_assert!(i < self.n_bits);
        self.words[i >> 6] |= 1u64 << (i & 63);
    }
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.n_bits);
        (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
    /// Set-bit offsets **relative to `start`** within `[start, start+len)`, ascending.
    pub fn iter_set_in(&self, start: usize, len: usize) -> impl Iterator<Item = usize> + '_ {
        (0..len).filter(move |&o| self.get(start + o))
    }
}
```
(Note: the simple `iter_set_in` is O(len); the occupancy slices are small (≤ 121) and it is called per firing neuron, which is acceptable for the first cut. A word-scan optimization can come later if the smoke benchmark shows it hot.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib wave_bitnet::bits` → Expected: PASS. Then `cargo build` → Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/wave_bitnet/mod.rs src/wave_bitnet/bits.rs
git commit -m "feat(wave_bitnet): scaffold module + minimal BitSet"
```

---

### Task 2: `synapse.rs` — hash helpers (copied) + cell decode + distinct-cell sampling

**Files:**
- Create: `src/wave_bitnet/synapse.rs`
- Modify: `src/wave_bitnet/mod.rs` (add `pub mod synapse;`)

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - Verbatim re-exports: `TopologyLevel { level: i32, radius: u32, count: u32 }`, `Synapse { target: u32, weight: i16 }`, `mix`, `key`, `map_range`, `local_of`, `xy_of`, `wrap`, `P_TARGET`.
  - `fn neigh_size(radius: u32) -> usize` = `(2*radius+1)²`.
  - `fn decode_cell(cell: usize, src_local: u32, radius: u32, size: u32) -> u32` — target local index for neighborhood `cell`.
  - `fn sample_distinct_cells(seed: u64, source_global: u32, level: i32, radius: u32, count: u32) -> Vec<u32>` — `count` distinct cell indices in `0..neigh_size(radius)`, deterministic.

- [ ] **Step 1: Copy the verbatim helpers**

Create `src/wave_bitnet/synapse.rs`. Copy **verbatim** from `src/wave_net/synapse.rs` (lines 1–98, i.e. through `target_of`, BUT do NOT copy `target_of` or `generate_into`): the `TopologyLevel` and `Synapse` structs, `P_TARGET`/`P_THRESHOLD`/`P_INPUT`, both `mix` variants with their `#[cfg]`, `GOLDEN`, `key`, `map_range`, `map_range24`, `local_of`, `xy_of`, `wrap`. Add `pub mod synapse;` to `src/wave_bitnet/mod.rs`.

- [ ] **Step 2: Write the failing tests for decode + sample**

Append to `src/wave_bitnet/synapse.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_center_is_self_and_corners_wrap() {
        let size = 8u32;
        let r = 2u32;
        let span = 2 * r + 1; // 5, so N = 25, center cell index = 12 (dx=dy=0)
        let src = local_of(3, 4, size);
        assert_eq!(decode_cell(12, src, r, size), src, "center cell maps to self");
        // cell 0 -> dx=-2, dy=-2 -> (3-2, 4-2) = (1, 2)
        assert_eq!(decode_cell(0, src, r, size), local_of(1, 2, size));
        // last cell (span*span-1 = 24) -> dx=+2, dy=+2 -> (5, 6)
        assert_eq!(decode_cell((span * span - 1) as usize, src, r, size), local_of(5, 6, size));
    }

    #[test]
    fn sample_is_distinct_bounded_and_deterministic() {
        let (seed, sg, level, r, count) = (0xABCDu64, 700u32, 1i32, 4u32, 48u32);
        let a = sample_distinct_cells(seed, sg, level, r, count);
        let b = sample_distinct_cells(seed, sg, level, r, count);
        assert_eq!(a, b, "deterministic");
        assert_eq!(a.len(), count as usize, "exactly count cells");
        let n = neigh_size(r);
        assert!(a.iter().all(|&c| (c as usize) < n), "all in range");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), count as usize, "all distinct");
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib wave_bitnet::synapse` → Expected: FAIL (`decode_cell`/`sample_distinct_cells`/`neigh_size` not found).

- [ ] **Step 4: Implement decode + sample**

Add to `src/wave_bitnet/synapse.rs` (above the tests):
```rust
/// Number of cells in a radius-`r` neighborhood: `(2r+1)²`.
#[inline]
pub fn neigh_size(radius: u32) -> usize {
    let span = (2 * radius + 1) as usize;
    span * span
}

/// Target local index for neighborhood `cell` of a source at `src_local`. Cell layout is
/// row-major over the `(2r+1)×(2r+1)` window centered on the source; pure arithmetic, no hash.
pub fn decode_cell(cell: usize, src_local: u32, radius: u32, size: u32) -> u32 {
    let (sx, sy) = xy_of(src_local, size);
    let span = 2 * radius + 1;
    let dx = (cell as u32 % span) as i32 - radius as i32;
    let dy = (cell as u32 / span) as i32 - radius as i32;
    local_of(wrap(sx, dx, size), wrap(sy, dy, size), size)
}

/// `count` DISTINCT cell indices in `0..neigh_size(radius)`, via a partial Fisher-Yates shuffle of the
/// cell indices seeded by the hash stream (one draw per swap). Deterministic; `count` must be
/// `<= neigh_size(radius)` (guaranteed by `Config::validate`).
pub fn sample_distinct_cells(seed: u64, source_global: u32, level: i32, radius: u32, count: u32) -> Vec<u32> {
    let n = neigh_size(radius);
    debug_assert!(count as usize <= n);
    let mut idx: Vec<u32> = (0..n as u32).collect();
    for k in 0..(count as usize) {
        let h = mix(key(seed, source_global, level, k as u32, P_TARGET));
        // pick j in [k, n) without modulo bias, swap into position k
        let j = k + map_range((h >> 32) as u32, (n - k) as u32) as usize;
        idx.swap(k, j);
    }
    idx.truncate(count as usize);
    idx
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib wave_bitnet::synapse` → PASS. `cargo build` → no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/wave_bitnet/synapse.rs src/wave_bitnet/mod.rs
git commit -m "feat(wave_bitnet): hash helpers + cell decode + distinct-cell sampling"
```

---

### Task 3: `config.rs` — Config/LayerConfig + validate

**Files:**
- Create: `src/wave_bitnet/config.rs`
- Modify: `src/wave_bitnet/mod.rs` (add `pub mod config;`)

**Interfaces:**
- Consumes: `synapse::{TopologyLevel, neigh_size}`.
- Produces: `LayerConfig { topology: Vec<TopologyLevel>, leak: (u8,u8), cooldown_base: u8, inhibitor_ratio: u32, threshold_jitter: u16, baseline_init: i16, adapt_bump: i16, adapt_decay: u8 }`, `Config { seed: u64, size: u32, layers: Vec<LayerConfig> }`, `Config::layer_size`, `Config::validate() -> Result<(), String>`, `Config::demo()`.

- [ ] **Step 1: Copy config structure and write the failing validate test**

Create `src/wave_bitnet/config.rs`. Copy the `THRESHOLD_JITTER_DEFAULT` const, `LayerConfig` struct, `Config` struct, `Config::layer_size`, `Config::n_total`, and `Config::demo()` **verbatim** from `src/wave_net/config.rs` (adjust the `use` to `use crate::wave_bitnet::synapse::TopologyLevel;`). Then copy `Config::validate()` verbatim and extend it. Add `pub mod config;` to mod.rs. Append this test:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::synapse::TopologyLevel;

    fn cfg_with(topology: Vec<TopologyLevel>) -> Config {
        let lc = LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: 5,
            adapt_decay: 6,
        };
        Config { seed: 1, size: 8, layers: vec![lc, LayerConfig { topology: vec![], ..cfg_lc() }] }
    }
    fn cfg_lc() -> LayerConfig {
        LayerConfig { topology: vec![], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 }
    }

    #[test]
    fn validate_accepts_fan_in_within_neighborhood() {
        // r2 -> N=25; count 16 <= 25 ok
        let c = cfg_with(vec![TopologyLevel { level: 1, radius: 2, count: 16 }]);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_rejects_fan_in_over_neighborhood() {
        // r2 -> N=25; count 30 > 25 -> Err
        let c = cfg_with(vec![TopologyLevel { level: 1, radius: 2, count: 30 }]);
        let e = c.validate().unwrap_err();
        assert!(e.contains("count") && e.contains("neighborhood"), "descriptive error: {e}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib wave_bitnet::config` → Expected: FAIL (the fan-in check rejects nothing yet).

- [ ] **Step 3: Extend validate() with the fan-in check**

In the copied `Config::validate()`, before the final `Ok(())`, add:
```rust
        for (z, lc) in self.layers.iter().enumerate() {
            for t in &lc.topology {
                let n = crate::wave_bitnet::synapse::neigh_size(t.radius);
                if t.count as usize > n {
                    return Err(format!(
                        "layer {z}: topology count {} exceeds neighborhood size {} for radius {} \
                         (a per-cell occupancy bitset caps fan-in at (2r+1)^2)",
                        t.count, n, t.radius
                    ));
                }
            }
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib wave_bitnet::config` → PASS. `cargo build` → no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/wave_bitnet/config.rs src/wave_bitnet/mod.rs
git commit -m "feat(wave_bitnet): Config/LayerConfig + fan-in<=neighborhood validation"
```

---

### Task 4: `neurons.rs` — Layer with the bitset representation

**Files:**
- Create: `src/wave_bitnet/neurons.rs`
- Modify: `src/wave_bitnet/mod.rs` (add `pub mod neurons;`)

**Interfaces:**
- Consumes: `bits::BitSet`, `config::LayerConfig`, `synapse::{sample_distinct_cells, neigh_size}`.
- Produces: `pub const ADAPT_SHIFT: u32 = 12;`, `pub const ADAPT_MAX: i32`, `struct Layer` with fields below, `Layer::new(cfg, layer_index, seed, size) -> Layer`, `Layer::weight_at(&self, widx: usize) -> i8`, `Layer::repack_row(&mut self, i: usize)`, `Layer::slot_base(&self, level_idx: usize) -> usize`, plus per-level `neigh` and `radius` accessors used by wave/network.

- [ ] **Step 1: Copy neuron state + constants, define the Layer struct**

Create `src/wave_bitnet/neurons.rs`. Copy **verbatim** from `src/wave_net/neurons.rs`: `ADAPT_SHIFT`, `ADAPT_MAX` consts and their doc comments. Then define the fork's `Layer` (this REPLACES wave_net's weight fields with the bitset representation; keep all neuron/eligibility state fields identical):
```rust
use crate::wave_bitnet::bits::BitSet;
use crate::wave_bitnet::config::LayerConfig;
use crate::wave_bitnet::synapse::{neigh_size, sample_distinct_cells, Synapse};

pub struct Layer {
    // neuron state (identical to wave_net::neurons::Layer)
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub adapt: Vec<i32>,
    pub threshold: Vec<i16>,
    pub inbox: Vec<Synapse>,
    // eligibility / decide-step state (identical to wave_net)
    pub elig_pre: Vec<i32>,
    pub elig_post: Vec<i32>,
    pub decide_potential: Vec<i16>,
    pub decide_eff: Vec<i32>,
    // config
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<crate::wave_bitnet::synapse::TopologyLevel>,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub readout: bool,
    pub ternary_threshold: f32,
    // derived layout
    pub total_slots: usize,          // Σ count
    pub slot_bases: Vec<usize>,      // per level: Σ_{ℓ'<ℓ} count
    pub neigh: Vec<usize>,           // per level: (2r+1)²
    // BITSET representation
    pub occupancy: Vec<BitSet>,      // per level: ls·neigh[ℓ] bits
    pub w_nonzero: BitSet,           // ls·total_slots bits
    pub w_sign: BitSet,              // ls·total_slots bits
    pub shadow: Vec<f32>,            // ls·total_slots
}
```

- [ ] **Step 2: Write failing tests for construction + repack**

Append:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::LayerConfig;
    use crate::wave_bitnet::synapse::TopologyLevel;

    fn lc(topology: Vec<TopologyLevel>) -> LayerConfig {
        LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 }
    }

    #[test]
    fn new_wires_exactly_count_distinct_cells_deterministically() {
        let size = 8u32;
        let ls = (size * size) as usize;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 16 }]);
        let a = Layer::new(&cfg, 0, 7, size);
        let b = Layer::new(&cfg, 0, 7, size);
        assert_eq!(a.total_slots, 16);
        // exactly 16 set bits per neuron in the single level's occupancy
        for i in 0..ls {
            let set = a.occupancy[0].iter_set_in(i * a.neigh[0], a.neigh[0]).count();
            assert_eq!(set, 16, "neuron {i} wires exactly count cells");
        }
        assert_eq!(a.occupancy[0].count_ones(), b.occupancy[0].count_ones());
    }

    #[test]
    fn repack_roundtrips_shadow_to_ternary() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let mut l = Layer::new(&cfg, 0, 7, size);
        // set neuron 0's row shadow to [2.0, -3.0, 0.05, 0.0]; γ = mean(|.|) = 1.2625; t=0.5
        // |x|/γ: 1.58, 2.38, 0.04, 0.0  -> nonzero mask [1,1,0,0], signs [+,-,.,.]
        let ts = l.total_slots;
        l.shadow[0 * ts + 0] = 2.0;
        l.shadow[0 * ts + 1] = -3.0;
        l.shadow[0 * ts + 2] = 0.05;
        l.shadow[0 * ts + 3] = 0.0;
        l.repack_row(0);
        assert_eq!(l.weight_at(0 * ts + 0), 1);
        assert_eq!(l.weight_at(0 * ts + 1), -1);
        assert_eq!(l.weight_at(0 * ts + 2), 0);
        assert_eq!(l.weight_at(0 * ts + 3), 0);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib wave_bitnet::neurons` → FAIL (`Layer::new`/`repack_row`/`weight_at` missing).

- [ ] **Step 4: Implement Layer::new, weight_at, slot_base, repack_row**

Add (above the tests). Copy the per-neuron threshold-jitter init logic from `wave_net::neurons::Layer::new` verbatim for the `threshold` vector; the rest is new:
```rust
impl Layer {
    pub fn slot_base(&self, level_idx: usize) -> usize {
        self.slot_bases[level_idx]
    }

    #[inline]
    pub fn weight_at(&self, widx: usize) -> i8 {
        if !self.w_nonzero.get(widx) { 0 } else if self.w_sign.get(widx) { 1 } else { -1 }
    }

    /// Requantise neuron `i`'s row (all `total_slots` shadow values) into `w_nonzero`/`w_sign`:
    /// γ = mean(|shadow|) over the row; `|shadow|/γ < t → 0`, else sign(shadow).
    pub fn repack_row(&mut self, i: usize) {
        let ts = self.total_slots;
        if ts == 0 { return; }
        let base = i * ts;
        let mut sum = 0.0f32;
        for s in 0..ts { sum += self.shadow[base + s].abs(); }
        let gamma = sum / ts as f32;
        let t = self.ternary_threshold;
        for s in 0..ts {
            let sh = self.shadow[base + s];
            let x = if gamma <= 0.0 { 0.0 } else { sh / gamma };
            let idx = base + s;
            if x.abs() < t {
                // clear nonzero (leave sign bit as-is; it is meaningless when nonzero=0)
                // BitSet has no clear; store via reconstruct — simplest: track and set below.
                self.set_weight_bits(idx, false, false);
            } else {
                self.set_weight_bits(idx, true, x > 0.0);
            }
        }
    }

    #[inline]
    fn set_weight_bits(&mut self, idx: usize, nonzero: bool, sign: bool) {
        // BitSet only sets; clearing needs a mutable word op. Add clear support in bits.rs (Step 4b).
        self.w_nonzero.put(idx, nonzero);
        self.w_sign.put(idx, sign);
    }

    pub fn new(cfg: &LayerConfig, layer_index: u32, seed: u64, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let n_levels = cfg.topology.len();
        let mut slot_bases = Vec::with_capacity(n_levels);
        let mut neigh = Vec::with_capacity(n_levels);
        let mut total_slots = 0usize;
        for t in &cfg.topology {
            slot_bases.push(total_slots);
            neigh.push(neigh_size(t.radius));
            total_slots += t.count as usize;
        }
        // occupancy: fill distinct cells per neuron per level
        let mut occupancy: Vec<BitSet> = neigh.iter().map(|&n| BitSet::zeros(ls * n)).collect();
        let base_global = layer_index as usize * ls;
        for (li, t) in cfg.topology.iter().enumerate() {
            let n = neigh[li];
            for i in 0..ls {
                let sg = (base_global + i) as u32;
                for &cell in &sample_distinct_cells(seed, sg, t.level, t.radius, t.count) {
                    occupancy[li].set(i * n + cell as usize);
                }
            }
        }
        // shadow init: ±1 sign from inhibitor_ratio (copy wave_net's sign rule), packed below
        let mut shadow = vec![0f32; ls * total_slots];
        for i in 0..ls {
            let sg = (base_global + i) as u32;
            for li in 0..n_levels {
                let t = &cfg.topology[li];
                for r in 0..(t.count as usize) {
                    // inhibitory iff (hash & 0xFFFF) < inhibitor_ratio  (mirror wave_net init)
                    let h = crate::wave_bitnet::synapse::mix(crate::wave_bitnet::synapse::key(seed, sg, t.level, r as u32, 7));
                    let sign = if (h & 0xFFFF) < cfg.inhibitor_ratio as u64 { -1.0 } else { 1.0 };
                    shadow[i * total_slots + slot_bases[li] + r] = sign;
                }
            }
        }
        // threshold: baseline_init + rand(0..threshold_jitter)  — COPY VERBATIM from wave_net::Layer::new
        let mut threshold = vec![0i16; ls];
        for i in 0..ls {
            let sg = (base_global + i) as u32;
            let h = crate::wave_bitnet::synapse::mix(crate::wave_bitnet::synapse::key(seed, sg, 0, 0, crate::wave_bitnet::synapse::P_THRESHOLD));
            let jitter = if cfg.threshold_jitter == 0 { 0 } else { (h % cfg.threshold_jitter as u64) as i16 };
            threshold[i] = cfg.baseline_init + jitter;
        }
        let mut layer = Layer {
            potential: vec![0i16; ls],
            cooldown: vec![0u8; ls],
            adapt: vec![0i32; ls],
            threshold,
            inbox: Vec::new(),
            elig_pre: vec![0i32; ls],
            elig_post: vec![0i32; ls],
            decide_potential: vec![0i16; ls],
            decide_eff: vec![0i32; ls],
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
            occupancy,
            w_nonzero: BitSet::zeros(ls * total_slots),
            w_sign: BitSet::zeros(ls * total_slots),
            shadow,
        };
        for i in 0..ls { layer.repack_row(i); }
        layer
    }
}
```

**Verify the threshold-jitter formula against `wave_net::neurons::Layer::new`** and match it exactly (the snippet above assumes `baseline_init + (hash % jitter)`; if wave_net differs, copy wave_net's exact expression).

- [ ] **Step 4b: Add `put` (set-or-clear) to BitSet**

In `src/wave_bitnet/bits.rs`, add to `impl BitSet`:
```rust
    #[inline]
    pub fn put(&mut self, i: usize, v: bool) {
        debug_assert!(i < self.n_bits);
        let mask = 1u64 << (i & 63);
        if v { self.words[i >> 6] |= mask; } else { self.words[i >> 6] &= !mask; }
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib wave_bitnet::neurons` and `cargo test --lib wave_bitnet::bits` → PASS. `cargo build` → no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/wave_bitnet/neurons.rs src/wave_bitnet/bits.rs src/wave_bitnet/mod.rs
git commit -m "feat(wave_bitnet): Layer with occupancy bitset + 2-bit weights + repack"
```

---

### Task 5: `wave.rs` — process_layer (forward pass, bitset scan)

**Files:**
- Create: `src/wave_bitnet/wave.rs`
- Modify: `src/wave_bitnet/mod.rs` (add `pub mod wave;`)

**Interfaces:**
- Consumes: `neurons::Layer`, `synapse::{Synapse, decode_cell}`, `neurons::{ADAPT_SHIFT, ADAPT_MAX}`.
- Produces: `pub fn process_layer(layer: &mut Layer, layer_index: u32, seed: u64, size: u32, input: &[u32], acc: &mut [i32], deliveries: &mut [Vec<Synapse>], fired: &mut Vec<u32>, record_elig: bool)`.

- [ ] **Step 1: Copy the neuron step verbatim, stub the generate step**

Create `src/wave_bitnet/wave.rs`. Copy `process_layer` **verbatim** from `src/wave_net/wave.rs` sections 1–3 (drain inbox, inject input, the fused decide/fire/ALIF/leak pass with `decide_potential`/`decide_eff`/`elig_*` accrual). For section 4 (generate), DELETE the `generate_into` call and leave the `for &local in fired.iter()` loop body empty for now. Adjust imports to `crate::wave_bitnet::...`. Add `pub mod wave;`.

- [ ] **Step 2: Write the failing test for delivery via bitset**

Append:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::LayerConfig;
    use crate::wave_bitnet::synapse::{decode_cell, Synapse, TopologyLevel};

    #[test]
    fn firing_neuron_delivers_nonzero_synapses_to_decoded_targets() {
        let size = 4u32;
        let ls = (size * size) as usize;
        let cfg = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        let mut l = crate::wave_bitnet::neurons::Layer::new(&cfg, 0, 5, size);
        // force neuron 0 to fire: low threshold, primed potential, cooldown 0
        l.threshold.iter_mut().for_each(|t| *t = 1);
        l.cooldown.iter_mut().for_each(|c| *c = 0);
        l.potential[0] = 100;
        // build expected: iterate neuron 0's wired cells, decode, skip weight 0
        let base = l.slot_base(0);
        let n = l.neigh[0];
        let mut expect: Vec<Synapse> = Vec::new();
        let mut r = 0;
        for c in l.occupancy[0].iter_set_in(0 * n, n) {
            let w = l.weight_at(0 * l.total_slots + base + r);
            r += 1;
            if w != 0 { expect.push(Synapse { target: decode_cell(c, 0, 1, size), weight: w as i16 }); }
        }
        let mut acc = vec![0i32; ls];
        let mut deliveries: Vec<Vec<Synapse>> = vec![Vec::new(); 2];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 5, size, &[], &mut acc, &mut deliveries, &mut fired, true);
        assert_eq!(fired, vec![0], "only neuron 0 fires");
        assert_eq!(deliveries[1], expect, "delivers decoded nonzero synapses to layer 1");
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib wave_bitnet::wave` → FAIL (deliveries empty — generate step is stubbed).

- [ ] **Step 4: Implement the bitset-scan generate step**

Replace the empty `for &local in fired.iter()` body with:
```rust
    let layer_count = deliveries.len() as i32;
    for &local in fired.iter() {
        let li_local = local as usize;
        for (lvl_idx, entry) in layer.topology.iter().enumerate() {
            let tl = layer_index as i32 + entry.level;
            if tl < 0 || tl >= layer_count { continue; }
            let n = layer.neigh[lvl_idx];
            let wbase = li_local * layer.total_slots + layer.slot_bases[lvl_idx];
            let radius = entry.radius;
            let mut r = 0usize;
            for c in layer.occupancy[lvl_idx].iter_set_in(li_local * n, n) {
                let w = layer.weight_at(wbase + r);
                r += 1;
                if w != 0 {
                    let target = decode_cell(c, local, radius, size);
                    deliveries[tl as usize].push(Synapse { target, weight: w as i16 });
                }
            }
        }
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib wave_bitnet::wave` → PASS. `cargo build` → no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/wave_bitnet/wave.rs src/wave_bitnet/mod.rs
git commit -m "feat(wave_bitnet): process_layer forward pass via occupancy bitset scan"
```

---

### Task 6: `network.rs` — Network, wave, L0 transducer, update primitive

**Files:**
- Create: `src/wave_bitnet/network.rs`
- Modify: `src/wave_bitnet/mod.rs` (add `pub mod network;`)

**Interfaces:**
- Consumes: `config::Config`, `neurons::Layer`, `wave::process_layer`, `synapse::{Synapse, decode_cell}`.
- Produces: `struct Network` with `Network::new(Config) -> Network`, `Network::new_with_readout(Config) -> Network`, `fn wave(&mut self, input: &[u32])`, `fn reset_state(&mut self)`, `fn layer_count(&self) -> usize`, `fn size(&self) -> u32`, `fn seed_val(&self) -> u64`, `fn on_layer`/`clear_listeners`, `fn with_layer`/`with_layer_mut`, `fn layer_decide_potential`/`layer_decide_effective_threshold`, and `fn eprop_update_synaptic(&mut self, source_z: usize, level_idx: usize, elig: &[f32], signal: &[f32], lr: f32)`.

- [ ] **Step 1: Copy Network scaffolding + wave loop, force L0 transducer**

Create `src/wave_bitnet/network.rs`. Copy **verbatim** from `src/wave_net/network.rs`: the `Network` struct, `Scratch`, `wave()`, `on_layer`/`clear_listeners`, `reset_state`, `with_layer`/`with_layer_mut`, `layer_decide_potential`/`layer_decide_effective_threshold`, `layer_count`/`size`/`seed_val`, and `build()`. In `build()`, keep the L0-transducer forcing block (`if z == 0 { threshold = i16::MAX; adapt_bump = 0; }`) and the `readout_last` block **exactly as wave_net has them**. Add `self.config.validate().expect(...)` (or return Result — match wave_net's `new` contract) at the top of `new`. Adjust imports to `crate::wave_bitnet::...`. Add `pub mod network;`.

- [ ] **Step 2: Write failing tests: determinism, L0 transducer, update raises a pruned synapse**

Append:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::{Config, LayerConfig};
    use crate::wave_bitnet::synapse::TopologyLevel;

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
        for _ in 0..5 { a.wave(&[0, 1, 2]); b.wave(&[0, 1, 2]); }
        a.with_layer(1, |la| b.with_layer(1, |lb| {
            assert_eq!(la.potential, lb.potential);
            assert_eq!(la.shadow, lb.shadow);
        }));
    }

    #[test]
    fn update_with_negative_signal_raises_pruned_synapse() {
        let mut net = Network::new(two_layer(8));
        let ls = 64usize;
        // pick neuron 0, level 0; zero its whole row shadow then repack (all pruned)
        net.with_layer_mut(0, |l| {
            let ts = l.total_slots;
            for s in 0..ts { l.shadow[0 * ts + s] = 0.0; }
            l.repack_row(0);
            assert_eq!(l.weight_at(0), 0, "row starts fully pruned");
        });
        // elig: length ls·count, positive for neuron 0 synapse 0; signal: negative at all targets
        let count = 8usize;
        let mut elig = vec![0f32; ls * count];
        elig[0 * count + 0] = 1.0;
        let signal = vec![-1.0f32; ls];
        net.eprop_update_synaptic(0, 0, &elig, &signal, 0.02);
        net.with_layer(0, |l| {
            assert!(l.shadow[0] > 0.0, "shadow raised by -lr·(-1)·1 > 0: {}", l.shadow[0]);
        });
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib wave_bitnet::network` → FAIL (`eprop_update_synaptic` missing; others may pass once `new` exists — that's fine, the update test drives Step 4).

- [ ] **Step 4: Implement eprop_update_synaptic**

Add to `impl Network`:
```rust
    /// Apply one e-prop update to layer `source_z`'s `level_idx` topology entry from a per-synapse
    /// eligibility `elig` (indexed `[i*count + r]`, r = wired-synapse rank) and per-target `signal`:
    /// `shadow[i*total_slots + slot_base + r] += -lr·signal[target]·elig[i*count+r]`, then repack
    /// each touched row. Targets are decoded from the occupancy bitset (no hash).
    pub fn eprop_update_synaptic(&mut self, source_z: usize, level_idx: usize, elig: &[f32], signal: &[f32], lr: f32) {
        let size = self.size();
        let ls = (size as usize) * (size as usize);
        let l = self.layer_count();
        self.with_layer_mut(source_z, |lz| {
            let entry = lz.topology[level_idx].clone();
            let tz = source_z as i32 + entry.level;
            if tz < 1 || tz as usize >= l { return; } // untrainable target
            let count = entry.count as usize;
            let n = lz.neigh[level_idx];
            let sbase = lz.slot_bases[level_idx];
            let ts = lz.total_slots;
            let radius = entry.radius;
            for i in 0..ls {
                let mut touched = false;
                let mut r = 0usize;
                // collect (r, cell) then apply — occupancy read is immutable, shadow write is mutable
                let cells: Vec<usize> = lz.occupancy[level_idx].iter_set_in(i * n, n).collect();
                for &c in &cells {
                    let e = elig[i * count + r];
                    if e != 0.0 {
                        touched = true;
                        let target = crate::wave_bitnet::synapse::decode_cell(c, i as u32, radius, size);
                        lz.shadow[i * ts + sbase + r] += -lr * signal[target as usize] * e;
                    }
                    r += 1;
                }
                if touched { lz.repack_row(i); }
            }
        });
    }
```
(If the borrow checker objects to reading `occupancy` while writing `shadow` inside the same `with_layer_mut` closure, the `cells` Vec already decouples them; keep it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib wave_bitnet::network` → PASS. `cargo build` → no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/wave_bitnet/network.rs src/wave_bitnet/mod.rs
git commit -m "feat(wave_bitnet): Network + wave loop + L0 transducer + shadow update"
```

---

### Task 7: `multilayer_dfa.rs` — temporal eligibility + DFA step (ported)

**Files:**
- Create: `src/wave_bitnet/multilayer_dfa.rs`
- Modify: `src/wave_bitnet/mod.rs` (add `pub mod multilayer_dfa;`)

**Interfaces:**
- Consumes: `network::Network`, `synapse::decode_cell`.
- Produces: `pub struct Edge { level: i32, count: usize, radius: u32 }`, `pub struct TrialRecords { spikes, pots, effs }`, `pub struct EligParams {...}`, `pub const PSI_WIDTH: f32`, `pub fn temporal_eligibility(net, entries, rec, p) -> Vec<Vec<Vec<f32>>>`, `pub fn multilayer_dfa_step(net, entries, rec, signal, lr, p)`.

- [ ] **Step 1: Copy the engine, adapting only the target lookup**

Create `src/wave_bitnet/multilayer_dfa.rs`. Copy **verbatim** from `src/wave_net/multilayer_dfa.rs`: the module doc (retitle), `Edge`, `TrialRecords`, `EligParams`, `PSI_WIDTH`, `PSI_GAMMA`, `elig_adapt_sum`, and the bodies of `temporal_eligibility` and `multilayer_dfa_step`. Then make TWO adaptations:

(a) In `temporal_eligibility`, change the per-layer edge loop from `for edge in &entries[z]` to `for (lvl, edge) in entries[z].iter().enumerate()` — `lvl` is the topology level index (the `entries`-order == topology-order invariant, same convention as `multilayer_dfa_step`'s `e_idx`). The inner loop currently computes `let j = target_of(seed, sg, i as u32, edge.level, k as u32, edge.radius, size)`. Replace that with a **decode from occupancy**, precomputing the ordered target list for this edge once:
```rust
                // targets for (layer z, level lvl) decoded from occupancy, in wired-synapse rank order
                let targets: Vec<usize> = net.with_layer(z, |lz| {
                    let n = lz.neigh[lvl];
                    let mut ts = Vec::with_capacity(ls * count);
                    for ii in 0..ls {
                        for c in lz.occupancy[lvl].iter_set_in(ii * n, n) {
                            ts.push(crate::wave_bitnet::synapse::decode_cell(c, ii as u32, edge.radius, size) as usize);
                        }
                    }
                    ts // length ls*count, indexed [i*count + r]
                });
```
and inside the `for i { for k { ... } }` loop rename `k`→`r` and use `let j = targets[i * count + r];` instead of `target_of(...)`. Keep the eligibility math (`elig_adapt_sum` / the `Σ_t pretr·ψ` branch) unchanged.

(b) In `multilayer_dfa_step`, change the call `net.eprop_update_synaptic(z, e_idx, &elig[z][e_idx], &signal[tz], lr)` — `e_idx` is the entry index, which must equal the **topology level index**. Since `entries[z]` is built in the same order as the layer's topology (invariant), `e_idx` is already the level index; the call is unchanged.

- [ ] **Step 2: Write failing tests**

Append (mirror wave_net's engine tests):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::{Config, LayerConfig};
    use crate::wave_bitnet::network::Network;
    use crate::wave_bitnet::synapse::TopologyLevel;

    fn net2(size: u32) -> (Network, Vec<Vec<Edge>>) {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        let net = Network::new(Config { seed: 3, size, layers: vec![up, top] });
        let entries = vec![vec![Edge { level: 1, count: 8, radius: 2 }], vec![]];
        (net, entries)
    }

    fn dense_records(ls: usize, l: usize, ttot: usize) -> TrialRecords {
        TrialRecords {
            spikes: (0..l).map(|_| (0..ttot).map(|t| if t % 2 == 0 { vec![0u32] } else { vec![] }).collect()).collect(),
            pots: (0..l).map(|_| (0..ttot).map(|_| vec![7i16; ls]).collect()).collect(),
            effs: (0..l).map(|_| (0..ttot).map(|_| vec![8i32; ls]).collect()).collect(),
        }
    }

    #[test]
    fn eligibility_is_shaped_and_deterministic() {
        let (net, entries) = net2(8);
        let ls = 64;
        let rec = dense_records(ls, 2, 6);
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let a = temporal_eligibility(&net, &entries, &rec, &p);
        let b = temporal_eligibility(&net, &entries, &rec, &p);
        assert_eq!(a[0][0].len(), ls * 8, "elig[layer0][edge0] length = ls*count");
        assert_eq!(a, b, "deterministic");
    }

    #[test]
    fn step_raises_weights_on_negative_signal() {
        let (mut net, entries) = net2(8);
        let ls = 64;
        let rec = dense_records(ls, 2, 6);
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]]; // negative on layer 1 (the target)
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let before: f32 = net.with_layer(0, |l| l.shadow.iter().sum());
        multilayer_dfa_step(&mut net, &entries, &rec, &signal, 0.02, &p);
        let after: f32 = net.with_layer(0, |l| l.shadow.iter().sum());
        assert!(after > before, "negative target signal + positive eligibility raises L0 shadow: {before}->{after}");
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib wave_bitnet::multilayer_dfa` → FAIL until Step 1 compiles + is correct.

- [ ] **Step 4: Make them pass**

Fix compile errors from Step 1 (imports, the `targets` precompute, the `k`→`r` rename). Re-run: `cargo test --lib wave_bitnet::multilayer_dfa` → PASS. `cargo build` → no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/wave_bitnet/multilayer_dfa.rs src/wave_bitnet/mod.rs
git commit -m "feat(wave_bitnet): temporal eligibility + DFA step (targets decoded from occupancy)"
```

---

### Task 8: Bench harness + FF depth-8 smoke benchmark + memory/throughput report

**Files:**
- Create: `src/bench/wave_bitnet_bench.rs`
- Modify: `src/bench/mod.rs` (add `pub mod wave_bitnet_bench;`)

**Interfaces:**
- Consumes: `crate::wave_bitnet::{config, network, multilayer_dfa, neurons, synapse}`.
- Produces (test-only harness): `make_ff`, `run_trial`, `softmax2`, `dfa_weight`, `build_signal`, `train_and_eval_best`, `single_task`, `ff_cfg`, `weight_sparsity`, `bytes_per_synapse(&Network) -> (f64 train, f64 rest)`; and tests `wave_bitnet_trains_above_chance` (non-ignored) + `wave_bitnet_ff_depth8_smoke` (`#[ignore]`).

- [ ] **Step 1: Port the minimal harness**

Create `src/bench/wave_bitnet_bench.rs`. Copy the harness pieces from `src/bench/multilayer_dfa.rs`'s `mod harness` — `CUE_P`, `P_DFA`, `cue_sites`, `softmax2`, `dfa_weight`, `run_trial`, `TaskCfg`, `build_signal`, `train_and_eval_best`, `single_task`, `ff_cfg`, `weight_sparsity` — **verbatim except** swap all `crate::wave_net::...` / `crate::bench::multilayer_dfa::...` references to `crate::wave_bitnet::...`, and change `make_ff` to build a wave_bitnet net:
```rust
    fn make_ff(seed: u64, size: u32, layers: usize, up_count: u32, up_radius: u32, adapt_bump: i16, adapt_decay: u8)
        -> (crate::wave_bitnet::network::Network, Vec<Vec<Edge>>) {
        use crate::wave_bitnet::config::{Config, LayerConfig};
        use crate::wave_bitnet::synapse::TopologyLevel;
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32,
            baseline_init: 6, adapt_bump, adapt_decay,
        };
        let net = Network::new(Config { seed, size, layers: vec![lc; layers] });
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: up_count as usize, radius: up_radius }] })
            .collect();
        (net, entries)
    }
```
`weight_sparsity` reads `w_nonzero`: fraction of `!w_nonzero` bits over `ls*total_slots` for layers `1..l`. Use `Edge`, `EligParams`, `PSI_WIDTH`, `temporal_eligibility`, `multilayer_dfa_step` from `crate::wave_bitnet::multilayer_dfa`. Add `pub mod wave_bitnet_bench;` to `src/bench/mod.rs`.

- [ ] **Step 2: Write the failing "trains above chance" test**

Append (non-ignored — cheap):
```rust
    #[test]
    fn wave_bitnet_trains_above_chance() {
        // 4-layer FF, size 16, generous fan-in; 2-class separable task must beat chance.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 5, 6);
        let mut cfg = ff_cfg(0, 0.004, 0.0);
        cfg.size = 16;
        let (best, _at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 3, 1000);
        assert!(best > 600, "wave_bitnet FF should train above chance: {best}");
    }
```

- [ ] **Step 3: Run test to verify it fails, then passes**

Run: `cargo test --release --lib wave_bitnet_trains_above_chance` → first FAIL if any harness wiring is off; fix compile/logic until PASS. (This is the integration proof that the whole engine trains.)

- [ ] **Step 4: Add the memory-accounting helper + smoke benchmark**

Append:
```rust
    /// (training bytes/synapse, at-rest bytes/synapse) for the whole net.
    fn bytes_per_synapse(net: &crate::wave_bitnet::network::Network) -> (f64, f64) {
        let l = net.layer_count();
        let (mut syn, mut occ_bits, mut ts_total) = (0usize, 0usize, 0usize);
        for z in 0..l {
            net.with_layer(z, |lz| {
                let ls = lz.potential.len();
                ts_total += ls * lz.total_slots;
                syn += ls * lz.total_slots; // wired synapses (count == distinct)
                for (li, _t) in lz.topology.iter().enumerate() {
                    occ_bits += ls * lz.neigh[li];
                }
            });
        }
        if syn == 0 { return (0.0, 0.0); }
        let weight_bits = 2.0 * ts_total as f64;      // nonzero + sign
        let occ = occ_bits as f64;
        let shadow_bytes = 4.0 * ts_total as f64;
        let train = ((weight_bits + occ) / 8.0 + shadow_bytes) / syn as f64;
        let rest = ((weight_bits + occ) / 8.0) / syn as f64;
        (train, rest)
    }

    #[test]
    #[ignore] // smoke: run manually in --release
    fn wave_bitnet_ff_depth8_smoke() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        eprintln!("== wave_bitnet FF depth-8 pure ternary smoke (r4/c48, adapt=5) ==");
        let mut bests = Vec::new();
        for &s in &seeds {
            let (mut net, entries) = make_ff(s, 32, 8, 48, 4, 5, 6);
            let mut cfg = ff_cfg(0, 0.004, 0.0);
            cfg.size = 32; cfg.present = 8; cfg.read = 8;
            let (best, at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, single_task, 300, 3, 3000);
            let (tr, rest) = bytes_per_synapse(&net);
            eprintln!("seed {s:#x}: best {best}@{at}, sparsity {:.0}%, {tr:.2} B/syn train, {rest:.2} B/syn at-rest", weight_sparsity(&net) * 100.0);
            bests.push(best);
        }
        let mean = bests.iter().sum::<u64>() / bests.len() as u64;
        let worst = *bests.iter().min().unwrap();
        eprintln!("mean {mean}, worst {worst} (target ~1000)");
        assert!(worst >= 900, "pure ternary FF depth-8 should hold ~1000 (worst {worst})");
    }
```

- [ ] **Step 5: Run the smoke benchmark**

Run: `cargo test --release --lib wave_bitnet_ff_depth8_smoke -- --ignored --nocapture`
Expected: prints best@trials + sparsity + bytes/synapse per seed; PASS with worst ≥ 900 (accuracy parity with wave_net's depth-8 result). At-rest bytes/synapse should be ≈ 0.4–0.5.

- [ ] **Step 6: Full suite green + commit**

Run: `cargo test --release --lib` → all pass (0 failed), warning-free.
```bash
git add src/bench/wave_bitnet_bench.rs src/bench/mod.rs
git commit -m "test(wave_bitnet): FF harness + depth-8 smoke benchmark + memory report"
```

---

## Self-review notes

- **Spec coverage:** module layout (T1–T7), config validate `count ≤ N` (T3), distinct-cell fill + decode (T2, T4), occupancy/2-bit/shadow storage (T4), forward bitset scan no-hash (T5), L0 transducer (T6), shadow update + repack (T6), ported eligibility/step with occupancy-decoded targets (T7), smoke benchmark + memory accounting + throughput note (T8). Change-bitset, scaled/int8, full-suite port, serialization all correctly absent (deferred). ✅
- **Throughput measurement:** the smoke benchmark reports accuracy + memory; the *forward throughput vs wave_net* comparison from the spec's success criteria is a nice-to-have — add a `criterion` micro-bench under `benches/` only if the smoke run's wall-clock doesn't already show the win. Flagged, not blocking.
- **Assumption to verify during T4:** the exact threshold-jitter expression and the inhibitor-sign init in `wave_net::neurons::Layer::new` — copy them verbatim rather than trusting the reconstructed formulas in the plan.
