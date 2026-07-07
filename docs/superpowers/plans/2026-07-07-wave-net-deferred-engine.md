# wave_net Deferred Engine (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline, per AGENTS.md — no subagent option). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A technically-functioning, deterministic, single-threaded `wave_net` engine: a stack of square spiking layers with hash-generated synapses, deferred one-hop propagation, sparse L0 spike input, and per-layer calibratable saturation.

**Architecture:** Each `Layer` owns its neuron state (`i16` potential, cooldown, per-neuron threshold) plus an `inbox`/`outbox` delivery pair. `wave::process_layer` runs one layer's integrate→fire→decay step and emits its firers' synapses grouped by *relative* level; `Network` owns all layers, does the absolute-layer routing into target `outbox`es, and swaps `inbox`/`outbox` each wave so signal advances exactly one layer per wave.

**Tech Stack:** Rust (edition 2024), standard library only, no `unsafe`, inline `#[cfg(test)]` tests.

**Spec:** `docs/superpowers/specs/2026-07-07-wave-net-deferred-engine-design.md`

## Global Constraints

- Standard library only; no external runtime deps. No `unsafe`. `cargo build` warning-free.
- Determinism: results are a pure function of `(seed, config, input sequence)`.
- Potentials/thresholds are `i16`; deliveries are `±1` summed in `i32` and clamped **once** per wave.
- Thresholds init near `i16::MAX` (silent start); `saturation` per layer, default `i16::MAX`; invariant `saturation ≥ max threshold` asserted in `Network::new`.
- All layers share one power-of-two `size` (square `size×size`); toroidal wrap by `& (size-1)`.
- Conventional commits, one per task, **no `Co-Authored-By` trailer** (AGENTS.md).
- Do not `git push` (local commits only) unless the user asks.

## File Structure

- `src/wave_net/synapse.rs` — wiring types (`TopologyLevel`, `Synapse`, `SynapseGroup`), hash primitives (`mix`/`key`/`map_range`/`map_range24`, `P_TARGET`/`P_THRESHOLD`), square-grid index helpers, and `generate_into`.
- `src/wave_net/config.rs` (new) — `Config`, `LayerConfig`, `THRESHOLD_JITTER_DEFAULT`, `Config::demo`, `Config::validate`.
- `src/wave_net/neurons.rs` — `Layer` + `Layer::new` (hash-jittered thresholds) + `max_threshold`.
- `src/wave_net/wave.rs` — `process_layer`.
- `src/wave_net/network.rs` — `Network` + public API.
- `src/wave_net/calibrate.rs` — no-op stub (doc comment only).
- `src/wave_net/mod.rs` — add `pub mod config;`.

The current files hold the user's sketch (u16 potentials, `update_buffer`, typo'd field names, `listeners` in `Layer`); each task replaces the relevant sketch content.

---

### Task 1: Synapse types, hash purpose constants, grid index helpers

**Files:**
- Modify: `src/wave_net/synapse.rs`

**Interfaces:**
- Produces: `pub struct Synapse { pub target: u32, pub inhibitory: bool }` (derives `Clone, Copy, Debug`); `pub struct TopologyLevel { pub level: i32, pub radius: u32, pub count: u32 }` (derives `Clone, Debug`); `pub struct SynapseGroup { pub level: i32, pub synapses: Vec<Synapse> }`; `pub const P_TARGET: u64 = 1;` `pub const P_THRESHOLD: u64 = 3;`; `pub fn local_of(x: u32, y: u32, size: u32) -> u32`; `pub fn xy_of(local: u32, size: u32) -> (u32, u32)`; `pub fn wrap(base: u32, off: i32, size: u32) -> u32`.

- [ ] **Step 1: Write failing tests** (append to `synapse.rs` `#[cfg(test)] mod tests`)

```rust
#[test]
fn index_roundtrip() {
    let size = 8;
    for y in 0..size { for x in 0..size {
        let l = local_of(x, y, size);
        assert_eq!(xy_of(l, size), (x, y));
    }}
}

#[test]
fn wrap_is_toroidal() {
    assert_eq!(wrap(0, -1, 8), 7);
    assert_eq!(wrap(7, 1, 8), 0);
    assert_eq!(wrap(0, -3, 8), 5);
}
```

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::synapse` → FAIL (unresolved `local_of`/`xy_of`/`wrap`).

- [ ] **Step 3: Implement.** Ensure `Synapse` derives `Clone, Copy, Debug`; `TopologyLevel` derives `Clone, Debug`. Add:

```rust
pub const P_TARGET: u64 = 1;
pub const P_THRESHOLD: u64 = 3;

#[derive(Clone, Debug)]
pub struct SynapseGroup {
    pub level: i32,
    pub synapses: Vec<Synapse>,
}

/// (x,y) -> local index in a `size`-wide square layer (`size` is a power of two).
#[inline]
pub fn local_of(x: u32, y: u32, size: u32) -> u32 {
    (y << size.trailing_zeros()) | x
}

/// local index -> (x,y).
#[inline]
pub fn xy_of(local: u32, size: u32) -> (u32, u32) {
    (local & (size - 1), local >> size.trailing_zeros())
}

/// Toroidal shift of one coordinate by `off`, wrapped into `0..size`.
#[inline]
pub fn wrap(base: u32, off: i32, size: u32) -> u32 {
    ((base as i32 + off) as u32) & (size - 1)
}
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::synapse` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wave_net): synapse types, hash purpose consts, grid index helpers"`

---

### Task 2: Procedural synapse generation

**Files:**
- Modify: `src/wave_net/synapse.rs`

**Interfaces:**
- Consumes: `mix`, `key`, `map_range24`, `wrap`, `local_of`, `xy_of`, `P_TARGET`, `TopologyLevel`, `Synapse`, `SynapseGroup`.
- Produces: `pub fn generate_into(seed: u64, source_global: u32, src_local: u32, size: u32, topology: &[TopologyLevel], inhibitor_ratio: u32, groups: &mut [SynapseGroup])`. Contract: `groups.len() == topology.len()`; caller sets each `groups[i].level = topology[i].level` and may leave `synapses` non-empty (this **appends**, enabling aggregation across firers).

- [ ] **Step 1: Write failing tests**

```rust
fn topo() -> Vec<TopologyLevel> {
    vec![
        TopologyLevel { level: 1, radius: 2, count: 6 },
        TopologyLevel { level: -1, radius: 0, count: 1 },
    ]
}

fn empty_groups(t: &[TopologyLevel]) -> Vec<SynapseGroup> {
    t.iter().map(|e| SynapseGroup { level: e.level, synapses: Vec::new() }).collect()
}

#[test]
fn generate_counts_per_level() {
    let t = topo();
    let mut g = empty_groups(&t);
    generate_into(42, 0, 0, 8, &t, 0, &mut g);
    assert_eq!(g[0].synapses.len(), 6);
    assert_eq!(g[1].synapses.len(), 1);
    // radius 0 targets the source cell itself
    assert_eq!(g[1].synapses[0].target, local_of(0, 0, 8));
}

#[test]
fn generate_targets_within_radius() {
    let t = topo();
    let mut g = empty_groups(&t);
    let (sx, sy) = (3u32, 5u32);
    generate_into(7, 100, local_of(sx, sy, 8), 8, &t, 0, &mut g);
    for s in &g[0].synapses {
        let (tx, ty) = xy_of(s.target, 8);
        // toroidal distance <= radius 2 on each axis
        let dx = ((tx + 8 - sx) & 7).min((sx + 8 - tx) & 7);
        let dy = ((ty + 8 - sy) & 7).min((sy + 8 - ty) & 7);
        assert!(dx <= 2 && dy <= 2, "target ({tx},{ty}) out of radius from ({sx},{sy})");
    }
}

#[test]
fn generate_is_deterministic_and_appends() {
    let t = topo();
    let mut a = empty_groups(&t);
    let mut b = empty_groups(&t);
    generate_into(1, 9, 9, 8, &t, 30000, &mut a);
    generate_into(1, 9, 9, 8, &t, 30000, &mut b);
    assert_eq!(a[0].synapses.len(), b[0].synapses.len());
    for (x, y) in a[0].synapses.iter().zip(&b[0].synapses) {
        assert_eq!((x.target, x.inhibitory), (y.target, y.inhibitory));
    }
    // second call appends (aggregation across firers)
    generate_into(1, 9, 9, 8, &t, 30000, &mut a);
    assert_eq!(a[0].synapses.len(), 12);
}
```

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::synapse::tests::generate` → FAIL (unresolved `generate_into`).

- [ ] **Step 3: Implement**

```rust
/// Append one firing neuron's synapses into `groups` (one per topology entry, same order).
/// Emits **relative** levels only; the caller (Network) resolves absolute target layers.
pub fn generate_into(
    seed: u64,
    source_global: u32,
    src_local: u32,
    size: u32,
    topology: &[TopologyLevel],
    inhibitor_ratio: u32,
    groups: &mut [SynapseGroup],
) {
    let (sx, sy) = xy_of(src_local, size);
    for (entry, group) in topology.iter().zip(groups.iter_mut()) {
        let span = 2 * entry.radius + 1;
        for k in 0..entry.count {
            let h = mix(key(seed, source_global, entry.level, k, P_TARGET));
            let dx = map_range24((h >> 40) as u32, span) as i32 - entry.radius as i32;
            let dy = map_range24(((h >> 16) as u32) & 0x00FF_FFFF, span) as i32 - entry.radius as i32;
            let tx = wrap(sx, dx, size);
            let ty = wrap(sy, dy, size);
            let inhibitory = ((h & 0xFFFF) as u32) < inhibitor_ratio;
            group.synapses.push(Synapse { target: local_of(tx, ty, size), inhibitory });
        }
    }
}
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::synapse` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wave_net): procedural synapse generation grouped by relative level"`

---

### Task 3: Config

**Files:**
- Create: `src/wave_net/config.rs`
- Modify: `src/wave_net/mod.rs` (add `pub mod config;`)

**Interfaces:**
- Consumes: `TopologyLevel`.
- Produces: `pub const THRESHOLD_JITTER_DEFAULT: u16 = 128;`; `pub struct LayerConfig { pub topology: Vec<TopologyLevel>, pub leak: (u8,u8), pub cooldown_base: u8, pub inhibitor_ratio: u32, pub threshold_jitter: u16, pub saturation: i16 }` (derives `Clone, Debug`); `pub struct Config { pub seed: u64, pub size: u32, pub layers: Vec<LayerConfig> }` (derives `Clone, Debug`); `Config::demo() -> Config`; `Config::validate(&self) -> Result<(), String>`; `Config::layer_size(&self) -> usize`; `Config::n_total(&self) -> usize`.

- [ ] **Step 1: Write failing tests** (in `config.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_is_valid() {
        assert!(Config::demo().validate().is_ok());
    }

    #[test]
    fn rejects_non_power_of_two_size() {
        let mut c = Config::demo();
        c.size = 12;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_empty_layers() {
        let mut c = Config::demo();
        c.layers.clear();
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_leak_and_cooldown() {
        let mut c = Config::demo();
        c.layers[0].leak = (0, 5);
        assert!(c.validate().is_err());
        let mut c = Config::demo();
        c.layers[0].cooldown_base = 0;
        assert!(c.validate().is_err());
    }
}
```

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::config` → FAIL (no module `config`).

- [ ] **Step 3: Implement.** Add `pub mod config;` to `mod.rs`, then create `config.rs`:

```rust
//! Construction input for the engine: a shared square `size`, a seed, and one
//! `LayerConfig` per layer. Thresholds are computed per neuron in `Layer::new`.

use crate::wave_net::synapse::TopologyLevel;

pub const THRESHOLD_JITTER_DEFAULT: u16 = 128;

#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub topology: Vec<TopologyLevel>,
    pub leak: (u8, u8),        // right-shift amounts a, b in `p -= (p>>a) + (p>>b)`
    pub cooldown_base: u8,     // refractory reload on fire
    pub inhibitor_ratio: u32,  // Q16: inhibitory iff (hash & 0xFFFF) < inhibitor_ratio
    pub threshold_jitter: u16, // threshold = i16::MAX - rand(0..threshold_jitter)
    pub saturation: i16,       // per-layer membrane clamp; default i16::MAX
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub size: u32,             // square side; power of two
    pub layers: Vec<LayerConfig>,
}

impl Config {
    pub fn layer_size(&self) -> usize {
        (self.size as usize) * (self.size as usize)
    }

    pub fn n_total(&self) -> usize {
        self.layer_size() * self.layers.len()
    }

    /// A small, valid, deterministic network for tests and bring-up.
    pub fn demo() -> Config {
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 2, radius: 1, count: 2 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
                TopologyLevel { level: -1, radius: 0, count: 1 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 9830, // ~0.15 * 65536
            threshold_jitter: THRESHOLD_JITTER_DEFAULT,
            saturation: i16::MAX,
        };
        Config { seed: 0x1234_5678_9ABC_DEF0, size: 16, layers: vec![layer; 6] }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.size < 1 || !self.size.is_power_of_two() {
            return Err(format!("size must be a power of two >= 1, got {}", self.size));
        }
        if self.layers.is_empty() {
            return Err("layers must not be empty".into());
        }
        for (z, lc) in self.layers.iter().enumerate() {
            if lc.leak.0 == 0 || lc.leak.1 == 0 {
                return Err(format!("layer {z}: leak shifts must be >= 1"));
            }
            if lc.cooldown_base == 0 {
                return Err(format!("layer {z}: cooldown_base must be >= 1"));
            }
            if lc.saturation < 1 {
                return Err(format!("layer {z}: saturation must be >= 1"));
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::config` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wave_net): Config/LayerConfig with demo and validation"`

---

### Task 4: Layer state + hash-jittered thresholds

**Files:**
- Modify: `src/wave_net/neurons.rs` (replace the sketch `Layer`)

**Interfaces:**
- Consumes: `LayerConfig`, `Synapse`, `TopologyLevel`, `mix`, `key`, `map_range`, `P_THRESHOLD`.
- Produces: `pub struct Layer { pub potential: Vec<i16>, pub cooldown: Vec<u8>, pub inbox: Vec<Synapse>, pub outbox: Vec<Synapse>, pub threshold: Vec<i16>, pub saturation: i16, pub leak: (u8,u8), pub cooldown_base: u8, pub topology: Vec<TopologyLevel>, pub inhibitor_ratio: u32 }`; `Layer::new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer`; `Layer::max_threshold(&self) -> i16`.

- [ ] **Step 1: Write failing tests** (replace `neurons.rs` test module)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::LayerConfig;
    use crate::wave_net::synapse::TopologyLevel;

    fn lc(jitter: u16) -> LayerConfig {
        LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0,
            threshold_jitter: jitter, saturation: i16::MAX,
        }
    }

    #[test]
    fn new_sizes_and_zeroes() {
        let l = Layer::new(&lc(128), 1, 0, 8);
        assert_eq!(l.potential.len(), 64);
        assert_eq!(l.cooldown.len(), 64);
        assert_eq!(l.threshold.len(), 64);
        assert!(l.potential.iter().all(|&p| p == 0));
        assert!(l.inbox.is_empty() && l.outbox.is_empty());
    }

    #[test]
    fn thresholds_near_i16_max_within_jitter() {
        let l = Layer::new(&lc(128), 1, 0, 8);
        for &t in &l.threshold {
            assert!(t > i16::MAX - 128 && t <= i16::MAX, "threshold {t} out of band");
        }
        assert_eq!(l.max_threshold(), *l.threshold.iter().max().unwrap());
    }

    #[test]
    fn thresholds_deterministic() {
        let a = Layer::new(&lc(128), 7, 2, 8);
        let b = Layer::new(&lc(128), 7, 2, 8);
        assert_eq!(a.threshold, b.threshold);
    }
}
```

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::neurons` → FAIL (Layer shape/`new` mismatch).

- [ ] **Step 3: Implement.** Replace the sketch `Layer` and its imports with:

```rust
//! `neurons` — a `Layer`'s per-neuron state, its delivery inbox/outbox pair, and its
//! per-layer parameters. Thresholds start near `i16::MAX` (silent) with a small hash jitter.

use crate::wave_net::config::LayerConfig;
use crate::wave_net::synapse::{key, map_range, mix, Synapse, TopologyLevel, P_THRESHOLD};

pub struct Layer {
    // wave-mutable hot state
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub inbox: Vec<Synapse>,   // drained THIS wave (filled last wave)
    pub outbox: Vec<Synapse>,  // filled for NEXT wave; swapped with inbox at wave end

    // tunable params (calibration/training will rewrite these between phases)
    pub threshold: Vec<i16>,
    pub saturation: i16,

    // fixed structure
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<TopologyLevel>,
    pub inhibitor_ratio: u32,
}

impl Layer {
    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;
        let mut threshold = vec![0i16; ls];
        for (local, th) in threshold.iter_mut().enumerate() {
            let global = (base + local) as u32;
            let h = mix(key(seed, global, 0, 0, P_THRESHOLD));
            let jitter = map_range(h as u32, cfg.threshold_jitter as u32) as i32; // [0, jitter)
            *th = (i16::MAX as i32 - jitter) as i16;
        }
        Layer {
            potential: vec![0; ls],
            cooldown: vec![0; ls],
            inbox: Vec::new(),
            outbox: Vec::new(),
            threshold,
            saturation: cfg.saturation,
            leak: cfg.leak,
            cooldown_base: cfg.cooldown_base,
            topology: cfg.topology.clone(),
            inhibitor_ratio: cfg.inhibitor_ratio,
        }
    }

    pub fn max_threshold(&self) -> i16 {
        self.threshold.iter().copied().max().unwrap_or(0)
    }
}
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::neurons` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wave_net): Layer state with inbox/outbox and hash-jittered thresholds"`

---

### Task 5: The wave step (`process_layer`)

**Files:**
- Modify: `src/wave_net/wave.rs` (replace the comment sketch)

**Interfaces:**
- Consumes: `Layer`, `generate_into`, `SynapseGroup`.
- Produces: `pub fn process_layer(layer: &mut Layer, layer_index: u32, seed: u64, size: u32, input: &[u32], acc: &mut [i32], out: &mut [SynapseGroup], fired: &mut Vec<u32>)`. Contract: `acc.len() >= layer_size`; `out.len() == layer.topology.len()` with each `out[i].level == layer.topology[i].level`; the caller clears each `out[i].synapses` before the call. `process_layer` drains+clears `layer.inbox`, fills `fired`, and appends generated synapses into `out`. It does **not** touch `outbox` (routing is the Network's job).

- [ ] **Step 1: Write failing tests** (in `wave.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::neurons::Layer;
    use crate::wave_net::config::LayerConfig;
    use crate::wave_net::synapse::{Synapse, SynapseGroup, TopologyLevel};

    // A layer with hand-set LOW thresholds so integration can actually cause firing.
    fn low_layer(size: u32, threshold: i16, cooldown_base: u8, topo: Vec<TopologyLevel>) -> Layer {
        let cfg = LayerConfig {
            topology: topo, leak: (3, 5), cooldown_base, inhibitor_ratio: 0,
            threshold_jitter: 0, saturation: i16::MAX,
        };
        let mut l = Layer::new(&cfg, 0, 0, size);
        for t in l.threshold.iter_mut() { *t = threshold; }
        l
    }

    fn groups_for(l: &Layer) -> Vec<SynapseGroup> {
        l.topology.iter().map(|e| SynapseGroup { level: e.level, synapses: Vec::new() }).collect()
    }

    #[test]
    fn integration_fires_and_resets() {
        let mut l = low_layer(4, 3, 2, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        // deliver +3 to neuron 0 via the inbox
        for _ in 0..3 { l.inbox.push(Synapse { target: 0, inhibitory: false }); }
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
        assert_eq!(fired, vec![0]);
        assert_eq!(l.potential[0], 0);       // reset on fire
        assert_eq!(l.cooldown[0], 2);        // reloaded
        assert_eq!(out[0].synapses.len(), 1);// one outgoing synapse generated
        assert!(l.inbox.is_empty());         // drained
    }

    #[test]
    fn refractory_blocks_refire() {
        let mut l = low_layer(1, 3, 2, vec![]);
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        // wave A: force fire via input injection (sets potential=saturation, cooldown=0)
        process_layer(&mut l, 0, 0, 1, &[0], &mut acc, &mut out, &mut fired);
        assert_eq!(fired, vec![0]);
        // wave B: strong drive but still refractory (cooldown 2 -> decremented to 1)
        for _ in 0..100 { l.inbox.push(Synapse { target: 0, inhibitory: false }); }
        process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        assert!(fired.is_empty(), "must not fire while refractory");
    }

    #[test]
    fn leak_decays_subthreshold_potential() {
        let mut l = low_layer(1, 20_000, 2, vec![]);
        l.potential[0] = 1000;
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        // leak (3,5): 1000 - 125 - 31 = 844
        assert_eq!(l.potential[0], 844);
    }

    #[test]
    fn inhibition_and_single_clamp() {
        let mut l = low_layer(1, 30_000, 2, vec![]);
        l.saturation = 50;
        l.potential[0] = 40;
        // +100 excitatory, -10 inhibitory -> raw 130, clamps once to 50
        for _ in 0..100 { l.inbox.push(Synapse { target: 0, inhibitory: false }); }
        for _ in 0..10  { l.inbox.push(Synapse { target: 0, inhibitory: true }); }
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        // clamped to 50 pre-decide (no fire), then leak (3,5): 50 - 6 - 1 = 43
        assert_eq!(l.potential[0], 43);
    }
}
```

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::wave` → FAIL (unresolved `process_layer`).

- [ ] **Step 3: Implement.** Replace `wave.rs` contents with:

```rust
//! `wave` — one layer's per-wave step: integrate (drain inbox) → inject → clamp →
//! decide → generate outgoing synapses → leak. Touches only this layer; the Network
//! routes the generated synapses into other layers' inboxes for the next wave.

use crate::wave_net::neurons::Layer;
use crate::wave_net::synapse::{generate_into, SynapseGroup};

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    seed: u64,
    size: u32,
    input: &[u32],
    acc: &mut [i32],
    out: &mut [SynapseGroup],
    fired: &mut Vec<u32>,
) {
    let ls = (size as usize) * (size as usize);

    // 1. cooldown decay
    for c in layer.cooldown.iter_mut() {
        *c = c.saturating_sub(1);
    }

    // 2. drain inbox: sum deliveries in i32, fold into potential (narrow to i16), clear inbox
    for a in acc[..ls].iter_mut() {
        *a = 0;
    }
    for s in layer.inbox.iter() {
        acc[s.target as usize] += if s.inhibitory { -1 } else { 1 };
    }
    layer.inbox.clear();
    for i in 0..ls {
        let v = layer.potential[i] as i32 + acc[i];
        layer.potential[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }

    // 3. inject forced-fire input (L0 only; other layers get &[])
    for &a in input {
        layer.potential[a as usize] = layer.saturation;
        layer.cooldown[a as usize] = 0;
    }

    // 4. one saturation clamp
    let sat = layer.saturation;
    for p in layer.potential.iter_mut() {
        *p = (*p).clamp(-sat, sat);
    }

    // 5. decide
    fired.clear();
    for i in 0..ls {
        if layer.cooldown[i] == 0 && layer.potential[i] >= layer.threshold[i] {
            layer.potential[i] = 0;
            layer.cooldown[i] = layer.cooldown_base;
            fired.push(i as u32);
        }
    }

    // 6. generate outgoing synapses, aggregated by relative level into `out`
    let base = layer_index as usize * ls;
    for &local in fired.iter() {
        generate_into(
            seed,
            (base + local as usize) as u32,
            local,
            size,
            &layer.topology,
            layer.inhibitor_ratio,
            out,
        );
    }

    // 7. leak survivors into the next wave
    let (la, lb) = layer.leak;
    for p in layer.potential.iter_mut() {
        let v = *p;
        *p = v - (v >> la) - (v >> lb);
    }
}
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::wave` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wave_net): process_layer wave step (integrate/fire/decay)"`

---

### Task 6: Network orchestration + calibrate stub + warning-free build

**Files:**
- Modify: `src/wave_net/network.rs` (replace the sketch)
- Modify: `src/wave_net/calibrate.rs` (doc-comment stub)

**Interfaces:**
- Consumes: `Config`, `Layer`, `SynapseGroup`, `process_layer`.
- Produces: `Network::new(config: Config) -> Network`; `wave(&self, input: &[u32])`; `on_layer(&mut self, layer: usize, Box<dyn Fn(usize, &[u32]) + Send + Sync>)`; `clear_listeners(&mut self)`; `reset_state(&self)`; `potential(&self, layer: usize, local: usize) -> i16`; `size(&self) -> u32`; `layer_count(&self) -> usize`; `n_total(&self) -> usize`.

- [ ] **Step 1: Write failing tests** (in `network.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::{local_of, TopologyLevel};
    use std::sync::{Arc, Mutex};

    // two 4x4 layers, L0 -> L1 straight up (level+1, radius 0), all excitatory
    fn two_layer() -> Config {
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 0, count: 1 }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0,
            threshold_jitter: 0, saturation: i16::MAX,
        };
        let l1 = LayerConfig { topology: vec![], ..l0.clone() };
        Config { seed: 99, size: 4, layers: vec![l0, l1] }
    }

    #[test]
    fn new_asserts_invariants() {
        assert_eq!(Network::new(two_layer()).n_total(), 32);
    }

    #[test]
    #[should_panic(expected = "saturation")]
    fn new_rejects_saturation_below_threshold() {
        let mut c = two_layer();
        c.layers[0].threshold_jitter = 0; // threshold == i16::MAX
        c.layers[0].saturation = 100;     // < max threshold -> panic
        Network::new(c);
    }

    #[test]
    fn injection_fires_exactly_l0_targets() {
        let net = Network::new(two_layer());
        let fired = Arc::new(Mutex::new(Vec::new()));
        {
            let f = fired.clone();
            let mut net = net;
            net.on_layer(0, Box::new(move |_w, locals| *f.lock().unwrap() = locals.to_vec()));
            net.wave(&[0, 5]);
            assert_eq!(*fired.lock().unwrap(), vec![0, 5]);
        }
    }

    #[test]
    fn deferred_delivery_is_one_hop() {
        let net = Network::new(two_layer());
        net.wave(&[0]);                       // L0 neuron 0 fires; delivery queued for L1
        assert_eq!(net.potential(1, local_of(0, 0, 4) as usize), 0, "not delivered same wave");
        net.wave(&[]);                        // L1 drains: +1 arrives
        assert_eq!(net.potential(1, local_of(0, 0, 4) as usize), 1, "delivered next wave");
    }

    #[test]
    fn deterministic_across_runs() {
        let inputs: [&[u32]; 3] = [&[0, 1, 2], &[], &[3]];
        let run = || {
            let net = Network::new(Config::demo());
            for inp in inputs { net.wave(inp); }
            (0..net.layer_count())
                .flat_map(|z| (0..(net.size() * net.size()) as usize).map(move |i| (z, i)))
                .map(|(z, i)| net.potential(z, i))
                .collect::<Vec<i16>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn reset_state_zeros_everything() {
        let net = Network::new(Config::demo());
        for _ in 0..5 { net.wave(&[0, 1, 2, 3]); }
        net.reset_state();
        for z in 0..net.layer_count() {
            for i in 0..(net.size() * net.size()) as usize {
                assert_eq!(net.potential(z, i), 0);
            }
        }
    }
}
```

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::network` → FAIL (Network API mismatch).

- [ ] **Step 3: Implement.** Replace `network.rs` with:

```rust
//! `network` — owns the layer stack, drives each wave, and routes each layer's
//! generated synapses into the target layers' inboxes for the next wave.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::wave_net::config::Config;
use crate::wave_net::neurons::Layer;
use crate::wave_net::synapse::SynapseGroup;
use crate::wave_net::wave::process_layer;

pub struct Network {
    seed: u64,
    size: u32,
    layers: Vec<Mutex<Layer>>,
    wave_id: AtomicUsize,
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

impl Network {
    pub fn new(config: Config) -> Network {
        config.validate().expect("invalid config");
        let size = config.size;
        let l = config.layers.len();
        let mut layers = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let layer = Layer::new(lc, config.seed, z as u32, size);
            assert!(
                layer.saturation >= layer.max_threshold(),
                "layer {z}: saturation {} must be >= max threshold {}",
                layer.saturation,
                layer.max_threshold()
            );
            layers.push(Mutex::new(layer));
        }
        Network {
            seed: config.seed,
            size,
            layers,
            wave_id: AtomicUsize::new(0),
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&self, input: &[u32]) {
        let w = self.wave_id.fetch_add(1, Ordering::Relaxed);
        let l = self.layers.len();
        let ls = (self.size as usize) * (self.size as usize);
        let mut acc = vec![0i32; ls];
        let mut fired: Vec<u32> = Vec::new();

        for z in 0..l {
            let mut out: Vec<SynapseGroup>;
            {
                let mut g = self.layers[z].lock().unwrap();
                out = g
                    .topology
                    .iter()
                    .map(|e| SynapseGroup { level: e.level, synapses: Vec::new() })
                    .collect();
                let inp: &[u32] = if z == 0 { input } else { &[] };
                process_layer(&mut g, z as u32, self.seed, self.size, inp, &mut acc, &mut out, &mut fired);
            }
            // route: Network resolves absolute target layers and feeds their outboxes
            for grp in out.iter() {
                let tl = z as i32 + grp.level;
                if tl >= 0 && (tl as usize) < l {
                    self.layers[tl as usize].lock().unwrap().outbox.extend(grp.synapses.iter().copied());
                }
            }
            if let Some(listener) = &self.listeners[z] {
                listener(w, &fired);
            }
        }

        // swap inbox <- outbox so this wave's deliveries drain next wave
        for layer in self.layers.iter() {
            let mut g = layer.lock().unwrap();
            std::mem::swap(&mut g.inbox, &mut g.outbox);
            g.outbox.clear();
        }
    }

    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>) {
        self.listeners[layer] = Some(listener);
    }

    pub fn clear_listeners(&mut self) {
        for l in self.listeners.iter_mut() {
            *l = None;
        }
    }

    pub fn reset_state(&self) {
        for layer in self.layers.iter() {
            let mut g = layer.lock().unwrap();
            g.potential.iter_mut().for_each(|p| *p = 0);
            g.cooldown.iter_mut().for_each(|c| *c = 0);
            g.inbox.clear();
            g.outbox.clear();
        }
        self.wave_id.store(0, Ordering::Relaxed);
    }

    pub fn potential(&self, layer: usize, local: usize) -> i16 {
        self.layers[layer].lock().unwrap().potential[local]
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn n_total(&self) -> usize {
        self.layers.len() * (self.size as usize) * (self.size as usize)
    }
}
```

Replace `calibrate.rs` with a documented stub:

```rust
//! `calibrate` — placeholder. v1 ships no calibration: thresholds stay near `i16::MAX`
//! (silent above L0). A later phase will lower per-layer thresholds toward a firing-rate
//! target and set each layer's saturation to a margin above its threshold band,
//! maintaining `saturation >= max threshold`.
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::network` → PASS.

- [ ] **Step 5: Full test + warning check** — `cargo test` → all pass; `cargo build 2>&1` → no warnings. Fix any unused-import/dead-code warnings in the touched files.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(wave_net): Network orchestration, routing, and deferred-wave swap"`

---

## Notes / deferred (not in v1)

- Scratch reuse: `acc`/`out`/`fired` are allocated per wave for simplicity; hoist onto the
  `Network` (behind the single-threaded path) if profiling ever calls for it.
- No threading, training, or calibration logic — see the spec's Future section.
- `AGENTS.md`'s "engine model" paragraph still describes the frozen `wave_reservoir` within-wave
  model; update it in a later docs pass once this engine is the primary one.
