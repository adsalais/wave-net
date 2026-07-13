# wave_bitnet save/load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two independent, std-only binary save/load formats to `wave_bitnet` — a self-contained inference **model** (`.wbm`) and a **runtime overlay** (`.wbr`) applied onto a loaded model.

**Architecture:** A new `src/wave_bitnet/persist.rs` module holds hand-rolled little-endian byte I/O, an FNV-1a fingerprint, and the two (de)serializers as `impl Network` methods. Loading a model reconstructs each `Layer` from stored fields via a new `Layer::from_parts`, which reuses an extracted `derive_layout` helper so a loaded layer's derived LUTs are byte-identical to a freshly-built one. The runtime overlay stores only the mutable per-neuron arrays + `wave_id` and is bound to its model by a fingerprint.

**Tech Stack:** Rust edition 2024, standard library only.

## Global Constraints

- **Standard library only** in `src/` — no serde/bincode/new deps.
- **Warning-free build** at every commit (`cargo build`).
- **No `unsafe`** anywhere in this feature (the one documented `unsafe` in the tree is unrelated, in `wave::process_layer`).
- **Determinism** — save→load→save is byte-stable; a loaded model runs identically.
- **One commit per task**, conventional-commit messages (`feat:`/`refactor:` …).
- **NEVER add a `Co-Authored-By` trailer.**
- Work happens on branch `feat/wave-bitnet-persist` (already created).
- `cargo test` must stay green.

---

### Task 1: Extract `derive_layout`; derive `PartialEq` for `TopologyLevel`

Pure refactor. Extract the `(topology, size)`-only layout derivation out of `Layer::new` into a shared helper so `Layer::from_parts` (Task 2) reuses the exact same code. Add `PartialEq` to `TopologyLevel` for later equality asserts.

**Files:**
- Modify: `src/wave_bitnet/synapse.rs:5` (derive line)
- Modify: `src/wave_bitnet/neurons.rs` (extract helper, rewire `Layer::new`)
- Test: inline `#[cfg(test)]` in `src/wave_bitnet/neurons.rs`

**Interfaces:**
- Produces: `pub(crate) struct DerivedLayout { total_slots: usize, slot_bases: Vec<usize>, neigh: Vec<usize>, occ_wpn: Vec<usize>, offsets: Vec<Vec<(i8,i8)>>, off_flat: Vec<Vec<i32>> }` and `pub(crate) fn derive_layout(topology: &[TopologyLevel], size: u32) -> DerivedLayout` in `neurons.rs`.

- [ ] **Step 1: Add `PartialEq` to `TopologyLevel`**

In `src/wave_bitnet/synapse.rs`, change line 5:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct TopologyLevel {
    pub level: i32,
    pub radius: u32,
    pub count: u32,
}
```

- [ ] **Step 2: Write the failing test for `derive_layout`**

Add to the `tests` module in `src/wave_bitnet/neurons.rs`:

```rust
#[test]
fn derive_layout_matches_expected() {
    let topo = vec![
        TopologyLevel { level: 1, radius: 2, count: 8 },
        TopologyLevel { level: 0, radius: 1, count: 3 },
    ];
    let d = derive_layout(&topo, 8);
    assert_eq!(d.total_slots, 11);
    assert_eq!(d.slot_bases, vec![0, 8]);
    assert_eq!(d.neigh, vec![25, 9]);
    assert_eq!(d.occ_wpn, vec![1, 1]);
    assert_eq!(d.offsets[0].len(), 25);
    assert_eq!(d.off_flat[1].len(), 9);
    assert_eq!(d.offsets[1][4], (0, 0)); // center of a radius-1 (span-3) level
    assert_eq!(d.off_flat[1][4], 0);
}
```

- [ ] **Step 3: Run it to confirm it fails to compile (no `derive_layout` yet)**

Run: `cargo test -p wave_state_machine derive_layout_matches_expected 2>&1 | tail -5`
(If the crate name differs, use `cargo test derive_layout_matches_expected`.)
Expected: compile error, `cannot find function derive_layout`.

- [ ] **Step 4: Implement `derive_layout` + `DerivedLayout` and rewire `Layer::new`**

In `src/wave_bitnet/neurons.rs`, add above `impl Layer` (after the `WCODE` const):

```rust
/// Layout quantities derived purely from `(topology, size)` — no seed, no RNG. Shared by
/// `Layer::new` (fresh build) and `Layer::from_parts` (load) so a loaded layer's LUTs are
/// byte-identical to a freshly-built one.
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
        offsets.push(
            (0..n)
                .map(|c| (((c as u32 % span) as i32 - r) as i8, ((c as u32 / span) as i32 - r) as i8))
                .collect(),
        );
        off_flat.push(
            (0..n)
                .map(|c| {
                    let dx = (c as u32 % span) as i32 - r;
                    let dy = (c as u32 / span) as i32 - r;
                    dy * size as i32 + dx
                })
                .collect(),
        );
        total_slots += t.count as usize;
    }
    DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat }
}
```

In `Layer::new`, replace the inline derived-layout block (currently `let mut slot_bases = …;` through the `for t in &cfg.topology { … }` loop that builds `slot_bases`/`neigh`/`occ_wpn`/`offsets`/`off_flat`/`total_slots`) with:

```rust
        let DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat } =
            derive_layout(&cfg.topology, size);
```

Leave the rest of `Layer::new` (threshold fill, occ fill, shadow init, struct literal, `repack_row` loop) unchanged — it reads `slot_bases[..]`, `occ_wpn[..]`, `total_slots` exactly as before.

- [ ] **Step 5: Run tests to confirm green (new test + all existing)**

Run: `cargo test 2>&1 | tail -20`
Expected: PASS, including `derive_layout_matches_expected` and every prior `neurons`/`network`/`multilayer_dfa` test.

- [ ] **Step 6: Confirm warning-free build**

Run: `cargo build 2>&1 | tail -5`
Expected: no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/wave_bitnet/synapse.rs src/wave_bitnet/neurons.rs
git commit -m "refactor(wave_bitnet): extract derive_layout shared by Layer::new and load"
```

---

### Task 2: `Layer::from_parts`

Construct a `Layer` from persisted parts, reusing `derive_layout`, reconstructing `shadow` as the decode of `codes`, zeroing runtime, and validating shapes.

**Files:**
- Modify: `src/wave_bitnet/neurons.rs` (add `Layer::from_parts`)
- Test: inline `#[cfg(test)]` in `src/wave_bitnet/neurons.rs`

**Interfaces:**
- Consumes: `derive_layout`, `DerivedLayout`, `WCODE` (Task 1 / existing).
- Produces: `pub fn Layer::from_parts(topology: Vec<TopologyLevel>, leak: (u8,u8), cooldown_base: u8, adapt_bump: i16, adapt_decay: u8, readout: bool, ternary_threshold: f32, threshold: Vec<i16>, occ: Vec<Vec<u64>>, codes: Vec<u64>, size: u32) -> Result<Layer, String>`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/wave_bitnet/neurons.rs`:

```rust
#[test]
fn from_parts_reproduces_built_layer() {
    let size = 8u32;
    let cfg = lc(vec![
        TopologyLevel { level: 1, radius: 2, count: 8 },
        TopologyLevel { level: 0, radius: 1, count: 3 },
    ]);
    let mut built = Layer::new(&cfg, 7, 1, size);
    // make neuron 0's row non-trivial (+1 / -1 / 0 mix) then repack so codes differ from init
    let ts = built.total_slots;
    for s in 0..ts {
        built.shadow[s] = match s % 3 { 0 => 0.0, 1 => 2.0, _ => -2.0 };
    }
    built.repack_row(0);
    let rebuilt = Layer::from_parts(
        built.topology.clone(),
        built.leak,
        built.cooldown_base,
        built.adapt_bump,
        built.adapt_decay,
        built.readout,
        built.ternary_threshold,
        built.threshold.clone(),
        built.occ.clone(),
        built.codes.clone(),
        size,
    )
    .unwrap();
    assert_eq!(rebuilt.topology, built.topology);
    assert_eq!(rebuilt.threshold, built.threshold);
    assert_eq!(rebuilt.occ, built.occ);
    assert_eq!(rebuilt.codes, built.codes);
    assert_eq!(rebuilt.total_slots, built.total_slots);
    assert_eq!(rebuilt.slot_bases, built.slot_bases);
    assert_eq!(rebuilt.neigh, built.neigh);
    assert_eq!(rebuilt.occ_wpn, built.occ_wpn);
    assert_eq!(rebuilt.offsets, built.offsets);
    assert_eq!(rebuilt.off_flat, built.off_flat);
    // shadow is the decode of codes (inference-authoritative), runtime zeroed
    for s in 0..rebuilt.shadow.len() {
        assert_eq!(rebuilt.shadow[s], rebuilt.weight_at(s) as f32);
    }
    assert!(rebuilt.potential.iter().all(|&p| p == 0));
    assert!(rebuilt.adapt.iter().all(|&a| a == 0));
}

#[test]
fn from_parts_rejects_bad_lengths() {
    let size = 8u32;
    let topo = vec![TopologyLevel { level: 1, radius: 2, count: 8 }];
    // threshold length 10 != ls 64
    let r = Layer::from_parts(topo, (3, 5), 2, 5, 6, false, 0.5, vec![0i16; 10], vec![vec![0u64; 64]], vec![0u64; 16], size);
    assert!(r.is_err());
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cargo test from_parts 2>&1 | tail -5`
Expected: compile error, `no function or associated item named from_parts`.

- [ ] **Step 3: Implement `Layer::from_parts`**

Add inside `impl Layer` in `src/wave_bitnet/neurons.rs`:

```rust
    /// Build a `Layer` directly from persisted parts (bypassing seed-based generation): rebuild the
    /// derived LUTs from `topology`+`size`, reconstruct `shadow` as the per-slot decode of `codes`
    /// (codes are authoritative for inference), and zero all runtime arrays. Validates array shapes;
    /// returns `Err(msg)` on any mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        topology: Vec<TopologyLevel>,
        leak: (u8, u8),
        cooldown_base: u8,
        adapt_bump: i16,
        adapt_decay: u8,
        readout: bool,
        ternary_threshold: f32,
        threshold: Vec<i16>,
        occ: Vec<Vec<u64>>,
        codes: Vec<u64>,
        size: u32,
    ) -> Result<Layer, String> {
        let ls = (size as usize) * (size as usize);
        let DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat } =
            derive_layout(&topology, size);
        if threshold.len() != ls {
            return Err(format!("threshold length {} != ls {ls}", threshold.len()));
        }
        if occ.len() != topology.len() {
            return Err(format!("occ levels {} != topology levels {}", occ.len(), topology.len()));
        }
        for (li, t) in topology.iter().enumerate() {
            if t.count as usize > neigh[li] {
                return Err(format!("level {li}: count {} exceeds neighborhood {}", t.count, neigh[li]));
            }
            let want = ls * occ_wpn[li];
            if occ[li].len() != want {
                return Err(format!("occ[{li}] length {} != ls*occ_wpn {want}", occ[li].len()));
            }
        }
        let want_codes = (ls * total_slots + 31) / 32;
        if codes.len() != want_codes {
            return Err(format!("codes length {} != {want_codes}", codes.len()));
        }
        // shadow = per-slot decode of codes (inference-authoritative; not the training master)
        let n = ls * total_slots;
        let mut shadow = vec![0f32; n];
        for s in 0..n {
            shadow[s] = WCODE[((codes[s >> 5] >> ((s & 31) * 2)) & 0b11) as usize] as f32;
        }
        Ok(Layer {
            potential: vec![0i16; ls],
            cooldown: vec![0u8; ls],
            adapt: vec![0i32; ls],
            threshold,
            pending: vec![0i32; ls],
            elig_pre: vec![0i32; ls],
            elig_post: vec![0i32; ls],
            decide_potential: vec![0i16; ls],
            decide_eff: vec![0i32; ls],
            leak,
            cooldown_base,
            topology,
            adapt_bump,
            adapt_decay,
            readout,
            ternary_threshold,
            total_slots,
            slot_bases,
            neigh,
            occ_wpn,
            occ,
            offsets,
            off_flat,
            codes,
            shadow,
        })
    }
```

- [ ] **Step 4: Run tests to confirm green**

Run: `cargo test from_parts 2>&1 | tail -8`
Expected: PASS (`from_parts_reproduces_built_layer`, `from_parts_rejects_bad_lengths`).

- [ ] **Step 5: Confirm warning-free build, then commit**

Run: `cargo build 2>&1 | tail -5` (expect no warnings), then:

```bash
git add src/wave_bitnet/neurons.rs
git commit -m "feat(wave_bitnet): Layer::from_parts (construct a layer from persisted parts)"
```

---

### Task 3: Model format — `save_model` / `load_model` (+ path variants)

Create `persist.rs` with byte I/O primitives, FNV-1a, and the self-contained model format. Add the `Network` accessors it needs.

**Files:**
- Create: `src/wave_bitnet/persist.rs`
- Modify: `src/wave_bitnet/mod.rs` (add `pub mod persist;`)
- Modify: `src/wave_bitnet/network.rs` (add `layers`, `from_layers` accessors)
- Test: inline `#[cfg(test)]` in `src/wave_bitnet/persist.rs`

**Interfaces:**
- Consumes: `Layer::from_parts` (Task 2), `Network::size/layer_count/with_layer` (existing).
- Produces (in `network.rs`): `pub(crate) fn Network::layers(&self) -> &[Layer]`; `pub(crate) fn Network::from_layers(size: u32, layers: Vec<Layer>) -> Network`.
- Produces (in `persist.rs`): `pub fn Network::save_model(&self, w: impl Write) -> io::Result<()>`; `pub fn Network::load_model(r: impl Read) -> io::Result<Network>`; `pub fn Network::save_model_path(&self, path: impl AsRef<Path>) -> io::Result<()>`; `pub fn Network::load_model_path(path: impl AsRef<Path>) -> io::Result<Network>`; module-private `fnv1a`, `write_model_identity`, `read_model_body`, and the byte primitives (`w_*`/`r_*`).

- [ ] **Step 1: Add `Network` accessors in `network.rs`**

Add inside `impl Network` in `src/wave_bitnet/network.rs`:

```rust
    /// Read-only access to the layer stack (used by persistence).
    pub(crate) fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Assemble a `Network` from already-built layers (used by `load_model`). Fresh runtime:
    /// `wave_id = 0`, zeroed delivery scratch, eligibility recording on, no listeners.
    pub(crate) fn from_layers(size: u32, layers: Vec<Layer>) -> Network {
        let l = layers.len();
        let ls = (size as usize) * (size as usize);
        Network {
            size,
            layers,
            wave_id: 0,
            scratch: Scratch { fired: Vec::new(), deliv: (0..l).map(|_| vec![0i32; ls]).collect() },
            record_eligibility: true,
            listeners: (0..l).map(|_| None).collect(),
        }
    }
```

- [ ] **Step 2: Register the module**

In `src/wave_bitnet/mod.rs`, add (keep alphabetical with the others):

```rust
pub mod persist;
```

- [ ] **Step 3: Write `persist.rs` with primitives, fingerprint, and the model format**

Create `src/wave_bitnet/persist.rs`:

```rust
//! `persist` — hand-rolled, std-only binary save/load for a `wave_bitnet` `Network`. Two independent
//! formats: a self-contained **model** (`b"WBNM"`: structure + 2-bit codes, inference-ready) and a
//! **runtime overlay** (`b"WBNR"`: the mutable per-neuron state only) applied onto a loaded model.
//! See docs/superpowers/specs/2026-07-13-wave-bitnet-persist-design.md.

use crate::wave_bitnet::network::Network;
use crate::wave_bitnet::neurons::Layer;
use crate::wave_bitnet::synapse::TopologyLevel;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC_MODEL: &[u8; 4] = b"WBNM";
const VERSION: u16 = 1;

fn inval(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// ---- FNV-1a 64-bit over raw bytes (file integrity + model fingerprint) ----
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---- byte primitives (little-endian). Some are first used by the runtime format (Task 4). ----
fn w_u8(w: &mut impl Write, v: u8) -> io::Result<()> { w.write_all(&[v]) }
fn w_u16(w: &mut impl Write, v: u16) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_u32(w: &mut impl Write, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_u64(w: &mut impl Write, v: u64) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_i16(w: &mut impl Write, v: i16) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_i32(w: &mut impl Write, v: i32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn w_f32(w: &mut impl Write, v: f32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }

fn r_u8(r: &mut impl Read) -> io::Result<u8> { let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0]) }
fn r_u16(r: &mut impl Read) -> io::Result<u16> { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn r_u32(r: &mut impl Read) -> io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn r_u64(r: &mut impl Read) -> io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn r_i16(r: &mut impl Read) -> io::Result<i16> { let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(i16::from_le_bytes(b)) }
fn r_i32(r: &mut impl Read) -> io::Result<i32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) }
fn r_f32(r: &mut impl Read) -> io::Result<f32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }

// length-prefixed vectors (u64 length; no pre-allocation, so a corrupt length fails fast on read).
fn w_vec_i16(w: &mut impl Write, v: &[i16]) -> io::Result<()> { w_u64(w, v.len() as u64)?; for &x in v { w_i16(w, x)?; } Ok(()) }
fn w_vec_u64(w: &mut impl Write, v: &[u64]) -> io::Result<()> { w_u64(w, v.len() as u64)?; for &x in v { w_u64(w, x)?; } Ok(()) }
fn r_vec_i16(r: &mut impl Read) -> io::Result<Vec<i16>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_i16(r)?); } Ok(v) }
fn r_vec_u64(r: &mut impl Read) -> io::Result<Vec<u64>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_u64(r)?); } Ok(v) }

#[allow(dead_code)] // used by the runtime format (Task 4)
fn w_vec_u8(w: &mut impl Write, v: &[u8]) -> io::Result<()> { w_u64(w, v.len() as u64)?; w.write_all(v) }
#[allow(dead_code)]
fn w_vec_i32(w: &mut impl Write, v: &[i32]) -> io::Result<()> { w_u64(w, v.len() as u64)?; for &x in v { w_i32(w, x)?; } Ok(()) }
#[allow(dead_code)]
fn r_vec_u8(r: &mut impl Read) -> io::Result<Vec<u8>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_u8(r)?); } Ok(v) }
#[allow(dead_code)]
fn r_vec_i32(r: &mut impl Read) -> io::Result<Vec<i32>> { let n = r_u64(r)? as usize; let mut v = Vec::new(); for _ in 0..n { v.push(r_i32(r)?); } Ok(v) }

/// Serialize the model IDENTITY: dims + per-layer structure + 2-bit codes. No magic/version/checksum,
/// no runtime state — this is exactly what `model_fingerprint` (Task 4) hashes and what the model file
/// carries between its header and trailing checksum.
fn write_model_identity(w: &mut impl Write, net: &Network) -> io::Result<()> {
    w_u32(w, net.size())?;
    w_u32(w, net.layer_count() as u32)?;
    for lz in net.layers() {
        w_u32(w, lz.topology.len() as u32)?;
        for t in &lz.topology {
            w_i32(w, t.level)?;
            w_u32(w, t.radius)?;
            w_u32(w, t.count)?;
        }
        w_u8(w, lz.leak.0)?;
        w_u8(w, lz.leak.1)?;
        w_u8(w, lz.cooldown_base)?;
        w_i16(w, lz.adapt_bump)?;
        w_u8(w, lz.adapt_decay)?;
        w_u8(w, lz.readout as u8)?;
        w_f32(w, lz.ternary_threshold)?;
        w_vec_i16(w, &lz.threshold)?;
        for level in &lz.occ {
            w_vec_u64(w, level)?;
        }
        w_vec_u64(w, &lz.codes)?;
    }
    Ok(())
}

/// Read a model identity from `r` (already past magic/version) and assemble a `Network`.
fn read_model_body(r: &mut impl Read) -> io::Result<Network> {
    let size = r_u32(r)?;
    let n_layers = r_u32(r)? as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let n_lv = r_u32(r)? as usize;
        let mut topology = Vec::with_capacity(n_lv);
        for _ in 0..n_lv {
            topology.push(TopologyLevel { level: r_i32(r)?, radius: r_u32(r)?, count: r_u32(r)? });
        }
        let leak = (r_u8(r)?, r_u8(r)?);
        let cooldown_base = r_u8(r)?;
        let adapt_bump = r_i16(r)?;
        let adapt_decay = r_u8(r)?;
        let readout = r_u8(r)? != 0;
        let ternary_threshold = r_f32(r)?;
        let threshold = r_vec_i16(r)?;
        let mut occ = Vec::with_capacity(n_lv);
        for _ in 0..n_lv {
            occ.push(r_vec_u64(r)?);
        }
        let codes = r_vec_u64(r)?;
        let layer = Layer::from_parts(
            topology, leak, cooldown_base, adapt_bump, adapt_decay, readout, ternary_threshold,
            threshold, occ, codes, size,
        )
        .map_err(inval)?;
        layers.push(layer);
    }
    Ok(Network::from_layers(size, layers))
}

impl Network {
    /// Serialize a self-contained, inference-ready model (`b"WBNM"`): materialized structure
    /// (occupancy + thresholds) + 2-bit weight codes. No seed dependence, no f32 shadow.
    pub fn save_model(&self, mut w: impl Write) -> io::Result<()> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_MODEL)?;
        w_u16(&mut buf, VERSION)?;
        write_model_identity(&mut buf, self)?;
        let cksum = fnv1a(&buf);
        w_u64(&mut buf, cksum)?;
        w.write_all(&buf)
    }

    /// Load a model saved by `save_model`. Verifies the trailing checksum, magic, and version;
    /// reconstructs each layer (shadow = decode of codes; runtime zeroed). Fails loud on any mismatch.
    pub fn load_model(mut r: impl Read) -> io::Result<Network> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        if buf.len() < MAGIC_MODEL.len() + 2 + 8 {
            return Err(inval("model file too short"));
        }
        let (body, cks) = buf.split_at(buf.len() - 8);
        let stored = u64::from_le_bytes(cks.try_into().unwrap());
        if fnv1a(body) != stored {
            return Err(inval("model checksum mismatch (corrupt file)"));
        }
        let mut c = io::Cursor::new(body);
        let mut magic = [0u8; 4];
        c.read_exact(&mut magic)?;
        if &magic != MAGIC_MODEL {
            return Err(inval("bad magic (not a wave_bitnet model file)"));
        }
        let ver = r_u16(&mut c)?;
        if ver != VERSION {
            return Err(inval(format!("unsupported model version {ver} (expected {VERSION})")));
        }
        read_model_body(&mut c)
    }

    /// Convenience: `save_model` to a file path.
    pub fn save_model_path(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_model(File::create(path)?)
    }

    /// Convenience: `load_model` from a file path.
    pub fn load_model_path(path: impl AsRef<Path>) -> io::Result<Network> {
        Network::load_model(File::open(path)?)
    }
}
```

- [ ] **Step 4: Write the model tests**

Append to `src/wave_bitnet/persist.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::{Config, LayerConfig};
    use crate::wave_bitnet::synapse::TopologyLevel;

    fn small_net() -> Network {
        let up = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 9830, threshold_jitter: 64,
            baseline_init: 6, adapt_bump: 5, adapt_decay: 6,
        };
        let mid = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 8 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
            ],
            ..up.clone()
        };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Network::new_with_readout(Config { seed: 0xABCD, size: 8, layers: vec![up, mid, top] })
    }

    /// Assert two networks have equal persisted + derived structure (not runtime; shadow checked as
    /// the decode of codes).
    fn assert_models_eq(a: &Network, b: &Network) {
        assert_eq!(a.size(), b.size());
        assert_eq!(a.layer_count(), b.layer_count());
        for z in 0..a.layer_count() {
            a.with_layer(z, |la| {
                b.with_layer(z, |lb| {
                    assert_eq!(la.topology, lb.topology, "layer {z} topology");
                    assert_eq!(la.leak, lb.leak);
                    assert_eq!(la.cooldown_base, lb.cooldown_base);
                    assert_eq!(la.adapt_bump, lb.adapt_bump);
                    assert_eq!(la.adapt_decay, lb.adapt_decay);
                    assert_eq!(la.readout, lb.readout);
                    assert_eq!(la.ternary_threshold, lb.ternary_threshold);
                    assert_eq!(la.threshold, lb.threshold, "layer {z} threshold");
                    assert_eq!(la.occ, lb.occ, "layer {z} occ");
                    assert_eq!(la.codes, lb.codes, "layer {z} codes");
                    assert_eq!(la.total_slots, lb.total_slots);
                    assert_eq!(la.slot_bases, lb.slot_bases);
                    assert_eq!(la.neigh, lb.neigh);
                    assert_eq!(la.occ_wpn, lb.occ_wpn);
                    assert_eq!(la.offsets, lb.offsets);
                    assert_eq!(la.off_flat, lb.off_flat);
                    // loaded shadow is the decode of codes
                    for s in 0..lb.shadow.len() {
                        assert_eq!(lb.shadow[s], lb.weight_at(s) as f32);
                    }
                })
            });
        }
    }

    #[test]
    fn model_roundtrip_fields() {
        let net = small_net();
        let mut bytes = Vec::new();
        net.save_model(&mut bytes).unwrap();
        let loaded = Network::load_model(&bytes[..]).unwrap();
        assert_models_eq(&net, &loaded);
    }

    #[test]
    fn model_roundtrip_inference_equivalent() {
        let net = small_net();
        let mut bytes = Vec::new();
        net.save_model(&mut bytes).unwrap();
        let mut a = Network::load_model(&bytes[..]).unwrap();
        let mut b = Network::load_model(&bytes[..]).unwrap();
        let inputs: [&[u32]; 6] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[], &[]];
        for inp in inputs {
            a.wave(inp);
            b.wave(inp);
            for z in 0..a.layer_count() {
                assert_eq!(a.layer_decide_potential(z), b.layer_decide_potential(z), "layer {z} decide_potential");
            }
        }
    }

    #[test]
    fn model_bytes_are_stable() {
        let net = small_net();
        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        net.save_model(&mut b1).unwrap();
        net.save_model(&mut b2).unwrap();
        assert_eq!(b1, b2);
    }

    #[test]
    fn model_rejects_bad_magic_version_and_corruption() {
        // bad magic (valid checksum so magic check is reached)
        let mut bad = b"ZZZZ".to_vec();
        w_u16(&mut bad, VERSION).unwrap();
        let ck = fnv1a(&bad);
        w_u64(&mut bad, ck).unwrap();
        assert_eq!(Network::load_model(&bad[..]).unwrap_err().kind(), io::ErrorKind::InvalidData);

        // unsupported version
        let mut badv = MAGIC_MODEL.to_vec();
        w_u16(&mut badv, VERSION + 1).unwrap();
        let ck = fnv1a(&badv);
        w_u64(&mut badv, ck).unwrap();
        assert_eq!(Network::load_model(&badv[..]).unwrap_err().kind(), io::ErrorKind::InvalidData);

        // corrupt body (flip a payload byte)
        let net = small_net();
        let mut good = Vec::new();
        net.save_model(&mut good).unwrap();
        good[8] ^= 0xFF;
        assert_eq!(Network::load_model(&good[..]).unwrap_err().kind(), io::ErrorKind::InvalidData);

        // truncated
        let net = small_net();
        let mut full = Vec::new();
        net.save_model(&mut full).unwrap();
        let half = &full[..full.len() / 2];
        assert!(Network::load_model(half).is_err());
    }

    #[test]
    fn model_path_roundtrip() {
        let net = small_net();
        let path = std::env::temp_dir().join("wave_bitnet_persist_model_test.wbm");
        net.save_model_path(&path).unwrap();
        let loaded = Network::load_model_path(&path).unwrap();
        assert_models_eq(&net, &loaded);
        let _ = std::fs::remove_file(&path);
    }
}
```

- [ ] **Step 5: Run to confirm failure first (module not yet compiling before Step 3), then green**

Run: `cargo test persist:: 2>&1 | tail -20`
Expected: after Steps 1-4, PASS — `model_roundtrip_fields`, `model_roundtrip_inference_equivalent`, `model_bytes_are_stable`, `model_rejects_bad_magic_version_and_corruption`, `model_path_roundtrip`.

- [ ] **Step 6: Confirm warning-free build**

Run: `cargo build 2>&1 | tail -5`
Expected: no warnings (the four runtime-only primitives carry `#[allow(dead_code)]`, removed in Task 4).

- [ ] **Step 7: Commit**

```bash
git add src/wave_bitnet/persist.rs src/wave_bitnet/mod.rs src/wave_bitnet/network.rs
git commit -m "feat(wave_bitnet): self-contained model save/load (.wbm)"
```

---

### Task 4: Runtime overlay — `save_runtime` / `apply_runtime` (+ path variants)

Add the runtime format: a fingerprint-bound overlay of the eight mutable per-neuron arrays + `wave_id`, applied in place onto a loaded model. Add the `Network` accessors it needs and remove the temporary `#[allow(dead_code)]`s.

**Files:**
- Modify: `src/wave_bitnet/persist.rs` (add `model_fingerprint`, runtime (de)serializers, remove dead-code allows)
- Modify: `src/wave_bitnet/network.rs` (add `layers_mut`, `wave_id`, `set_wave_id`)
- Test: inline `#[cfg(test)]` in `src/wave_bitnet/persist.rs`

**Interfaces:**
- Consumes: `Network::layers` (Task 3), `write_model_identity`/`fnv1a`/byte primitives (Task 3).
- Produces (in `network.rs`): `pub(crate) fn Network::layers_mut(&mut self) -> &mut [Layer]`; `pub(crate) fn Network::wave_id(&self) -> usize`; `pub(crate) fn Network::set_wave_id(&mut self, w: usize)`.
- Produces (in `persist.rs`): `pub fn Network::model_fingerprint(&self) -> u64`; `pub fn Network::save_runtime(&self, w: impl Write) -> io::Result<()>`; `pub fn Network::apply_runtime(&mut self, r: impl Read) -> io::Result<()>`; `pub fn Network::save_runtime_path(&self, path: impl AsRef<Path>) -> io::Result<()>`; `pub fn Network::apply_runtime_path(&mut self, path: impl AsRef<Path>) -> io::Result<()>`.

- [ ] **Step 1: Add `Network` accessors in `network.rs`**

Add inside `impl Network` in `src/wave_bitnet/network.rs`:

```rust
    /// Mutable access to the layer stack (used by the runtime overlay).
    pub(crate) fn layers_mut(&mut self) -> &mut [Layer] {
        &mut self.layers
    }

    /// The wave counter (persisted/restored by the runtime overlay).
    pub(crate) fn wave_id(&self) -> usize {
        self.wave_id
    }

    /// Restore the wave counter (used by `apply_runtime`).
    pub(crate) fn set_wave_id(&mut self, w: usize) {
        self.wave_id = w;
    }
```

- [ ] **Step 2: Remove the four `#[allow(dead_code)]`s in `persist.rs`**

In `src/wave_bitnet/persist.rs`, delete the four `#[allow(dead_code)]` attribute lines above `w_vec_u8`, `w_vec_i32`, `r_vec_u8`, `r_vec_i32` (they are used starting this task). Also delete the trailing `// used by the runtime format (Task 4)` comment on the first one.

- [ ] **Step 3: Write the failing runtime tests**

Add these tests inside the existing `mod tests` in `src/wave_bitnet/persist.rs`:

```rust
    #[test]
    fn runtime_resume_equivalent() {
        let net = small_net();
        let mut model_bytes = Vec::new();
        net.save_model(&mut model_bytes).unwrap();

        // A: load, drive several waves, snapshot runtime
        let mut a = Network::load_model(&model_bytes[..]).unwrap();
        let inputs: [&[u32]; 5] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[]];
        for inp in inputs {
            a.wave(inp);
        }
        let mut rt_bytes = Vec::new();
        a.save_runtime(&mut rt_bytes).unwrap();

        // B: fresh load of the SAME model, apply A's runtime overlay
        let mut b = Network::load_model(&model_bytes[..]).unwrap();
        b.apply_runtime(&rt_bytes[..]).unwrap();

        // immediate state equality
        assert_eq!(a.wave_id(), b.wave_id());
        for z in 0..a.layer_count() {
            a.with_layer(z, |la| {
                b.with_layer(z, |lb| {
                    assert_eq!(la.potential, lb.potential, "layer {z} potential");
                    assert_eq!(la.cooldown, lb.cooldown, "layer {z} cooldown");
                    assert_eq!(la.adapt, lb.adapt, "layer {z} adapt");
                    assert_eq!(la.pending, lb.pending, "layer {z} pending");
                    assert_eq!(la.elig_pre, lb.elig_pre, "layer {z} elig_pre");
                    assert_eq!(la.elig_post, lb.elig_post, "layer {z} elig_post");
                    assert_eq!(la.decide_potential, lb.decide_potential, "layer {z} decide_potential");
                    assert_eq!(la.decide_eff, lb.decide_eff, "layer {z} decide_eff");
                })
            });
        }

        // continuing both must stay identical (true resume)
        let more: [&[u32]; 3] = [&[1, 4], &[], &[7]];
        for inp in more {
            a.wave(inp);
            b.wave(inp);
            for z in 0..a.layer_count() {
                assert_eq!(a.layer_decide_potential(z), b.layer_decide_potential(z), "resumed layer {z}");
            }
        }
    }

    #[test]
    fn runtime_binding_rejects_wrong_model() {
        // source model + its runtime
        let a = small_net();
        let mut model_a = Vec::new();
        a.save_model(&mut model_a).unwrap();
        let mut a = Network::load_model(&model_a[..]).unwrap();
        a.wave(&[0, 1, 2]);
        let mut rt = Vec::new();
        a.save_runtime(&mut rt).unwrap();

        // a DIFFERENT model (different seed → different occupancy/codes → different fingerprint)
        let other = {
            let up = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }],
                leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 9830, threshold_jitter: 64,
                baseline_init: 6, adapt_bump: 5, adapt_decay: 6,
            };
            let mid = LayerConfig {
                topology: vec![
                    TopologyLevel { level: 1, radius: 2, count: 8 },
                    TopologyLevel { level: 0, radius: 1, count: 2 },
                ],
                ..up.clone()
            };
            let top = LayerConfig { topology: vec![], ..up.clone() };
            Network::new_with_readout(Config { seed: 0x9999, size: 8, layers: vec![up, mid, top] })
        };
        let mut other_bytes = Vec::new();
        other.save_model(&mut other_bytes).unwrap();
        let mut other = Network::load_model(&other_bytes[..]).unwrap();

        let err = other.apply_runtime(&rt[..]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn runtime_fingerprint_stable_across_save_load() {
        let net = small_net();
        let fp = net.model_fingerprint();
        let mut bytes = Vec::new();
        net.save_model(&mut bytes).unwrap();
        let loaded = Network::load_model(&bytes[..]).unwrap();
        assert_eq!(fp, loaded.model_fingerprint());
    }

    #[test]
    fn runtime_path_roundtrip() {
        let net = small_net();
        let mut model_bytes = Vec::new();
        net.save_model(&mut model_bytes).unwrap();
        let mut a = Network::load_model(&model_bytes[..]).unwrap();
        a.wave(&[0, 1, 2]);
        a.wave(&[]);

        let path = std::env::temp_dir().join("wave_bitnet_persist_runtime_test.wbr");
        a.save_runtime_path(&path).unwrap();
        let mut b = Network::load_model(&model_bytes[..]).unwrap();
        b.apply_runtime_path(&path).unwrap();
        assert_eq!(a.wave_id(), b.wave_id());
        a.with_layer(1, |la| b.with_layer(1, |lb| assert_eq!(la.potential, lb.potential)));
        let _ = std::fs::remove_file(&path);
    }
```

- [ ] **Step 4: Run to confirm failure**

Run: `cargo test persist::tests::runtime 2>&1 | tail -8`
Expected: compile error — `no method named save_runtime`/`model_fingerprint`.

- [ ] **Step 5: Implement the fingerprint + runtime format**

In `src/wave_bitnet/persist.rs`, add the runtime magic constant next to `MAGIC_MODEL`:

```rust
const MAGIC_RUNTIME: &[u8; 4] = b"WBNR";
```

Then add these methods to the `impl Network` block (after `load_model_path`):

```rust
    /// Stable 64-bit fingerprint of a model's identity (structure + codes), independent of runtime
    /// state. Written into a runtime overlay and re-checked on `apply_runtime` so an overlay can only
    /// be applied to the model it was taken from.
    pub fn model_fingerprint(&self) -> u64 {
        let mut buf = Vec::new();
        write_model_identity(&mut buf, self).expect("Vec<u8> write is infallible");
        fnv1a(&buf)
    }

    /// Serialize ONLY the mutable per-neuron runtime state + `wave_id` (`b"WBNR"`). Not standalone:
    /// `apply_runtime` restores it onto a matching model. `scratch.deliv` is provably all-zero between
    /// waves, so it is not stored.
    pub fn save_runtime(&self, mut w: impl Write) -> io::Result<()> {
        let mut buf = Vec::new();
        buf.write_all(MAGIC_RUNTIME)?;
        w_u16(&mut buf, VERSION)?;
        w_u64(&mut buf, self.model_fingerprint())?;
        w_u32(&mut buf, self.size())?;
        w_u32(&mut buf, self.layer_count() as u32)?;
        w_u64(&mut buf, self.wave_id() as u64)?;
        for lz in self.layers() {
            w_vec_i16(&mut buf, &lz.potential)?;
            w_vec_u8(&mut buf, &lz.cooldown)?;
            w_vec_i32(&mut buf, &lz.adapt)?;
            w_vec_i32(&mut buf, &lz.pending)?;
            w_vec_i32(&mut buf, &lz.elig_pre)?;
            w_vec_i32(&mut buf, &lz.elig_post)?;
            w_vec_i16(&mut buf, &lz.decide_potential)?;
            w_vec_i32(&mut buf, &lz.decide_eff)?;
        }
        let cksum = fnv1a(&buf);
        w_u64(&mut buf, cksum)?;
        w.write_all(&buf)
    }

    /// Restore a runtime overlay (from `save_runtime`) onto this network, in place. Verifies checksum,
    /// magic, version, that the overlay's `model_fingerprint` matches this model, and that every array
    /// length matches, all BEFORE mutating any state. Fails loud on any mismatch.
    pub fn apply_runtime(&mut self, mut r: impl Read) -> io::Result<()> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        if buf.len() < MAGIC_RUNTIME.len() + 2 + 8 {
            return Err(inval("runtime file too short"));
        }
        let (body, cks) = buf.split_at(buf.len() - 8);
        let stored = u64::from_le_bytes(cks.try_into().unwrap());
        if fnv1a(body) != stored {
            return Err(inval("runtime checksum mismatch (corrupt file)"));
        }
        let mut c = io::Cursor::new(body);
        let mut magic = [0u8; 4];
        c.read_exact(&mut magic)?;
        if &magic != MAGIC_RUNTIME {
            return Err(inval("bad magic (not a wave_bitnet runtime file)"));
        }
        let ver = r_u16(&mut c)?;
        if ver != VERSION {
            return Err(inval(format!("unsupported runtime version {ver} (expected {VERSION})")));
        }
        let model_fp = r_u64(&mut c)?;
        if model_fp != self.model_fingerprint() {
            return Err(inval("runtime overlay does not belong to this model (fingerprint mismatch)"));
        }
        let size = r_u32(&mut c)?;
        let n_layers = r_u32(&mut c)? as usize;
        if size != self.size() || n_layers != self.layer_count() {
            return Err(inval("runtime dims do not match this model"));
        }
        let wave_id = r_u64(&mut c)? as usize;
        let ls = (size as usize) * (size as usize);
        // read + validate every layer's arrays BEFORE mutating (keep apply all-or-nothing)
        type LayerRt = (Vec<i16>, Vec<u8>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i16>, Vec<i32>);
        let mut staged: Vec<LayerRt> = Vec::with_capacity(n_layers);
        for z in 0..n_layers {
            let potential = r_vec_i16(&mut c)?;
            let cooldown = r_vec_u8(&mut c)?;
            let adapt = r_vec_i32(&mut c)?;
            let pending = r_vec_i32(&mut c)?;
            let elig_pre = r_vec_i32(&mut c)?;
            let elig_post = r_vec_i32(&mut c)?;
            let decide_potential = r_vec_i16(&mut c)?;
            let decide_eff = r_vec_i32(&mut c)?;
            for (name, len) in [
                ("potential", potential.len()), ("cooldown", cooldown.len()), ("adapt", adapt.len()),
                ("pending", pending.len()), ("elig_pre", elig_pre.len()), ("elig_post", elig_post.len()),
                ("decide_potential", decide_potential.len()), ("decide_eff", decide_eff.len()),
            ] {
                if len != ls {
                    return Err(inval(format!("layer {z} {name} length {len} != ls {ls}")));
                }
            }
            staged.push((potential, cooldown, adapt, pending, elig_pre, elig_post, decide_potential, decide_eff));
        }
        // commit
        let layers = self.layers_mut();
        for (z, (potential, cooldown, adapt, pending, elig_pre, elig_post, decide_potential, decide_eff)) in staged.into_iter().enumerate() {
            layers[z].potential = potential;
            layers[z].cooldown = cooldown;
            layers[z].adapt = adapt;
            layers[z].pending = pending;
            layers[z].elig_pre = elig_pre;
            layers[z].elig_post = elig_post;
            layers[z].decide_potential = decide_potential;
            layers[z].decide_eff = decide_eff;
        }
        self.set_wave_id(wave_id);
        Ok(())
    }

    /// Convenience: `save_runtime` to a file path.
    pub fn save_runtime_path(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.save_runtime(File::create(path)?)
    }

    /// Convenience: `apply_runtime` from a file path.
    pub fn apply_runtime_path(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.apply_runtime(File::open(path)?)
    }
```

- [ ] **Step 6: Run the runtime tests to confirm green**

Run: `cargo test persist::tests::runtime 2>&1 | tail -12`
Expected: PASS — `runtime_resume_equivalent`, `runtime_binding_rejects_wrong_model`, `runtime_fingerprint_stable_across_save_load`, `runtime_path_roundtrip`.

- [ ] **Step 7: Full test suite + warning-free build**

Run: `cargo test 2>&1 | tail -20` (expect all green) then `cargo build 2>&1 | tail -5` (expect no warnings — the dead-code allows are gone and every primitive is now used).

- [ ] **Step 8: Commit**

```bash
git add src/wave_bitnet/persist.rs src/wave_bitnet/network.rs
git commit -m "feat(wave_bitnet): runtime overlay save/apply (.wbr), fingerprint-bound to its model"
```

---

## Self-Review

**Spec coverage:**
- Two independent formats, layered (model + overlay) → Tasks 3, 4. ✓
- Self-contained structure (occ + threshold stored) → `write_model_identity` (Task 3) + `Layer::from_parts` (Task 2). ✓
- 2-bit codes only; shadow = decode → `Layer::from_parts` (Task 2), asserted in `assert_models_eq` (Task 3). ✓
- Std-only hand-rolled LE binary → primitives (Task 3). ✓
- Fingerprint integrity + binding → `fnv1a` file checksum (Tasks 3/4) + `model_fingerprint` binding (Task 4). ✓
- Fail loud on mismatch → magic/version/checksum/length/fingerprint checks return `InvalidData` (Tasks 3/4). ✓
- New module `persist.rs`; `Layer::from_parts` + `Network::from_layers` construction path → Tasks 2, 3. ✓
- API: `save_model`/`load_model`/`save_runtime`/`apply_runtime` (+ path variants) → Tasks 3, 4. ✓
- Runtime stores eight arrays + `wave_id`, not `deliv`/`listeners`/`record_eligibility` → `save_runtime` (Task 4). ✓
- All spec test categories (model field round-trip, inference-equivalence, runtime resume, binding mismatch, malformed input, byte-stability) → Tasks 3, 4. ✓

**Placeholder scan:** none — every code and test step is complete.

**Type consistency:** `derive_layout`/`DerivedLayout` (Task 1) consumed identically in `Layer::new` (Task 1) and `Layer::from_parts` (Task 2); `write_model_identity` (Task 3) reused by `model_fingerprint` (Task 4); accessor names (`layers`/`from_layers` Task 3, `layers_mut`/`wave_id`/`set_wave_id` Task 4) match their call sites; `MAGIC_MODEL`/`MAGIC_RUNTIME`/`VERSION` consistent. ✓
