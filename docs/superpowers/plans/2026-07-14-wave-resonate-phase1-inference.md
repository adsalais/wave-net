# wave_resonate Phase 1 — BRF inference engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task (this repo mandates inline execution, not subagent-driven). Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Stand up a correct, deterministic **BRF (complex Resonate-and-Fire) forward inference engine**
as an island in `src/wave_resonate/`, integrating integer ternary spike-deliveries on the copied topology
substrate. No training yet.

**Architecture:** Duplicate `wave_driven`'s topology/ternary substrate (occupancy bitset + 2-bit codes +
deferred one-hop routing) but replace LIF integration with the BRF complex oscillator (f32 `x,y,q` +
per-neuron `ω,b′`). The membrane update is **dense** (every neuron every wave); spike delivery is
**sparse** (firer-gated). No frontier, no sparse/dense Mode split — `process_layer` is always the same.

**Tech Stack:** Rust edition 2024, standard library only, `#[cfg(test)]` inline tests.

## Global Constraints

- **Standard library only** in `src/`; **warning-free build**. No `unsafe` in Phase 1 (delivery scan may
  adopt the sanctioned `unsafe` later; Phase 1 uses safe indexing).
- **Determinism is a hard requirement** — every result a pure function of `(seed, config, input)`.
  Single-threaded f32; no nondeterministic ops.
- **Ternary weights stay** ±1/0 as 2-bit codes (`0b00`→0, `0b01`→+1, `0b11`→−1), init to the procedural
  ±1 sign; O(synapses).
- **f32 membrane** state is O(neurons) only.
- **BRF equations (exact, from the spec):** `p(ω)=(−1+√(1−(δω)²))/δ`; `b=p(ω)−|b′|−q`;
  `x←x+δ(b·x−ω·y+I)`; `y←y+δ(ω·x+b·y)`; `z=Θ(x−ϑ_c−q)`; `q←γ·q+z`. Constants `γ=0.9`, `ϑ_c=1`,
  divergence boundary requires `δ·ω≤1`.
- **`size` a power of two** (toroidal wrap is a bitmask); local index `y*size+x`.
- **One commit per task**, conventional-commit messages, **no `Co-Authored-By` trailer**.

---

## File Structure

- `src/wave_resonate/mod.rs` — module wiring
- `src/wave_resonate/synapse.rs` — copied verbatim from `wave_driven/synapse.rs` (engine-agnostic hashing,
  topology, `decode_cell`, `sample_distinct_cells`, `random_l0_input`)
- `src/wave_resonate/config.rs` — `Config` (global `dt,gamma,theta_c`) + `LayerConfig` (per-layer
  topology/init) + `validate`
- `src/wave_resonate/neurons.rs` — `pw()` + `Layer` SoA (BRF state + copied occupancy substrate + codes)
  + construction + accessors
- `src/wave_resonate/wave.rs` — `process_layer` (drain → transducer/readout/compute → generate)
- `src/wave_resonate/network.rs` — `Network` (build, `wave`, `reset_state`, listeners, accessors)
- Modify `src/lib.rs` — register `pub mod wave_resonate;`

---

### Task 1: Module scaffold + copied synapse substrate + registration

**Files:**
- Create: `src/wave_resonate/mod.rs`
- Create: `src/wave_resonate/synapse.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces: module `wave_resonate` with `pub mod synapse;`; `synapse` re-exports (verbatim from
  wave_driven) `TopologyLevel{level,radius,count}`, `mix`, `key`, `map_range`, `local_of`, `xy_of`,
  `wrap`, `neigh_size`, `decode_cell`, `sample_distinct_cells`, `random_l0_input`, `P_TARGET`,
  `P_THRESHOLD`, `P_INPUT`.

- [ ] **Step 1: Copy `synapse.rs` verbatim.** Copy `src/wave_driven/synapse.rs` byte-for-byte to
  `src/wave_resonate/synapse.rs` (it is engine-agnostic; its inline tests come with it).

- [ ] **Step 2: Create `mod.rs`** with only the modules that exist so far:

```rust
//! `wave_resonate` — an independent engine (island duplicated from `wave_driven`) whose neuron is the
//! Balanced Resonate-and-Fire (BRF) complex-membrane oscillator (Higuchi et al., ICML 2024), integrating
//! integer ternary spike-deliveries. f32 membrane + ternary ±1/0 weights. Phase 1: forward inference.
//! Spec: docs/superpowers/specs/2026-07-14-wave-resonate-brf-hypr-design.md.

pub mod config;
pub mod network;
pub mod neurons;
pub mod synapse;
pub mod wave;
```

Note: `config`/`neurons`/`wave`/`network` don't exist yet — this will not compile until Task 2+. To keep
each task green, temporarily comment out the not-yet-created modules and uncomment them as each task lands.
(Alternative: create empty stub files. Prefer stubs so `cargo build` stays green — see Step 3.)

- [ ] **Step 3: Create empty stub files** so the tree compiles: `config.rs`, `neurons.rs`, `wave.rs`,
  `network.rs` each containing only a module doc comment line (e.g. `//! (stub — filled in a later task)`).

- [ ] **Step 4: Register in `lib.rs`.** Add after the `wave_driven` line:

```rust
pub mod wave_resonate; // BRF resonate-and-fire engine (island); Phase 1: inference
```

- [ ] **Step 5: Run tests to verify the copied substrate is green**

Run: `cargo test wave_resonate::synapse`
Expected: PASS (the two copied tests `decode_center_is_self_and_corners_wrap`,
`sample_is_distinct_bounded_and_deterministic`).

- [ ] **Step 6: Verify warning-free build**

Run: `cargo build 2>&1 | grep -i warning; echo done`
Expected: `done` with no warnings (stubs are just doc comments).

- [ ] **Step 7: Commit**

```bash
git add src/wave_resonate/mod.rs src/wave_resonate/synapse.rs src/wave_resonate/config.rs \
        src/wave_resonate/neurons.rs src/wave_resonate/wave.rs src/wave_resonate/network.rs src/lib.rs
git commit -m "feat(wave_resonate): module scaffold + copied synapse substrate"
```

---

### Task 2: `config.rs` — BRF Config/LayerConfig + validate

**Files:**
- Modify: `src/wave_resonate/config.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct LayerConfig {
      pub topology: Vec<TopologyLevel>,
      pub inhibitor_ratio: u32,     // Q16: inhibitory iff (hash & 0xFFFF) < inhibitor_ratio
      pub omega_init: (f32, f32),   // per-neuron ω ~ U[lo, hi]
      pub b_offset_init: (f32, f32),// per-neuron b' ~ U[lo, hi], b' >= 0
      pub tau_out: f32,             // readout leaky-integrator time constant (used only if readout)
  }
  pub struct Config {
      pub seed: u64,
      pub size: u32,      // power of two
      pub dt: f32,        // δ integration step (global)
      pub gamma: f32,     // refractory decay γ (global)
      pub theta_c: f32,   // base threshold ϑ_c (global)
      pub layers: Vec<LayerConfig>,
  }
  impl Config { pub fn layer_size(&self)->usize; pub fn n_total(&self)->usize;
                pub fn demo()->Config; pub fn validate(&self)->Result<(),String>; }
  ```

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::synapse::TopologyLevel;

    #[test]
    fn demo_is_valid() { assert!(Config::demo().validate().is_ok()); }

    #[test]
    fn rejects_non_power_of_two_size() {
        let mut c = Config::demo(); c.size = 12; assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_divergence_boundary_violation() {
        // δ·ω must be ≤ 1; push omega_init.hi over 1/dt
        let mut c = Config::demo();
        c.dt = 0.1; c.layers[0].omega_init = (5.0, 20.0); // 0.1*20 = 2 > 1
        let e = c.validate().unwrap_err();
        assert!(e.contains("divergence") || e.contains("δ·ω") || e.contains("dt*omega"), "descriptive: {e}");
    }

    #[test]
    fn rejects_fan_in_over_neighborhood() {
        let mut c = Config::demo();
        c.layers[0].topology = vec![TopologyLevel { level: 1, radius: 2, count: 30 }]; // 30 > 25
        let e = c.validate().unwrap_err();
        assert!(e.contains("count") && e.contains("neighborhood"), "descriptive: {e}");
    }

    #[test]
    fn rejects_bad_omega_or_gamma() {
        let mut c = Config::demo(); c.gamma = 1.5; assert!(c.validate().is_err());
        let mut c2 = Config::demo(); c2.layers[0].omega_init = (10.0, 5.0); // lo > hi
        assert!(c2.validate().is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test wave_resonate::config -- --nocapture`
Expected: FAIL (compile error — `Config`/`LayerConfig` not defined).

- [ ] **Step 3: Implement `config.rs`**

```rust
//! Construction input for the BRF engine: a shared square `size`, a seed, global BRF constants
//! (`dt, gamma, theta_c`), and one `LayerConfig` per layer. Mirrors wave_driven::config but drops all
//! LIF/adaptation fields (leak/cooldown/adapt_*) and adds the resonator init ranges + divergence-boundary
//! validation (`δ·ω ≤ 1`).

use crate::wave_resonate::synapse::{neigh_size, TopologyLevel};

#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub topology: Vec<TopologyLevel>,
    pub inhibitor_ratio: u32,
    pub omega_init: (f32, f32),
    pub b_offset_init: (f32, f32),
    pub tau_out: f32,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub size: u32,
    pub dt: f32,
    pub gamma: f32,
    pub theta_c: f32,
    pub layers: Vec<LayerConfig>,
}

impl Config {
    pub fn layer_size(&self) -> usize { (self.size as usize) * (self.size as usize) }
    pub fn n_total(&self) -> usize { self.layer_size() * self.layers.len() }

    /// A small, valid, deterministic network for tests and bring-up. δ·ω_hi = 0.05·10 = 0.5 ≤ 1.
    pub fn demo() -> Config {
        let mk = |topology: Vec<TopologyLevel>| LayerConfig {
            topology, inhibitor_ratio: 9830, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0,
        };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 6 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 6 }]),
            mk(vec![]),
        ];
        Config { seed: 0x1234_5678_9ABC_DEF0, size: 16, dt: 0.05, gamma: 0.9, theta_c: 1.0, layers }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.size < 1 || !self.size.is_power_of_two() {
            return Err(format!("size must be a power of two >= 1, got {}", self.size));
        }
        if self.layers.is_empty() { return Err("layers must not be empty".into()); }
        if !(self.dt > 0.0) { return Err(format!("dt must be > 0, got {}", self.dt)); }
        if !(self.gamma >= 0.0 && self.gamma <= 1.0) { return Err(format!("gamma must be in [0,1], got {}", self.gamma)); }
        if !(self.theta_c > 0.0) { return Err(format!("theta_c must be > 0, got {}", self.theta_c)); }
        for (z, lc) in self.layers.iter().enumerate() {
            let (olo, ohi) = lc.omega_init;
            if !(olo > 0.0 && ohi >= olo) { return Err(format!("layer {z}: omega_init must satisfy 0 < lo <= hi, got {olo}..{ohi}")); }
            if self.dt * ohi > 1.0 { return Err(format!("layer {z}: divergence boundary requires δ·ω ≤ 1 (dt*omega); dt={} · omega_hi={} = {} > 1", self.dt, ohi, self.dt*ohi)); }
            let (blo, bhi) = lc.b_offset_init;
            if !(blo >= 0.0 && bhi >= blo) { return Err(format!("layer {z}: b_offset_init must satisfy 0 <= lo <= hi, got {blo}..{bhi}")); }
            if !(lc.tau_out > 0.0) { return Err(format!("layer {z}: tau_out must be > 0, got {}", lc.tau_out)); }
            for t in &lc.topology {
                let n = neigh_size(t.radius);
                if t.count as usize > n {
                    return Err(format!("layer {z}: topology count {} exceeds neighborhood size {} for radius {} (a per-cell occupancy bitset caps fan-in at (2r+1)^2)", t.count, n, t.radius));
                }
            }
        }
        Ok(())
    }
}
```

Then append the test module from Step 1.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test wave_resonate::config`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/config.rs
git commit -m "feat(wave_resonate): BRF Config/LayerConfig + divergence-boundary validate"
```

---

### Task 3: `neurons.rs` — `pw()` + `Layer` SoA + construction + accessors

**Files:**
- Modify: `src/wave_resonate/neurons.rs`

**Interfaces:**
- Consumes: `LayerConfig` (Task 2), `synapse::*` (Task 1).
- Produces:
  ```rust
  pub fn pw(omega: f32, dt: f32) -> f32;            // (−1+√(1−(δω)²))/δ
  pub(crate) const WCODE: [i8; 4];                  // [0,1,0,-1]
  pub struct Layer {
      pub x: Vec<f32>, pub y: Vec<f32>, pub q: Vec<f32>,     // BRF state (readout reuses x as accumulator)
      pub omega: Vec<f32>, pub b_off: Vec<f32>,              // per-neuron params (b_off >= 0)
      pub pending: Vec<i32>,                                 // integer delivery accumulator
      pub dt: f32, pub gamma: f32, pub theta_c: f32, pub kappa: f32, // dynamics constants (kappa = exp(-dt/tau_out))
      pub transducer: bool, pub readout: bool,
      pub topology: Vec<TopologyLevel>,
      pub total_slots: usize, pub slot_bases: Vec<usize>, pub neigh: Vec<usize>, pub occ_wpn: Vec<usize>,
      pub occ: Vec<Vec<u64>>, pub offsets: Vec<Vec<(i8,i8)>>, pub off_flat: Vec<Vec<i32>>, pub codes: Vec<u64>,
  }
  impl Layer {
      pub fn new(cfg:&LayerConfig, dt:f32, gamma:f32, theta_c:f32, seed:u64, layer_index:u32, size:u32) -> Layer;
      pub fn weight_at(&self, widx: usize) -> i8;
      pub fn synapse_count(&self) -> usize;
      pub fn slot_base(&self, level_idx: usize) -> usize;
      pub fn for_wired(&self, lvl: usize, i: usize, f: impl FnMut(usize, usize));
      pub fn decode(&self, lvl: usize, src_local: u32, cell: usize, size: u32) -> u32;
  }
  ```

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::LayerConfig;
    use crate::wave_resonate::synapse::TopologyLevel;

    fn lc(topology: Vec<TopologyLevel>) -> LayerConfig {
        LayerConfig { topology, inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0 }
    }

    #[test]
    fn new_wires_exactly_count_distinct_cells_and_is_deterministic() {
        let size = 8u32; let ls = (size*size) as usize;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 16 }]);
        let a = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        let b = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        assert_eq!(a.total_slots, 16);
        for i in 0..ls {
            let mut cells = Vec::new();
            a.for_wired(0, i, |_r, c| cells.push(c));
            assert_eq!(cells.len(), 16);
            assert!(cells.windows(2).all(|w| w[0] < w[1]), "ascending cell order");
        }
        assert_eq!(a.occ, b.occ, "deterministic occupancy");
        assert_eq!(a.codes, b.codes, "deterministic ±1 codes");
        assert_eq!(a.omega, b.omega, "deterministic omega");
        assert_eq!(a.b_off, b.b_off, "deterministic b_off");
    }

    #[test]
    fn omega_and_b_off_in_range() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let l = Layer::new(&cfg, 0.05, 0.9, 1.0, 3, 0, size);
        assert!(l.omega.iter().all(|&o| o >= 5.0 && o <= 10.0), "omega in [5,10]");
        assert!(l.b_off.iter().all(|&b| b >= 0.1 && b <= 1.0), "b_off in [0.1,1.0]");
    }

    #[test]
    fn weight_at_decodes_pm1_from_codes() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]);
        let l = Layer::new(&cfg, 0.05, 0.9, 1.0, 3, 0, size); // inhibitor_ratio 0 => all +1
        for s in 0..l.synapse_count() { assert!(matches!(l.weight_at(s), 1 | -1)); }
    }

    #[test]
    fn decode_center_is_self() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let l = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        let src = crate::wave_resonate::synapse::local_of(3, 4, size);
        assert_eq!(l.decode(0, src, 12, size), src, "center cell (idx 12, span 5) maps to self");
    }

    #[test]
    fn pw_matches_formula() {
        let (omega, dt) = (10.0f32, 0.05f32);
        let want = (-1.0 + (1.0 - (dt*omega)*(dt*omega)).sqrt()) / dt;
        assert_eq!(pw(omega, dt), want);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test wave_resonate::neurons`
Expected: FAIL (compile error).

- [ ] **Step 3: Implement `neurons.rs`** (occupancy/codes wiring copied from wave_driven; BRF state + init
  new). Use safe indexing.

```rust
//! `neurons` — a BRF `Layer`'s per-neuron SoA state (f32 x,y,q + per-neuron ω,b') and the copied bitset
//! topology substrate (occupancy + offset LUTs + 2-bit ±1/0 codes). Structure duplicated from
//! wave_driven::neurons; LIF/adaptation state is replaced by the resonator state.

use crate::wave_resonate::config::LayerConfig;
use crate::wave_resonate::synapse::{key, local_of, map_range, mix, neigh_size, sample_distinct_cells, wrap, xy_of, TopologyLevel, P_TARGET, P_THRESHOLD};

/// 2-bit weight code decode LUT: 0b00→0, 0b01→+1, 0b11→−1.
pub(crate) const WCODE: [i8; 4] = [0, 1, 0, -1];

/// Divergence boundary p(ω) = (−1 + √(1 − (δω)²)) / δ. Caller guarantees δ·ω ≤ 1 (Config::validate).
#[inline]
pub fn pw(omega: f32, dt: f32) -> f32 {
    (-1.0 + (1.0 - (dt * omega) * (dt * omega)).sqrt()) / dt
}

pub struct Layer {
    // BRF neuron state (readout layers reuse `x` as the leaky-integrator accumulator)
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub q: Vec<f32>,
    pub omega: Vec<f32>,
    pub b_off: Vec<f32>,
    pub pending: Vec<i32>,
    // dynamics constants
    pub dt: f32,
    pub gamma: f32,
    pub theta_c: f32,
    pub kappa: f32,
    // role
    pub transducer: bool,
    pub readout: bool,
    // topology substrate
    pub topology: Vec<TopologyLevel>,
    pub total_slots: usize,
    pub slot_bases: Vec<usize>,
    pub neigh: Vec<usize>,
    pub occ_wpn: Vec<usize>,
    pub occ: Vec<Vec<u64>>,
    pub offsets: Vec<Vec<(i8, i8)>>,
    pub off_flat: Vec<Vec<i32>>,
    pub codes: Vec<u64>,
}

impl Layer {
    #[inline]
    pub fn slot_base(&self, level_idx: usize) -> usize { self.slot_bases[level_idx] }

    #[inline]
    pub fn weight_at(&self, widx: usize) -> i8 {
        WCODE[((self.codes[widx >> 5] >> ((widx & 31) * 2)) & 0b11) as usize]
    }

    #[inline]
    pub fn synapse_count(&self) -> usize { self.total_slots * self.x.len() }

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

    #[inline]
    pub fn decode(&self, lvl: usize, src_local: u32, cell: usize, size: u32) -> u32 {
        let (sx, sy) = xy_of(src_local, size);
        let (dx, dy) = self.offsets[lvl][cell];
        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size)
    }

    pub fn new(cfg: &LayerConfig, dt: f32, gamma: f32, theta_c: f32, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;

        // derived layout (copied from wave_driven::derive_layout, inlined)
        let n_levels = cfg.topology.len();
        let mut slot_bases = Vec::with_capacity(n_levels);
        let mut neigh = Vec::with_capacity(n_levels);
        let mut occ_wpn = Vec::with_capacity(n_levels);
        let mut offsets: Vec<Vec<(i8, i8)>> = Vec::with_capacity(n_levels);
        let mut off_flat: Vec<Vec<i32>> = Vec::with_capacity(n_levels);
        let mut total_slots = 0usize;
        for t in &cfg.topology {
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

        // per-neuron ω, b' from the init ranges (deterministic hash streams; reuse P_THRESHOLD/P_TARGET tags)
        let (olo, ohi) = cfg.omega_init;
        let (blo, bhi) = cfg.b_offset_init;
        let mut omega = vec![0f32; ls];
        let mut b_off = vec![0f32; ls];
        for local in 0..ls {
            let g = (base + local) as u32;
            let ho = mix(key(seed, g, 0, 0, P_THRESHOLD));
            let hb = mix(key(seed, g, 0, 1, P_THRESHOLD));
            let fo = ((ho >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
            let fb = ((hb >> 40) as f32) / ((1u64 << 24) as f32);
            omega[local] = olo + (ohi - olo) * fo;
            b_off[local] = blo + (bhi - blo) * fb;
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

        // codes: init each wired synapse to the procedural ±1 sign (wired-rank order)
        let mut codes = vec![0u64; (ls * total_slots + 31) / 32];
        for i in 0..ls {
            let sg = (base + i) as u32;
            for (li, t) in cfg.topology.iter().enumerate() {
                for r in 0..(t.count as usize) {
                    let h = mix(key(seed, sg, t.level, r as u32, P_TARGET));
                    let sign_code: u64 = if ((h & 0xFFFF) as u32) < cfg.inhibitor_ratio { 0b11 } else { 0b01 };
                    let idx = i * total_slots + slot_bases[li] + r;
                    codes[idx >> 5] |= sign_code << ((idx & 31) * 2);
                }
            }
        }

        let kappa = (-dt / cfg.tau_out).exp();
        Layer {
            x: vec![0f32; ls], y: vec![0f32; ls], q: vec![0f32; ls], omega, b_off,
            pending: vec![0i32; ls],
            dt, gamma, theta_c, kappa,
            transducer: false, readout: false,
            topology: cfg.topology.clone(),
            total_slots, slot_bases, neigh, occ_wpn, occ, offsets, off_flat, codes,
        }
    }
}
```

Then append the test module from Step 1.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test wave_resonate::neurons`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/neurons.rs
git commit -m "feat(wave_resonate): BRF Layer SoA + ω/b' init + copied topology substrate"
```

---

### Task 4: `wave.rs` — `process_layer` compute dynamics (drain / transducer / readout / integrate+decide)

**Files:**
- Modify: `src/wave_resonate/wave.rs`

**Interfaces:**
- Consumes: `Layer` (Task 3), `pw` (Task 3), `synapse::*`.
- Produces: `pub fn process_layer(layer:&mut Layer, layer_index:u32, size:u32, input:&[u32], deliv:&mut [Vec<i32>], fired:&mut Vec<u32>)` — Task 4 fills drain/transducer/readout/integrate-decide (leaves `fired` correct); Task 5 adds the `generate` delivery into `deliv`.

- [ ] **Step 1: Write the failing tests** (single-neuron oracle is the core correctness gate)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::LayerConfig;
    use crate::wave_resonate::neurons::{pw, Layer};
    use crate::wave_resonate::synapse::TopologyLevel;

    fn compute_layer(size: u32, omega: f32, b_off: f32, dt: f32, gamma: f32, theta_c: f32) -> Layer {
        let cfg = LayerConfig { topology: vec![], inhibitor_ratio: 0, omega_init: (omega, omega), b_offset_init: (b_off, b_off), tau_out: 20.0 };
        Layer::new(&cfg, dt, gamma, theta_c, 5, 0, size)
    }

    // hand-rolled reference BRF neuron (plain f32 loop) — the fidelity oracle
    fn ref_brf(dt: f32, gamma: f32, theta_c: f32, omega: f32, b_off: f32, i_seq: &[f32]) -> Vec<(f32, f32, f32, u8)> {
        let (mut x, mut y, mut q) = (0f32, 0f32, 0f32);
        let mut out = Vec::new();
        for &i in i_seq {
            let p = (-1.0 + (1.0 - (dt * omega) * (dt * omega)).sqrt()) / dt;
            let b = p - b_off.abs() - q;
            let nx = x + dt * (b * x - omega * y + i);
            let ny = y + dt * (omega * x + b * y);
            let z = if nx - theta_c - q > 0.0 { 1u8 } else { 0u8 };
            let nq = gamma * q + z as f32;
            x = nx; y = ny; q = nq;
            out.push((x, y, q, z));
        }
        out
    }

    #[test]
    fn single_neuron_matches_reference_bit_exact() {
        let (dt, gamma, theta_c, omega, b_off) = (0.05f32, 0.9f32, 1.0f32, 10.0f32, 0.3f32);
        let i_seq: Vec<f32> = vec![3.0, 3.0, 3.0, 0.0, 0.0, 5.0, 0.0, -2.0, 0.0, 0.0, 4.0, 4.0];
        let want = ref_brf(dt, gamma, theta_c, omega, b_off, &i_seq);
        let mut l = compute_layer(1, omega, b_off, dt, gamma, theta_c);
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
        let mut fired = Vec::new();
        for (t, &i) in i_seq.iter().enumerate() {
            l.pending[0] = i as i32;
            process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
            let (wx, wy, wq, wz) = want[t];
            assert_eq!(l.x[0], wx, "x @ t={t}");
            assert_eq!(l.y[0], wy, "y @ t={t}");
            assert_eq!(l.q[0], wq, "q @ t={t}");
            assert_eq!(fired.contains(&0), wz == 1, "spike @ t={t}");
        }
        let _ = pw; // keep import used
    }

    #[test]
    fn divergence_free_stays_bounded_under_strong_drive() {
        let mut l = compute_layer(1, 10.0, 0.3, 0.05, 0.9, 1.0);
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
        let mut fired = Vec::new();
        for _ in 0..2000 {
            l.pending[0] = 50; // sustained strong drive
            process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
            assert!(l.x[0].abs() < 1e4 && l.y[0].abs() < 1e4, "BRF stays bounded: x={} y={}", l.x[0], l.y[0]);
        }
    }

    #[test]
    fn vanilla_rf_reference_diverges_documenting_control() {
        // Documenting control: a fixed positive-b RF (no p(ω) boundary) grows unbounded, whereas the BRF
        // reference above stays bounded. Pure hand-rolled ref (no engine change) — motivates p(ω).
        let (dt, omega) = (0.05f32, 10.0f32);
        let (mut x, mut y) = (0.1f32, 0.0f32);
        for _ in 0..3000 { let b = 0.5f32; let nx = x + dt*(b*x - omega*y); let ny = y + dt*(omega*x + b*y); x = nx; y = ny; }
        assert!(x.abs() > 1e6 || y.abs() > 1e6, "fixed +b RF diverges");
    }

    #[test]
    fn resonance_prefers_matched_frequency() {
        // A neuron with ω → discrete period P ≈ 2π/(δω) waves. Impulse-drive at P (resonant) vs P/3
        // (off), matched total input; resonant reaches a larger peak |x|.
        let (dt, omega) = (0.05f32, 10.0f32);
        let period = (2.0 * std::f32::consts::PI / (dt * omega)).round() as usize; // ≈ 13
        let run = |stride: usize| -> f32 {
            let mut l = compute_layer(1, omega, 0.05, dt, 0.9, 1.0);
            let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
            let mut fired = Vec::new();
            let mut peak = 0f32;
            for t in 0..(period * 8) {
                l.pending[0] = if t % stride == 0 { 6 } else { 0 };
                process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
                peak = peak.max(l.x[0].abs());
            }
            peak
        };
        let resonant = run(period);
        let off = run((period / 3).max(1));
        assert!(resonant > off, "resonant peak {resonant} should exceed off-frequency peak {off}");
    }

    #[test]
    fn transducer_fires_iff_injected_and_does_not_oscillate() {
        let mut l = compute_layer(4, 10.0, 0.3, 0.05, 0.9, 1.0);
        l.transducer = true;
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 16]; 1];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 4, &[2, 5, 9], &mut deliv, &mut fired);
        let mut f = fired.clone(); f.sort_unstable();
        assert_eq!(f, vec![2, 5, 9], "transducer fires exactly the injected sites");
        assert!(l.x.iter().all(|&v| v == 0.0) && l.y.iter().all(|&v| v == 0.0), "no oscillation");
    }

    #[test]
    fn readout_integrates_and_never_fires() {
        let mut l = compute_layer(4, 10.0, 0.3, 0.05, 0.9, 1.0);
        l.readout = true;
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 16]; 1];
        let mut fired = Vec::new();
        l.pending[0] = 10;
        process_layer(&mut l, 0, 4, &[], &mut deliv, &mut fired);
        assert!(fired.is_empty(), "readout never fires");
        assert!(l.x[0] > 0.0, "readout integrated input into x (leaky accumulator)");
        let x1 = l.x[0];
        l.pending[0] = 0;
        process_layer(&mut l, 0, 4, &[], &mut deliv, &mut fired);
        assert!(l.x[0] < x1 && l.x[0] > 0.0, "with no input the accumulator leaks toward 0: {} < {}", l.x[0], x1);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test wave_resonate::wave`
Expected: FAIL (compile error — `process_layer` not defined).

- [ ] **Step 3: Implement `process_layer` (compute dynamics; generate stubbed empty for now)**

```rust
//! `wave` — one BRF layer's per-wave step: drain the integer delivery accumulator into the input current,
//! (L0) act as a pass-through transducer or (last) leaky-integrator readout, else run the complex
//! resonator (dense over all neurons) and collect firers. `generate` (firer-gated delivery) is added in
//! the delivery task.

use crate::wave_resonate::neurons::{pw, Layer, WCODE};
use crate::wave_resonate::synapse::{local_of, wrap, xy_of};

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    size: u32,
    input: &[u32],
    deliv: &mut [Vec<i32>],
    fired: &mut Vec<u32>,
) {
    let ls = (size as usize) * (size as usize);
    fired.clear();

    // --- transducer (L0): fire exactly the injected sites; no membrane, clear any pending ---
    if layer.transducer {
        for p in layer.pending.iter_mut() { *p = 0; }
        for &a in input { fired.push(a); }
        generate(layer, layer_index, size, deliv, fired);
        return;
    }

    // --- readout: leaky-integrate the drained current into x; never fire ---
    if layer.readout {
        let k = layer.kappa;
        for i in 0..ls {
            let cur = layer.pending[i] as f32;
            layer.x[i] = k * layer.x[i] + cur;
            layer.pending[i] = 0;
        }
        return;
    }

    // --- compute: dense BRF oscillator update + decide ---
    let (dt, gamma, theta_c) = (layer.dt, layer.gamma, layer.theta_c);
    for i in 0..ls {
        let cur = layer.pending[i] as f32;
        layer.pending[i] = 0;
        let (x, y, q, omega, b_off) = (layer.x[i], layer.y[i], layer.q[i], layer.omega[i], layer.b_off[i]);
        let b = pw(omega, dt) - b_off.abs() - q;
        let nx = x + dt * (b * x - omega * y + cur);
        let ny = y + dt * (omega * x + b * y);
        let spike = nx - theta_c - q > 0.0;
        layer.x[i] = nx;
        layer.y[i] = ny;
        layer.q[i] = gamma * q + if spike { 1.0 } else { 0.0 };
        if spike { fired.push(i as u32); }
    }

    generate(layer, layer_index, size, deliv, fired);
}

// Filled in the delivery task; a no-op placeholder keeps Task 4 tests (which don't inspect deliv) green.
fn generate(_layer: &Layer, _layer_index: u32, _size: u32, _deliv: &mut [Vec<i32>], _fired: &[u32]) {
    let _ = (local_of as fn(u32, u32, u32) -> u32, wrap as fn(u32, i32, u32) -> u32, xy_of as fn(u32, u32) -> (u32, u32), WCODE);
}
```

Then append the test module from Step 1.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test wave_resonate::wave`
Expected: PASS (6 tests). If `resonance_prefers_matched_frequency` is flaky, tune `stride`/`period`/drive
until the resonant peak robustly exceeds the off-frequency peak (physically it must for a lightly-damped
resonator) — do not weaken the assertion to trivially true.

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/wave.rs
git commit -m "feat(wave_resonate): process_layer BRF oscillator + transducer/readout (single-neuron oracle)"
```

---

### Task 5: `wave.rs` — `generate` firer-gated ternary delivery

**Files:**
- Modify: `src/wave_resonate/wave.rs`

**Interfaces:**
- Consumes: `process_layer`/`generate` (Task 4), `Layer` accessors.
- Produces: `generate` fully implemented — each firer word-scans its occupancy bitset, decodes each wired
  cell to its target local, and scatter-adds the packed ±1/0 weight into `deliv[target_layer][target]`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn firer_scatters_decoded_weights_into_target_accumulator() {
        use crate::wave_resonate::config::LayerConfig;
        let size = 4u32; let ls = (size*size) as usize;
        let cfg = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }], inhibitor_ratio: 0, omega_init: (10.0,10.0), b_offset_init: (0.3,0.3), tau_out: 20.0 };
        let mut l = Layer::new(&cfg, 0.05, 0.9, 1.0, 5, 0, size);
        // force neuron 0 to fire: huge drive, zero q
        l.pending[0] = 1000;
        // expected: sum decoded nonzero weights per target for neuron 0
        let base = l.slot_base(0);
        let mut expect = vec![0i32; ls];
        l.for_wired(0, 0, |r, cell| {
            let wt = l.weight_at(0 * l.total_slots + base + r);
            if wt != 0 { expect[l.decode(0, 0, cell, size) as usize] += wt as i32; }
        });
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; ls]; 2];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, size, &[], &mut deliv, &mut fired);
        assert!(fired.contains(&0), "neuron 0 fires under strong drive");
        assert_eq!(deliv[1], expect, "scatter-adds decoded ±1 weights into layer 1's accumulator");
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test wave_resonate::wave::tests::firer_scatters_decoded_weights_into_target_accumulator`
Expected: FAIL (deliv[1] all zeros — generate is a no-op).

- [ ] **Step 3: Implement `generate`** (word-scan delivery, copied shape from wave_driven; safe indexing)

```rust
fn generate(layer: &Layer, layer_index: u32, size: u32, deliv: &mut [Vec<i32>], fired: &[u32]) {
    let layer_count = deliv.len() as i32;
    for &local in fired.iter() {
        let li = local as usize;
        let (sx, sy) = xy_of(local, size);
        for (lvl, entry) in layer.topology.iter().enumerate() {
            let tl = layer_index as i32 + entry.level;
            if tl < 0 || tl >= layer_count { continue; }
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
                    let wt = WCODE[((layer.codes[widx >> 5] >> ((widx & 31) * 2)) & 0b11) as usize] as i32;
                    let target = if interior {
                        (li_i + flat[cell]) as usize
                    } else {
                        let (dx, dy) = lut[cell];
                        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize
                    };
                    deliv[tz][target] += wt;
                    rank += 1;
                    word &= word - 1;
                }
            }
        }
    }
}
```

Remove the placeholder body / unused-import shim from Task 4.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test wave_resonate::wave`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/wave.rs
git commit -m "feat(wave_resonate): firer-gated ternary delivery (occupancy word-scan)"
```

---

### Task 6: `network.rs` — Network orchestration (build / wave / reset / listeners / accessors)

**Files:**
- Modify: `src/wave_resonate/network.rs`

**Interfaces:**
- Consumes: `Config` (Task 2), `Layer` (Task 3), `process_layer` (Task 4/5).
- Produces:
  ```rust
  pub struct Network { /* private */ }
  impl Network {
      pub fn new(config: Config) -> Network;              // last layer computes
      pub fn new_with_readout(config: Config) -> Network; // last layer is a leaky-integrator readout
      pub fn wave(&mut self, input: &[u32]);
      pub fn reset_state(&mut self);
      pub fn size(&self) -> u32;
      pub fn layer_count(&self) -> usize;
      pub fn with_layer<R>(&self, z: usize, f: impl FnOnce(&Layer) -> R) -> R;
      pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>);
      pub fn clear_listeners(&mut self);
  }
  ```

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;

    fn three_layer(size: u32) -> Config {
        let mk = |topology: Vec<TopologyLevel>| LayerConfig { topology, inhibitor_ratio: 0, omega_init: (5.0,10.0), b_offset_init: (0.1,1.0), tau_out: 20.0 };
        Config { seed: 9, size, dt: 0.05, gamma: 0.9, theta_c: 1.0, layers: vec![
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![]),
        ] }
    }

    #[test]
    fn l0_is_transducer_last_is_compute_by_default() {
        let net = Network::new(three_layer(8));
        net.with_layer(0, |l| assert!(l.transducer && !l.readout));
        net.with_layer(2, |l| assert!(!l.transducer && !l.readout));
    }

    #[test]
    fn new_with_readout_flags_last_layer() {
        let net = Network::new_with_readout(three_layer(8));
        net.with_layer(2, |l| assert!(l.readout && !l.transducer));
    }

    #[test]
    fn wave_is_deterministic() {
        let mut a = Network::new(three_layer(8));
        let mut b = Network::new(three_layer(8));
        let inputs: [&[u32]; 6] = [&[0,1,2], &[0,1,2], &[], &[5,9], &[], &[1]];
        for inp in inputs { a.wave(inp); b.wave(inp); }
        a.with_layer(1, |la| b.with_layer(1, |lb| {
            assert_eq!(la.x, lb.x); assert_eq!(la.y, lb.y); assert_eq!(la.q, lb.q);
        }));
    }

    #[test]
    fn activity_propagates_up_the_stack() {
        // Drive L0 for many waves; a middle compute layer should develop nonzero membrane state
        // (signal climbed the deferred one-hop stack).
        let mut net = Network::new(three_layer(16));
        for w in 0..60 { net.wave(if w % 2 == 0 { &[0,1,2,16,17,18,32,33] } else { &[] }); }
        let any = net.with_layer(1, |l| l.x.iter().any(|&v| v != 0.0) || l.y.iter().any(|&v| v != 0.0));
        assert!(any, "layer 1 developed oscillator activity from L0 drive");
    }

    #[test]
    fn readout_never_fires_but_integrates() {
        let mut net = Network::new_with_readout(three_layer(8));
        let fired_top = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let ft = fired_top.clone();
        net.on_layer(2, Box::new(move |_w, f| *ft.lock().unwrap() += f.len()));
        for w in 0..40 { net.wave(if w % 2 == 0 { &[0,1,2,8,9,10] } else { &[] }); }
        assert_eq!(*fired_top.lock().unwrap(), 0, "readout never fires");
        let any = net.with_layer(2, |l| l.x.iter().any(|&v| v != 0.0));
        assert!(any, "readout integrated some signal");
    }

    #[test]
    fn reset_state_clears_membrane() {
        let mut net = Network::new(three_layer(8));
        for _ in 0..10 { net.wave(&[0,1,2]); }
        net.reset_state();
        net.with_layer(1, |l| {
            assert!(l.x.iter().all(|&v| v == 0.0) && l.y.iter().all(|&v| v == 0.0) && l.q.iter().all(|&v| v == 0.0));
            assert!(l.pending.iter().all(|&p| p == 0));
        });
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test wave_resonate::network`
Expected: FAIL (compile error — `Network` not defined).

- [ ] **Step 3: Implement `network.rs`**

```rust
//! `network` — owns the BRF layer stack and drives each wave: process every layer (dense membrane +
//! firer-gated delivery), route generated deliveries one hop, swap into each layer's `pending` at wave
//! end (deferred propagation). L0 is the transducer; the last layer is either compute or a readout.

use crate::wave_resonate::config::Config;
use crate::wave_resonate::neurons::Layer;
use crate::wave_resonate::wave::process_layer;

pub struct Network {
    size: u32,
    layers: Vec<Layer>,
    wave_id: u32,
    fired: Vec<u32>,
    deliv: Vec<Vec<i32>>, // per layer: NEXT wave's incoming accumulator
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

impl Network {
    pub fn new(config: Config) -> Network { Network::build(config, false) }
    pub fn new_with_readout(config: Config) -> Network { Network::build(config, true) }

    fn build(config: Config, readout_last: bool) -> Network {
        config.validate().expect("invalid config");
        let size = config.size;
        let l = config.layers.len();
        let ls = (size as usize) * (size as usize);
        let mut layers = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let mut layer = Layer::new(lc, config.dt, config.gamma, config.theta_c, config.seed, z as u32, size);
            if z == 0 { layer.transducer = true; }
            if readout_last && z == l - 1 { layer.readout = true; }
            layers.push(layer);
        }
        Network {
            size, layers, wave_id: 0,
            fired: Vec::new(),
            deliv: (0..l).map(|_| vec![0i32; ls]).collect(),
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&mut self, input: &[u32]) {
        let w = self.wave_id;
        self.wave_id = self.wave_id.wrapping_add(1);
        let l = self.layers.len();
        let size = self.size;
        let Self { layers, deliv, fired, listeners, .. } = self;
        for z in 0..l {
            let inp: &[u32] = if z == 0 { input } else { &[] };
            process_layer(&mut layers[z], z as u32, size, inp, deliv, fired);
            if let Some(cb) = &listeners[z] { cb(w as usize, fired); }
        }
        // deferred one hop: this wave's deliveries become next wave's pending
        for z in 0..l { std::mem::swap(&mut layers[z].pending, &mut deliv[z]); }
    }

    pub fn reset_state(&mut self) {
        for g in self.layers.iter_mut() {
            g.x.iter_mut().for_each(|v| *v = 0.0);
            g.y.iter_mut().for_each(|v| *v = 0.0);
            g.q.iter_mut().for_each(|v| *v = 0.0);
            g.pending.iter_mut().for_each(|p| *p = 0);
        }
        for d in self.deliv.iter_mut() { d.iter_mut().for_each(|x| *x = 0); }
        self.wave_id = 0;
    }

    pub fn size(&self) -> u32 { self.size }
    pub fn layer_count(&self) -> usize { self.layers.len() }
    pub fn with_layer<R>(&self, z: usize, f: impl FnOnce(&Layer) -> R) -> R { f(&self.layers[z]) }
    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>) { self.listeners[layer] = Some(listener); }
    pub fn clear_listeners(&mut self) { for l in self.listeners.iter_mut() { *l = None; } }
}
```

Note the delivery routing subtlety: `process_layer` scatters into `deliv[target_layer]`; the swap at wave
end moves `deliv[z]` into `layers[z].pending`. Because a firer in layer `z` delivers into `deliv[z+level]`,
and we swap all layers after processing all layers, the one-hop deferral holds (matches wave_driven).

Then append the test module from Step 1.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test wave_resonate::network`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/network.rs
git commit -m "feat(wave_resonate): Network orchestration — wave, reset, readout, listeners"
```

---

### Task 7: Full-suite green + warning-free + docs cross-check

**Files:**
- Modify: `src/wave_resonate/mod.rs` (ensure all modules uncommented; remove any stubs' leftover)
- Modify: `AGENTS.md` (architecture map: add the `wave_resonate/` entry + a one-line engine note)

- [ ] **Step 1: Run the whole test suite**

Run: `cargo test`
Expected: PASS — all existing wave_bitnet/wave_driven tests plus the new `wave_resonate` tests green.

- [ ] **Step 2: Verify warning-free build**

Run: `cargo build 2>&1 | grep -i warning; echo done`
Expected: `done`, no warnings.

- [ ] **Step 3: Update `AGENTS.md`** — under "The two modules" add a sentence noting the third engine
  `wave_resonate` (BRF complex resonate-and-fire, f32 membrane + ternary weights, Phase 1 inference), and
  add the `src/wave_resonate/` block to the Architecture map mirroring the `wave_driven/` block
  (`synapse.rs` copied; `config.rs` BRF constants; `neurons.rs` BRF SoA + `pw`; `wave.rs` dense oscillator
  + firer-gated delivery; `network.rs` orchestration).

- [ ] **Step 4: Commit**

```bash
git add src/wave_resonate/mod.rs AGENTS.md
git commit -m "docs(wave_resonate): register Phase 1 engine in the architecture map"
```

---

## Self-Review

**Spec coverage (Phase 1 scope):**
- BRF complex dynamics (`p(ω)`, `b`, `x/y` update, spike on real part, `q`) → Tasks 3 (`pw`) + 4
  (`process_layer`), gated by the single-neuron reference oracle. ✓
- f32 membrane + ternary ±1/0 weights preserved → Task 3 (`x,y,q` f32; `codes` 2-bit; `weight_at`). ✓
- Dense membrane / sparse firer-gated delivery → Task 4 (dense loop) + Task 5 (`generate` over `fired`). ✓
- L0 transducer + leaky-integrator readout → Task 4 (`transducer`/`readout` branches) + Task 6 (flags). ✓
- Divergence-free stability + resonance/frequency selectivity → Task 4 tests. ✓
- Determinism (pure fn of seed/config/input) → Tasks 3 + 6 tests. ✓
- Divergence-boundary validation (`δ·ω ≤ 1`) → Task 2. ✓
- Copied topology substrate (occupancy bitset, codes, decode) → Tasks 1 + 3. ✓
- Timescale mapping as a hyperparameter (`dt`, `omega_init`) → Task 2 config fields; demo uses δ·ω=0.5. ✓
- **Deferred to Phase 2 (correctly absent here):** `TrainState`, eligibility, `dfa_update`, bench harness,
  bit-exact online-vs-dense eligibility oracle, trainable ω/b′ updates. ✓

**Placeholder scan:** the Task 4 `generate` no-op is an explicit, temporary placeholder replaced in Task 5
(documented), not a plan placeholder. No TBD/TODO in shipped code. ✓

**Type consistency:** `Layer::new(cfg, dt, gamma, theta_c, seed, index, size)` used identically in Tasks 4
& 6; `process_layer(layer, layer_index, size, input, deliv, fired)` signature identical across Tasks 4–6;
`pw(omega, dt)` consistent; readout accumulator is `x` throughout. ✓
