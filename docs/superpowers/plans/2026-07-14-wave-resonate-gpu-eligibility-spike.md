# wave_resonate GPU Eligibility Kernel Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: this repo's AGENTS.md mandates **inline, autonomous** execution — do NOT use the subagent-driven option here.

**Goal:** Port `wave_resonate`'s `accrue_eligibility` (the profiled 85% training hot path) to a GPU kernel on two hand-written backends (CUDA + WebGPU), validate each against the CPU `dense_eligibility` oracle, and measure throughput vs CPU across size {32,64,128} — producing a GO/NO-GO for a later full-engine spec.

**Architecture:** Precompute a flat per-synapse edge list (`tgt_g`/`src_g` global neuron ids) from a `Network`, turning the accrual into a one-thread-per-synapse map. A std-only `cpu_accrue` reference over that layout is the exact semantics the CUDA `.cu` and WGSL kernels mirror. A `GpuBackend` trait (impls: `CpuBackend`, `CudaBackend`, `WgpuBackend`) makes the harness backend-generic. Tasks 1–3 are std-only and green in the default build; Tasks 4–5 are feature-gated GPU backends that only need to match the reference.

**Tech Stack:** Rust (edition 2024, std-only default); `cudarc` (feature `cuda`, NVRTC-compiled `.cu` at runtime); `wgpu` + `pollster` + `bytemuck` (feature `wgpu`, WGSL); `criterion` already present (unused here — throughput lives in a feature-gated example).

## Global Constraints

- **Default build stays std-only + warning-free.** GPU deps and the GPU backend modules compile **only** under `--features cuda` / `--features wgpu`. `cargo build` and `cargo test` (no features) must be unaffected.
- **Determinism:** eligibility is per-synapse independent — no atomics, no cross-thread contention; the kernel is deterministic per backend.
- **Validation bar:** GPU `elig` vs CPU `dense_eligibility` oracle: `max |Δ| < 1e-5` AND `max rel-err < 1e-5`, per backend, per size.
- **Config (the fixed operating point):** size ∈ {32,64,128}, 5 layers, FF `TopologyLevel{level:1,radius:3,count:32}` on layers 0..4 (top layer empty topology), `dt=0.05`, `gamma=0.9`, `theta_c=0.1`, `omega_init=(5.0,10.0)`, `b_offset_init=(0.0,0.2)`, `tau_out=20.0`, `inhibitor_ratio=0`, `EligParams{dt:0.05, eps_cut:1e-6, train_omega_b:false}`.
- **Kernel scalars:** `dt=0.05`, `cut=1e-6` (the `eps_cut`).
- **No engine changes:** nothing under `src/wave_resonate/{neurons,wave,network,training,config,synapse}.rs` is modified. All new code lives in `src/wave_resonate_gpu/` (feature-gated GPU parts) + `examples/`.
- **Commits:** one per task, conventional messages, **no `Co-Authored-By` trailer**, on branch `feat/wave-resonate-gpu`.

## File Structure

```
src/lib.rs                         MODIFY: add `#[cfg(...)] pub mod wave_resonate_gpu;`
src/wave_resonate_gpu/
  mod.rs        Layout, Captured, GpuBackend trait, CpuBackend, build_layout, capture_inputs, cpu_accrue   [std-only; always compiles]
  cuda.rs       CudaBackend (cudarc host)                                                                   #[cfg(feature="cuda")]
  elig.cu       CUDA C kernel (NVRTC source string, included via include_str!)                              [built at runtime]
  wgpu.rs       WgpuBackend (wgpu host)                                                                     #[cfg(feature="wgpu")]
  elig.wgsl     WGSL kernel (included via include_str!)                                                     [built at runtime]
examples/profile_resonate_gpu.rs   size sweep + CPU/CUDA/wgpu waves/s + max_err table                       #[cfg gpu features]
Cargo.toml                         MODIFY: optional deps + [features] cuda / wgpu
```

Rationale: `mod.rs` holds everything std-only (layout, reference, trait, CPU backend) so the correctness anchor is testable with zero GPU. Each GPU backend is one file + one kernel source, kept small so the `.cu`/`.wgsl` pair is easy to keep in sync.

---

### Task 1: Flat edge-list layout (`Layout` + `build_layout`)

**Files:**
- Create: `src/wave_resonate_gpu/mod.rs`
- Modify: `src/lib.rs` (register the module)
- Test: inline `#[cfg(test)]` in `src/wave_resonate_gpu/mod.rs`

**Interfaces:**
- Consumes: `wave_resonate::network::Network` (`size()`, `layer_count()`, `with_layer`), `wave_resonate::neurons::Layer` (`total_slots`, `slot_bases`, `topology`, `for_wired`, `decode`).
- Produces:
  - `pub struct Layout { pub n: usize, pub ls: usize, pub l: usize, pub syn_base: Vec<usize>, pub tgt_g: Vec<u32>, pub src_g: Vec<u32> }`
  - `pub fn build_layout(net: &Network) -> Layout`

- [ ] **Step 1: Register the module (default build, no GPU).** Add to `src/lib.rs` after the other `pub mod` lines:

```rust
pub mod wave_resonate_gpu;
```

- [ ] **Step 2: Write `mod.rs` with `Layout` + `build_layout` + a failing test.**

```rust
//! `wave_resonate_gpu` — a de-risking spike: GPU port of the HYPR eligibility accrual
//! (`Network::accrue_eligibility`), validated vs the CPU `dense_eligibility` oracle. The std-only parts
//! here (flat edge-list layout, the `cpu_accrue` reference, the `GpuBackend` trait + `CpuBackend`) compile
//! in the default build; the CUDA/wgpu backends are feature-gated. See
//! docs/superpowers/specs/2026-07-14-wave-resonate-gpu-eligibility-spike-design.md.

use crate::wave_resonate::network::Network;

/// Flat per-synapse edge list derived once from a `Network`. Synapse `e` corresponds to the CPU's
/// `widx = i*ts + sbase + rank` within layer `z`, offset by `syn_base[z]`. `tgt_g[e]`/`src_g[e]` are the
/// GLOBAL neuron ids (`layer*ls + local`) of the synapse's target and source.
pub struct Layout {
    pub n: usize,            // total synapses across the stack
    pub ls: usize,           // size*size
    pub l: usize,            // layer count
    pub syn_base: Vec<usize>, // len l+1; synapse-index base per layer (prefix sum of ls*ts)
    pub tgt_g: Vec<u32>,
    pub src_g: Vec<u32>,
}

/// Build the flat edge list: walk every source's wired cells once (setup-time `for_wired` + `decode`,
/// folding the target-layer offset `z+level` into a global id). Every slot of every synapse-bearing layer
/// is a wired synapse (Σ count_e == total_slots), so no sentinels are needed for the FF spike config.
pub fn build_layout(net: &Network) -> Layout {
    let size = net.size();
    let ls = (size as usize) * (size as usize);
    let l = net.layer_count();
    let ts_by: Vec<usize> = (0..l).map(|z| net.with_layer(z, |lz| lz.total_slots)).collect();
    let mut syn_base = vec![0usize; l + 1];
    for z in 0..l {
        syn_base[z + 1] = syn_base[z] + ls * ts_by[z];
    }
    let n = syn_base[l];
    let mut tgt_g = vec![0u32; n];
    let mut src_g = vec![0u32; n];
    for z in 0..l {
        let ts = ts_by[z];
        if ts == 0 {
            continue;
        }
        let base_z = syn_base[z];
        net.with_layer(z, |lz| {
            for i in 0..ls {
                for (e_idx, edge) in lz.topology.iter().enumerate() {
                    let tz_i = z as i32 + edge.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let sbase = lz.slot_bases[e_idx];
                    lz.for_wired(e_idx, i, |rank, cell| {
                        let j = lz.decode(e_idx, i as u32, cell, size) as usize;
                        let e = base_z + i * ts + sbase + rank;
                        tgt_g[e] = (tz * ls + j) as u32;
                        src_g[e] = (z * ls + i) as u32;
                    });
                }
            }
        });
    }
    Layout { n, ls, l, syn_base, tgt_g, src_g }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;

    fn ff_cfg(size: u32, layers: usize) -> Config {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
            inhibitor_ratio: 0,
            omega_init: (5.0, 10.0),
            b_offset_init: (0.0, 0.2),
            tau_out: 20.0,
        };
        Config { seed: 7, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: vec![lc; layers] }
    }

    #[test]
    fn layout_indices_line_up_with_widx_and_decode() {
        let size = 8u32;
        let ls = (size * size) as usize;
        let net = Network::new(ff_cfg(size, 3));
        let layout = build_layout(&net);
        // top layer (empty topology) contributes 0 synapses; layers 0,1 contribute ls*32 each.
        assert_eq!(layout.syn_base[3], layout.n);
        assert_eq!(layout.n, 2 * ls * 32);
        // spot-check: for layer 0, source i, edge 0, the flat entry matches decode() and the global ids.
        net.with_layer(0, |lz| {
            let ts = lz.total_slots;
            for &i in &[0usize, 5, ls - 1] {
                lz.for_wired(0, i, |rank, cell| {
                    let e = layout.syn_base[0] + i * ts + lz.slot_bases[0] + rank;
                    let j = lz.decode(0, i as u32, cell, size) as usize;
                    assert_eq!(layout.tgt_g[e] as usize, 1 * ls + j, "target in layer 1");
                    assert_eq!(layout.src_g[e] as usize, 0 * ls + i, "source in layer 0");
                });
            }
        });
    }
}
```

- [ ] **Step 3: Run the test to verify it passes (default build).**

Run: `cargo test --lib wave_resonate_gpu::tests::layout_indices_line_up -- --nocapture`
Expected: PASS. Also run `cargo build` and confirm **no warnings**.

- [ ] **Step 4: Commit.**

```bash
git add src/lib.rs src/wave_resonate_gpu/mod.rs
git commit -m "feat(wave_resonate_gpu): flat per-synapse edge-list layout builder"
```

---

### Task 2: `cpu_accrue` reference + input capture (validated vs the oracle)

**Files:**
- Modify: `src/wave_resonate_gpu/mod.rs`
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Consumes: `Layout` (Task 1); `wave_resonate::training::{EligParams, dense_eligibility}`; `wave_resonate::config::Config`.
- Produces:
  - `pub struct Captured { pub b_eff_g: Vec<f32>, pub omega_g: Vec<f32>, pub psi_g: Vec<f32>, pub prev_fired_g: Vec<u32> }`
  - `pub fn capture_inputs(cfg: &Config, inputs: &[Vec<u32>]) -> (Layout, Vec<Captured>)`
  - `pub fn cpu_accrue(layout: &Layout, seq: &[Captured], dt: f32, cut: f32) -> Vec<f32>`

- [ ] **Step 1: Write the failing test** (append to the `tests` module in `mod.rs`). It drives a small net, captures inputs, runs `cpu_accrue`, and asserts it matches `dense_eligibility` flattened.

```rust
#[test]
fn cpu_accrue_matches_dense_eligibility_oracle() {
    use crate::wave_resonate::training::{dense_eligibility, EligParams};
    let size = 8u32;
    let ls = (size * size) as usize;
    let cfg = ff_cfg(size, 3);
    // a representative drive sequence (some cue waves, some silent)
    let inputs: Vec<Vec<u32>> = (0..40)
        .map(|w| if w % 3 == 0 { vec![0u32, 1, 2, 8, 9, 10] } else { vec![] })
        .collect();
    let p = EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: false };

    let oracle = dense_eligibility(&cfg, &inputs, &p); // Vec<Vec<f32>> per layer, indexed by widx
    let (layout, seq) = capture_inputs(&cfg, &inputs);
    let flat = cpu_accrue(&layout, &seq, p.dt, p.eps_cut);

    // compare flat[syn_base[z]+widx] to oracle[z][widx]
    let l = cfg.layers.len();
    let ts_by: Vec<usize> = (0..l).map(|z| oracle[z].len() / ls.max(1)).collect();
    let mut max_abs = 0f32;
    for z in 0..l {
        for widx in 0..oracle[z].len() {
            let e = layout.syn_base[z] + widx;
            max_abs = max_abs.max((flat[e] - oracle[z][widx]).abs());
        }
        let _ = ts_by[z];
    }
    assert!(max_abs < 1e-6, "cpu_accrue matches oracle, max_abs={max_abs}");
    assert!(flat.iter().any(|&e| e != 0.0), "some eligibility accrued");
}
```

- [ ] **Step 2: Run it to confirm it fails** (functions not defined).

Run: `cargo test --lib wave_resonate_gpu::tests::cpu_accrue_matches -- --nocapture`
Expected: FAIL to compile — `capture_inputs`/`cpu_accrue`/`Captured` not found.

- [ ] **Step 3: Implement `Captured`, `capture_inputs`, `cpu_accrue`** (add above the `tests` module in `mod.rs`).

```rust
use crate::wave_resonate::config::Config;
use std::sync::{Arc, Mutex};

/// Per-wave forward outputs, packed by GLOBAL neuron id `g = z*ls + local` (length `L*ls` each).
/// `prev_fired_g` is the PREVIOUS wave's firers (the source injection `z_i^{t-1}`); `omega_g` is constant
/// while `train_omega_b=false` but captured per wave for generality.
pub struct Captured {
    pub b_eff_g: Vec<f32>,
    pub omega_g: Vec<f32>,
    pub psi_g: Vec<f32>,
    pub prev_fired_g: Vec<u32>,
}

/// Drive a fresh training net on `inputs`, capturing each wave's (b_eff, psi, omega, prev_fired) into the
/// global-id layout. Mirrors what `dense_eligibility` consumes, so `cpu_accrue`/the GPU kernels fed this
/// reproduce the oracle. Returns the `Layout` (same net) alongside the per-wave captures.
pub fn capture_inputs(cfg: &Config, inputs: &[Vec<u32>]) -> (Layout, Vec<Captured>) {
    use crate::wave_resonate::training::EligParams;
    let size = cfg.size;
    let ls = (size as usize) * (size as usize);
    let l = cfg.layers.len();
    let mut net = Network::new(cfg.clone());
    net.set_elig_params(EligParams { dt: cfg.dt, eps_cut: 1.0 / 1024.0, train_omega_b: false });
    net.enable_training();
    let layout = build_layout(&net);

    // capture each wave's firers per layer via listeners (listener overwrites its slot each wave)
    let cur: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 0..l {
        let c = cur.clone();
        net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] = f.to_vec()));
    }

    let mut prev_fired_g = vec![0u32; l * ls]; // wave 0: no previous firers
    let mut seq = Vec::with_capacity(inputs.len());
    for inp in inputs {
        net.wave(inp);
        let mut b_eff_g = vec![0f32; l * ls];
        let mut omega_g = vec![0f32; l * ls];
        let mut psi_g = vec![0f32; l * ls];
        for z in 0..l {
            net.with_layer(z, |lz| {
                omega_g[z * ls..(z + 1) * ls].copy_from_slice(&lz.omega);
                if let Some(t) = lz.train.as_ref() {
                    b_eff_g[z * ls..(z + 1) * ls].copy_from_slice(&t.b_eff);
                    psi_g[z * ls..(z + 1) * ls].copy_from_slice(&t.psi);
                }
            });
        }
        seq.push(Captured { b_eff_g, omega_g, psi_g, prev_fired_g: prev_fired_g.clone() });
        // roll prev_fired ← THIS wave's firers for the next wave
        prev_fired_g.iter_mut().for_each(|v| *v = 0);
        let fired_now = cur.lock().unwrap();
        for z in 0..l {
            for &i in &fired_now[z] {
                prev_fired_g[z * ls + i as usize] = 1;
            }
        }
    }
    net.clear_listeners();
    (layout, seq)
}

/// The flat-edge-list eligibility reference — the EXACT per-synapse recursion the CUDA/WGSL kernels mirror.
/// Per synapse `e`: advance the 2-state ε from the TARGET's (b_eff, ω, ψ) and the SOURCE's prev-spike,
/// apply `eps_cut`, accumulate `elig += ψ·εˣ`. Per-synapse independent → order-free (matches the oracle).
pub fn cpu_accrue(layout: &Layout, seq: &[Captured], dt: f32, cut: f32) -> Vec<f32> {
    let n = layout.n;
    let mut eps_x = vec![0f32; n];
    let mut eps_y = vec![0f32; n];
    let mut elig = vec![0f32; n];
    for cap in seq {
        for e in 0..n {
            let j = layout.tgt_g[e] as usize;
            let i = layout.src_g[e] as usize;
            let inj = if cap.prev_fired_g[i] != 0 { dt } else { 0.0 };
            let b = cap.b_eff_g[j];
            let om = cap.omega_g[j];
            let psi = cap.psi_g[j];
            let ex = eps_x[e];
            let ey = eps_y[e];
            let coef = 1.0 + dt * b;
            let mut nex = coef * ex - dt * om * ey + inj;
            let mut ney = dt * om * ex + coef * ey;
            if nex.abs() < cut {
                nex = 0.0;
            }
            if ney.abs() < cut {
                ney = 0.0;
            }
            eps_x[e] = nex;
            eps_y[e] = ney;
            if psi != 0.0 && nex != 0.0 {
                elig[e] += psi * nex;
            }
        }
    }
    elig
}
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `cargo test --lib wave_resonate_gpu::tests::cpu_accrue_matches -- --nocapture`
Expected: PASS (`max_abs` well under `1e-6`). Confirm `cargo build` stays warning-free.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_resonate_gpu/mod.rs
git commit -m "feat(wave_resonate_gpu): cpu_accrue flat reference + input capture, validated vs oracle"
```

---

### Task 3: `GpuBackend` trait + `CpuBackend`

**Files:**
- Modify: `src/wave_resonate_gpu/mod.rs`
- Test: inline `#[cfg(test)]`

**Interfaces:**
- Consumes: `Layout`, `Captured`, `cpu_accrue`.
- Produces:
  - `pub trait GpuBackend { fn new(layout: &Layout, dt: f32, cut: f32) -> Self where Self: Sized; fn reset(&mut self); fn step(&mut self, cap: &Captured); fn download_elig(&self) -> Vec<f32>; }`
  - `pub struct CpuBackend { … }` implementing it.
  - `pub fn run_backend<B: GpuBackend>(layout: &Layout, seq: &[Captured], dt: f32, cut: f32) -> (Vec<f32>, std::time::Duration)`

- [ ] **Step 1: Write the failing test** — `CpuBackend` driven wave-by-wave equals the whole-sequence `cpu_accrue`.

```rust
#[test]
fn cpu_backend_stepwise_equals_batch() {
    let size = 8u32;
    let cfg = ff_cfg(size, 3);
    let inputs: Vec<Vec<u32>> = (0..30).map(|w| if w % 3 == 0 { vec![0u32, 1, 2] } else { vec![] }).collect();
    let (layout, seq) = capture_inputs(&cfg, &inputs);
    let batch = cpu_accrue(&layout, &seq, 0.05, 1e-6);
    let (stepwise, _dur) = run_backend::<CpuBackend>(&layout, &seq, 0.05, 1e-6);
    assert_eq!(batch, stepwise, "stepwise CpuBackend == batch cpu_accrue");
}
```

- [ ] **Step 2: Run it to confirm it fails.**

Run: `cargo test --lib wave_resonate_gpu::tests::cpu_backend_stepwise -- --nocapture`
Expected: FAIL — `GpuBackend`/`CpuBackend`/`run_backend` not found.

- [ ] **Step 3: Implement the trait, `CpuBackend`, and `run_backend`** (add to `mod.rs`).

```rust
use std::time::{Duration, Instant};

/// A backend advances the per-synapse eligibility one wave at a time and can read back `elig`. `step`
/// must be synchronous (complete before returning) so `run_backend` measures true wall time.
pub trait GpuBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self
    where
        Self: Sized;
    fn reset(&mut self);
    fn step(&mut self, cap: &Captured);
    fn download_elig(&self) -> Vec<f32>;
}

/// Reference backend: the same per-synapse recursion as `cpu_accrue`, but one wave per `step` with
/// resident `eps_x/eps_y/elig` — the shape every GPU backend implements. Also the harness baseline.
pub struct CpuBackend {
    tgt_g: Vec<u32>,
    src_g: Vec<u32>,
    eps_x: Vec<f32>,
    eps_y: Vec<f32>,
    elig: Vec<f32>,
    dt: f32,
    cut: f32,
}

impl GpuBackend for CpuBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self {
        CpuBackend {
            tgt_g: layout.tgt_g.clone(),
            src_g: layout.src_g.clone(),
            eps_x: vec![0.0; layout.n],
            eps_y: vec![0.0; layout.n],
            elig: vec![0.0; layout.n],
            dt,
            cut,
        }
    }
    fn reset(&mut self) {
        self.eps_x.iter_mut().for_each(|v| *v = 0.0);
        self.eps_y.iter_mut().for_each(|v| *v = 0.0);
        self.elig.iter_mut().for_each(|v| *v = 0.0);
    }
    fn step(&mut self, cap: &Captured) {
        let (dt, cut) = (self.dt, self.cut);
        for e in 0..self.elig.len() {
            let j = self.tgt_g[e] as usize;
            let i = self.src_g[e] as usize;
            let inj = if cap.prev_fired_g[i] != 0 { dt } else { 0.0 };
            let (b, om, psi) = (cap.b_eff_g[j], cap.omega_g[j], cap.psi_g[j]);
            let (ex, ey) = (self.eps_x[e], self.eps_y[e]);
            let coef = 1.0 + dt * b;
            let mut nex = coef * ex - dt * om * ey + inj;
            let mut ney = dt * om * ex + coef * ey;
            if nex.abs() < cut {
                nex = 0.0;
            }
            if ney.abs() < cut {
                ney = 0.0;
            }
            self.eps_x[e] = nex;
            self.eps_y[e] = ney;
            if psi != 0.0 && nex != 0.0 {
                self.elig[e] += psi * nex;
            }
        }
    }
    fn download_elig(&self) -> Vec<f32> {
        self.elig.clone()
    }
}

/// Run a full capture sequence through a backend, returning (final elig, wall-clock over the wave loop).
/// Construction is excluded from timing; the loop is the per-wave `step` (upload + launch for GPU).
pub fn run_backend<B: GpuBackend>(layout: &Layout, seq: &[Captured], dt: f32, cut: f32) -> (Vec<f32>, Duration) {
    let mut b = B::new(layout, dt, cut);
    let t = Instant::now();
    for cap in seq {
        b.step(cap);
    }
    let dur = t.elapsed();
    (b.download_elig(), dur)
}
```

- [ ] **Step 4: Run the test to verify it passes.**

Run: `cargo test --lib wave_resonate_gpu -- --nocapture`
Expected: all three `wave_resonate_gpu` tests PASS. `cargo build` warning-free.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_resonate_gpu/mod.rs
git commit -m "feat(wave_resonate_gpu): GpuBackend trait + CpuBackend reference backend"
```

---

### Task 4: CUDA backend (`CudaBackend` + `elig.cu`)

**Files:**
- Create: `src/wave_resonate_gpu/cuda.rs`, `src/wave_resonate_gpu/elig.cu`
- Modify: `src/wave_resonate_gpu/mod.rs` (feature-gated `pub mod cuda;` + re-export), `Cargo.toml`
- Test: `#[cfg(all(test, feature = "cuda"))]` in `cuda.rs`

**Interfaces:**
- Consumes: `Layout`, `Captured`, `GpuBackend`, `capture_inputs`, `cpu_accrue`.
- Produces: `pub struct CudaBackend` implementing `GpuBackend`.

> **API note:** `cudarc` moves fast. Pin the exact version in `Cargo.toml` and verify each call against docs.rs for that version before running — the calls below target the `cudarc 0.12`-era driver API (`CudaDevice`, `htod_sync_copy`, `alloc_zeros`, `load_ptx`, `get_func`, `LaunchAsync::launch`, `dtoh_sync_copy`). Adjust names if the pinned version differs; the semantics (H2D copy, launch grid = ceil(n/256), D2H copy) do not change.

- [ ] **Step 1: Add the optional dep + feature to `Cargo.toml`.**

```toml
[dependencies]
cudarc = { version = "0.12", optional = true, features = ["nvrtc", "driver"] }

[features]
cuda = ["dep:cudarc"]
```

- [ ] **Step 2: Write the CUDA kernel `src/wave_resonate_gpu/elig.cu`.**

```cuda
// One thread per synapse: advance the 2-state HYPR eligibility and accumulate elig. Mirrors cpu_accrue.
extern "C" __global__ void accrue(
    const unsigned int* tgt_g, const unsigned int* src_g,
    const float* b_eff_g, const float* omega_g, const float* psi_g, const unsigned int* prev_fired_g,
    float* eps_x, float* eps_y, float* elig,
    unsigned int n, float dt, float cut)
{
    unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int j = tgt_g[e];
    unsigned int i = src_g[e];
    float inj = prev_fired_g[i] != 0u ? dt : 0.0f;
    float b = b_eff_g[j], om = omega_g[j], psi = psi_g[j];
    float ex = eps_x[e], ey = eps_y[e];
    float coef = 1.0f + dt * b;
    float nex = coef * ex - dt * om * ey + inj;
    float ney = dt * om * ex + coef * ey;
    if (fabsf(nex) < cut) nex = 0.0f;
    if (fabsf(ney) < cut) ney = 0.0f;
    eps_x[e] = nex; eps_y[e] = ney;
    if (psi != 0.0f && nex != 0.0f) elig[e] += psi * nex;
}
```

- [ ] **Step 3: Write `src/wave_resonate_gpu/cuda.rs`** (host: NVRTC-compile the kernel, resident device buffers, per-wave H2D of the 4 neuron arrays + launch, final D2H of `elig`).

```rust
//! CUDA backend for the eligibility kernel (cudarc + NVRTC). `eps_x/eps_y/elig/tgt_g/src_g` stay resident
//! on the device; each `step` uploads only the per-neuron arrays (b_eff/omega/psi/prev_fired) and launches.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::{Captured, GpuBackend, Layout};

pub struct CudaBackend {
    dev: Arc<CudaDevice>,
    func: CudaFunction,
    n: u32,
    dt: f32,
    cut: f32,
    // resident device buffers
    tgt_g: CudaSlice<u32>,
    src_g: CudaSlice<u32>,
    eps_x: CudaSlice<f32>,
    eps_y: CudaSlice<f32>,
    elig: CudaSlice<f32>,
    b_eff: CudaSlice<f32>,
    omega: CudaSlice<f32>,
    psi: CudaSlice<f32>,
    prev_fired: CudaSlice<u32>,
    ls_l: usize, // L*ls (length of the per-neuron arrays)
}

impl GpuBackend for CudaBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self {
        let dev = CudaDevice::new(0).expect("cuda device 0");
        let ptx = compile_ptx(include_str!("elig.cu")).expect("nvrtc compile elig.cu");
        dev.load_ptx(ptx, "elig", &["accrue"]).expect("load ptx");
        let func = dev.get_func("elig", "accrue").expect("get accrue");
        let n = layout.n;
        let ls_l = layout.l * layout.ls; // per-neuron array length = L*ls (global-id domain)
        let tgt_g = dev.htod_sync_copy(&layout.tgt_g).unwrap();
        let src_g = dev.htod_sync_copy(&layout.src_g).unwrap();
        let eps_x = dev.alloc_zeros::<f32>(n).unwrap();
        let eps_y = dev.alloc_zeros::<f32>(n).unwrap();
        let elig = dev.alloc_zeros::<f32>(n).unwrap();
        let b_eff = dev.alloc_zeros::<f32>(ls_l).unwrap();
        let omega = dev.alloc_zeros::<f32>(ls_l).unwrap();
        let psi = dev.alloc_zeros::<f32>(ls_l).unwrap();
        let prev_fired = dev.alloc_zeros::<u32>(ls_l).unwrap();
        CudaBackend {
            dev, func, n: n as u32, dt, cut,
            tgt_g, src_g, eps_x, eps_y, elig, b_eff, omega, psi, prev_fired, ls_l,
        }
    }

    fn reset(&mut self) {
        self.dev.memset_zeros(&mut self.eps_x).unwrap();
        self.dev.memset_zeros(&mut self.eps_y).unwrap();
        self.dev.memset_zeros(&mut self.elig).unwrap();
    }

    fn step(&mut self, cap: &Captured) {
        debug_assert_eq!(cap.b_eff_g.len(), self.ls_l);
        self.dev.htod_sync_copy_into(&cap.b_eff_g, &mut self.b_eff).unwrap();
        self.dev.htod_sync_copy_into(&cap.omega_g, &mut self.omega).unwrap();
        self.dev.htod_sync_copy_into(&cap.psi_g, &mut self.psi).unwrap();
        self.dev.htod_sync_copy_into(&cap.prev_fired_g, &mut self.prev_fired).unwrap();
        let cfg = LaunchConfig::for_num_elems(self.n);
        unsafe {
            self.func.clone().launch(
                cfg,
                (
                    &self.tgt_g, &self.src_g,
                    &self.b_eff, &self.omega, &self.psi, &self.prev_fired,
                    &mut self.eps_x, &mut self.eps_y, &mut self.elig,
                    self.n, self.dt, self.cut,
                ),
            ).unwrap();
        }
        self.dev.synchronize().unwrap(); // synchronous step for honest timing
    }

    fn download_elig(&self) -> Vec<f32> {
        self.dev.dtoh_sync_copy(&self.elig).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;
    use crate::wave_resonate_gpu::{capture_inputs, cpu_accrue, run_backend};

    fn ff_cfg(size: u32, layers: usize) -> Config {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
            inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.0, 0.2), tau_out: 20.0,
        };
        Config { seed: 7, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: vec![lc; layers] }
    }

    #[test]
    fn cuda_matches_oracle_within_tolerance() {
        let cfg = ff_cfg(16, 4);
        let inputs: Vec<Vec<u32>> = (0..40).map(|w| if w % 3 == 0 { vec![0u32, 1, 2, 16, 17] } else { vec![] }).collect();
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        let cpu = cpu_accrue(&layout, &seq, 0.05, 1e-6);
        let (gpu, _dur) = run_backend::<CudaBackend>(&layout, &seq, 0.05, 1e-6);
        let (mut max_abs, mut max_rel) = (0f32, 0f32);
        for e in 0..cpu.len() {
            let d = (cpu[e] - gpu[e]).abs();
            max_abs = max_abs.max(d);
            if cpu[e].abs() > 1e-6 {
                max_rel = max_rel.max(d / cpu[e].abs());
            }
        }
        assert!(max_abs < 1e-5 && max_rel < 1e-5, "cuda vs oracle: max_abs={max_abs} max_rel={max_rel}");
    }
}
```

- [ ] **Step 4: Gate the module in `mod.rs`.** Add near the top of `src/wave_resonate_gpu/mod.rs`:

```rust
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
```

- [ ] **Step 5: Verify default build is untouched, then run the CUDA test.**

Run: `cargo build` → warning-free, no cudarc compiled.
Run: `cargo test --features cuda --lib wave_resonate_gpu::cuda -- --nocapture`
Expected: `cuda_matches_oracle_within_tolerance` PASS (`max_abs`/`max_rel` < 1e-5).
If cudarc API names differ from the pinned version, fix per docs.rs (see API note) and re-run — the SEMANTICS are fixed by `cpu_accrue`.

- [ ] **Step 6: Commit.**

```bash
git add Cargo.toml src/wave_resonate_gpu/mod.rs src/wave_resonate_gpu/cuda.rs src/wave_resonate_gpu/elig.cu
git commit -m "feat(wave_resonate_gpu): CUDA backend (cudarc + NVRTC) validated vs oracle <1e-5"
```

---

### Task 5: WebGPU backend (`WgpuBackend` + `elig.wgsl`)

**Files:**
- Create: `src/wave_resonate_gpu/wgpu.rs`, `src/wave_resonate_gpu/elig.wgsl`
- Modify: `src/wave_resonate_gpu/mod.rs` (feature-gated), `Cargo.toml`
- Test: `#[cfg(all(test, feature = "wgpu"))]` in `wgpu.rs`

**Interfaces:**
- Consumes: `Layout`, `Captured`, `GpuBackend`.
- Produces: `pub struct WgpuBackend` implementing `GpuBackend`.

> **API note:** `wgpu` also churns. Pin the version; the code targets the `wgpu 0.20`-era API (`Instance::default`, `request_adapter`, `request_device`, `create_shader_module`, `create_compute_pipeline`, storage `Buffer`s + a `BindGroup`, `CommandEncoder` + `ComputePass`, `queue.write_buffer`, map-read for D2H). `pollster::block_on` drives the async calls. Verify signatures against docs.rs for the pinned version.

- [ ] **Step 1: Add deps + feature to `Cargo.toml`.**

```toml
[dependencies]
wgpu = { version = "0.20", optional = true }
pollster = { version = "0.3", optional = true }
bytemuck = { version = "1", optional = true, features = ["derive"] }

[features]
wgpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]
```

- [ ] **Step 2: Write the WGSL kernel `src/wave_resonate_gpu/elig.wgsl`** (semantics identical to `elig.cu`; `params` uniform carries `n`, `dt`, `cut`).

```wgsl
struct Params { n: u32, dt: f32, cut: f32, _pad: u32, };
@group(0) @binding(0) var<storage, read>        tgt_g: array<u32>;
@group(0) @binding(1) var<storage, read>        src_g: array<u32>;
@group(0) @binding(2) var<storage, read>        b_eff_g: array<f32>;
@group(0) @binding(3) var<storage, read>        omega_g: array<f32>;
@group(0) @binding(4) var<storage, read>        psi_g: array<f32>;
@group(0) @binding(5) var<storage, read>        prev_fired_g: array<u32>;
@group(0) @binding(6) var<storage, read_write>  eps_x: array<f32>;
@group(0) @binding(7) var<storage, read_write>  eps_y: array<f32>;
@group(0) @binding(8) var<storage, read_write>  elig: array<f32>;
@group(0) @binding(9) var<uniform>              params: Params;

@compute @workgroup_size(256)
fn accrue(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= params.n) { return; }
    let j = tgt_g[e];
    let i = src_g[e];
    var inj = 0.0;
    if (prev_fired_g[i] != 0u) { inj = params.dt; }
    let b = b_eff_g[j]; let om = omega_g[j]; let psi = psi_g[j];
    let ex = eps_x[e]; let ey = eps_y[e];
    let coef = 1.0 + params.dt * b;
    var nex = coef * ex - params.dt * om * ey + inj;
    var ney = params.dt * om * ex + coef * ey;
    if (abs(nex) < params.cut) { nex = 0.0; }
    if (abs(ney) < params.cut) { ney = 0.0; }
    eps_x[e] = nex; eps_y[e] = ney;
    if (psi != 0.0 && nex != 0.0) { elig[e] = elig[e] + psi * nex; }
}
```

- [ ] **Step 3: Write `src/wave_resonate_gpu/wgpu.rs`** (device/pipeline setup in `new`; per-wave `queue.write_buffer` for the 4 neuron arrays + a `ComputePass` dispatch; D2H via a mapped staging buffer in `download_elig`).

```rust
//! WebGPU backend for the eligibility kernel (wgpu + WGSL). Storage buffers for tgt/src/eps/elig stay
//! resident; each `step` writes the 4 per-neuron arrays and dispatches ceil(n/256) workgroups.

use super::{Captured, GpuBackend, Layout};

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    b_eff: wgpu::Buffer,
    omega: wgpu::Buffer,
    psi: wgpu::Buffer,
    prev_fired: wgpu::Buffer,
    elig: wgpu::Buffer,
    readback: wgpu::Buffer,
    n: u32,
    elig_bytes: u64,
}

fn storage_init<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T], rw: bool) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let usage = wgpu::BufferUsages::STORAGE
        | if rw { wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST } else { wgpu::BufferUsages::COPY_DST };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

impl GpuBackend for WgpuBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self {
        pollster::block_on(async move {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
                .await
                .expect("no wgpu adapter");
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .expect("request_device");

            let n = layout.n;
            let ls_l = layout.l * layout.ls;
            let zeros_n = vec![0f32; n];
            let zeros_neu_f = vec![0f32; ls_l];
            let zeros_neu_u = vec![0u32; ls_l];

            let tgt_g = storage_init(&device, &layout.tgt_g, false);
            let src_g = storage_init(&device, &layout.src_g, false);
            let b_eff = storage_init(&device, &zeros_neu_f, false);
            let omega = storage_init(&device, &zeros_neu_f, false);
            let psi = storage_init(&device, &zeros_neu_f, false);
            let prev_fired = storage_init(&device, &zeros_neu_u, false);
            let eps_x = storage_init(&device, &zeros_n, true);
            let eps_y = storage_init(&device, &zeros_n, true);
            let elig = storage_init(&device, &zeros_n, true);
            // params uniform: n, dt, cut, pad
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct Params { n: u32, dt: f32, cut: f32, _pad: u32 }
            use wgpu::util::DeviceExt;
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params { n: n as u32, dt, cut, _pad: 0 }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(include_str!("elig.wgsl").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None, layout: None, module: &shader, entry_point: "accrue",
                compilation_options: Default::default(), cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tgt_g.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src_g.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b_eff.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: omega.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: psi.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: prev_fired.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: eps_x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: eps_y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: elig.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: params.as_entire_binding() },
                ],
            });
            let elig_bytes = (n * std::mem::size_of::<f32>()) as u64;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: elig_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            WgpuBackend { device, queue, pipeline, bind_group, b_eff, omega, psi, prev_fired, elig, readback, n: n as u32, elig_bytes }
        })
    }

    fn reset(&mut self) {
        // simplest correct reset: zero eps/elig by writing zeros to elig (eps handled by re-`new` in the
        // harness per size). For the spike, prefer constructing a fresh backend per run instead of reset.
        let zeros = vec![0f32; self.n as usize];
        self.queue.write_buffer(&self.elig, 0, bytemuck::cast_slice(&zeros));
        self.device.poll(wgpu::Maintain::Wait);
    }

    fn step(&mut self, cap: &Captured) {
        self.queue.write_buffer(&self.b_eff, 0, bytemuck::cast_slice(&cap.b_eff_g));
        self.queue.write_buffer(&self.omega, 0, bytemuck::cast_slice(&cap.omega_g));
        self.queue.write_buffer(&self.psi, 0, bytemuck::cast_slice(&cap.psi_g));
        self.queue.write_buffer(&self.prev_fired, 0, bytemuck::cast_slice(&cap.prev_fired_g));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((self.n + 255) / 256, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait); // synchronous step for honest timing
    }

    fn download_elig(&self) -> Vec<f32> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.elig, 0, &self.readback, 0, self.elig_bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.readback.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;
    use crate::wave_resonate_gpu::{capture_inputs, cpu_accrue, run_backend};

    #[test]
    fn wgpu_matches_oracle_within_tolerance() {
        let lc = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }], inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.0, 0.2), tau_out: 20.0 };
        let cfg = Config { seed: 7, size: 16, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: vec![lc; 4] };
        let inputs: Vec<Vec<u32>> = (0..40).map(|w| if w % 3 == 0 { vec![0u32, 1, 2, 16, 17] } else { vec![] }).collect();
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        let cpu = cpu_accrue(&layout, &seq, 0.05, 1e-6);
        let (gpu, _dur) = run_backend::<WgpuBackend>(&layout, &seq, 0.05, 1e-6);
        let (mut max_abs, mut max_rel) = (0f32, 0f32);
        for e in 0..cpu.len() {
            let d = (cpu[e] - gpu[e]).abs();
            max_abs = max_abs.max(d);
            if cpu[e].abs() > 1e-6 { max_rel = max_rel.max(d / cpu[e].abs()); }
        }
        assert!(max_abs < 1e-5 && max_rel < 1e-5, "wgpu vs oracle: max_abs={max_abs} max_rel={max_rel}");
    }
}
```

- [ ] **Step 4: Gate the module in `mod.rs`.**

```rust
#[cfg(feature = "wgpu")]
pub mod wgpu;
#[cfg(feature = "wgpu")]
pub use wgpu::WgpuBackend;
```

- [ ] **Step 5: Default build untouched, then run the wgpu test.**

Run: `cargo build` → warning-free.
Run: `cargo test --features wgpu --lib wave_resonate_gpu::wgpu -- --nocapture`
Expected: `wgpu_matches_oracle_within_tolerance` PASS. Adjust API per the pinned version if needed.

- [ ] **Step 6: Commit.**

```bash
git add Cargo.toml src/wave_resonate_gpu/mod.rs src/wave_resonate_gpu/wgpu.rs src/wave_resonate_gpu/elig.wgsl
git commit -m "feat(wave_resonate_gpu): WebGPU backend (wgpu + WGSL) validated vs oracle <1e-5"
```

---

### Task 6: Throughput example + record the GO/NO-GO

**Files:**
- Create: `examples/profile_resonate_gpu.rs`
- Modify: `docs/experiments_results.md` (append the results section)

**Interfaces:**
- Consumes: `capture_inputs`, `cpu_accrue`, `run_backend`, `CpuBackend`, and (feature-gated) `CudaBackend`/`WgpuBackend`.

- [ ] **Step 1: Write `examples/profile_resonate_gpu.rs`** — for each size in {32,64,128}: build the FF cfg, capture a 256-wave random-drive sequence, then time `CpuBackend` (baseline), and each enabled GPU backend; print waves/s and `max_abs` vs the CpuBackend result.

```rust
//! GPU eligibility spike throughput: CPU vs CUDA vs wgpu waves/s across size {32,64,128}, plus max error
//! vs the CPU reference (itself validated <1e-6 vs the dense_eligibility oracle in the unit tests).
//! Run: `cargo run --release --features cuda --example profile_resonate_gpu`
//!  or: `cargo run --release --features "cuda wgpu" --example profile_resonate_gpu`

use wave_net::wave_resonate::config::{Config, LayerConfig};
use wave_net::wave_resonate::synapse::{random_l0_input, TopologyLevel};
use wave_net::wave_resonate_gpu::{capture_inputs, run_backend, CpuBackend, GpuBackend, Layout, Captured};

const SEED: u64 = 0xC0FFEE_1234_5678;
const WAVES: usize = 256;

fn ff_cfg(size: u32) -> Config {
    let lc = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }], inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.0, 0.2), tau_out: 20.0 };
    Config { seed: SEED, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: vec![lc; 5] }
}

fn waves_per_s(dur: std::time::Duration) -> f64 { WAVES as f64 / dur.as_secs_f64() }

fn max_abs(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn run_named<B: GpuBackend>(name: &str, layout: &Layout, seq: &[Captured], base: &[f32]) {
    let (elig, dur) = run_backend::<B>(layout, seq, 0.05, 1e-6);
    println!("    {name:<6} {:>10.0} waves/s   max_abs_vs_cpu={:.2e}", waves_per_s(dur), max_abs(base, &elig));
}

fn main() {
    for &size in &[32u32, 64, 128] {
        let cfg = ff_cfg(size);
        let input = random_l0_input(SEED, size, 8000);
        let inputs: Vec<Vec<u32>> = (0..WAVES).map(&input).collect();
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        println!("size {size}: {} synapses, {} waves", layout.n, WAVES);
        let (cpu, cdur) = run_backend::<CpuBackend>(&layout, &seq, 0.05, 1e-6);
        println!("    {:<6} {:>10.0} waves/s   (reference)", "cpu", waves_per_s(cdur));
        #[cfg(feature = "cuda")]
        run_named::<wave_net::wave_resonate_gpu::CudaBackend>("cuda", &layout, &seq, &cpu);
        #[cfg(feature = "wgpu")]
        run_named::<wave_net::wave_resonate_gpu::WgpuBackend>("wgpu", &layout, &seq, &cpu);
        let _ = &cpu;
    }
}
```

- [ ] **Step 2: Build + run each enabled backend.**

Run: `cargo build --features cuda --example profile_resonate_gpu` → compiles.
Run: `cargo run --release --features cuda --example profile_resonate_gpu`
Then (if the wgpu backend is landed): `cargo run --release --features "cuda wgpu" --example profile_resonate_gpu`
Expected: a table per size; every `max_abs_vs_cpu` < 1e-5; GPU waves/s rising relative to CPU as size grows.

- [ ] **Step 3: Record results + the GO/NO-GO** in `docs/experiments_results.md`.

Append a `## wave_resonate GPU eligibility spike` section: the CPU/CUDA/wgpu waves/s table for {32,64,128}, the max errors, the end-to-end projection (`(forward + cpu_accrue) / (forward + gpu)` using the profiled forward ~7%), and the GO/NO-GO call against the ~5×-at-size≥64 bar. State plainly if a backend was unavailable in the run environment.

- [ ] **Step 4: Commit.**

```bash
git add examples/profile_resonate_gpu.rs docs/experiments_results.md
git commit -m "feat(wave_resonate_gpu): throughput example + spike results/GO-NO-GO"
```

---

## Self-Review

**Spec coverage:** flat edge-list layout → Task 1; per-synapse kernel semantics → `cpu_accrue` (Task 2) mirrored by `.cu` (Task 4) and `.wgsl` (Task 5); data flow (resident eps/elig, per-wave neuron-array upload, synchronous step) → Tasks 4–5 `step`; tolerance vs `dense_eligibility` oracle → Tasks 2/4/5 tests; `GpuBackend` trait + two hand-written backends + `CpuBackend` → Tasks 3–5; feature gates + std-only default → Cargo/`mod.rs` gating (Tasks 4–5), verified by `cargo build` at every GPU step; size sweep {32,64,128} + waves/s table + GO/NO-GO → Task 6; determinism (per-synapse, no atomics) → holds by construction in all backends. All spec sections map to a task.

**Placeholder scan:** no TBD/TODO; every code step is complete. The two "API note" callouts (cudarc/wgpu version pinning) are deliberate — external crate APIs churn past the knowledge cutoff, so the plan fixes the *semantics* (`cpu_accrue`) and flags "verify signatures against the pinned version," which is the correct instruction rather than a fabricated exact API.

**Type consistency:** `GpuBackend::{new,reset,step,download_elig}` signatures identical across `CpuBackend`/`CudaBackend`/`WgpuBackend`; `Layout`/`Captured` field names (`n,ls,l,syn_base,tgt_g,src_g` / `b_eff_g,omega_g,psi_g,prev_fired_g`) used consistently in every task and both kernels; `run_backend`/`capture_inputs`/`cpu_accrue` signatures match their call sites in Tasks 3/4/5/6; kernel bindings (WGSL `@binding` order) match the Rust `BindGroupEntry` order and the `.cu` parameter order matches the cudarc `launch` tuple order.

**Known follow-ups (out of spike scope):** async kernel overlap (non-synchronous `step`) is a deferred optimization; synchronous `step` is intentionally conservative for GO/NO-GO timing. Sorting edges by target id (to reduce `b_eff_g[j]` gather divergence) is likewise deferred.
