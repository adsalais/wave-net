# wave_resonate Phase 2a — HYPR training (weights-only) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline; this repo does not
> use subagent-driven execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train the BRF engine's **ternary weights** online with a HYPR forward eligibility (no BPTT):
a per-synapse 2-state trace `(ε^x, ε^y)` through the BRF Jacobian × a double-Gaussian surrogate `ψ`,
accumulated into `elig`, applied by a multi-layer-DFA update to the f32 shadow, then requantized.

**Architecture:** Mirror `wave_driven`'s online-eligibility + `dfa_update` shape. The forward pass captures
each compute neuron's `b_eff` and `ψ` per wave; after each wave, an active-set accrual advances the
per-synapse `(ε^x, ε^y)` recursion (source-driven scan, target-indexed Jacobian) and accumulates `elig`.
A dense oracle re-runs the deterministic forward and recomputes `elig` bit-for-bit. `ω/b′` stay frozen
(`train_omega_b=false` path); trainable `ω/b′` is Phase 2b.

**Tech Stack:** Rust edition 2024, std only, inline `#[cfg(test)]`.

## Global Constraints

- Std only; **warning-free**; no `unsafe` in Phase 2a (safe indexing; the sanctioned `unsafe` accrual is a
  later perf pass).
- **Determinism** — pure fn of `(seed, config, input)`; single-threaded f32.
- **Bit-exact online == dense eligibility** — the online accrual and the dense oracle must produce
  identical `elig` f32 bits. This requires identical operation order and an identical per-slot cutoff.
- **Ternary weights** stay ±1/0; training moves the f32 `shadow` and `repack_row` requantizes it.
- **HYPR eligibility (from the spec), per synapse i→j (source i in layer z, target j in layer tz=z+level):**
  ```
  ε^x_ji(t) = (1 + δ·b_j^t)·ε^x_ji(t−1) − δ·ω_j·ε^y_ji(t−1) + δ·z_i^{t−1}
  ε^y_ji(t) = δ·ω_j·ε^x_ji(t−1) + (1 + δ·b_j^t)·ε^y_ji(t−1)
  elig_ji  += ψ_j^t · ε^x_ji(t)
  ```
  `b_j^t, ω_j, ψ_j^t` are the TARGET j's (captured this wave); `z_i^{t−1}` is the SOURCE i's spike LAST
  wave (deferred one hop — matches the forward `I` current). Weight update: `shadow_ji += −lr·signal_tz[j]·elig_ji`.
- **Surrogate default (reference `StepDoubleGaussianGrad`):**
  `ψ(v) = γ·[(1+p)·N(v;0,σ₁) − 2p·N(v;0,σ₂)]`, `N(v;μ,σ)=exp(−(v−μ)²/(2σ²))/(σ√(2π))`,
  `p=0.15, σ₁=0.5, σ₂=3.0, γ=0.5`, evaluated at `v = x_j^t − ϑ_c − q_j^{t−1}`.
- One commit per task; conventional-commit messages; **no `Co-Authored-By` trailer**.

---

## File Structure

- Modify `src/wave_resonate/neurons.rs` — add `TrainState`, `enable/disable_training`, `repack_row`,
  `ternary_threshold`; extend `Layer` with `train: Option<TrainState>`.
- Create `src/wave_resonate/training.rs` — `surrogate`, `EligParams`, `Edge`, `dense_eligibility` oracle.
- Modify `src/wave_resonate/wave.rs` — capture `b_eff`/`ψ`/`spike_count` in the compute loop when training.
- Modify `src/wave_resonate/network.rs` — training toggles, per-wave `accrue_eligibility`, `dfa_update`,
  `reset_eligibility`, `set_elig_params`, active-set/prev-fired bookkeeping.
- Modify `src/wave_resonate/mod.rs` — add `pub mod training;`.
- Create `src/bench/wave_resonate_bench.rs` + register in `src/bench/mod.rs` — FF training smoke.

---

### Task 1: `TrainState` + training toggles + `repack_row`

**Files:** Modify `src/wave_resonate/neurons.rs`.

**Interfaces:**
- Produces on `Layer`:
  ```rust
  pub ternary_threshold: f32,          // new field, default 0.5
  pub train: Option<TrainState>,       // new field, None on inference-lean
  pub struct TrainState {
      pub shadow: Vec<f32>,   // ls*total_slots — ternary master (decode(codes) on enable)
      pub elig: Vec<f32>,     // ls*total_slots — Σ_t ψ_j·ε^x
      pub eps_x: Vec<f32>,    // ls*total_slots — HYPR trace (real)
      pub eps_y: Vec<f32>,    // ls*total_slots — HYPR trace (imag)
      pub b_eff: Vec<f32>,    // ls — this wave's b_j^t (captured in forward)
      pub psi: Vec<f32>,      // ls — this wave's ψ_j^t (captured in forward)
      pub spike_count: Vec<u32>, // ls — spikes since reset (liveness / future rate_reg)
  }
  impl Layer {
      pub fn enable_training(&mut self);
      pub fn disable_training(&mut self);
      pub fn repack_row(&mut self, i: usize);
  }
  ```

- [ ] **Step 1: Add the two `Layer` fields.** In the struct add `pub ternary_threshold: f32,` and
  `pub train: Option<TrainState>,`; in `Layer::new(...)` initializer add `ternary_threshold: 0.5,` and
  `train: None,`.

- [ ] **Step 2: Write the failing tests** (append to `neurons.rs` tests module)

```rust
    #[test]
    fn enable_training_builds_shadow_from_codes() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]);
        let mut l = Layer::new(&cfg, 0.05, 0.9, 1.0, 3, 0, size);
        assert!(l.train.is_none());
        l.enable_training();
        let t = l.train.as_ref().unwrap();
        assert_eq!(t.shadow.len(), l.synapse_count());
        assert_eq!(t.eps_x.len(), l.synapse_count());
        assert_eq!(t.eps_y.len(), l.synapse_count());
        assert_eq!(t.b_eff.len(), l.x.len());
        for s in 0..t.shadow.len() {
            assert_eq!(t.shadow[s], l.weight_at(s) as f32, "shadow == decode(codes)");
        }
        assert!(t.elig.iter().all(|&e| e == 0.0) && t.eps_x.iter().all(|&e| e == 0.0));
    }

    #[test]
    fn repack_row_roundtrips_shadow_to_ternary() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let mut l = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        l.enable_training();
        {
            let sh = &mut l.train.as_mut().unwrap().shadow;
            sh[0] = 2.0; sh[1] = -3.0; sh[2] = 0.05; sh[3] = 0.0;
        }
        l.repack_row(0);
        assert_eq!(l.weight_at(0), 1);
        assert_eq!(l.weight_at(1), -1);
        assert_eq!(l.weight_at(2), 0);
        assert_eq!(l.weight_at(3), 0);
    }

    #[test]
    fn disable_training_frees_state() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let mut l = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        l.enable_training();
        l.disable_training();
        assert!(l.train.is_none());
    }
```

- [ ] **Step 3: Run to verify failure** — `cargo test wave_resonate::neurons` → FAIL (no `enable_training`).

- [ ] **Step 4: Implement `TrainState` + methods** (add above `#[cfg(test)]`)

```rust
/// Per-layer TRAINING state — allocated only while training (see `enable_training`). `shadow` is the f32
/// master requantized into `codes` by `repack_row`; `eps_x/eps_y` are the per-synapse HYPR 2-state
/// eligibility trace (layout == `shadow`); `elig` accumulates `ψ·ε^x` over a trial; `b_eff/psi` are the
/// per-neuron values captured each wave by the forward pass.
pub struct TrainState {
    pub shadow: Vec<f32>,
    pub elig: Vec<f32>,
    pub eps_x: Vec<f32>,
    pub eps_y: Vec<f32>,
    pub b_eff: Vec<f32>,
    pub psi: Vec<f32>,
    pub spike_count: Vec<u32>,
}

impl Layer {
    pub fn enable_training(&mut self) {
        if self.train.is_some() {
            return;
        }
        let n = self.synapse_count();
        let ls = self.x.len();
        let mut shadow = vec![0f32; n];
        for s in 0..n {
            shadow[s] = self.weight_at(s) as f32;
        }
        self.train = Some(TrainState {
            shadow,
            elig: vec![0f32; n],
            eps_x: vec![0f32; n],
            eps_y: vec![0f32; n],
            b_eff: vec![0f32; ls],
            psi: vec![0f32; ls],
            spike_count: vec![0u32; ls],
        });
    }

    pub fn disable_training(&mut self) {
        self.train = None;
    }

    #[inline]
    fn set_code(&mut self, idx: usize, code: u64) {
        let w = idx >> 5;
        let shift = (idx & 31) * 2;
        self.codes[w] = (self.codes[w] & !(0b11u64 << shift)) | (code << shift);
    }

    /// Requantise neuron `i`'s row into `codes`: γ = mean(|shadow|); `|shadow|/γ < ternary_threshold → 0`,
    /// else sign. Requires training enabled.
    pub fn repack_row(&mut self, i: usize) {
        let ts = self.total_slots;
        if ts == 0 {
            return;
        }
        let base = i * ts;
        let t = self.ternary_threshold;
        let gamma = {
            let shadow = &self.train.as_ref().expect("repack_row requires training enabled").shadow;
            let mut sum = 0.0f32;
            for s in 0..ts {
                sum += shadow[base + s].abs();
            }
            sum / ts as f32
        };
        for s in 0..ts {
            let sh = self.train.as_ref().unwrap().shadow[base + s];
            let x = if gamma <= 0.0 { 0.0 } else { sh / gamma };
            let code: u64 = if x.abs() < t { 0b00 } else if x > 0.0 { 0b01 } else { 0b11 };
            self.set_code(base + s, code);
        }
    }
}
```

- [ ] **Step 5: Run to verify pass** — `cargo test wave_resonate::neurons` → PASS (8 tests). Warning check.

- [ ] **Step 6: Commit**

```bash
git add src/wave_resonate/neurons.rs
git commit -m "feat(wave_resonate): TrainState (shadow + HYPR eps traces) + repack_row"
```

---

### Task 2: `training.rs` — surrogate + `EligParams` + `Edge`

**Files:** Create `src/wave_resonate/training.rs`; modify `src/wave_resonate/mod.rs` (add `pub mod training;`).

**Interfaces:**
- Produces:
  ```rust
  pub fn surrogate(v: f32) -> f32;                     // double-Gaussian, reference constants
  pub struct EligParams { pub dt: f32, pub eps_cut: f32, pub train_omega_b: bool }
  impl Default for EligParams { /* dt matches Config; eps_cut = 1/1024; train_omega_b = false (2a) */ }
  pub struct Edge { pub level: i32, pub count: usize, pub radius: u32 }
  ```

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_peaks_at_zero_and_is_symmetric() {
        let z = surrogate(0.0);
        assert!(z > 0.0, "positive at threshold");
        assert!((surrogate(0.3) - surrogate(-0.3)).abs() < 1e-6, "symmetric");
        assert!(surrogate(0.0) > surrogate(0.5), "peaks near 0");
        assert!(surrogate(50.0).abs() < 1e-3, "≈0 far from threshold");
    }

    #[test]
    fn elig_params_default_is_weights_only() {
        let p = EligParams::default();
        assert!(!p.train_omega_b, "Phase 2a default: ω/b′ frozen");
        assert!(p.eps_cut > 0.0);
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test wave_resonate::training` → FAIL (module missing).

- [ ] **Step 3: Add `pub mod training;` to `mod.rs`, then implement `training.rs`** (leave `dense_eligibility` for Task 5 — a stub is not needed since Task 5 appends it)

```rust
//! `training` — HYPR online-eligibility training for the BRF engine: the double-Gaussian surrogate, the
//! eligibility knobs, and (Task 5) the bit-exact dense_eligibility oracle. The per-wave accrual and the
//! shadow update live on `Network` (they need the layer stack + per-wave fired sets).

/// Reference double-Gaussian surrogate `∂z/∂x` at `v = x − ϑ_c − q` (StepDoubleGaussianGrad).
#[inline]
pub fn surrogate(v: f32) -> f32 {
    const P: f32 = 0.15;
    const S1: f32 = 0.5;
    const S2: f32 = 3.0;
    const G: f32 = 0.5;
    let inv_sqrt_2pi = 1.0f32 / (2.0 * std::f32::consts::PI).sqrt();
    let n = |mu: f32, sigma: f32| (inv_sqrt_2pi / sigma) * (-((v - mu) * (v - mu)) / (2.0 * sigma * sigma)).exp();
    G * ((1.0 + P) * n(0.0, S1) - 2.0 * P * n(0.0, S2))
}

/// HYPR eligibility knobs. `dt` mirrors `Config::dt` (the eligibility recursion uses the same δ). `eps_cut`
/// zeroes a trace slot once `|ε^x|,|ε^y|` fall below it (bounds the trace + keeps the dense oracle exact).
/// `train_omega_b` gates the per-neuron ω/b′ updates (Phase 2b; false here).
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub dt: f32,
    pub eps_cut: f32,
    pub train_omega_b: bool,
}

impl Default for EligParams {
    fn default() -> Self {
        EligParams { dt: 0.05, eps_cut: 1.0 / 1024.0, train_omega_b: false }
    }
}

/// One topology edge of a source layer, in built `LayerConfig` topology order (so `entries[z][e]` lines up
/// with the layer's `e`-th level). Mirrors the DFA credit wiring.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub level: i32,
    pub count: usize,
    pub radius: u32,
}
```

Then append the Step-1 test module.

- [ ] **Step 4: Run to verify pass** — `cargo test wave_resonate::training` → PASS (2 tests). Warning check.

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/training.rs src/wave_resonate/mod.rs
git commit -m "feat(wave_resonate): double-Gaussian surrogate + EligParams/Edge"
```

---

### Task 3: Forward capture of `b_eff` / `ψ` / `spike_count`

**Files:** Modify `src/wave_resonate/wave.rs`.

**Interfaces:**
- Consumes: `Layer.train` (Task 1), `surrogate` (Task 2).
- Produces: when `layer.train.is_some()` and the layer is a compute layer, `process_layer` writes
  `train.b_eff[i] = b_j^t`, `train.psi[i] = surrogate(nx − θ_c − q_old)`, and `train.spike_count[i] += 1`
  on a spike. No behavior change when `train` is `None`.

- [ ] **Step 1: Write the failing test** (append to `wave.rs` tests)

```rust
    #[test]
    fn training_capture_fills_b_eff_and_psi() {
        use crate::wave_resonate::training::surrogate;
        let dt = 0.05f32;
        let mut l = compute_layer(1, 10.0, 0.3, dt, 0.9, 1.0);
        l.enable_training();
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
        let mut fired = Vec::new();
        l.pending[0] = 40; // drive toward threshold
        process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
        let t = l.train.as_ref().unwrap();
        // b_eff should equal pw(omega) - |b_off| - q_old (q_old was 0 on the first wave)
        let expect_b = crate::wave_resonate::neurons::pw(l.omega[0], dt) - l.b_off[0].abs() - 0.0;
        assert!((t.b_eff[0] - expect_b).abs() < 1e-6, "b_eff captured");
        assert_eq!(t.psi[0], surrogate(l.x[0] - l.theta_c - 0.0), "psi captured at (x - θ_c - q_old)");
    }
```

- [ ] **Step 2: Run to verify failure** — FAIL (b_eff stays 0.0).

- [ ] **Step 3: Implement capture in the compute loop.** In `process_layer`'s compute branch, before the
  loop take an optional split of the train fields, and inside the loop (after computing `b`, `nx`, using
  the pre-update `q`) write the captures. Use `use crate::wave_resonate::training::surrogate;` at top.
  Replace the compute loop with:

```rust
    // --- compute: dense BRF oscillator update + decide (+ training capture) ---
    let (dt, gamma, theta_c) = (layer.dt, layer.gamma, layer.theta_c);
    let training = layer.train.is_some();
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
        if spike {
            fired.push(i as u32);
        }
        if training {
            let t = layer.train.as_mut().unwrap();
            t.b_eff[i] = b;
            t.psi[i] = surrogate(nx - theta_c - q);
            if spike {
                t.spike_count[i] += 1;
            }
        }
    }
```

- [ ] **Step 4: Run to verify pass** — `cargo test wave_resonate::wave` → PASS (8 tests). Warning check.

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/wave.rs
git commit -m "feat(wave_resonate): forward capture of b_eff/ψ/spike_count when training"
```

---

### Task 4: Online eligibility accrual on `Network`

**Files:** Modify `src/wave_resonate/network.rs`.

**Interfaces:**
- Consumes: `Layer.train`, `EligParams`, `Edge`, `process_layer`.
- Produces on `Network`:
  ```rust
  pub fn enable_training(&mut self);
  pub fn disable_training(&mut self);
  pub fn is_training(&self) -> bool;
  pub fn set_elig_params(&mut self, p: EligParams);
  pub fn layer_spike_count(&self, z: usize) -> &[u32];
  // wave() now calls accrue_eligibility() after the layer sweep+swap when training
  ```
- New private state: `elig_params: EligParams`, `fired_by_layer: Vec<Vec<u32>>`,
  `prev_fired_bitset: Vec<Vec<u64>>` (source spikes from the PREVIOUS wave), `elig_active: Vec<Vec<u32>>`
  + `elig_mark: Vec<Vec<u64>>` (source active-set + dedup), `dirty_rows: Vec<Vec<u32>>` +
  `dirty_mark: Vec<Vec<u64>>`, `entries: Vec<Vec<Edge>>` (topology edges per layer, built in `build`).

**Accrual algorithm (canonical order — the dense oracle mirrors it exactly):** after the layer sweep and
the pending/deliv swap, for wave `t`:
1. For each source layer `z`, add this-wave firers to `elig_active[z]` (dedup via `elig_mark`).
2. For each source layer `z`, scan `elig_active[z]` (compacting survivors). For source `i`, for each edge
   `e` with `tz=z+level ∈ [0,L)`: snapshot the TARGET layer's `b_eff`, `psi` (from `train[tz]`) and
   `omega` (from `layers[tz]`) — read-only. For each wired cell → target `j`:
   ```
   widx = i*ts + slot_base(e) + rank
   ex = eps_x[widx]; ey = eps_y[widx]
   inj = δ · (prev_fired_bitset[z] has i ? 1 : 0)
   nex = (1 + δ·b_eff[j])·ex − δ·omega[j]·ey + inj
   ney = δ·omega[j]·ex + (1 + δ·b_eff[j])·ey
   if nex.abs() < eps_cut { nex = 0 }   // per-slot cutoff (bit-exact in dense too)
   if ney.abs() < eps_cut { ney = 0 }
   eps_x[widx] = nex; eps_y[widx] = ney
   if psi[j] != 0 && nex != 0 { elig[widx] += psi[j]·nex; mark dirty_rows[z] i }
   ```
   Keep source `i` in `elig_active[z]` iff its eps row has any nonzero slot; else drop.
3. Rebuild `prev_fired_bitset` from this wave's `fired_by_layer` (set bits), for use next wave.

Borrow handling: because the scan mutates `layers[z].train.eps_x/elig` while reading `layers[tz].train.b_eff/psi`
and `layers[tz].omega`, snapshot the per-target arrays into owned `Vec<f32>` (`b_eff`, `psi`, `omega`) for
all layers up front each wave (O(n_total) f32 copies — acceptable; the dense membrane is already O(n_total)),
then the scan borrows only `layers[z]` mutably. This is the same "precompute a per-layer read-only vector"
device `wave_driven` uses for `rho`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod training_tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;
    use crate::wave_resonate::training::EligParams;

    fn two_layer(size: u32) -> Config {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Config { seed: 9, size, dt: 0.05, gamma: 0.9, theta_c: 1.0, layers: vec![up, top] }
    }

    #[test]
    fn training_toggles() {
        let mut net = Network::new(two_layer(8));
        assert!(!net.is_training());
        net.enable_training();
        assert!(net.is_training());
        net.with_layer(0, |l| assert_eq!(l.train.as_ref().unwrap().shadow.len(), l.synapse_count()));
        net.disable_training();
        assert!(!net.is_training());
    }

    #[test]
    fn accrual_is_deterministic_and_nonzero() {
        let cfg = two_layer(8);
        let run = || {
            let mut net = Network::new(cfg.clone());
            net.enable_training();
            for _ in 0..40 { net.wave(&[0, 1, 2, 8, 9, 10]); }
            net.with_layer(0, |l| l.train.as_ref().unwrap().elig.clone())
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "deterministic elig");
        assert!(a.iter().any(|&e| e != 0.0), "some L0→L1 eligibility accrued once L1 fires");
    }
}
```

- [ ] **Step 2: Run to verify failure** — FAIL (`enable_training` etc. not on `Network`).

- [ ] **Step 3: Implement.** Add the new fields to `Network` + init in `build` (build `entries` from each
  layer's topology). Add the training methods and `accrue_eligibility`; call it at the end of `wave` when
  training. Capture `fired` per layer during the sweep into `fired_by_layer`. (Full method bodies: follow
  the algorithm above; model the active-set/compaction and dirty-row bookkeeping on
  `wave_driven::network::accrue_eligibility`, but with the 2-state `eps_x/eps_y` recursion and the
  precomputed per-layer `b_eff/psi/omega` snapshots instead of `rho`/`pretr`.)

Key code for the per-wave snapshot + scan (inside `accrue_eligibility`):

```rust
fn accrue_eligibility(&mut self) {
    let l = self.layers.len();
    let size = self.size;
    let dt = self.elig_params.dt;
    let cut = self.elig_params.eps_cut;
    // 1. per-layer read-only target snapshots (b_eff, psi, omega)
    let b_snap: Vec<Vec<f32>> = self.layers.iter().map(|lz| lz.train.as_ref().map(|t| t.b_eff.clone()).unwrap_or_default()).collect();
    let psi_snap: Vec<Vec<f32>> = self.layers.iter().map(|lz| lz.train.as_ref().map(|t| t.psi.clone()).unwrap_or_default()).collect();
    let om_snap: Vec<Vec<f32>> = self.layers.iter().map(|lz| lz.omega.clone()).collect();
    let entries = self.entries.clone();
    let Self { layers, fired_by_layer, prev_fired_bitset, elig_active, elig_mark, dirty_rows, dirty_mark, .. } = self;

    // 2. add this-wave firers to elig_active (dedup)
    for z in 0..l {
        for &i in &fired_by_layer[z] {
            let w = (i >> 6) as usize; let bit = 1u64 << (i & 63);
            if elig_mark[z][w] & bit == 0 { elig_mark[z][w] |= bit; elig_active[z].push(i); }
        }
    }

    // 3. scan each source layer's active set
    for z in 0..l {
        if layers[z].train.is_none() { continue; }
        let ts = layers[z].total_slots;
        let mut scan = std::mem::take(&mut elig_active[z]);
        let mut keep = 0usize;
        for r in 0..scan.len() {
            let iu = scan[r]; let i = iu as usize;
            let src_fired_prev = prev_fired_bitset[z][(iu >> 6) as usize] & (1u64 << (iu & 63)) != 0;
            let inj = if src_fired_prev { dt } else { 0.0 };
            let mut any_live = false;
            let (sx, sy) = crate::wave_resonate::synapse::xy_of(iu, size);
            // borrow this source layer mutably
            let Layer { topology, slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
            let tr = train.as_mut().unwrap();
            for (e_idx, entry) in topology.iter().enumerate() {
                let tz_i = z as i32 + entry.level;
                if tz_i < 0 || tz_i as usize >= l { continue; }
                let tz = tz_i as usize;
                let (b_t, psi_t, om_t) = (&b_snap[tz], &psi_snap[tz], &om_snap[tz]);
                let sbase = slot_bases[e_idx];
                let wpn = occ_wpn[e_idx];
                let words = &occ[e_idx][i * wpn..i * wpn + wpn];
                let lut = &offsets[e_idx];
                let mut rank = 0usize;
                for (wi, &w0) in words.iter().enumerate() {
                    let mut word = w0; let cbase = wi * 64;
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        let cell = cbase + bit;
                        let (dx, dy) = lut[cell];
                        let j = crate::wave_resonate::synapse::local_of(
                            crate::wave_resonate::synapse::wrap(sx, dx as i32, size),
                            crate::wave_resonate::synapse::wrap(sy, dy as i32, size), size) as usize;
                        let widx = i * ts + sbase + rank;
                        let ex = tr.eps_x[widx]; let ey = tr.eps_y[widx];
                        let coef = 1.0 + dt * b_t[j];
                        let mut nex = coef * ex - dt * om_t[j] * ey + inj;
                        let mut ney = dt * om_t[j] * ex + coef * ey;
                        if nex.abs() < cut { nex = 0.0; }
                        if ney.abs() < cut { ney = 0.0; }
                        tr.eps_x[widx] = nex; tr.eps_y[widx] = ney;
                        if psi_t[j] != 0.0 && nex != 0.0 {
                            tr.elig[widx] += psi_t[j] * nex;
                            let w = (iu >> 6) as usize; let b = 1u64 << (iu & 63);
                            if dirty_mark[z][w] & b == 0 { dirty_mark[z][w] |= b; dirty_rows[z].push(iu); }
                        }
                        if nex != 0.0 || ney != 0.0 { any_live = true; }
                        rank += 1; word &= word - 1;
                    }
                }
            }
            if any_live { scan[keep] = iu; keep += 1; }
            else { elig_mark[z][(iu >> 6) as usize] &= !(1u64 << (iu & 63)); }
        }
        scan.truncate(keep);
        elig_active[z] = scan;
    }

    // 4. prev_fired_bitset := this wave's fired (for next wave's injection)
    for z in 0..l {
        for w in prev_fired_bitset[z].iter_mut() { *w = 0; }
        for &i in &fired_by_layer[z] { prev_fired_bitset[z][(i >> 6) as usize] |= 1u64 << (i & 63); }
    }
}
```

In `wave`, after the swap, capture fired per layer (during the sweep, `fired_by_layer[z].clone_from(fired)`)
and, if `self.is_training()`, call `self.accrue_eligibility()`. Note `fired_by_layer` must be filled inside
the sweep loop right after each `process_layer` (as `wave_driven` does).

- [ ] **Step 4: Run to verify pass** — `cargo test wave_resonate::network` → PASS. Warning check.

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/network.rs
git commit -m "feat(wave_resonate): online HYPR eligibility accrual (2-state ε^x/ε^y, active-set)"
```

---

### Task 5: Dense eligibility oracle + bit-exact test

**Files:** Modify `src/wave_resonate/training.rs` (append `dense_eligibility` + expose a capture hook if
needed); add the bit-exact test in `src/wave_resonate/equivalence_tests.rs` (create + register in `mod.rs`
under `#[cfg(test)] mod equivalence_tests;`).

**Interfaces:**
- Produces: `pub fn dense_eligibility(cfg: &Config, inputs: &[Vec<u32>], p: &EligParams) -> Vec<Vec<f32>>`
  — builds a FRESH training net, runs `wave` for each input capturing per-wave `(fired[z], b_eff[z], psi[z])`
  via `with_layer`, then computes `elig` per layer with the SAME recursion/order/cutoff as the online
  accrual but scanning ALL sources every wave (no active-set). Returns per-layer `elig` vectors.

Because the online net and the oracle both build from the same `(seed,config,input)`, their forward passes
are bit-identical, so `b_eff/psi/fired` match and the eligibility matches bit-for-bit.

- [ ] **Step 1: Write the failing test** (in `equivalence_tests.rs`)

```rust
use crate::wave_resonate::config::{Config, LayerConfig};
use crate::wave_resonate::network::Network;
use crate::wave_resonate::synapse::{random_l0_input, TopologyLevel};
use crate::wave_resonate::training::{dense_eligibility, EligParams};

fn deep_cfg(size: u32) -> Config {
    let mk = |topology: Vec<TopologyLevel>| LayerConfig { topology, inhibitor_ratio: 6553, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0 };
    Config { seed: 0x0E11, size, dt: 0.05, gamma: 0.9, theta_c: 1.0, layers: vec![
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
        mk(vec![]),
    ] }
}

#[test]
fn online_equals_dense_eligibility_bit_exact() {
    let size = 16u32;
    let cfg = deep_cfg(size);
    let p = EligParams::default();
    let gen = random_l0_input(0x0E11, size, 15000);
    let inputs: Vec<Vec<u32>> = (0..120).map(|w| gen(w)).collect();

    let mut net = Network::new(cfg.clone());
    net.set_elig_params(p);
    net.enable_training();
    for inp in &inputs { net.wave(inp); }

    let dense = dense_eligibility(&cfg, &inputs, &p);
    for z in 0..net.layer_count() {
        net.with_layer(z, |lz| {
            assert_eq!(lz.train.as_ref().unwrap().elig, dense[z], "layer {z} online == dense elig (bit-exact)");
        });
    }
}
```

- [ ] **Step 2: Run to verify failure** — FAIL (`dense_eligibility` missing).

- [ ] **Step 3: Implement `dense_eligibility`** in `training.rs`:

```rust
use crate::wave_resonate::config::Config;
use crate::wave_resonate::network::Network;
use crate::wave_resonate::synapse::{local_of, wrap, xy_of};

/// Bit-exact reference eligibility. Builds a fresh training net, drives it, and after EACH wave captures
/// (fired[z], b_eff[z], psi[z]) via the layer state, then advances the SAME ε^x/ε^y recursion (same order,
/// same per-slot cutoff) over ALL sources. Because the forward is deterministic, this matches the online
/// accrual bit-for-bit.
pub fn dense_eligibility(cfg: &Config, inputs: &[Vec<u32>], p: &EligParams) -> Vec<Vec<f32>> {
    let size = cfg.size;
    let ls = (size as usize) * (size as usize);
    let l = cfg.layers.len();
    let dt = p.dt;
    let cut = p.eps_cut;

    let mut net = Network::new(cfg.clone());
    net.set_elig_params(*p);
    net.enable_training();

    let entries: Vec<Vec<Edge>> = cfg.layers.iter().map(|lc| lc.topology.iter().map(|t| Edge { level: t.level, count: t.count as usize, radius: t.radius }).collect()).collect();
    let ts_by: Vec<usize> = (0..l).map(|z| net.with_layer(z, |lz| lz.total_slots)).collect();
    let mut elig: Vec<Vec<f32>> = (0..l).map(|z| vec![0f32; ls * ts_by[z]]).collect();
    let mut eps_x: Vec<Vec<f32>> = elig.iter().map(|e| vec![0f32; e.len()]).collect();
    let mut eps_y: Vec<Vec<f32>> = elig.iter().map(|e| vec![0f32; e.len()]).collect();
    let mut prev_fired = vec![vec![false; ls]; l];

    for inp in inputs {
        net.wave(inp);
        // capture this wave's fired / b_eff / psi / omega per layer
        let mut fired = vec![vec![false; ls]; l];
        let mut b_eff = vec![vec![0f32; ls]; l];
        let mut psi = vec![vec![0f32; ls]; l];
        let mut omega = vec![vec![0f32; ls]; l];
        for z in 0..l {
            net.with_layer(z, |lz| {
                omega[z].copy_from_slice(&lz.omega);
                if let Some(t) = lz.train.as_ref() {
                    b_eff[z].copy_from_slice(&t.b_eff);
                    psi[z].copy_from_slice(&t.psi);
                }
            });
        }
        // NOTE: `fired` for the *source injection* uses the PREVIOUS wave (prev_fired). We need this wave's
        // spikes for next iteration; capture them via a listener-free reconstruction is unavailable, so use
        // a spike listener set up before the loop (see Step 3b).
        let _ = &mut fired;
        // accrue: all sources, same order as online
        for z in 0..l {
            let ts = ts_by[z];
            for i in 0..ls {
                let (sx, sy) = xy_of(i as u32, size);
                let inj = if prev_fired[z][i] { dt } else { 0.0 };
                for (e_idx, edge) in entries[z].iter().enumerate() {
                    let tz_i = z as i32 + edge.level;
                    if tz_i < 0 || tz_i as usize >= l { continue; }
                    let tz = tz_i as usize;
                    net_dense_accrue(net_slot_base(&net, z, e_idx), edge, i, ts, sx, sy, size, dt, cut,
                        &b_eff[tz], &psi[tz], &omega[tz], &net_occ(&net, z, e_idx, i), &net_off(&net, z, e_idx),
                        inj, &mut eps_x[z], &mut eps_y[z], &mut elig[z]);
                }
            }
        }
        // roll prev_fired ← this wave's spikes (captured via listener; see 3b)
    }
    elig
}
```

- [ ] **Step 3b: Simplify the capture.** The pseudo-helpers above (`net_occ`, `net_off`, …) are awkward.
  Instead, have `dense_eligibility` read occupancy/offsets/slot_bases directly via a single
  `net.with_layer(z, |lz| …)` that, for each wave, does the full source scan inside the closure (it has
  `&Layer`, so `for_wired`/`decode`/`slot_base` are available). Capture this wave's fired via a listener
  registered before the loop (mirror `wave_driven`'s `dense_eligibility` test which records fired through
  `on_layer`). Concretely: register `on_layer(z)` closures pushing fired bitsets into a shared
  `Vec<Vec<bool>>`; after each `wave`, run the scan using `with_layer` for occupancy/`for_wired`/`decode`
  and the captured `b_eff/psi/omega`; then roll `prev_fired` from the just-captured fired. Keep the inner
  arithmetic byte-identical to Task 4 (same `coef`, same cutoff order, same `elig += psi*nex` guard).

  (Implementer: prefer this closure-based form; it removes the pseudo-helpers and keeps one source of the
  scan arithmetic. The `for_wired(e_idx, i, |rank, cell| …)` + `decode(e_idx, i as u32, cell, size)` pair
  gives `(rank, j)` exactly as the online scan's `rank`/`j`.)

- [ ] **Step 4: Run to verify pass** — `cargo test wave_resonate::equivalence_tests` → PASS. If not
  bit-exact, diff the first mismatching `widx`: the usual causes are (a) injection using this-wave instead
  of prev-wave fired, (b) cutoff applied in a different order, (c) `eps_y` computed from the already-updated
  `eps_x` (must use the OLD `ex`). Fix until identical.

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/training.rs src/wave_resonate/equivalence_tests.rs src/wave_resonate/mod.rs
git commit -m "test(wave_resonate): bit-exact online-vs-dense HYPR eligibility oracle"
```

---

### Task 6: `dfa_update` + `reset_eligibility`

**Files:** Modify `src/wave_resonate/network.rs`.

**Interfaces:**
- Produces on `Network`:
  ```rust
  pub fn dfa_update(&mut self, entries: &[Vec<Edge>], signal: &[Vec<f32>], lr: f32);
  pub fn reset_eligibility(&mut self);   // also called by reset_state
  ```

`dfa_update`: for each source layer `z`, over `dirty_rows[z]`, for each edge with `tz=z+level ∈ [1,L)`,
for each wired cell → `j`: `shadow[widx] += −lr·signal[tz][j]·elig[widx]` (only when `elig != 0`); then
`repack_row(i)` for touched rows. Copy the shape from `wave_driven::dfa_update`.

`reset_eligibility`: zero `elig`/`eps_x`/`eps_y` over `dirty_rows`/`elig_active`, zero `spike_count`
densely, clear `elig_active`/`elig_mark`/`dirty_rows`/`dirty_mark`/`prev_fired_bitset`/`fired_by_layer`.
Wire it into `reset_state`.

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn dfa_update_with_negative_signal_raises_eligible_synapse() {
        let cfg = two_layer(8);
        let entries = vec![vec![Edge { level: 1, count: 8, radius: 2 }], vec![]];
        let mut net = Network::new(cfg);
        net.enable_training();
        for _ in 0..40 { net.wave(&[0, 1, 2, 8, 9, 10]); }
        let ls = 64usize;
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
        let before: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow.iter().sum());
        net.dfa_update(&entries, &signal, 0.05);
        let after: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow.iter().sum());
        assert!(after != before, "a negative signal × accrued eligibility moves the shadow: {before}->{after}");
    }

    #[test]
    fn reset_eligibility_clears_accumulators() {
        let mut net = Network::new(two_layer(8));
        net.enable_training();
        for _ in 0..20 { net.wave(&[0, 1, 2, 8, 9, 10]); }
        net.reset_state();
        net.with_layer(0, |l| {
            let t = l.train.as_ref().unwrap();
            assert!(t.elig.iter().all(|&e| e == 0.0));
            assert!(t.eps_x.iter().all(|&e| e == 0.0) && t.eps_y.iter().all(|&e| e == 0.0));
            assert!(t.spike_count.iter().all(|&c| c == 0));
        });
    }
```

- [ ] **Step 2: Run to verify failure** — FAIL (`dfa_update` missing).

- [ ] **Step 3: Implement** `dfa_update` + `reset_eligibility` (shape copied from `wave_driven::network`,
  reading `elig`, writing `shadow`, then `repack_row`). Ensure `reset_state` calls `reset_eligibility`.

- [ ] **Step 4: Run to verify pass** — `cargo test wave_resonate::network` → PASS. Warning check.

- [ ] **Step 5: Commit**

```bash
git add src/wave_resonate/network.rs
git commit -m "feat(wave_resonate): multi-layer-DFA shadow update + reset_eligibility"
```

---

### Task 7: End-to-end FF training smoke (`bench`)

**Files:** Create `src/bench/wave_resonate_bench.rs`; modify `src/bench/mod.rs` (add `pub mod wave_resonate_bench;`).

**Interfaces:** Consumes the public `Network` training API. Mirrors `wave_driven_bench.rs`: cue-site task
generator, softmax2 readout on the top compute layer's read-window spike counts, DFA feedback
(`dfa_weight`) per layer, `run_trial` → `dfa_update`. Produces an `#[ignore]` smoke test proving FF
single-cue training rises above chance.

- [ ] **Step 1: Write the smoke test** — port `wave_driven_bench.rs`'s single-cue FF test (two classes,
  present/delay/read windows, `on_layer` capture of top-layer spikes, softmax2 + cross-entropy, per-neuron
  DFA signal `signal[tz][j] = (p − target)·dfa_weight(...)`, `net.dfa_update(&entries, &signal, lr)` each
  trial), swapping in the `wave_resonate` `Config/Network/EligParams`. A BRF-appropriate config: `dt=0.05`,
  `omega_init=(5,10)`, `b_offset_init=(0.1,1.0)`, 3 compute layers + read the top, generous fan-in
  (`count≈16`, `radius 3` at size 16). Mark `#[ignore]` (long; run in `--release`). Assert held-out
  accuracy > 0.6 (above the 0.5 chance floor) after enough trials.

```rust
    #[test]
    #[ignore] // run manually: cargo test --release wave_resonate_bench -- --ignored --nocapture
    fn ff_single_cue_trains_above_chance() {
        // ... (port from wave_driven_bench::tests, using wave_resonate types) ...
        // assert!(acc > 0.6, "FF BRF+HYPR single-cue accuracy {acc} above chance");
    }
```

- [ ] **Step 2: Run** — `cargo test --release wave_resonate_bench -- --ignored --nocapture`. Tune
  `lr`, trial count, and `ω`-init/`dt` until it clears 0.6 (capability gate). If it sits at chance, first
  verify liveness (top layer fires above a floor) — bump fan-in per AGENTS.md before touching the rule.

- [ ] **Step 3: Commit**

```bash
git add src/bench/wave_resonate_bench.rs src/bench/mod.rs
git commit -m "test(wave_resonate): FF single-cue HYPR training smoke (above chance)"
```

---

### Task 8: Full-suite green + warning-free + docs

- [ ] **Step 1:** `cargo test` → all green (new + existing). `cargo build` warning-free.
- [ ] **Step 2:** Update `AGENTS.md` `wave_resonate/` map: mark Phase 2a landed; add `training.rs`
  (surrogate + EligParams + dense oracle) and `equivalence_tests.rs`; note `neurons.rs` now has TrainState.
- [ ] **Step 3: Commit** `docs(wave_resonate): register Phase 2a (HYPR training) in the architecture map`.

---

## Self-Review

**Spec coverage (Phase 2a scope):**
- HYPR per-synapse 2-state eligibility `(ε^x,ε^y)` + recursion → Task 4 (online) + Task 5 (dense). ✓
- Double-Gaussian surrogate `ψ` at `x−ϑ_c−q`, reference constants → Task 2 + Task 3 capture. ✓
- `elig += ψ·ε^x`, multi-layer-DFA shadow update + repack → Tasks 4 + 6. ✓
- Bit-exact online-vs-dense oracle → Task 5. ✓
- Ternary weights preserved (shadow master, repack) → Task 1. ✓
- Training toggles (enable/disable) + reset → Tasks 1, 4, 6. ✓
- End-to-end FF training above chance → Task 7. ✓
- **Deferred to Phase 2b (correctly absent):** trainable `ω/b′` (per-neuron param eligibilities + clamp);
  `EligParams.train_omega_b` is present and defaults false. ✓
- **Deferred (spec Non-goals):** `q→b` 2nd-order eligibility term; `unsafe` accrual perf pass; persistence.

**Placeholder scan:** Task 5 Step 3 shows a first-cut `dense_eligibility` with pseudo-helpers, immediately
replaced by the closure-based form in Step 3b — the implementer writes the 3b form (the pseudo-helpers are
illustrative of the wrong path, explicitly called out). No shipped TBDs. Task 7 ports an existing, known
harness rather than inlining 200 lines verbatim — the port source (`wave_driven_bench.rs`) is exact.

**Type consistency:** `EligParams{dt,eps_cut,train_omega_b}`, `Edge{level,count,radius}`, `surrogate(v)`,
`TrainState{shadow,elig,eps_x,eps_y,b_eff,psi,spike_count}`, `dfa_update(entries,signal,lr)`,
`dense_eligibility(cfg,inputs,p)` used consistently across Tasks 1–7. The eligibility recursion (coef
`1+δ·b`, `eps_y` from OLD `ex`, per-slot cutoff, `elig += ψ·nex` guarded by `ψ≠0 && nex≠0`) is written
identically in Task 4 and Task 5 (the bit-exact requirement).
