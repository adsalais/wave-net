# wave_driven Phase 2a (activity-scaled training) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (this repo mandates **inline** execution — never subagent-driven; see AGENTS.md). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give `wave_driven` an **online, activity-scaled** multi-layer-DFA trainer — membrane e-prop eligibility with spike-ψ, accrued during `wave()` over only active synapses, replacing `wave_bitnet`'s offline `O(size²·waves·count)` post-hoc pass.

**Architecture:** An optional per-`Layer` `TrainState` (`shadow`/`elig`/`pretr`/`spike_count`) allocated only while training. Each wave, a source-driven scan accrues `e_ij += pretr_i` for every synapse whose target fired (spike-ψ), with an ε-thresholded presynaptic trace that keeps the work activity-scaled and lets an offline oracle match it bit-for-bit. At trial end the bench applies `Δshadow = −lr·signal·elig` over the touched rows and repacks. Feed-forward only; the ALIF `εᵃ` term is Phase 2b.

**Tech Stack:** Rust edition 2024, std only, deterministic, single-threaded. Inline `#[cfg(test)]` tests; `#[ignore]`d smokes in `--release`.

## Global Constraints

- **Standard library only** in `src/`; **warning-free** `cargo build`; `cargo test` green. (AGENTS.md)
- **Determinism is a hard requirement** — pure function of `(seed, config, input, task-seed)`. (AGENTS.md)
- **No `unsafe`** in new code (the one allowed `unsafe` is `wave_bitnet`'s generate loop; `wave_driven` uses only the safe path).
- **NEVER add a `Co-Authored-By` trailer.** Conventional-commit messages. **One commit per task.** **NEVER push.**
- `wave_driven` does **not** depend on `wave_bitnet` — port/copy code as needed.
- Membrane-only, **spike-ψ** (`ψ_j = 1` iff `j` fires), `elig_beta = 0`. No `decide_potential`/`decide_eff` snapshots.
- Eligibility trace: `pretr_i(t) = clamp0(pretr_i(t−1)·decay + fired_i)`, `decay = 1 − 1/rec_tau`, `clamp0(x)= if x<ε {0} else {x}`. **Canonical pretr order: decay → ε-drop → bump firers** (online and the oracle MUST use this identical order, or they won't match bit-for-bit).
- Branch: `feat/wave-driven-training` (already checked out).

---

### Task 1: `TrainState` on the Layer (shadow/elig/pretr/spike_count + repack)

**Files:**
- Modify: `src/wave_driven/neurons.rs`

**Interfaces:**
- Produces: `struct TrainState { shadow: Vec<f32>, elig: Vec<f32>, pretr: Vec<f32>, spike_count: Vec<u32> }`; `Layer.train: Option<TrainState>`; `Layer::enable_training(&mut self)`, `Layer::disable_training(&mut self)`, `Layer::repack_row(&mut self, i: usize)`; private `Layer::set_code`.
- Consumes: existing `Layer`, `WCODE`, `weight_at`, `synapse_count`.

- [ ] **Step 1: Write failing tests** — append to the `tests` module in `neurons.rs`:

```rust
#[test]
fn enable_training_builds_shadow_from_codes_and_zeros_the_rest() {
    let size = 8u32;
    let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]);
    let mut l = Layer::new(&cfg, 3, 0, size);
    assert!(l.train.is_none(), "fresh layer is inference-lean");
    l.enable_training();
    let t = l.train.as_ref().unwrap();
    assert_eq!(t.shadow.len(), l.synapse_count());
    assert_eq!(t.elig.len(), l.synapse_count());
    assert_eq!(t.pretr.len(), l.threshold.len());
    assert_eq!(t.spike_count.len(), l.threshold.len());
    for s in 0..t.shadow.len() {
        assert_eq!(t.shadow[s], l.weight_at(s) as f32, "shadow == decode(codes)");
    }
    assert!(t.elig.iter().all(|&e| e == 0.0) && t.pretr.iter().all(|&p| p == 0.0));
}

#[test]
fn repack_row_roundtrips_shadow_to_ternary() {
    let size = 8u32;
    let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
    let mut l = Layer::new(&cfg, 7, 0, size);
    l.enable_training();
    let ts = l.total_slots;
    {
        let sh = &mut l.train.as_mut().unwrap().shadow;
        sh[0] = 2.0; sh[1] = -3.0; sh[2] = 0.05; sh[3] = 0.0;
    }
    l.repack_row(0);
    assert_eq!(l.weight_at(0), 1);
    assert_eq!(l.weight_at(1), -1);
    assert_eq!(l.weight_at(2), 0);
    assert_eq!(l.weight_at(3), 0);
    let _ = ts;
}

#[test]
fn disable_training_frees_state() {
    let size = 8u32;
    let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
    let mut l = Layer::new(&cfg, 7, 0, size);
    l.enable_training();
    l.disable_training();
    assert!(l.train.is_none());
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::neurons`
Expected: FAIL (`train` field / `enable_training` not found).

- [ ] **Step 3: Add `TrainState`, the `train` field, and methods.** In `neurons.rs`:

Add the struct after the constants:

```rust
/// Per-layer TRAINING state — allocated only while training (see `enable_training`). `shadow` is the
/// f32 master requantized into `codes` by `repack_row`; `elig` is the per-synapse eligibility
/// accumulated over a trial (SAME layout as `shadow`); `pretr`/`spike_count` are per-neuron.
pub struct TrainState {
    pub shadow: Vec<f32>,      // ls * total_slots
    pub elig: Vec<f32>,        // ls * total_slots
    pub pretr: Vec<f32>,       // ls
    pub spike_count: Vec<u32>, // ls
}
```

Add `pub train: Option<TrainState>,` as the **last** field of `struct Layer`. In `Layer::new`, add `train: None,` as the last field of the returned struct literal. Then add these methods inside `impl Layer`:

```rust
pub fn enable_training(&mut self) {
    if self.train.is_some() {
        return;
    }
    let n = self.synapse_count();
    let ls = self.threshold.len();
    let mut shadow = vec![0f32; n];
    for s in 0..n {
        shadow[s] = self.weight_at(s) as f32;
    }
    self.train = Some(TrainState { shadow, elig: vec![0f32; n], pretr: vec![0f32; ls], spike_count: vec![0u32; ls] });
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
```

- [ ] **Step 4: Run tests.**

Run: `cargo test wave_driven::neurons`
Expected: PASS (all, including the three new). `cargo build` warning-free.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/neurons.rs
git commit -m "feat(wave_driven): optional per-Layer TrainState (shadow/elig/pretr/spike_count) + repack"
```

---

### Task 2: `training` types + Network training scaffolding (toggle, params, reset)

**Files:**
- Create: `src/wave_driven/training.rs`
- Modify: `src/wave_driven/mod.rs` (add `pub mod training;`)
- Modify: `src/wave_driven/network.rs`

**Interfaces:**
- Produces: `training::EligParams { rec_tau: f32, epsilon: f32 }` (+ `Default`); `training::Edge { level: i32, count: usize, radius: u32 }`; on `Network`: `enable_training`, `disable_training`, `is_training`, `set_elig_params`, `reset_eligibility`, `layer_spike_count`. New `Network` fields `fired_by_layer`, `fired_bitset`, `pretr_active`, `dirty_rows`, `elig_params`.
- Consumes: `Layer::{enable_training, disable_training}`, `Frontier`.

- [ ] **Step 1: Create `src/wave_driven/training.rs`** with the types and the offline oracle stub (oracle body lands in Task 4):

```rust
//! `training` — online, activity-scaled multi-layer-DFA training for `wave_driven`: membrane e-prop
//! eligibility with spike-ψ, accrued on the frontier during `wave()`. Types here; the accrual and
//! shadow-update live on `Network` (they need the layer stack + per-wave fired sets).

/// Eligibility knobs (membrane-only, spike-ψ). `rec_tau` sets the presynaptic-trace decay
/// (`decay = 1 − 1/rec_tau`); `epsilon` is the hard trace cutoff (activity-scaling + exact oracle).
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub rec_tau: f32,
    pub epsilon: f32,
}

impl Default for EligParams {
    fn default() -> Self {
        EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 }
    }
}

/// One topology edge of a source layer, in built `LayerConfig` topology order (so `entries[z][e]`
/// lines up with the layer's `e`-th level). Mirrors the DFA credit wiring.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub level: i32,
    pub count: usize,
    pub radius: u32,
}
```

- [ ] **Step 2: Add `pub mod training;`** to `src/wave_driven/mod.rs` (keep the list alphabetical: after `synapse` / before `wave`, or anywhere — order is cosmetic).

- [ ] **Step 3: Write failing tests** in `network.rs` `tests` module:

```rust
#[test]
fn training_toggles_and_reports() {
    let mut net = Network::new(two_layer(8));
    assert!(!net.is_training());
    net.enable_training();
    assert!(net.is_training());
    net.with_layer(0, |l| assert_eq!(l.train.as_ref().unwrap().shadow.len(), l.synapse_count()));
    net.disable_training();
    assert!(!net.is_training());
}

#[test]
fn reset_eligibility_clears_accumulators() {
    let mut net = Network::new(two_layer(8));
    net.enable_training();
    // dirty a bit of state by hand, then reset
    net.with_layer_mut_test(0, |l| {
        let t = l.train.as_mut().unwrap();
        t.elig[0] = 5.0; t.pretr[0] = 2.0; t.spike_count[0] = 7;
    });
    net.reset_eligibility();
    net.with_layer(0, |l| {
        let t = l.train.as_ref().unwrap();
        // spike_count is cleared densely; elig/pretr cleared over (now-empty) work-sets means index 0
        // may remain unless it was tracked — so reset_eligibility clears spike_count densely and
        // elig/pretr are cleared here because we seed the work-sets below in the real flow. Assert the
        // dense guarantee:
        assert!(t.spike_count.iter().all(|&c| c == 0));
    });
}
```

Add a tiny test-only mutator to `Network` (guard with `#[cfg(test)]`) so the test can seed state:

```rust
#[cfg(test)]
pub(crate) fn with_layer_mut_test<R>(&mut self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R {
    f(&mut self.layers[z])
}
```

- [ ] **Step 4: Run to verify failure.**

Run: `cargo test wave_driven::network`
Expected: FAIL (`enable_training` / `reset_eligibility` not found).

- [ ] **Step 5: Add the fields, imports, and methods to `network.rs`.** At the top:

```rust
use crate::wave_driven::training::{Edge, EligParams};
```

Add fields to `struct Network` (after `frontier_next`):

```rust
    fired_by_layer: Vec<Vec<u32>>,  // this wave's fired ids per layer (captured during wave, training only)
    fired_bitset: Vec<Vec<u64>>,    // per layer: "did neuron fire this wave" (ceil(ls/64) words)
    pretr_active: Vec<Frontier>,    // per layer: sources with a live presynaptic trace
    dirty_rows: Vec<Frontier>,      // per layer: source neurons whose elig row got accrual
    elig_params: EligParams,
```

In **both** `build` and `from_layers` (wherever the `Network { ... }` literal is constructed), add:

```rust
            fired_by_layer: (0..l).map(|_| Vec::new()).collect(),
            fired_bitset: (0..l).map(|_| vec![0u64; (ls + 63) / 64]).collect(),
            pretr_active: (0..l).map(|_| Frontier::new(ls)).collect(),
            dirty_rows: (0..l).map(|_| Frontier::new(ls)).collect(),
            elig_params: EligParams::default(),
```

Add the methods inside `impl Network`:

```rust
pub fn enable_training(&mut self) {
    for l in self.layers.iter_mut() {
        l.enable_training();
    }
}

pub fn disable_training(&mut self) {
    for l in self.layers.iter_mut() {
        l.disable_training();
    }
}

pub fn is_training(&self) -> bool {
    self.layers.first().map(|l| l.train.is_some()).unwrap_or(false)
}

pub fn set_elig_params(&mut self, p: EligParams) {
    self.elig_params = p;
}

/// Per-neuron spike count accumulated since the last reset (for rate_reg). Requires training.
pub fn layer_spike_count(&self, z: usize) -> &[u32] {
    &self.layers[z].train.as_ref().expect("layer_spike_count requires training enabled").spike_count
}

/// Clear all per-trial training accumulators (elig over dirty rows, pretr over the active set,
/// spike_count densely) and the per-wave work-sets. Called by `reset_state`.
pub fn reset_eligibility(&mut self) {
    let l = self.layers.len();
    let Self { layers, pretr_active, dirty_rows, fired_by_layer, fired_bitset, .. } = self;
    for z in 0..l {
        let ts = layers[z].total_slots;
        if let Some(t) = layers[z].train.as_mut() {
            for &i in &dirty_rows[z].list {
                let base = i as usize * ts;
                for s in 0..ts {
                    t.elig[base + s] = 0.0;
                }
            }
            for &i in &pretr_active[z].list {
                t.pretr[i as usize] = 0.0;
            }
            t.spike_count.iter_mut().for_each(|c| *c = 0);
        }
        dirty_rows[z].clear();
        pretr_active[z].clear();
        for &j in &fired_by_layer[z] {
            fired_bitset[z][(j >> 6) as usize] &= !(1u64 << (j & 63));
        }
        fired_by_layer[z].clear();
    }
}
```

Finally, make `reset_state` also reset training: add `self.reset_eligibility();` as the **last** line of `reset_state` (before it returns). For the test's `reset_eligibility_clears_accumulators`, note the elig/pretr at index 0 won't be in the work-sets (we set them by hand), so only the dense `spike_count` guarantee is asserted — that matches the code. Update that test's body to seed via the work-sets instead, so elig/pretr clearing is also exercised:

```rust
// replace the body's seeding with one that also registers the dirty row / active source:
net.with_layer_mut_test(0, |l| {
    let t = l.train.as_mut().unwrap();
    t.elig[0] = 5.0; t.pretr[0] = 2.0; t.spike_count[0] = 7;
});
net.seed_worksets_test(0, 0); // register neuron 0 as dirty + pretr-active
net.reset_eligibility();
net.with_layer(0, |l| {
    let t = l.train.as_ref().unwrap();
    assert_eq!(t.elig[0], 0.0);
    assert_eq!(t.pretr[0], 0.0);
    assert!(t.spike_count.iter().all(|&c| c == 0));
});
```

and add the test helper:

```rust
#[cfg(test)]
pub(crate) fn seed_worksets_test(&mut self, z: usize, i: u32) {
    self.dirty_rows[z].push(i);
    self.pretr_active[z].push(i);
}
```

- [ ] **Step 6: Run tests.**

Run: `cargo test wave_driven::network`
Expected: PASS. `cargo build` warning-free.

- [ ] **Step 7: Commit.**

```bash
git add src/wave_driven/training.rs src/wave_driven/mod.rs src/wave_driven/network.rs
git commit -m "feat(wave_driven): training types + Network toggle/params/reset scaffolding"
```

---

### Task 3: Online eligibility accrual on the frontier

**Files:**
- Modify: `src/wave_driven/network.rs`

**Interfaces:**
- Produces: private `Network::accrue_eligibility(&mut self)`, invoked at the end of a **sparse** `wave()` when `is_training()`; `wave()` captures `fired_by_layer` during the layer loop.
- Consumes: `elig_params`, `fired_by_layer`, `fired_bitset`, `pretr_active`, `dirty_rows`, `Layer.train`, `Layer.occ/offsets/topology`.

- [ ] **Step 1: Write a failing test** in `network.rs` `tests`. A hand-built 2-layer net where L0 neuron 0 is forced to fire feeding L1: after driving, L0→L1 eligibility rows for L0's live sources are nonzero, and it is deterministic:

```rust
#[test]
fn accrual_marks_eligibility_and_is_deterministic() {
    // Two builds, identical drive, must match; and some L0-row eligibility must be nonzero after L1 fires.
    let cfg = {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 3, adapt_bump: 0, adapt_decay: 6 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Config { seed: 21, size: 8, layers: vec![up, top] }
    };
    let mut a = Network::new(cfg.clone());
    let mut b = Network::new(cfg);
    a.enable_training();
    b.enable_training();
    for _ in 0..12 {
        a.wave(&[0, 1, 2, 8, 9, 10]);
        b.wave(&[0, 1, 2, 8, 9, 10]);
    }
    a.with_layer(0, |la| {
        b.with_layer(0, |lb| {
            assert_eq!(la.train.as_ref().unwrap().elig, lb.train.as_ref().unwrap().elig, "deterministic elig");
        })
    });
    let any = a.with_layer(0, |l| l.train.as_ref().unwrap().elig.iter().any(|&e| e > 0.0));
    assert!(any, "some L0->L1 eligibility accrued once L1 neurons fire");
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::network::tests::accrual_marks_eligibility_and_is_deterministic`
Expected: FAIL (elig all zero — no accrual yet).

- [ ] **Step 3: Capture fired in `wave()` and add `accrue_eligibility`.** In `wave()`'s `Mode::Sparse` arm, compute `let training = self.is_training();` **before** the destructure, add `fired_by_layer` to the destructured `Self { .. }`, and after each `process_layer` capture:

```rust
        if training {
            fired_by_layer[z].clear();
            fired_by_layer[z].extend_from_slice(fired);
        }
```

After the `Mode::Sparse` arm's frontier swap/clear (i.e. after the whole `match`), add:

```rust
        if self.is_training() {
            self.accrue_eligibility();
        }
```

Add imports at top if not present: `use crate::wave_driven::synapse::{local_of, wrap, xy_of};` and `use crate::wave_driven::neurons::Layer;` (Layer likely already imported). Add the method inside `impl Network`:

```rust
/// Accrue membrane spike-ψ eligibility for this wave (source-driven scan). Called after the wave's
/// layer step, when training. `e_ij += pretr_i` for every synapse whose target fired this wave.
fn accrue_eligibility(&mut self) {
    let size = self.size;
    let l = self.layers.len();
    let decay = 1.0 - 1.0 / self.elig_params.rec_tau.max(1.0);
    let eps = self.elig_params.epsilon;
    let Self { layers, fired_by_layer, fired_bitset, pretr_active, dirty_rows, .. } = self;

    // 1. fired bitset + spike_count
    for z in 0..l {
        for &j in &fired_by_layer[z] {
            fired_bitset[z][(j >> 6) as usize] |= 1u64 << (j & 63);
        }
        if let Some(t) = layers[z].train.as_mut() {
            for &j in &fired_by_layer[z] {
                t.spike_count[j as usize] += 1;
            }
        }
    }

    // 2. pretr update: decay -> eps-drop -> bump firers (canonical order; matches the dense oracle)
    for z in 0..l {
        let Some(t) = layers[z].train.as_mut() else { continue };
        let pretr = &mut t.pretr;
        let old: Vec<u32> = std::mem::take(&mut pretr_active[z].list);
        for &i in &old {
            pretr_active[z].mark[(i >> 6) as usize] &= !(1u64 << (i & 63));
        }
        for &i in &old {
            let iu = i as usize;
            pretr[iu] *= decay;
            if pretr[iu] < eps {
                pretr[iu] = 0.0;
            } else {
                pretr_active[z].push(i);
            }
        }
        for &j in &fired_by_layer[z] {
            pretr[j as usize] += 1.0;
            pretr_active[z].push(j);
        }
    }

    // 3. accrue: for each source with a live trace, scan its fan-out, add pretr where the target fired
    for z in 0..l {
        if layers[z].train.is_none() {
            continue;
        }
        let ts = layers[z].total_slots;
        let Layer { topology, slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
        let tr = train.as_mut().unwrap();
        for &iu in &pretr_active[z].list {
            let i = iu as usize;
            let pr = tr.pretr[i];
            if pr == 0.0 {
                continue;
            }
            let (sx, sy) = xy_of(iu, size);
            for (e_idx, entry) in topology.iter().enumerate() {
                let tz_i = z as i32 + entry.level;
                if tz_i < 0 || tz_i as usize >= l {
                    continue;
                }
                let tz = tz_i as usize;
                let sbase = slot_bases[e_idx];
                let wpn = occ_wpn[e_idx];
                let words = &occ[e_idx][i * wpn..i * wpn + wpn];
                let fb = &fired_bitset[tz];
                let mut rank = 0usize;
                for (wi, &w0) in words.iter().enumerate() {
                    let mut word = w0;
                    let cbase = wi * 64;
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        let cell = cbase + bit;
                        let (dx, dy) = offsets[e_idx][cell];
                        let j = local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size);
                        if fb[(j >> 6) as usize] & (1u64 << (j & 63)) != 0 {
                            tr.elig[i * ts + sbase + rank] += pr;
                            dirty_rows[z].push(iu);
                        }
                        rank += 1;
                        word &= word - 1;
                    }
                }
            }
        }
    }

    // 4. clear this wave's fired bitset for reuse next wave
    for z in 0..l {
        for &j in &fired_by_layer[z] {
            fired_bitset[z][(j >> 6) as usize] &= !(1u64 << (j & 63));
        }
    }
}
```

> Borrow note: in step 3, destructuring `&mut layers[z]` into `{ topology, .., occ, offsets, train, .. }` gives disjoint field references, so the shared borrow of `occ` (via `words`) coexists with the mutable borrow of `train` (via `tr`). `fired_bitset[tz]` and `dirty_rows[z]` are disjoint `Self` fields. Decode is inlined (offset LUT + wrap) because `train` is destructured out, so the `Layer::decode` method (needs `&self`) isn't callable here.

- [ ] **Step 4: Run the test.**

Run: `cargo test wave_driven::network::tests::accrual_marks_eligibility_and_is_deterministic`
Expected: PASS. Then full `cargo test wave_driven::` PASS, `cargo build` warning-free.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/network.rs
git commit -m "feat(wave_driven): online spike-ψ eligibility accrual on the frontier"
```

---

### Task 4: Dense-eligibility oracle + bit-exact `online ≡ dense`

**Files:**
- Modify: `src/wave_driven/training.rs` (add `dense_eligibility`)
- Create: test in `training.rs` (or a `#[cfg(test)]` block) exercising the oracle against a live run.

**Interfaces:**
- Produces: `training::dense_eligibility(net: &Network, entries: &[Vec<Edge>], fired: &[Vec<Vec<u32>>], p: &EligParams) -> Vec<Vec<f32>>` — per-layer `elig`-layout (`ls·total_slots`) eligibility computed offline from full fired records.
- Consumes: `Network::{size, layer_count, with_layer}`, `Layer::{for_wired, decode, total_slots, slot_bases}`.

- [ ] **Step 1: Write the failing oracle test** in `training.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{random_l0_input, TopologyLevel};
    use std::sync::{Arc, Mutex};

    fn deep_cfg(size: u32) -> (Config, Vec<Vec<Edge>>) {
        let mk = |topology: Vec<TopologyLevel>| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 4, adapt_bump: 5, adapt_decay: 6 };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: 0, radius: 1, count: 3 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![]),
        ];
        let entries = vec![
            vec![Edge { level: 1, count: 8, radius: 2 }],
            vec![Edge { level: 1, count: 8, radius: 2 }, Edge { level: 0, count: 3, radius: 1 }],
            vec![Edge { level: 1, count: 8, radius: 2 }],
            vec![],
        ];
        (Config { seed: 0x0E11, size, layers }, entries)
    }

    #[test]
    fn online_equals_dense_eligibility_bit_exact() {
        let size = 16u32;
        let (cfg, entries) = deep_cfg(size);
        let mut net = Network::new(cfg);
        net.enable_training();
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });

        // record fired per layer per wave via listeners, in lockstep with the online accrual
        let l = net.layer_count();
        let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
        for z in 0..l {
            let r = rec.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| r.lock().unwrap()[z].push(f.to_vec())));
        }
        net.reset_state();
        let input = random_l0_input(0x0E11, size, 15000);
        for w in 0..120 {
            net.wave(&input(w));
        }
        net.clear_listeners();
        let fired = rec.lock().unwrap().clone();

        let dense = dense_eligibility(&net, &entries, &fired, &EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });
        for z in 0..l {
            net.with_layer(z, |lz| {
                assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "layer {z} online == dense elig (bit-exact)");
            });
        }
    }
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::training`
Expected: FAIL (`dense_eligibility` not defined).

- [ ] **Step 3: Implement `dense_eligibility`** in `training.rs` (uses the **canonical decay → ε-drop → bump** order, so it matches the online accrual exactly):

```rust
use crate::wave_driven::network::Network;

/// Offline reference eligibility from full fired records: `e_ij = Σ_t pretr_i(t)·[j fires at t]`,
/// `pretr` maintained with the canonical decay → ε-drop → bump order. Returns per-layer `elig`-layout
/// vectors (`ls·total_slots`) for direct comparison to the engine's online `elig`.
pub fn dense_eligibility(net: &Network, entries: &[Vec<Edge>], fired: &[Vec<Vec<u32>>], p: &EligParams) -> Vec<Vec<f32>> {
    let size = net.size();
    let ls = (size as usize) * (size as usize);
    let l = net.layer_count();
    let ttot = fired.iter().map(|f| f.len()).max().unwrap_or(0);
    let decay = 1.0 - 1.0 / p.rec_tau.max(1.0);
    let eps = p.epsilon;
    let mut out: Vec<Vec<f32>> = (0..l).map(|z| net.with_layer(z, |lz| vec![0f32; ls * lz.total_slots])).collect();
    let mut pretr = vec![vec![0f32; ls]; l];
    for t in 0..ttot {
        // pretr: decay -> eps-drop -> bump firers (identical order to Network::accrue_eligibility)
        for z in 0..l {
            for i in 0..ls {
                pretr[z][i] *= decay;
                if pretr[z][i] < eps {
                    pretr[z][i] = 0.0;
                }
            }
            if t < fired[z].len() {
                for &i in &fired[z][t] {
                    pretr[z][i as usize] += 1.0;
                }
            }
        }
        // fired bitset per layer at wave t
        let mut fb = vec![vec![0u64; (ls + 63) / 64]; l];
        for z in 0..l {
            if t < fired[z].len() {
                for &j in &fired[z][t] {
                    fb[z][(j >> 6) as usize] |= 1u64 << (j & 63);
                }
            }
        }
        // accrue: source-driven, add pretr where the target fired
        for z in 0..l {
            net.with_layer(z, |lz| {
                let ts = lz.total_slots;
                for (e_idx, edge) in entries[z].iter().enumerate() {
                    let tz_i = z as i32 + edge.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let sbase = lz.slot_bases[e_idx];
                    for i in 0..ls {
                        let pr = pretr[z][i];
                        if pr == 0.0 {
                            continue;
                        }
                        lz.for_wired(e_idx, i, |r, c| {
                            let j = lz.decode(e_idx, i as u32, c, size);
                            if fb[tz][(j >> 6) as usize] & (1u64 << (j & 63)) != 0 {
                                out[z][i * ts + sbase + r] += pr;
                            }
                        });
                    }
                }
            });
        }
    }
    out
}
```

- [ ] **Step 4: Run the oracle test.**

Run: `cargo test wave_driven::training`
Expected: PASS (`online_equals_dense_eligibility_bit_exact`). If it fails on a mismatch, the cause is almost always a pretr-order divergence — verify both use decay → ε-drop → bump and the SAME `epsilon`. Fix the engine/oracle, never loosen the assert to a tolerance.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/training.rs
git commit -m "test(wave_driven): dense-eligibility oracle + bit-exact online==dense"
```

---

### Task 5: `dfa_update` — apply the eligibility to the shadow

**Files:**
- Modify: `src/wave_driven/network.rs`

**Interfaces:**
- Produces: `Network::dfa_update(&mut self, entries: &[Vec<Edge>], signal: &[Vec<f32>], lr: f32)` — `shadow[i,edge,r] += −lr·signal[tz][j]·elig[i,edge,r]` over dirty rows (`tz = z+level ∈ [1, L)`), then `repack_row` each touched row.
- Consumes: `dirty_rows`, `Layer.train.{shadow, elig}`, `repack_row`.

- [ ] **Step 1: Write a failing test** in `network.rs` `tests` — a negative learning signal on a pruned synapse raises its shadow (ported from `wave_bitnet`'s `update_with_negative_signal_raises_pruned_synapse`), driven through the real accrual:

```rust
#[test]
fn dfa_update_with_negative_signal_raises_eligible_synapse() {
    let cfg = {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 3, adapt_bump: 0, adapt_decay: 6 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Config { seed: 5, size: 8, layers: vec![up, top] }
    };
    let entries = vec![vec![Edge { level: 1, count: 8, radius: 2 }], vec![]];
    let mut net = Network::new(cfg);
    net.enable_training();
    // zero L0 row 0's shadow then repack -> fully pruned
    net.with_layer_mut_test(0, |l| {
        let ts = l.total_slots;
        for s in 0..ts { l.train.as_mut().unwrap().shadow[s] = 0.0; }
        l.repack_row(0);
    });
    net.reset_state();
    for _ in 0..12 { net.wave(&[0, 1, 2, 8, 9, 10]); }
    // negative signal everywhere on the target layer
    let ls = 64usize;
    let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
    let before: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow[0..l.total_slots].iter().sum());
    net.dfa_update(&entries, &signal, 0.05);
    let after: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow[0..l.total_slots].iter().sum());
    assert!(after > before, "negative target signal + positive eligibility raises L0 row-0 shadow: {before}->{after}");
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::network::tests::dfa_update_with_negative_signal_raises_eligible_synapse`
Expected: FAIL (`dfa_update` not found).

- [ ] **Step 3: Implement `dfa_update`** in `impl Network`:

```rust
/// Apply one multi-layer-DFA update from the accumulated eligibility: for each trainable edge
/// (`tz = z + level ∈ [1, L)`), `shadow[i,edge,r] += −lr·signal[tz][j]·elig[i,edge,r]` over the dirty
/// rows, then repack each touched row. Targets decoded from the occupancy (inlined).
pub fn dfa_update(&mut self, entries: &[Vec<Edge>], signal: &[Vec<f32>], lr: f32) {
    let size = self.size;
    let l = self.layers.len();
    let Self { layers, dirty_rows, .. } = self;
    for z in 0..l {
        if layers[z].train.is_none() {
            continue;
        }
        for ri in 0..dirty_rows[z].list.len() {
            let iu = dirty_rows[z].list[ri];
            let i = iu as usize;
            let mut touched = false;
            {
                let ts = layers[z].total_slots;
                let Layer { topology, slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
                let tr = train.as_mut().unwrap();
                let (sx, sy) = xy_of(iu, size);
                for (e_idx, entry) in topology.iter().enumerate() {
                    let tz_i = z as i32 + entry.level;
                    if tz_i < 1 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let sbase = slot_bases[e_idx];
                    let wpn = occ_wpn[e_idx];
                    let words = &occ[e_idx][i * wpn..i * wpn + wpn];
                    let mut rank = 0usize;
                    for (wi, &w0) in words.iter().enumerate() {
                        let mut word = w0;
                        let cbase = wi * 64;
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            let cell = cbase + bit;
                            let (dx, dy) = offsets[e_idx][cell];
                            let j = local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize;
                            let widx = i * ts + sbase + rank;
                            let e = tr.elig[widx];
                            if e != 0.0 {
                                tr.shadow[widx] += -lr * signal[tz][j] * e;
                                touched = true;
                            }
                            rank += 1;
                            word &= word - 1;
                        }
                    }
                }
            }
            if touched {
                layers[z].repack_row(i);
            }
        }
    }
}
```

> Note: `dirty_rows[z].list[ri]` is indexed (not iterated by reference) so the shared borrow of `dirty_rows` doesn't overlap the `&mut layers[z]` inside the block. The inner `{ }` scope ends the layer destructure before `repack_row` re-borrows `layers[z]`.

- [ ] **Step 4: Run the test + full module.**

Run: `cargo test wave_driven::`
Expected: PASS all. `cargo build` warning-free.

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/network.rs
git commit -m "feat(wave_driven): dfa_update — apply eligibility to the shadow over dirty rows"
```

---

### Task 6: Bench harness port + FF-trains-above-chance

**Files:**
- Create: `src/bench/wave_driven_bench.rs`
- Modify: `src/bench/mod.rs` (add `mod wave_driven_bench;` under `#[cfg(test)]` if that's the pattern, else `pub mod`)

**Interfaces:**
- Consumes: `wave_driven::{config, network, synapse, training}` public API only. Mirrors `bench::wave_bitnet_bench` shape.

- [ ] **Step 1: Inspect `src/bench/mod.rs`** to match the module-declaration style:

Run: `sed -n '1,20p' src/bench/mod.rs`
Expected: shows how `wave_bitnet_bench` is declared; replicate that for `wave_driven_bench`.

- [ ] **Step 2: Create `src/bench/wave_driven_bench.rs`** (ported harness; no `pots`/`effs` recording, rate from the engine, `dfa_update` instead of `multilayer_dfa_step`):

```rust
//! FF training harness for the `wave_driven` engine (test-only). Ports the `wave_bitnet_bench` shape
//! onto the event-driven engine's online eligibility: no per-wave pot/eff recording, per-neuron rate
//! read from the engine's `spike_count`, and `Network::dfa_update` applied from the accumulated
//! eligibility. Proves the activity-scaled trainer learns end-to-end (FF single-cue above chance).

#[cfg(test)]
mod tests {
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{key, mix, TopologyLevel};
    use crate::wave_driven::training::{Edge, EligParams};
    use std::sync::{Arc, Mutex};

    const CUE_P: u64 = 0xC0E;
    const P_DFA: u64 = 61;

    fn cue_sites(task_seed: u64, size: u32, class: usize) -> Vec<u32> {
        let ls = (size * size) as u32;
        (0..ls).filter(|&loc| mix(key(task_seed, loc, class as i32, 0, CUE_P)) & 3 == 0).collect()
    }

    fn softmax2(z0: f32, z1: f32) -> (f32, f32) {
        let m = z0.max(z1);
        let (e0, e1) = ((z0 - m).exp(), (z1 - m).exp());
        let s = (e0 + e1).max(1e-30);
        (e0 / s, e1 / s)
    }

    fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
        if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
    }

    /// Run one trial; returns (top-layer read-window spike counts `act`, total waves `ttot`).
    fn run_trial(net: &mut Network, size: u32, classes: &[usize], task_seed: u64, present: usize, delay: usize, read: usize) -> (Vec<f32>, usize) {
        let l = net.layer_count();
        let ls = (size * size) as usize;
        let top = l - 1;
        let top_spikes: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let ts = top_spikes.clone();
        net.on_layer(top, Box::new(move |_w, fired: &[u32]| ts.lock().unwrap().push(fired.to_vec())));
        net.reset_state();
        let mut ttot = 0usize;
        for (pos, &class) in classes.iter().enumerate() {
            if pos > 0 {
                for _ in 0..delay {
                    net.wave(&[]);
                    ttot += 1;
                }
            }
            for _ in 0..present {
                let sites = cue_sites(task_seed, size, class);
                net.wave(&sites);
                ttot += 1;
            }
        }
        let read_start = top_spikes.lock().unwrap().len();
        for _ in 0..read {
            net.wave(&[]);
            ttot += 1;
        }
        net.clear_listeners();
        let mut act = vec![0f32; ls];
        for wv in top_spikes.lock().unwrap().iter().skip(read_start) {
            for &loc in wv {
                act[loc as usize] += 1.0;
            }
        }
        (act, ttot)
    }

    struct TaskCfg {
        size: u32,
        present: usize,
        delay: usize,
        read: usize,
        holdout: usize,
        readout_lr: f32,
        hidden_lr: f32,
        rate_reg: f32,
        rate_target: f32,
    }

    /// Learning signal per computational layer/neuron: DFA feedback + rate_reg (rate from the engine).
    fn build_signal(net: &Network, w: &[Vec<f32>], err: &[f32], seed: u64, ttot: usize, cfg: &TaskCfg) -> Vec<Vec<f32>> {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let denom = ttot.max(1) as f32;
        let mut signal = vec![vec![0f32; ls]; l];
        for tz in 1..l {
            let sc = net.layer_spike_count(tz);
            for j in 0..ls {
                let task_sig: f32 = (0..2)
                    .map(|c| {
                        let b = if tz == top { w[c][j] } else { dfa_weight(seed, (tz * ls + j) as u32, c) };
                        b * err[c]
                    })
                    .sum();
                let rate = sc[j] as f32 / denom;
                signal[tz][j] = task_sig + cfg.rate_reg * (rate - cfg.rate_target);
            }
        }
        signal
    }

    fn train_and_eval_best(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize), eval_every: usize, patience: usize, max_trials: usize) -> (u64, usize) {
        const EVAL_OFFSET: usize = 10_000_000;
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let mut w = vec![vec![0f32; ls]; 2];
        let score = |w: &[Vec<f32>], a: &[f32]| -> (f32, f32) {
            (w[0].iter().zip(a).map(|(x, y)| x * y).sum(), w[1].iter().zip(a).map(|(x, y)| x * y).sum())
        };
        let (mut best, mut best_at, mut stale, mut t) = (0u64, 0usize, 0usize, 0usize);
        while t < max_trials {
            let stop = (t + eval_every).min(max_trials);
            while t < stop {
                let (classes, label) = task(task_seed, t);
                let (act, ttot) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                let (p0, p1) = softmax2(s0, s1);
                let err = [p0 - if label == 0 { 1.0 } else { 0.0 }, p1 - if label == 1 { 1.0 } else { 0.0 }];
                for c in 0..2 {
                    for j in 0..ls {
                        w[c][j] -= cfg.readout_lr * err[c] * act[j];
                    }
                }
                if cfg.hidden_lr != 0.0 {
                    let signal = build_signal(net, &w, &err, seed, ttot, cfg);
                    net.dfa_update(entries, &signal, cfg.hidden_lr);
                }
                t += 1;
            }
            let mut correct = 0usize;
            for i in 0..cfg.holdout {
                let (classes, label) = task(task_seed, EVAL_OFFSET + i);
                let (act, _) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                if ((s1 > s0) as usize) == label {
                    correct += 1;
                }
            }
            let acc = (correct as u64 * 1000) / cfg.holdout as u64;
            if acc > best {
                best = acc;
                best_at = t;
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }
        (best, best_at)
    }

    fn single_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let c = (mix(key(seed, t as u32, 0, 0, 71)) & 1) as usize;
        (vec![c], c)
    }

    fn make_ff(seed: u64, size: u32, layers: usize, up_count: u32, up_radius: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay,
        };
        let mut net = Network::new(Config { seed, size, layers: vec![lc; layers] });
        net.enable_training();
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: up_count as usize, radius: up_radius }] })
            .collect();
        (net, entries)
    }

    fn ff_cfg() -> TaskCfg {
        TaskCfg { size: 16, present: 6, delay: 4, read: 6, holdout: 200, readout_lr: 0.02, hidden_lr: 0.004, rate_reg: 5.0, rate_target: 0.1 }
    }

    #[test]
    fn wave_driven_ff_trains_above_chance() {
        // 4-layer FF, size 16, generous fan-in; single-cue 2-class must beat chance. Integration proof
        // that online eligibility + shadow update + repack + readout trains end-to-end.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 5, 6);
        let cfg = ff_cfg();
        let (best, _at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 3, 1000);
        assert!(best > 600, "wave_driven FF should train above chance: {best}");
    }
}
```

- [ ] **Step 3: Declare the module** in `src/bench/mod.rs` matching the existing style (e.g. `mod wave_driven_bench;` if `wave_bitnet_bench` is declared that way; keep it test-scoped as that file already is via its inner `#[cfg(test)]`).

- [ ] **Step 4: Run the acceptance test** (may take a few seconds):

Run: `cargo test wave_driven_ff_trains_above_chance -- --nocapture`
Expected: PASS (`best > 600`). If it sits at chance, first confirm the oracle (Task 4) still passes, then check `rate_reg`/`hidden_lr` and that `layer_spike_count`'s `ttot` denominator matches the recorded window — do NOT weaken the threshold.

- [ ] **Step 5: Commit.**

```bash
git add src/bench/wave_driven_bench.rs src/bench/mod.rs
git commit -m "test(wave_driven): FF trainer harness — trains above chance end-to-end"
```

---

### Task 7: Stretch experiments (`#[ignore]`) — depth-8 smoke + training throughput

**Files:**
- Modify: `src/bench/wave_driven_bench.rs`

**Interfaces:** none new (uses Task 6 helpers).

- [ ] **Step 1: Add a depth-8 FF smoke** (`#[ignore]`, `--release`) after the acceptance test, mirroring `wave_bitnet_ff_depth8_smoke`:

```rust
#[test]
#[ignore] // smoke: run manually in --release
fn wave_driven_ff_depth8_smoke() {
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    eprintln!("== wave_driven FF depth-8 pure ternary smoke (r4/c48, adapt=5, online elig) ==");
    let mut bests = Vec::new();
    for &s in &seeds {
        let (mut net, entries) = make_ff(s, 32, 8, 48, 4, 5, 6);
        let mut cfg = ff_cfg();
        cfg.size = 32;
        cfg.present = 8;
        cfg.read = 8;
        cfg.holdout = 300;
        let (best, at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, single_task, 300, 3, 3000);
        eprintln!("seed {s:#x}: best {best}@{at}");
        bests.push(best);
    }
    let worst = *bests.iter().min().unwrap();
    eprintln!("worst {worst} (target ~1000)");
    assert!(worst >= 900, "pure ternary FF depth-8 should hold ~1000 (worst {worst})");
}
```

- [ ] **Step 2: Add a training-throughput comparison** (`#[ignore]`): time `N` training trials (online accrual) vs the same trials with eligibility computed via the **offline `dense_eligibility`** oracle over recorded fired sets, at size 16 and 32, and print trials/s for each. This makes the activity-scaling win visible. Use `std::time::Instant`:

```rust
#[test]
#[ignore] // experiment: online vs offline-eligibility trainer throughput (run in --release)
fn wave_driven_training_throughput() {
    use crate::wave_driven::training::dense_eligibility;
    use std::time::Instant;
    for &size in &[16u32, 32u32] {
        let seed = 0xC0FFEEu64;
        let (mut net, entries) = make_ff(seed, size, 4, 32, 3, 5, 6);
        let cfg = ff_cfg();
        let trials = 200usize;

        // online: run trials, accrual happens inside wave(); dfa_update reads engine elig
        let mut w = vec![vec![0f32; (size * size) as usize]; 2];
        let t0 = Instant::now();
        for t in 0..trials {
            let (classes, _label) = single_task(seed, t);
            let (_act, ttot) = run_trial(&mut net, size, &classes, seed, cfg.present, cfg.delay, cfg.read);
            let err = [0.1f32, -0.1f32];
            let signal = build_signal(&net, &w, &err, seed, ttot, &cfg);
            net.dfa_update(&entries, &signal, cfg.hidden_lr);
            for c in 0..2 { for j in 0..(size * size) as usize { w[c][j] += 0.0; } }
        }
        let online = t0.elapsed().as_secs_f64();

        // offline: record fired every wave, compute dense_eligibility per trial (the size-bound path)
        let (mut net2, entries2) = make_ff(seed, size, 4, 32, 3, 5, 6);
        let t1 = Instant::now();
        for t in 0..trials {
            let (classes, _label) = single_task(seed, t);
            let l = net2.layer_count();
            let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
            for z in 0..l { let r = rec.clone(); net2.on_layer(z, Box::new(move |_w, f: &[u32]| r.lock().unwrap()[z].push(f.to_vec()))); }
            net2.reset_state();
            for _ in 0..(cfg.present + cfg.read) { let sites = cue_sites(seed, size, classes[0]); net2.wave(&sites); }
            net2.clear_listeners();
            let fired = rec.lock().unwrap().clone();
            let _e = dense_eligibility(&net2, &entries2, &fired, &EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });
        }
        let offline = t1.elapsed().as_secs_f64();
        eprintln!("size {size}: online {:.1} trials/s, offline-eligibility {:.1} trials/s ({:.1}x)", trials as f64 / online, trials as f64 / offline, offline / online);
    }
}
```

- [ ] **Step 3: Verify they compile and the smokes run.**

Run: `cargo build --tests && cargo test wave_driven_ff_depth8_smoke -- --ignored --nocapture --release` (optional; long) and `cargo test wave_driven_training_throughput -- --ignored --nocapture --release`
Expected: compile clean; smoke prints per-seed bests (worst ≥ 900); throughput prints an online-vs-offline ratio.

- [ ] **Step 4: Commit.**

```bash
git add src/bench/wave_driven_bench.rs
git commit -m "test(wave_driven): ignored depth-8 FF smoke + online-vs-offline training throughput"
```

---

### Task 8: Documentation

**Files:**
- Modify: `AGENTS.md`

**Interfaces:** docs only.

- [ ] **Step 1: Update the `wave_driven/` architecture-map block in `AGENTS.md`** to add the two new files and the training note. Replace the `wave_driven/` block's file list so it includes:

```
    training.rs          # online activity-scaled multi-layer-DFA: spike-ψ membrane eligibility accrued on the frontier + dense oracle
```

and note in `neurons.rs`'s line that it now carries the optional `TrainState` (shadow/elig/pretr/spike_count), and add to `bench/`:

```
    wave_driven_bench.rs # wave_driven FF training harness (online eligibility) — trains above chance
```

- [ ] **Step 2: Add one sentence** to the `wave_driven` paragraph in the "two modules" section noting Phase 2a is landed: online membrane e-prop (spike-ψ), FF training scales with activity, `εᵃ`/recurrence is Phase 2b. Reference `docs/superpowers/specs/2026-07-13-wave-driven-phase2-training-design.md`.

- [ ] **Step 3: Full build + test.**

Run: `cargo build && cargo test`
Expected: warning-free; all tests pass (both engines).

- [ ] **Step 4: Commit.**

```bash
git add AGENTS.md
git commit -m "docs(wave_driven): document Phase 2a online training in the architecture map"
```

---

## Self-Review

**Spec coverage:**
- `TrainState` (shadow/elig/pretr/spike_count) → Task 1. `enable/disable_training` toggle + `EligParams`/`Edge` + reset → Tasks 1–2. Online source-driven spike-ψ accrual on the frontier → Task 3. ε-thresholded `pretr` (canonical decay→drop→bump) → Tasks 3 & 4. `dense_eligibility` oracle + **bit-exact online≡dense** → Task 4. `dfa_update` over dirty rows + repack → Task 5. Bench port (`run_trial`/readout/`build_signal`/`train_and_eval_best`) + **FF-above-chance acceptance** → Task 6. `#[ignore]` depth-8 smoke + **training-throughput** win → Task 7. Determinism → Task 3 test. `layer_spike_count` for `rate_reg` → Tasks 2 & 6. Non-goals (no `εᵃ`/bump-ψ, no decide snapshots, no persistence, spiking top not readout) → honored (none built; FF uses `Network::new`, spiking top). **All spec sections map to a task.**

**Placeholder scan:** No "TBD/TODO". Every code step shows full code. The throughput experiment (Task 7) uses a synthetic fixed `err` — intentional (it measures machinery cost, not learning) and labelled as such.

**Type consistency:** `TrainState { shadow, elig, pretr, spike_count }` identical across Tasks 1–6. `EligParams { rec_tau, epsilon }` and `Edge { level, count, radius }` consistent (Tasks 2–6). `dense_eligibility(net, entries, fired, p) -> Vec<Vec<f32>>` matches its call site (Task 4). `dfa_update(entries: &[Vec<Edge>], signal: &[Vec<f32>], lr: f32)` matches the bench call (Task 6). `layer_spike_count(z) -> &[u32]`, `set_elig_params`, `reset_eligibility`, `is_training` consistent (Tasks 2, 6). The `#[cfg(test)]` helpers `with_layer_mut_test`/`seed_worksets_test` are used only in tests. Canonical pretr order (decay→ε-drop→bump) stated identically in the constraints, Task 3, and Task 4 — the bit-exactness hinge.

**Known follow-ups (out of scope):** the ALIF `εᵃ` term + bump-ψ + recurrence re-validation (Phase 2b); shadow persistence; the reverse-adjacency (Approach B) optimization; GPU.
