# wave_driven Phase 2b (ALIF adaptation eligibility, εᵃ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (this repo mandates **inline** execution — never subagent-driven; see AGENTS.md). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the online, activity-scaled ALIF adaptation-eligibility term `εᵃ` (spike-ψ) to `wave_driven`'s trainer, and run the side-car FF-vs-recurrence experiment (width × rec_count, σ-instrumented) to test whether spike-ψ + `εᵃ` unlocks recurrence.

**Architecture:** A per-synapse `εᵃ` trace (`TrainState.eps_a`, allocated only when `β≠0`) recursed at the target layer's adaptation rate `ρ`; the eligibility gains the silent-source coupling `−β·εᵃ`. Accrual extends Phase 2a's source-driven scan over a larger `elig_active` set (sources fired since reset). `β=0` takes the unchanged Phase-2a path. Validated by a bit-exact online-vs-dense oracle extended with the `εᵃ` recursion.

**Tech Stack:** Rust edition 2024, std only, deterministic, single-threaded. Inline `#[cfg(test)]` tests; `#[ignore]`d experiments in `--release`.

## Global Constraints

- **Standard library only** in `src/`; **warning-free** `cargo build`; `cargo test` green. (AGENTS.md)
- **Determinism is a hard requirement** — pure function of `(seed, config, input, task-seed)`. (AGENTS.md)
- **No `unsafe`** in new code.
- **NEVER add a `Co-Authored-By` trailer.** Conventional-commit messages. **One commit per task.** **NEVER push.**
- **spike-ψ only** (`ψ_j = [j fires] ∈ {0,1}`); bump-ψ / decide snapshots are out of scope (Phase 2b-2).
- εᵃ rule (per synapse `i→j`, target `tz`, `ρ = 1 − 2^(−adapt_decay)` of layer `tz`): **on `j` fire** `e_ij += pretr_i − β·εᵃ_ij` and `εᵃ_ij := pretr_i + (ρ−β)·εᵃ_ij`; **on `j` silent** `εᵃ_ij := ρ·εᵃ_ij`; then `if |εᵃ_ij| < ε_a → 0`. **`β=0` ≡ Phase 2a exactly.**
- Branch: `feat/wave-driven-adaptation-elig` (already checked out).

---

### Task 1: `EligParams` fields + `TrainState.eps_a`

**Files:**
- Modify: `src/wave_driven/training.rs`
- Modify: `src/wave_driven/neurons.rs`
- Modify: `src/wave_driven/network.rs`

**Interfaces:**
- Produces: `EligParams { rec_tau, epsilon, elig_beta, epsilon_a }`; `TrainState.eps_a: Vec<f32>`; `Layer::enable_training(&mut self, alloc_eps_a: bool)`; `Network::enable_training` allocates `eps_a` iff the active `elig_beta ≠ 0`.

- [ ] **Step 1: Extend `EligParams`** in `training.rs`:

```rust
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub rec_tau: f32,
    pub epsilon: f32,
    pub elig_beta: f32,  // β: ALIF adaptation-eligibility coupling (0 = membrane-only = Phase 2a)
    pub epsilon_a: f32,  // εᵃ magnitude cutoff (bounds εᵃ + keeps the offline oracle exact)
}

impl Default for EligParams {
    fn default() -> Self {
        EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 }
    }
}
```

- [ ] **Step 2: Add `eps_a` to `TrainState`** and change `Layer::enable_training` in `neurons.rs`. In `struct TrainState` add the field:

```rust
pub struct TrainState {
    pub shadow: Vec<f32>,      // ls * total_slots
    pub elig: Vec<f32>,        // ls * total_slots
    pub eps_a: Vec<f32>,       // ls * total_slots when β≠0, else empty (Phase 2a footprint)
    pub pretr: Vec<f32>,       // ls
    pub spike_count: Vec<u32>, // ls
}
```

Change `enable_training` to take `alloc_eps_a`:

```rust
pub fn enable_training(&mut self, alloc_eps_a: bool) {
    if self.train.is_some() {
        return;
    }
    let n = self.synapse_count();
    let ls = self.threshold.len();
    let mut shadow = vec![0f32; n];
    for s in 0..n {
        shadow[s] = self.weight_at(s) as f32;
    }
    let eps_a = if alloc_eps_a { vec![0f32; n] } else { Vec::new() };
    self.train = Some(TrainState { shadow, elig: vec![0f32; n], eps_a, pretr: vec![0f32; ls], spike_count: vec![0u32; ls] });
}
```

- [ ] **Step 3: Update the 3 `neurons.rs` test call sites** — `enable_training_builds_shadow_from_codes_and_zeros_the_rest`, `repack_row_roundtrips_shadow_to_ternary`, `disable_training_frees_state` — change `l.enable_training()` to `l.enable_training(false)`. In the first test, also assert eps_a is empty at `false`:

```rust
// in enable_training_builds_shadow_from_codes_and_zeros_the_rest, after existing asserts:
assert!(t.eps_a.is_empty(), "eps_a not allocated when alloc_eps_a=false");
```

- [ ] **Step 4: Update `Network::enable_training`** in `network.rs` to pass the flag:

```rust
pub fn enable_training(&mut self) {
    let alloc = self.elig_params.elig_beta != 0.0;
    for l in self.layers.iter_mut() {
        l.enable_training(alloc);
    }
}
```

- [ ] **Step 5: Run tests.**

Run: `cargo test wave_driven::neurons wave_driven::network`  *(run separately: `cargo test wave_driven::neurons` then `cargo test wave_driven::network`)*
Expected: PASS. `cargo build` warning-free.

- [ ] **Step 6: Commit.**

```bash
git add src/wave_driven/training.rs src/wave_driven/neurons.rs src/wave_driven/network.rs
git commit -m "feat(wave_driven): EligParams β/εᵃ knobs + TrainState.eps_a (β-gated alloc)"
```

---

### Task 2: `elig_active` set + online `εᵃ` accrual

**Files:**
- Modify: `src/wave_driven/network.rs`

**Interfaces:**
- Produces: `Network.elig_active: Vec<Frontier>`; `accrue_eligibility` branches to the `εᵃ` path when `elig_beta ≠ 0`; `reset_eligibility` zeroes `eps_a` and clears `elig_active`.
- Consumes: `EligParams.{elig_beta, epsilon_a}`, `TrainState.eps_a`, `Layer.adapt_decay`.

- [ ] **Step 1: Write failing tests** in `network.rs` `tests`:

```rust
#[test]
fn eps_a_accrual_changes_elig_and_is_deterministic() {
    // Same net/input trained at β=0 vs β=0.4 must produce DIFFERENT elig (εᵃ has an effect),
    // and two β=0.4 builds must match (determinism).
    let cfg = {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: 0, radius: 1, count: 3 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 3, adapt_bump: 5, adapt_decay: 6 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Config { seed: 21, size: 8, layers: vec![up, top] }
    };
    let run = |beta: f32| {
        let mut net = Network::new(cfg.clone());
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: beta, epsilon_a: 1.0 / 1024.0 });
        net.enable_training();
        for _ in 0..16 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        net.with_layer(0, |l| l.train.as_ref().unwrap().elig.clone())
    };
    let b0 = run(0.0);
    let b4a = run(0.4);
    let b4b = run(0.4);
    assert_eq!(b4a, b4b, "β=0.4 deterministic");
    assert_ne!(b0, b4a, "εᵃ (β=0.4) changes the eligibility vs β=0");
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::network::tests::eps_a_accrual_changes_elig_and_is_deterministic`
Expected: FAIL (`elig_active` not found / β path absent → `assert_ne` fails because β currently ignored).

- [ ] **Step 3: Add the `elig_active` field.** In `struct Network`, after `dirty_rows`:

```rust
    elig_active: Vec<Frontier>,   // per layer: sources fired since reset (εᵃ scan set, β≠0)
```

In `build` (the `Network { ... }` literal), after `dirty_rows: ...`:

```rust
            elig_active: (0..l).map(|_| Frontier::new(ls)).collect(),
```

- [ ] **Step 4: Extend `accrue_eligibility`.** Replace the current method body so step 2 also feeds `elig_active` (when `β≠0`) and step 3 branches. Full replacement:

```rust
fn accrue_eligibility(&mut self) {
    let size = self.size;
    let l = self.layers.len();
    let decay = 1.0 - 1.0 / self.elig_params.rec_tau.max(1.0);
    let eps = self.elig_params.epsilon;
    let beta = self.elig_params.elig_beta;
    let eps_a_cut = self.elig_params.epsilon_a;
    let use_ea = beta != 0.0;
    let rho: Vec<f32> = self.layers.iter().map(|lz| 1.0 - 2f32.powi(-(lz.adapt_decay as i32))).collect();
    let Self { layers, fired_by_layer, fired_bitset, pretr_active, dirty_rows, elig_active, .. } = self;

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

    // 2. pretr update (decay -> eps-drop -> bump firers); also feed elig_active when using εᵃ
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
    if use_ea {
        for z in 0..l {
            for &j in &fired_by_layer[z] {
                elig_active[z].push(j);
            }
        }
    }

    // 3. accrual
    if !use_ea {
        // Phase 2a: membrane, over pretr_active
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
    } else {
        // εᵃ: over elig_active, with the adaptation-eligibility recursion (spike-ψ)
        for z in 0..l {
            if layers[z].train.is_none() {
                continue;
            }
            let ts = layers[z].total_slots;
            let Layer { topology, slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
            let tr = train.as_mut().unwrap();
            for &iu in &elig_active[z].list {
                let i = iu as usize;
                let pr = tr.pretr[i]; // 0 if the presynaptic trace already decayed (silent-source coupling)
                let (sx, sy) = xy_of(iu, size);
                for (e_idx, entry) in topology.iter().enumerate() {
                    let tz_i = z as i32 + entry.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let r_tz = rho[tz];
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
                            let widx = i * ts + sbase + rank;
                            let ea = tr.eps_a[widx];
                            let fired = fb[(j >> 6) as usize] & (1u64 << (j & 63)) != 0;
                            let new_ea = if fired {
                                tr.elig[widx] += pr - beta * ea;
                                dirty_rows[z].push(iu);
                                pr + (r_tz - beta) * ea
                            } else {
                                r_tz * ea
                            };
                            tr.eps_a[widx] = if new_ea.abs() < eps_a_cut { 0.0 } else { new_ea };
                            rank += 1;
                            word &= word - 1;
                        }
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

- [ ] **Step 5: Update `reset_eligibility`** to zero `eps_a` and clear `elig_active`. In the per-layer body, after zeroing `elig` over `dirty_rows`, add eps_a zeroing over `elig_active`, and after the `pretr_active[z].clear()` add the `elig_active` clear. Replace the method body's loop with:

```rust
    pub fn reset_eligibility(&mut self) {
        let l = self.layers.len();
        let Self { layers, pretr_active, dirty_rows, elig_active, fired_by_layer, fired_bitset, .. } = self;
        for z in 0..l {
            let ts = layers[z].total_slots;
            if let Some(t) = layers[z].train.as_mut() {
                for &i in &dirty_rows[z].list {
                    let base = i as usize * ts;
                    for s in 0..ts {
                        t.elig[base + s] = 0.0;
                    }
                }
                if !t.eps_a.is_empty() {
                    for &i in &elig_active[z].list {
                        let base = i as usize * ts;
                        for s in 0..ts {
                            t.eps_a[base + s] = 0.0;
                        }
                    }
                }
                for &i in &pretr_active[z].list {
                    t.pretr[i as usize] = 0.0;
                }
                t.spike_count.iter_mut().for_each(|c| *c = 0);
            }
            dirty_rows[z].clear();
            pretr_active[z].clear();
            elig_active[z].clear();
            for &j in &fired_by_layer[z] {
                fired_bitset[z][(j >> 6) as usize] &= !(1u64 << (j & 63));
            }
            fired_by_layer[z].clear();
        }
    }
```

- [ ] **Step 6: Run tests.**

Run: `cargo test wave_driven::network`
Expected: PASS (incl. `eps_a_accrual_changes_elig_and_is_deterministic`, and the Phase-2a tests unchanged). `cargo build` warning-free.

- [ ] **Step 7: Commit.**

```bash
git add src/wave_driven/network.rs
git commit -m "feat(wave_driven): online εᵃ adaptation-eligibility accrual (spike-ψ) over elig_active"
```

---

### Task 3: dense-oracle `εᵃ` extension + bit-exact `online ≡ dense`

**Files:**
- Modify: `src/wave_driven/training.rs`

**Interfaces:**
- Produces: `dense_eligibility` handles `p.elig_beta ≠ 0` via the `εᵃ` recursion (target-layer `ρ`, `p.epsilon_a` cutoff).
- Consumes: `Network::{size, layer_count, with_layer}`, `Layer.{adapt_decay, total_slots, slot_bases, for_wired, decode}`.

- [ ] **Step 1: Write failing tests** in `training.rs` `tests` (bit-exact at β=0.4 AND β=0 regression):

```rust
#[test]
fn online_equals_dense_eligibility_with_eps_a_bit_exact() {
    let size = 16u32;
    let (cfg, entries) = deep_cfg(size);
    let params = EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 };
    let mut net = Network::new(cfg);
    net.set_elig_params(params);
    net.enable_training();

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

    let dense = dense_eligibility(&net, &entries, &fired, &params);
    for z in 0..l {
        net.with_layer(z, |lz| {
            assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "layer {z} online == dense elig with εᵃ (bit-exact)");
        });
    }
}

#[test]
fn beta_zero_dense_matches_membrane() {
    // β=0 dense_eligibility must equal the Phase-2a membrane result (the regression gate).
    let size = 16u32;
    let (cfg, entries) = deep_cfg(size);
    let membrane = EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 };
    let mut net = Network::new(cfg);
    net.set_elig_params(membrane);
    net.enable_training();
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
    let dense = dense_eligibility(&net, &entries, &fired, &membrane);
    for z in 0..l {
        net.with_layer(z, |lz| {
            assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "β=0 online==dense (membrane)");
        });
    }
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven::training`
Expected: `online_equals_dense_eligibility_with_eps_a_bit_exact` FAILs (oracle ignores β); `beta_zero_*` passes.

- [ ] **Step 3: Extend `dense_eligibility`** to maintain `εᵃ` when `β≠0`. Replace its body:

```rust
pub fn dense_eligibility(net: &Network, entries: &[Vec<Edge>], fired: &[Vec<Vec<u32>>], p: &EligParams) -> Vec<Vec<f32>> {
    let size = net.size();
    let ls = (size as usize) * (size as usize);
    let l = net.layer_count();
    let ttot = fired.iter().map(|f| f.len()).max().unwrap_or(0);
    let decay = 1.0 - 1.0 / p.rec_tau.max(1.0);
    let eps = p.epsilon;
    let beta = p.elig_beta;
    let eps_a_cut = p.epsilon_a;
    let use_ea = beta != 0.0;
    let rho: Vec<f32> = (0..l).map(|z| net.with_layer(z, |lz| 1.0 - 2f32.powi(-(lz.adapt_decay as i32)))).collect();
    let mut out: Vec<Vec<f32>> = (0..l).map(|z| net.with_layer(z, |lz| vec![0f32; ls * lz.total_slots])).collect();
    let mut epsa: Vec<Vec<f32>> = if use_ea { out.iter().map(|o| vec![0f32; o.len()]).collect() } else { Vec::new() };
    let mut pretr = vec![vec![0f32; ls]; l];
    for t in 0..ttot {
        // pretr: decay -> eps-drop -> bump firers
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
        // accrue
        for z in 0..l {
            let out_z = &mut out[z];
            let epsa_z = if use_ea { Some(&mut epsa[z]) } else { None };
            net.with_layer(z, |lz| {
                let ts = lz.total_slots;
                accrue_dense_layer(lz, z, l, size, entries, &pretr[z], &fb, &rho, beta, eps_a_cut, use_ea, ts, out_z, epsa_z);
            });
        }
    }
    out
}

// One layer's dense accrual for wave t (mirrors Network::accrue_eligibility exactly).
#[allow(clippy::too_many_arguments)]
fn accrue_dense_layer(
    lz: &crate::wave_driven::neurons::Layer,
    z: usize,
    l: usize,
    size: u32,
    entries: &[Vec<Edge>],
    pretr_z: &[f32],
    fb: &[Vec<u64>],
    rho: &[f32],
    beta: f32,
    eps_a_cut: f32,
    use_ea: bool,
    ts: usize,
    out_z: &mut [f32],
    mut epsa_z: Option<&mut Vec<f32>>,
) {
    let ls = (size as usize) * (size as usize);
    for (e_idx, edge) in entries[z].iter().enumerate() {
        let tz_i = z as i32 + edge.level;
        if tz_i < 0 || tz_i as usize >= l {
            continue;
        }
        let tz = tz_i as usize;
        let r_tz = rho[tz];
        let sbase = lz.slot_bases[e_idx];
        for i in 0..ls {
            let pr = pretr_z[i];
            if !use_ea && pr == 0.0 {
                continue;
            }
            lz.for_wired(e_idx, i, |r, c| {
                let j = lz.decode(e_idx, i as u32, c, size);
                let fired = fb[tz][(j >> 6) as usize] & (1u64 << (j & 63)) != 0;
                let widx = i * ts + sbase + r;
                if use_ea {
                    let epsa_z = epsa_z.as_deref_mut().unwrap();
                    let ea = epsa_z[widx];
                    let new_ea = if fired {
                        out_z[widx] += pr - beta * ea;
                        pr + (r_tz - beta) * ea
                    } else {
                        r_tz * ea
                    };
                    epsa_z[widx] = if new_ea.abs() < eps_a_cut { 0.0 } else { new_ea };
                } else if fired {
                    out_z[widx] += pr;
                }
            });
        }
    }
}
```

> Note: `accrue_dense_layer` is a free function (not a closure) so the `epsa_z` mutable borrow is threaded cleanly through `for_wired`. It reproduces the online recursion line-for-line; the shared `use_ea`/`ρ`/`ε_a` guarantee bit-exactness.

- [ ] **Step 4: Run tests.**

Run: `cargo test wave_driven::training`
Expected: PASS (`online_equals_dense_eligibility_with_eps_a_bit_exact`, `beta_zero_dense_matches_membrane`, and the existing `online_equals_dense_eligibility_bit_exact`). If the εᵃ test mismatches, the cause is a recursion-order or `ρ`-layer divergence — compare `accrue_dense_layer` against `Network::accrue_eligibility`'s εᵃ branch line-for-line (same `ρ = rho[tz]`, same cutoff, same `(ρ−β)` on fire).

- [ ] **Step 5: Commit.**

```bash
git add src/wave_driven/training.rs
git commit -m "test(wave_driven): dense oracle εᵃ extension + bit-exact online==dense (β>0 and β=0)"
```

---

### Task 4: Side-car builder + parity tasks + FF-vs-recurrence experiment

**Files:**
- Modify: `src/bench/wave_driven_bench.rs`

**Interfaces:**
- Consumes: `wave_driven` public API + Task-1..3 machinery. Adds a side-car builder, an N-bit parity task (temporal XOR = N2, parity N=4), a per-layer rate/σ reporter, and an `#[ignore]` experiment sweeping recurrent width × `rec_count`.

- [ ] **Step 1: Add the side-car builder + parity task + rate reporter** to `wave_driven_bench.rs`'s `tests` module (after `make_ff`). `make_ff` already sets `EligParams`; the side-car sets `elig_beta = 0.4`, `rec_tau = 20.0`:

```rust
// Backward-fed side-car (ported from benches/throughput_bitnet.rs make_sidecar):
// L0→L1(+1); L1→L3(+2 skip); L2 self(0)+→L3(+1); L3→L2(−1)+→L4(+1); L4 read.
fn make_sidecar(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
    let mk = |topology| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump, adapt_decay };
    let layers = vec![
        mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]),
        mk(vec![TopologyLevel { level: 2, radius: ur, count: uc }]),
        mk(vec![TopologyLevel { level: 0, radius: r, count: n }, TopologyLevel { level: 1, radius: r, count: n }]),
        mk(vec![TopologyLevel { level: -1, radius: r, count: n }, TopologyLevel { level: 1, radius: ur, count: uc }]),
        mk(vec![]),
    ];
    let mut net = Network::new(Config { seed, size, layers });
    net.set_elig_params(EligParams { rec_tau: 20.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 });
    net.enable_training();
    let entries = vec![
        vec![Edge { level: 1, count: uc as usize, radius: ur }],
        vec![Edge { level: 2, count: uc as usize, radius: ur }],
        vec![Edge { level: 0, count: n as usize, radius: r }, Edge { level: 1, count: n as usize, radius: r }],
        vec![Edge { level: -1, count: n as usize, radius: r }, Edge { level: 1, count: uc as usize, radius: ur }],
        vec![],
    ];
    (net, entries)
}

/// N-bit sequential parity: N deterministic cue bits, label = their XOR. (N=2 is temporal XOR.)
fn task_parity(seed: u64, t: usize, n: usize) -> (Vec<usize>, usize) {
    let bits: Vec<usize> = (0..n).map(|i| (mix(key(seed, t as u32, 0, i as u32, 51)) & 1) as usize).collect();
    let label = bits.iter().fold(0usize, |a, &b| a ^ b);
    (bits, label)
}

/// Per-layer firing rate (%/neuron/wave) over a counted window, and a coarse σ (mean consecutive-layer
/// spike ratio) — the dynamics diagnostic that separates σ-supercritical collapse from credit collapse.
fn rate_profile(net: &mut Network, size: u32, task_seed: u64, class: usize, warmup: usize, waves: usize) -> (Vec<f64>, f64) {
    let l = net.layer_count();
    let counts = Arc::new(Mutex::new(vec![0u64; l]));
    for z in 0..l {
        let c = counts.clone();
        net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] += f.len() as u64));
    }
    net.reset_state();
    let sites = cue_sites(task_seed, size, class);
    for _ in 0..warmup {
        net.wave(&sites);
    }
    counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
    for _ in 0..waves {
        net.wave(&sites);
    }
    net.clear_listeners();
    let counts = std::mem::take(&mut *counts.lock().unwrap());
    let denom = ((size as u64) * (size as u64) * waves as u64) as f64;
    let pct: Vec<f64> = counts.iter().map(|&s| (s as f64 / denom * 1000.0).round() / 10.0).collect();
    // coarse σ: geometric-ish mean of spikes[z+1]/spikes[z] over computational layers
    let mut ratios = Vec::new();
    for z in 1..l - 1 {
        if counts[z] > 0 {
            ratios.push(counts[z + 1] as f64 / counts[z] as f64);
        }
    }
    let sigma = if ratios.is_empty() { 0.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
    (pct, sigma)
}
```

- [ ] **Step 2: Add the FF-vs-side-car experiment** (`#[ignore]`, `--release`) sweeping width × `rec_count`, with σ/profile instrumentation. It trains an FF baseline and side-cars at each `(size, rec_count)`, on parity N=2 (temporal XOR) and N=4:

```rust
#[test]
#[ignore] // experiment: does spike-ψ εᵃ unlock recurrence? (run in --release; minutes)
fn wave_driven_sidecar_vs_ff() {
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678];
    eprintln!("== wave_driven side-car vs FF (spike-ψ εᵃ, β=0.4) — width × rec_count, σ-instrumented ==");
    for &parity_n in &[2usize, 4usize] {
        eprintln!("-- parity N={parity_n} ({}) --", if parity_n == 2 { "temporal XOR" } else { "parity-4" });
        let task = move |s: u64, t: usize| task_parity(s, t, parity_n);
        for &size in &[16u32, 32u32] {
            // FF baseline (β=0, membrane) at this width
            let mut ff_bests = Vec::new();
            for &s in &seeds {
                let (mut net, entries) = make_ff(s, size, 5, 32, 3, 5, 6);
                let mut cfg = ff_cfg();
                cfg.size = size;
                cfg.present = 6;
                cfg.delay = 8;
                cfg.read = 8;
                cfg.holdout = 200;
                let (best, _at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, &task, 300, 3, 2400);
                ff_bests.push(best);
            }
            eprintln!("  size {size} FF          : worst {} mean {}", ff_bests.iter().min().unwrap(), ff_bests.iter().sum::<u64>() / ff_bests.len() as u64);

            // side-car at a sweep of rec_count (into/beyond the historical bump-ψ cliff ~12)
            for &rec_count in &[8u32, 16u32, 24u32] {
                let mut sc_bests = Vec::new();
                let mut sigmas = Vec::new();
                for &s in &seeds {
                    let (mut net, entries) = make_sidecar(s, size, 32, 3, rec_count, 4, 5, 6);
                    let mut cfg = ff_cfg();
                    cfg.size = size;
                    cfg.present = 6;
                    cfg.delay = 8;
                    cfg.read = 8;
                    cfg.holdout = 200;
                    let (best, _at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, &task, 300, 3, 2400);
                    sc_bests.push(best);
                    let (_pct, sigma) = rate_profile(&mut net, size, s, 0, 16, 64);
                    sigmas.push(sigma);
                }
                let profile = {
                    let (mut net, _e) = make_sidecar(seeds[0], size, 32, 3, rec_count, 4, 5, 6);
                    rate_profile(&mut net, size, seeds[0], 0, 16, 64).0
                };
                eprintln!(
                    "  size {size} side rec{rec_count:>2}: worst {} mean {} | σ≈{:.2} | rate% {:?}",
                    sc_bests.iter().min().unwrap(),
                    sc_bests.iter().sum::<u64>() / sc_bests.len() as u64,
                    sigmas.iter().sum::<f64>() / sigmas.len() as f64,
                    profile
                );
            }
        }
    }
    // No hard assertion: this is the research readout. Interpret per the spec's convergence ladder
    // (σ super-critical ⇒ dynamics collapse, density too high; healthy σ + poor acc ⇒ credit-limited).
}
```

> `train_and_eval_best` and `run_trial` already accept a multi-cue `classes` sequence and a `task: impl Fn(u64,usize)->(Vec<usize>,usize)`, so the parity task drops in. The FF baseline uses `make_ff` (β=0); the side-car uses `make_sidecar` (β=0.4).

- [ ] **Step 3: Verify it compiles and runs a short slice.** Full compile, then run the experiment (it is long; allow it, or Ctrl-C after the first few lines to confirm output shape):

Run: `cargo build --tests 2>&1 | grep -iE "warning|error" || echo clean`
Then (optional, long): `cargo test --release wave_driven_sidecar_vs_ff -- --ignored --nocapture`
Expected: compiles clean; the experiment prints FF vs side-car worst/mean, σ, and the per-layer rate profile per `(N, size, rec_count)`.

- [ ] **Step 4: Commit.**

```bash
git add src/bench/wave_driven_bench.rs
git commit -m "test(wave_driven): side-car FF-vs-recurrence experiment (spike-ψ εᵃ; width × rec_count, σ-instrumented)"
```

---

### Task 5: Documentation

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the `wave_driven` paragraph and the `training.rs` map line** in `AGENTS.md` to note Phase 2b: the `εᵃ` term (spike-ψ) is online, `β=0` recovers Phase 2a, and the side-car experiment tests recurrence-beats-FF (width × rec_count, σ-instrumented). Reference `docs/superpowers/specs/2026-07-13-wave-driven-phase2b-adaptation-eligibility-design.md`. Keep it to ~2 sentences added to the existing Phase-2a text, and extend the `training.rs` architecture-map line to mention the `εᵃ` recursion + `eps_a`.

- [ ] **Step 2: Full build + test.**

Run: `cargo build && cargo test`
Expected: warning-free; all tests pass (both engines).

- [ ] **Step 3: Commit.**

```bash
git add AGENTS.md
git commit -m "docs(wave_driven): document Phase 2b εᵃ adaptation eligibility"
```

---

## Self-Review

**Spec coverage:**
- `εᵃ` rule (spike-ψ, target-layer ρ, ε_a cutoff, silent-source coupling) → Task 2 (accrual) + Task 3 (oracle). `TrainState.eps_a` β-gated alloc → Task 1. `EligParams{elig_beta, epsilon_a}` → Task 1. `elig_active` (fired-since-reset) work-set → Task 2. `β=0` ≡ Phase 2a (regression) → Task 2 test (`assert_ne` for β≠0) + Task 3 `beta_zero_dense_matches_membrane`. Bit-exact online≡dense with εᵃ → Task 3. Determinism → Task 2 test. Side-car experiment with **width × rec_count** and **σ + spiking-profile** instrumentation → Task 4. Convergence ladder (density above the old cliff; σ distinguishes dynamics vs credit collapse) → Task 4 experiment + its interpretation note. Non-goals (bump-ψ/decide snapshots, streaming horizon, lazy εᵃ) → none built. **All spec sections map to a task.**

**Placeholder scan:** No "TBD/TODO". Every code step shows full code. Task 4's experiment has no hard assertion by design (research readout) — stated explicitly, not a placeholder.

**Type consistency:** `EligParams { rec_tau, epsilon, elig_beta, epsilon_a }` consistent (Tasks 1–4). `TrainState.eps_a` used identically in accrual (Task 2), reset (Task 2), and alloc (Task 1). `Layer::enable_training(alloc_eps_a: bool)` — the signature change is applied at all call sites (Task 1: 3 neurons tests + `Network::enable_training`). `dense_eligibility(net, entries, fired, p)` signature unchanged; behavior branches on `p.elig_beta` (Task 3). `accrue_dense_layer(..)` free-fn signature matches its single call site (Task 3). `make_sidecar` / `task_parity` / `rate_profile` names consistent (Task 4). The εᵃ recursion is written **identically** in `Network::accrue_eligibility` (Task 2) and `accrue_dense_layer` (Task 3): `on fire out += pr − β·ea, ea := pr + (ρ−β)·ea; else ea := ρ·ea; cutoff ε_a` — the bit-exactness hinge.

**Known follow-ups (out of scope):** bump-ψ + decide snapshots (Phase 2b-2, if the ladder needs it); flip-flop / distractor-XOR tasks (parity N=2/N=4 is the decisive subset); streaming `εᵃ` horizon expiry; lazy `εᵃ`.
