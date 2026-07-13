# Optional `TrainState` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (Per AGENTS.md this repo executes plans **inline and autonomously** — do not use the subagent-driven option.)

**Goal:** Move all engine training data into an `Option<TrainState>` that exists only while training is toggled on, so inference and profiling pay zero memory for it.

**Architecture:** `Layer` gains `train: Option<TrainState>` holding `shadow` + the two decide-time snapshots; `Network` gets `enable_training`/`disable_training`/`is_training`. Fresh and loaded nets are lean by default; `shadow` is reconstructed as `decode(codes)` on enable (identical to a `.wbm` round-trip). Dead `elig_pre`/`elig_post` are deleted and the `.wbr` overlay is slimmed to the genuinely-resumable arrays.

**Tech Stack:** Rust edition 2024, standard library only, inline `#[cfg(test)]` tests.

## Global Constraints

- **Standard library only** in `src/`; **warning-free build** (`cargo build`).
- **No `unsafe`** outside the one documented `process_layer` forward loop; do not add any.
- **Determinism** is a hard requirement — results are a pure function of `(seed, config, input)`.
- **Conventional-commit** messages; **one commit per task**. **NEVER** add a `Co-Authored-By` trailer. **NEVER** push.
- Tests are inline `#[cfg(test)]` per module; keep `cargo test` green at every commit.
- `ls == size*size == threshold.len()`; a synapse row is `total_slots` wide; weight index `i*total_slots + slot_base + rank`.

---

### Task 1: Slim the `.wbr` runtime overlay + split the format version

Remove the four training/scratch vectors from the runtime overlay (it should carry only genuinely-resumable forward state) and split the shared `VERSION` so the unchanged `.wbm` model stays valid while the `.wbr` overlay bumps to v2. `elig_pre`/`elig_post`/`decide_potential`/`decide_eff` still exist as `Layer` fields after this task — we simply stop persisting them.

**Files:**
- Modify: `src/wave_bitnet/persist.rs`

**Interfaces:**
- Produces: `MODEL_VERSION: u16 = 1`, `RUNTIME_VERSION: u16 = 2` (replacing `VERSION`). New `.wbr` layout per layer: `potential(i16), cooldown(u8), adapt(i32), pending(i32)`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/wave_bitnet/persist.rs`:

```rust
    #[test]
    fn runtime_version_is_bumped_and_distinct_from_model() {
        assert_eq!(MODEL_VERSION, 1, "model format unchanged");
        assert_eq!(RUNTIME_VERSION, 2, "runtime overlay bumped for the slimmer layout");
    }

    #[test]
    fn runtime_rejects_old_version() {
        // hand-build a runtime header with the OLD version byte (1); expect a loud reject.
        let net = small_net();
        let mut bad = MAGIC_RUNTIME.to_vec();
        w_u16(&mut bad, 1).unwrap(); // stale runtime version
        w_u64(&mut bad, net.model_fingerprint()).unwrap();
        w_u32(&mut bad, net.size()).unwrap();
        w_u32(&mut bad, net.layer_count() as u32).unwrap();
        w_u64(&mut bad, 0).unwrap();
        let ck = fnv1a(&bad);
        w_u64(&mut bad, ck).unwrap();
        let mut n = small_net();
        assert_eq!(err_kind_rt(n.apply_runtime(&bad[..])), io::ErrorKind::InvalidData);
    }
```

Add this helper next to `err_kind` in the same `tests` module (it adapts `err_kind` to the `io::Result<()>` that `apply_runtime` returns):

```rust
    fn err_kind_rt(r: io::Result<()>) -> io::ErrorKind {
        match r {
            Ok(()) => panic!("expected an error, got Ok"),
            Err(e) => e.kind(),
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib wave_bitnet::persist 2>&1 | tail -20`
Expected: FAIL — compile error, `MODEL_VERSION`/`RUNTIME_VERSION` not found.

- [ ] **Step 3: Split the version constants**

In `src/wave_bitnet/persist.rs`, replace line 15 (`const VERSION: u16 = 1;`) with:

```rust
const MODEL_VERSION: u16 = 1;
const RUNTIME_VERSION: u16 = 2;
```

In `save_model` change `w_u16(&mut buf, VERSION)?;` → `w_u16(&mut buf, MODEL_VERSION)?;`.
In `load_model` change the check to:

```rust
        let ver = r_u16(&mut c)?;
        if ver != MODEL_VERSION {
            return Err(inval(format!("unsupported model version {ver} (expected {MODEL_VERSION})")));
        }
```

In `save_runtime` change `w_u16(&mut buf, VERSION)?;` → `w_u16(&mut buf, RUNTIME_VERSION)?;`.
In `apply_runtime` change the check to:

```rust
        let ver = r_u16(&mut c)?;
        if ver != RUNTIME_VERSION {
            return Err(inval(format!("unsupported runtime version {ver} (expected {RUNTIME_VERSION})")));
        }
```

Update the `model_rejects_bad_magic_version_and_corruption` test: its two `w_u16(..., VERSION)` / `VERSION + 1` uses become `MODEL_VERSION` / `MODEL_VERSION + 1`.

- [ ] **Step 4: Slim `save_runtime`**

In `save_runtime`, replace the per-layer write loop (currently writing 8 vectors) with only the four resumable arrays:

```rust
        for lz in self.layers() {
            w_vec_i16(&mut buf, &lz.potential)?;
            w_vec_u8(&mut buf, &lz.cooldown)?;
            w_vec_i32(&mut buf, &lz.adapt)?;
            w_vec_i32(&mut buf, &lz.pending)?;
        }
```

- [ ] **Step 5: Slim `apply_runtime`**

Replace the `LayerRt` type alias, the staged read/validate loop, and the commit loop:

```rust
        // read + validate every layer's arrays BEFORE mutating (keep apply all-or-nothing)
        type LayerRt = (Vec<i16>, Vec<u8>, Vec<i32>, Vec<i32>);
        let mut staged: Vec<LayerRt> = Vec::with_capacity(n_layers);
        for z in 0..n_layers {
            let potential = r_vec_i16(&mut c)?;
            let cooldown = r_vec_u8(&mut c)?;
            let adapt = r_vec_i32(&mut c)?;
            let pending = r_vec_i32(&mut c)?;
            for (name, len) in [
                ("potential", potential.len()), ("cooldown", cooldown.len()),
                ("adapt", adapt.len()), ("pending", pending.len()),
            ] {
                if len != ls {
                    return Err(inval(format!("layer {z} {name} length {len} != ls {ls}")));
                }
            }
            staged.push((potential, cooldown, adapt, pending));
        }
        // commit
        let layers = self.layers_mut();
        for (z, (potential, cooldown, adapt, pending)) in staged.into_iter().enumerate() {
            layers[z].potential = potential;
            layers[z].cooldown = cooldown;
            layers[z].adapt = adapt;
            layers[z].pending = pending;
        }
        self.set_wave_id(wave_id);
        Ok(())
```

Update the module doc comment (top of file) if it enumerates the overlay contents — the overlay now carries "only `potential`, `cooldown`, `adapt`, `pending` (+ `wave_id`)".

- [ ] **Step 6: Trim the resume test to the persisted arrays**

In `runtime_resume_equivalent`, delete the four assertion lines for `elig_pre`, `elig_post`, `decide_potential`, `decide_eff` in the immediate-equality block (keep `potential`, `cooldown`, `adapt`, `pending`). The "continuing both must stay identical" block (comparing `layer_decide_potential`) is unchanged — both nets recompute decide state identically from equal forward state.

- [ ] **Step 7: Run the persist tests**

Run: `cargo test --lib wave_bitnet::persist 2>&1 | tail -20`
Expected: PASS (all persist tests green).

- [ ] **Step 8: Commit**

```bash
git add src/wave_bitnet/persist.rs
git commit -m "refactor(persist): slim .wbr overlay to resumable state; split model/runtime version"
```

---

### Task 2: Delete the dead `elig_pre` / `elig_post` state

Nothing reads these (the DFA path builds eligibility from `TrialRecords`; `eprop_update_synaptic` takes an external `elig` slice). Remove the fields, their hot-path write loops, and the now-unused `PSI_BAND`.

**Files:**
- Modify: `src/wave_bitnet/neurons.rs`, `src/wave_bitnet/wave.rs`, `src/wave_bitnet/network.rs`

**Interfaces:**
- Produces: `Layer` no longer has `elig_pre` / `elig_post`; `process_layer` no longer touches them.

- [ ] **Step 1: Confirm nothing reads them**

Run: `grep -rn "elig_pre\|elig_post\|PSI_BAND" src/`
Expected: only the definition/write/reset sites listed below — **no reads** feeding a computation. (Persist references were removed in Task 1.)

- [ ] **Step 2: Remove the fields from `Layer` and both constructors**

In `src/wave_bitnet/neurons.rs`:
- Delete the two struct fields (lines ~74–75):
  ```rust
      pub elig_pre: Vec<i32>,
      pub elig_post: Vec<i32>,
  ```
- In `from_parts` (the returned `Layer { … }`) delete the two initializers:
  ```rust
      elig_pre: vec![0i32; ls],
      elig_post: vec![0i32; ls],
  ```
- In `new` (the `Layer { … }` literal) delete the same two initializers.

- [ ] **Step 3: Remove the hot-path writes in `wave.rs`**

In `src/wave_bitnet/wave.rs`:
- Delete the two reslices:
  ```rust
      let elig_pre = &mut layer.elig_pre[..ls];
      let elig_post = &mut layer.elig_post[..ls];
  ```
- Delete the `PSI_BAND` const (`const PSI_BAND: i32 = 8;`).
- In the `(A0)` block, delete the `elig_post` bump so it becomes exactly:
  ```rust
      if record_elig {
          for i in 0..ls {
              let p = potential[i];
              let eff = threshold[i] as i32 + (adapt[i] >> ADAPT_SHIFT);
              decide_potential[i] = p; // snapshot pre fire-reset/leak
              decide_eff[i] = eff; // pre-bump effective threshold
          }
      }
  ```
- Delete the entire `(A2)` block:
  ```rust
      if record_elig {
          for &i in fired.iter() {
              elig_pre[i as usize] += 1;
          }
      }
  ```
  Update the pass-comment above `(A0)` (line ~53) to drop the `(A2)` mention.

- [ ] **Step 4: Remove from `reset_state`**

In `src/wave_bitnet/network.rs` `reset_state`, delete the two lines:

```rust
            g.elig_pre.iter_mut().for_each(|e| *e = 0);
            g.elig_post.iter_mut().for_each(|e| *e = 0);
```

- [ ] **Step 5: Build + test**

Run: `cargo build 2>&1 | tail -5 && cargo test --lib 2>&1 | tail -15`
Expected: warning-free build; all lib tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/wave_bitnet/neurons.rs src/wave_bitnet/wave.rs src/wave_bitnet/network.rs
git commit -m "refactor(wave_bitnet): delete dead elig_pre/elig_post eligibility state"
```

---

### Task 3: Introduce `TrainState` and relocate `shadow` + decide snapshots (behavior-identical)

Pure mechanical relocation: the five remaining training fields become a `TrainState` that every net still allocates (`train: Some(...)`), so behavior is byte-identical and all tests pass unchanged. The toggle and lean-default come in Task 4.

**Files:**
- Modify: `src/wave_bitnet/neurons.rs`, `src/wave_bitnet/wave.rs`, `src/wave_bitnet/network.rs`, `src/wave_bitnet/multilayer_dfa.rs`, `src/bench/wave_bitnet_bench.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct TrainState {
      pub shadow: Vec<f32>,           // ls * total_slots
      pub decide_potential: Vec<i16>, // ls
      pub decide_eff: Vec<i32>,       // ls
  }
  ```
  `Layer` field `pub train: Option<TrainState>`. New method `Layer::synapse_count(&self) -> usize`.

- [ ] **Step 1: Define `TrainState` and swap the `Layer` fields**

In `src/wave_bitnet/neurons.rs`, add above `pub struct Layer`:

```rust
/// All per-layer TRAINING state — allocated only while training is enabled (see
/// `Network::enable_training`). Absent (`Layer.train == None`) on an inference-lean net.
/// `shadow` is the f32 training master requantized into `codes` by `repack_row`; the two
/// decide-time snapshots are the credit-assignment records the bench reads each wave.
pub struct TrainState {
    pub shadow: Vec<f32>,           // ls * total_slots
    pub decide_potential: Vec<i16>, // ls
    pub decide_eff: Vec<i32>,       // ls
}
```

In `struct Layer`, remove the `shadow`, `decide_potential`, `decide_eff` fields and the two comment lines that head the weights/eligibility groups mentioning shadow; add at the end:

```rust
    // TRAINING state — present only while training is enabled (None on an inference-lean net).
    pub train: Option<TrainState>,
```

- [ ] **Step 2: Add `synapse_count` and route `repack_row` through `train`**

In `impl Layer`, add:

```rust
    /// Number of stored synapses (`ls * total_slots`) — independent of whether training is enabled.
    #[inline]
    pub fn synapse_count(&self) -> usize {
        self.total_slots * self.threshold.len()
    }
```

Rewrite `repack_row` to read the shadow from `train` (scoped immutable borrow, then `set_code`):

```rust
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

- [ ] **Step 3: Populate `train: Some(...)` in both constructors**

In `from_parts`, the shadow reconstruction stays, but wrap it into the returned `Layer`'s `train`. Replace the three initializers (`shadow`, and the `decide_potential`/`decide_eff` zeros) with:

```rust
            train: Some(TrainState {
                shadow,
                decide_potential: vec![0i16; ls],
                decide_eff: vec![0i32; ls],
            }),
```

In `new`, the `shadow` local is already built before the `Layer { … }` literal. Replace the `shadow`, `decide_potential`, `decide_eff` initializers with:

```rust
            train: Some(TrainState {
                shadow,
                decide_potential: vec![0i16; ls],
                decide_eff: vec![0i32; ls],
            }),
```

(The trailing `for i in 0..ls { layer.repack_row(i); }` still works — `train` is `Some`.)

- [ ] **Step 4: Route `process_layer` decide-snapshot through `train`**

In `src/wave_bitnet/wave.rs`, delete the two top-level reslices `let decide_potential = &mut layer.decide_potential[..ls];` and `let decide_eff = &mut layer.decide_eff[..ls];`. Replace the `(A0)` block with a train-gated version:

```rust
    // (A0) eligibility snapshot — records decide-time potential/eff into the training scratch, and only
    // when training is enabled (train == Some). Reads potential/adapt BEFORE the fire loop mutates them.
    if record_elig {
        if let Some(t) = layer.train.as_mut() {
            let decide_potential = &mut t.decide_potential[..ls];
            let decide_eff = &mut t.decide_eff[..ls];
            for i in 0..ls {
                let p = potential[i];
                let eff = threshold[i] as i32 + (adapt[i] >> ADAPT_SHIFT);
                decide_potential[i] = p;
                decide_eff[i] = eff;
            }
        }
    }
```

(`potential`, `adapt`, `threshold` are borrows of disjoint `Layer` fields, so borrowing `layer.train` alongside them is fine.)

- [ ] **Step 5: Route `eprop_update_synaptic` + `reset_state` + decide accessors through `train`**

In `src/wave_bitnet/network.rs`:

`eprop_update_synaptic` — scope the shadow borrow to the write loop:

```rust
                let mut touched = false;
                {
                    let shadow = &mut lz.train.as_mut().expect("eprop requires training enabled").shadow;
                    for &(r, target) in &wired {
                        let e = elig[i * count + r];
                        if e != 0.0 {
                            touched = true;
                            shadow[i * ts + sbase + r] += -lr * signal[target] * e;
                        }
                    }
                }
                if touched {
                    lz.repack_row(i);
                }
```

`reset_state` — replace the `decide_potential` zero line with a train-gated block (also zero `decide_eff` for consistency):

```rust
            if let Some(t) = g.train.as_mut() {
                t.decide_potential.iter_mut().for_each(|p| *p = 0);
                t.decide_eff.iter_mut().for_each(|e| *e = 0);
            }
```

`layer_decide_potential` / `layer_decide_effective_threshold`:

```rust
    pub fn layer_decide_potential(&self, z: usize) -> Vec<i16> {
        self.layers[z].train.as_ref().expect("layer_decide_potential requires training enabled").decide_potential.clone()
    }
    pub fn layer_decide_effective_threshold(&self, z: usize) -> Vec<i32> {
        self.layers[z].train.as_ref().expect("layer_decide_effective_threshold requires training enabled").decide_eff.clone()
    }
```

- [ ] **Step 6: Update the remaining `.shadow` readers (tests + bench)**

Mechanical: every direct `X.shadow` becomes `X.train.as_ref().unwrap().shadow` (or `as_mut()` for writes). Sites:

- `src/bench/wave_bitnet_bench.rs` `weight_sparsity`: replace `let n = lz.shadow.len();` with `let n = lz.synapse_count();`.
- `src/wave_bitnet/neurons.rs` tests `from_parts_reproduces_built_layer`: `built.shadow[s] = …` → `built.train.as_mut().unwrap().shadow[s] = …`; the `for s in 0..rebuilt.shadow.len()` / `rebuilt.shadow[s]` → `rebuilt.train.as_ref().unwrap().shadow`.
- `src/wave_bitnet/neurons.rs` test `repack_roundtrips_shadow_to_ternary`: the four `l.shadow[…] = …` → `l.train.as_mut().unwrap().shadow[…] = …`.
- `src/wave_bitnet/network.rs` test `wave_is_deterministic`: `la.shadow` / `lb.shadow` → `…train.as_ref().unwrap().shadow`.
- `src/wave_bitnet/network.rs` test `update_with_negative_signal_raises_pruned_synapse`: the `l.shadow[…] = 0.0` and the `l.shadow[0] > 0.0` assertion → through `train.as_mut()/as_ref().unwrap().shadow`.
- `src/wave_bitnet/multilayer_dfa.rs` test `step_raises_weights_on_negative_signal`: both `l.shadow.iter().sum()` → `l.train.as_ref().unwrap().shadow.iter().sum()`.
- `src/wave_bitnet/persist.rs` tests `assert_models_eq` (loaded shadow == decode) and `model_roundtrip_…` shadow loop → through `train.as_ref().unwrap().shadow` (loaded nets are still `Some` in this task).

- [ ] **Step 7: Build + test**

Run: `cargo build 2>&1 | tail -5 && cargo test --lib 2>&1 | tail -15`
Expected: warning-free build; all lib tests PASS (behavior identical — `train` is always `Some`).

- [ ] **Step 8: Commit**

```bash
git add src/wave_bitnet/ src/bench/wave_bitnet_bench.rs
git commit -m "refactor(wave_bitnet): relocate shadow + decide snapshots into Layer.train (TrainState)"
```

---

### Task 4: Add the enable/disable toggle and make lean the default

Flip both constructors to `train: None`, add the `Network` toggle, drive decide-recording purely off `train.is_some()` (removing `record_eligibility`), and update every training site to call `enable_training()`.

**Files:**
- Modify: `src/wave_bitnet/neurons.rs`, `src/wave_bitnet/wave.rs`, `src/wave_bitnet/network.rs`, `src/wave_bitnet/persist.rs`, `src/wave_bitnet/multilayer_dfa.rs`, `src/bench/wave_bitnet_bench.rs`, `benches/throughput_bitnet.rs`, `examples/profile_bitnet.rs`

**Interfaces:**
- Consumes: `TrainState`, `Layer.train`, `Layer::synapse_count` (Task 3).
- Produces: `Network::enable_training(&mut self)`, `disable_training(&mut self)`, `is_training(&self) -> bool`; `Layer::enable_training(&mut self)`, `disable_training(&mut self)`. `process_layer` drops its `record_elig: bool` parameter.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/wave_bitnet/network.rs`:

```rust
    #[test]
    fn fresh_net_is_lean_and_toggles() {
        let mut net = Network::new(two_layer(8));
        assert!(!net.is_training(), "fresh net is inference-lean");
        net.with_layer(0, |l| assert!(l.train.is_none()));
        net.enable_training();
        assert!(net.is_training());
        net.with_layer(1, |l| {
            let t = l.train.as_ref().expect("enabled");
            assert_eq!(t.shadow.len(), l.synapse_count());
        });
        net.disable_training();
        assert!(!net.is_training());
        net.with_layer(1, |l| assert!(l.train.is_none()));
    }

    #[test]
    fn enable_is_idempotent_and_preserves_shadow() {
        let mut net = Network::new(two_layer(8));
        net.enable_training();
        net.with_layer_mut(1, |l| l.train.as_mut().unwrap().shadow[0] = 3.5);
        net.enable_training(); // second call must not clobber
        net.with_layer(1, |l| assert_eq!(l.train.as_ref().unwrap().shadow[0], 3.5));
    }

    #[test]
    fn enable_reconstructs_decode_of_codes() {
        let mut net = Network::new(two_layer(8));
        net.enable_training();
        net.with_layer(1, |l| {
            let sh = &l.train.as_ref().unwrap().shadow;
            for s in 0..sh.len() {
                assert_eq!(sh[s], l.weight_at(s) as f32, "fresh enabled shadow == decode(codes)");
            }
        });
    }

    #[test]
    fn lean_and_trained_inference_match() {
        let mut lean = Network::new(two_layer(8));
        let mut trained = Network::new(two_layer(8));
        trained.enable_training();
        let inputs: [&[u32]; 5] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[]];
        for inp in inputs {
            lean.wave(inp);
            trained.wave(inp);
            for z in 0..lean.layer_count() {
                lean.with_layer(z, |a| trained.with_layer(z, |b| {
                    assert_eq!(a.potential, b.potential, "layer {z} potential matches");
                    assert_eq!(a.codes, b.codes, "layer {z} codes matches");
                }));
            }
        }
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib wave_bitnet::network 2>&1 | tail -20`
Expected: FAIL — `is_training`/`enable_training`/`disable_training` not found.

- [ ] **Step 3: Add the `Layer` toggle helpers**

In `src/wave_bitnet/neurons.rs` `impl Layer`, add:

```rust
    /// Allocate this layer's training state (idempotent — keeps any existing accumulated shadow).
    /// A fresh state reconstructs `shadow` as the per-slot decode of `codes` and zeroes the decide
    /// snapshots — identical to the shadow a `.wbm` load would rebuild (codes are the master).
    pub fn enable_training(&mut self) {
        if self.train.is_some() {
            return;
        }
        let n = self.synapse_count();
        let mut shadow = vec![0f32; n];
        for s in 0..n {
            shadow[s] = self.weight_at(s) as f32;
        }
        let ls = self.threshold.len();
        self.train = Some(TrainState { shadow, decide_potential: vec![0i16; ls], decide_eff: vec![0i32; ls] });
    }

    /// Free this layer's training state. LOSSY for in-flight sub-threshold shadow: re-enabling
    /// rebuilds `shadow = decode(codes)`, exactly like a `.wbm` save/load round-trip.
    pub fn disable_training(&mut self) {
        self.train = None;
    }
```

- [ ] **Step 4: Make both constructors lean**

In `src/wave_bitnet/neurons.rs`:
- `from_parts`: delete the `shadow` reconstruction block (the `let n = ls * total_slots; let mut shadow = …` loop) and set the field to `train: None`.
- `new`: the procedural `shadow` local is still needed to seed `codes` via `repack_row`. Keep building it, keep `train: Some(TrainState { shadow, … })` **temporarily** so `repack_row` works, then after the `for i in 0..ls { layer.repack_row(i); }` loop add `layer.train = None;` before `layer` is returned. (Construction briefly holds a shadow to seed codes, then drops it — the returned layer is lean.)

- [ ] **Step 5: Add the `Network` toggle + drop `record_eligibility`**

In `src/wave_bitnet/network.rs`:
- Remove the `record_eligibility: bool` field from `struct Network`, from the `Network { … }` literal in `build`, and from the one in `from_layers`.
- Delete `set_record_eligibility` and the `record_elig` locals in `wave` (`let record_elig = self.record_eligibility;`). The `process_layer` call becomes:
  ```rust
              process_layer(&mut layers[z], z as u32, size, inp, deliv, fired);
  ```
- Add the toggle methods:
  ```rust
      /// Allocate training state on every layer (idempotent). Required before any training update.
      pub fn enable_training(&mut self) {
          for l in self.layers.iter_mut() {
              l.enable_training();
          }
      }

      /// Free training state on every layer (lossy for sub-threshold shadow; see `Layer::disable_training`).
      pub fn disable_training(&mut self) {
          for l in self.layers.iter_mut() {
              l.disable_training();
          }
      }

      /// True if training state is currently allocated (checks layer 0).
      pub fn is_training(&self) -> bool {
          self.layers.first().map(|l| l.train.is_some()).unwrap_or(false)
      }
  ```

- [ ] **Step 6: Drop the `record_elig` parameter from `process_layer`**

In `src/wave_bitnet/wave.rs`:
- Change the signature: remove `record_elig: bool,` from `process_layer`.
- Replace the `(A0)` `if record_elig { if let Some(t) = layer.train.as_mut() { … } }` with just the inner gate:
  ```rust
      if let Some(t) = layer.train.as_mut() {
          let decide_potential = &mut t.decide_potential[..ls];
          let decide_eff = &mut t.decide_eff[..ls];
          for i in 0..ls {
              let p = potential[i];
              let eff = threshold[i] as i32 + (adapt[i] >> ADAPT_SHIFT);
              decide_potential[i] = p;
              decide_eff[i] = eff;
          }
      }
  ```
- In the module's test `firing_neuron_scatters_nonzero_weights_to_decoded_targets`, drop the trailing `true` arg: `process_layer(&mut l, 0, size, &[], &mut deliv, &mut fired);` (the test checks the forward delivery path, which needs no training state).

- [ ] **Step 7: Add `enable_training()` at every training site + fix lean-load tests**

- `src/wave_bitnet/network.rs` tests: in `wave_is_deterministic` add `a.enable_training(); b.enable_training();` after construction (it compares shadow); in `update_with_negative_signal_raises_pruned_synapse` add `net.enable_training();` after `Network::new` (it uses shadow + eprop).
- `src/wave_bitnet/neurons.rs` tests: in `from_parts_reproduces_built_layer` add `built.enable_training();` after `Layer::new` and `rebuilt.enable_training();` after `from_parts` (before the shadow-decode assertion); in `repack_roundtrips_shadow_to_ternary` add `l.enable_training();` after `Layer::new`.
- `src/wave_bitnet/multilayer_dfa.rs` test `step_raises_weights_on_negative_signal`: add `net.enable_training();` after `net2(8)`.
- `src/wave_bitnet/persist.rs`:
  - `assert_models_eq`: delete the loaded-shadow-decode loop (the `for s in 0..lb.shadow…` block) — loaded nets are lean; `codes` equality already proves weight identity.
  - `model_roundtrip_inference_equivalent`: replace the per-layer `layer_decide_potential` comparison with a lean-inference comparison via `with_layer`:
    ```rust
            for z in 0..a.layer_count() {
                a.with_layer(z, |la| b.with_layer(z, |lb| assert_eq!(la.potential, lb.potential, "layer {z} potential")));
            }
    ```
  - the `model_roundtrip_…` shadow loop referenced in Task 3 Step 6 (lines ~346) is inside `assert_models_eq`, already deleted here.
- `src/bench/wave_bitnet_bench.rs`: in `make_ff` and the side-car builder (each `let net = Network::new(…)` at ~line 211 and ~line 249), change to `let mut net = Network::new(…)` and add `net.enable_training();` on the line before the `(net, entries)` return.
- `benches/throughput_bitnet.rs`: in `make_sidecar` change `let net = Network::new(Config { seed, size, layers });` to `let mut net = …` and add `net.enable_training();` before `(net, entries)`. In `setup_net`, delete the line `net.set_record_eligibility(false);` and its comment.
- `examples/profile_bitnet.rs`: delete the line `net.set_record_eligibility(false);` (net is `Network::new` → lean → forward reads only codes).

- [ ] **Step 8: Build + test (lib, example, bench compile)**

Run: `cargo build 2>&1 | tail -5 && cargo test --lib 2>&1 | tail -15 && cargo build --benches --examples 2>&1 | tail -5`
Expected: warning-free build; all lib tests PASS; benches + examples compile.

- [ ] **Step 9: Verify a real training run still converges (behavior guard)**

Run: `cargo test --lib --release wave_bitnet_ff_depth8_smoke -- --ignored --nocapture 2>&1 | tail -25`
Expected: the FF depth-8 smoke benchmark still trains to its usual accuracy (≈1000) — training state is allocated via `enable_training()` and updates land.

- [ ] **Step 10: Commit**

```bash
git add src/ benches/throughput_bitnet.rs examples/profile_bitnet.rs
git commit -m "feat(wave_bitnet): toggleable training state — lean inference by default"
```

---

### Task 5: Update AGENTS.md

Reflect the optional `TrainState` and the enable/disable toggle in the guidance doc.

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the engine description + architecture map**

In `AGENTS.md`:
- In "The two modules", change the `wave_bitnet/` bullet clause "weights are stored as **2-bit packed ±1/0 codes** with an `f32` training shadow" to note the shadow (and decide snapshots) live in an **optional per-layer `TrainState`**, allocated only while training is enabled.
- In "The one idea that explains the engine", adjust "plus an `f32` shadow for training" to "plus, **when training is enabled**, an `f32` shadow (in `Layer.train`) the trainer moves and requantizes."
- In the architecture map, update the `neurons.rs` line from "…2-bit codes + f32 shadow + elig/decide state…" to "…2-bit codes + optional `TrainState` (f32 shadow + decide snapshots)…".
- In "Persistence", note the `.wbr` overlay now carries only `potential/cooldown/adapt/pending` (+ `wave_id`); the model (`.wbm`) is unchanged.
- Add a one-line note under "Reading & training" or Persistence: **training is toggled with `Network::enable_training()` / `disable_training()`; inference nets are lean (`train: None`) and pay nothing for shadow; disabling is lossy for sub-threshold shadow, like a `.wbm` round-trip.**

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: describe optional TrainState toggle in AGENTS.md"
```

---

## Self-Review

**Spec coverage:**
- Decision 1 (Approach A, `Layer` owns `Option<TrainState>`) → Task 3.
- Decision 2 (`TrainState` = shadow + decide snapshots) → Task 3 Step 1.
- Decision 3 (lean by default, explicit enable) → Task 4 Steps 4–7.
- Decision 4 (delete `elig_pre`/`elig_post` + `(A2)`) → Task 2.
- Decision 5 (remove `record_eligibility`) → Task 4 Steps 5–6.
- Decision 6 (`.wbm` unchanged, `.wbr` slimmed, version split) → Task 1.
- Decision 7 (lossy disable, documented) → Task 4 Step 3 (doc-comment) + Task 5.
- API (enable/disable/is_training; panic-on-lean) → Task 4 Steps 3, 5 + Task 3 Steps 2, 5.
- Persistence test switch (potential/spikes) → Task 4 Step 7.
- Bench/harness/doc ripple → Task 4 Step 7, Task 5.

**Placeholder scan:** No TBD/TODO; every code step shows concrete code or an explicit enumerated site list.

**Type consistency:** `TrainState { shadow, decide_potential, decide_eff }`, `Layer.train: Option<TrainState>`, `Layer::synapse_count`, `Network::{enable,disable}_training`/`is_training`, `Layer::{enable,disable}_training` used consistently across tasks. `process_layer` loses `record_elig` in Task 4 with both call sites (network `wave`, wave.rs test) updated.
