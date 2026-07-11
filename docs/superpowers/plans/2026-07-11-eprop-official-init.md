# e-prop as an official `wave_net` init + training method — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline, per this repo's AGENTS.md — never the subagent-driven option). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the σ-eprop criticality init and the core e-prop rule from `bench/` into the `wave_net/` engine as official init + training (FF), downgrade the firing-rate calibration to a `bench/` tool, and make `critical_init` the default FF init — gated on an end-to-end FF training comparison.

**Architecture:** Engine gains a generic `eprop_update` primitive, a `forward_avalanche` σ diagnostic, `critical_init` (built on both), and a feed-forward `train_ff` driver. `bench` keeps task/DFA/rate/σ learning-signal computation, the side-car/sequence loops (now calling `eprop_update`), the experiments, and the downgraded `calibrate`.

**Tech Stack:** Rust edition 2024, standard library only, single-threaded deterministic integer engine.

## Global Constraints

- **Standard library only in `src/`**; **no `unsafe`**; **warning-free build**.
- **Determinism** — every result a pure function of `(seed, config, input)`.
- **All existing tests stay green** at every task (the moves are mechanical; the suite is the regression guard). Baseline: `cargo test` = 164 passed, 0 failed.
- **Never modify `wave_state_machine`** (frozen reference).
- **Conventional-commit messages**, **one commit per task**, **no `Co-Authored-By` trailer**, **never push**.
- Branch `feat/critical-init-spike` is already checked out (has the criticality-init work + spec).
- External crate import path doubles up: `wave_net::wave_net::…`.

## File Structure

```
wave_net/
  synapse.rs        # + target_of (moved from rsnn) — Task 1
  eprop.rs          # NEW: eprop_update, windowed_eligibility (Task 2), train_ff (Task 4)
  critical_init.rs  # NEW: forward_avalanche, critical_init, CriticalInitParams, random_l0_input (Task 3)
  network.rs        # (methods live in eprop.rs/critical_init.rs via `impl Network`)
  mod.rs            # + pub mod eprop; pub mod critical_init;  (Task 2/3)
  calibrate.rs      # REMOVED (Task 5)
bench/
  calibrate.rs      # NEW: calibrate() free fn + CalibrateParams (Task 5)
  rsnn.rs           # target_of removed; calibrate call sites updated (Task 5); FF default flip + side-car uses eprop_update (Task 7)
  critical_init.rs  # experiments only, call engine API (Task 3); + validation gate (Task 6)
  mod.rs            # + pub mod calibrate;  (Task 5)
```

---

### Task 1: Move `target_of` into `wave_net/synapse.rs`

**Files:**
- Modify: `src/wave_net/synapse.rs` (add `target_of`), `src/bench/rsnn.rs` (remove it, import from engine)

**Interfaces:**
- Produces: `pub fn wave_net::wave_net::synapse::target_of(seed, source_global, src_local, level, k, radius, size) -> u32`

- [ ] **Step 1: Add `target_of` to `synapse.rs`** (it uses `xy_of/mix/key/map_range24/wrap/local_of/P_TARGET`, all already in that module):

```rust
/// Target local of one synapse (source `src_local`, topology `level`, slot `k`, `radius`) — the same
/// hash `generate_into` uses. Lets a learning rule recover a synapse's target without re-scattering.
pub fn target_of(seed: u64, source_global: u32, src_local: u32, level: i32, k: u32, radius: u32, size: u32) -> u32 {
    let (sx, sy) = xy_of(src_local, size);
    let h = mix(key(seed, source_global, level, k, P_TARGET));
    let span = 2 * radius + 1;
    let dx = map_range24((h >> 40) as u32, span) as i32 - radius as i32;
    let dy = map_range24(((h >> 16) as u32) & 0x00FF_FFFF, span) as i32 - radius as i32;
    local_of(wrap(sx, dx, size), wrap(sy, dy, size), size)
}
```

- [ ] **Step 2: Delete `target_of` from `rsnn.rs`** and add `use crate::wave_net::synapse::target_of;` near the other `wave_net::synapse` imports. Remove now-unused `map_range24`/`P_TARGET`/etc. imports from `rsnn.rs` only if they become unused.

- [ ] **Step 3: Verify**

Run: `cargo build 2>&1 | grep -c warning` → `0`; `cargo test 2>&1 | grep "test result:" | head -1` → `164 passed`.

- [ ] **Step 4: Commit**

```bash
git add src/wave_net/synapse.rs src/bench/rsnn.rs
git commit -m "refactor: move target_of into wave_net/synapse (procedural addressing belongs with the hash)"
```

---

### Task 2: Add the generic `eprop_update` primitive + `windowed_eligibility` (`wave_net/eprop.rs`)

**Files:**
- Create: `src/wave_net/eprop.rs`
- Modify: `src/wave_net/mod.rs` (`pub mod eprop;`)

**Interfaces:**
- Produces:
  - `pub fn Network::eprop_update(&mut self, source_z: usize, entry_idx: usize, pre: &[i32], psi: &[i32], signal: &[f32], lr: f32, use_psi: bool)`
  - `pub fn Network::windowed_eligibility(&mut self, warmup: usize, waves: usize, input: &impl Fn(usize)->Vec<u32>) -> (Vec<Vec<i32>>, Vec<Vec<i32>>)`

- [ ] **Step 1: Write the failing test** (`src/wave_net/eprop.rs`, `#[cfg(test)]`): a hand-computed update. Build a 2-layer net (L0→L1, level+1, radius 0, count 1 so target of local `i` is `i`), set `elig_pre[0]=[…]` implicitly via known pre/psi passed in, and assert `out_shadow` moved by exactly `-lr·signal[j]·pre_i·psi_j`.

```rust
#[test]
fn eprop_update_applies_expected_delta() {
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::TopologyLevel;
    let lc = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 0, count: 1 }],
        leak: (3,5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 5 };
    let mut net = Network::new(Config { seed: 1, size: 4, layers: vec![lc; 2] });
    let ls = 16;
    let pre = vec![2i32; ls];      // source pre-trace
    let psi = vec![3i32; ls];      // target ψ
    let signal = vec![0.5f32; ls]; // per-target learning signal
    let before = net.with_layer(0, |l| l.out_shadow[0]);         // slot 0 of source local 0 (target = local 0)
    net.eprop_update(0, 0, &pre, &psi, &signal, 0.1, true);
    let after = net.with_layer(0, |l| l.out_shadow[0]);
    // Δ = -lr·signal[0]·pre[0]·psi[0] = -0.1·0.5·2·3 = -0.3
    assert!((after - before + 0.3).abs() < 1e-4, "{before} -> {after}");
}
```

- [ ] **Step 2: Run — verify it fails** (`eprop_update` not defined).

Run: `cargo test --lib eprop_update_applies_expected_delta 2>&1 | tail -5` → compile error / FAIL.

- [ ] **Step 3: Implement `eprop.rs`:**

```rust
//! `eprop` — the official e-prop learning rule on the live engine: a generic per-layer weight-update
//! primitive from the stored eligibility, and a feed-forward training driver. Learning *signals*
//! (task error, DFA, rate, σ) are computed by the caller (`bench`) and passed in.

use crate::wave_net::network::Network;
use crate::wave_net::synapse::target_of;

impl Network {
    /// Apply one e-prop weight update to the `entry_idx`-th topology entry of layer `source_z`, using
    /// caller-supplied `pre` (source pre-trace), `psi` (target ψ) and per-target `signal`:
    /// `out_shadow[i, slot] += -lr · signal[j] · pre_i · (psi_j if use_psi else 1)`, then requantise.
    /// `target_of` recovers each synapse's target `j` (no re-scatter). Generic over the entry, so FF
    /// (the up entry) and side-car edges both reuse it.
    pub fn eprop_update(&mut self, source_z: usize, entry_idx: usize, pre: &[i32], psi: &[i32], signal: &[f32], lr: f32, use_psi: bool) {
        let seed = self.seed_val();
        let size = self.size();
        let l = self.layer_count();
        let ls = (size as usize) * (size as usize);
        let (level, radius, count, slot_base, total_slots) = self.with_layer(source_z, |lz| {
            let e = &lz.topology[entry_idx];
            let slot_base: usize = lz.topology[..entry_idx].iter().map(|t| t.count as usize).sum();
            (e.level, e.radius, e.count as usize, slot_base, lz.total_slots)
        });
        let tz = source_z as i32 + level;
        if tz < 0 || tz as usize >= l {
            return;
        }
        let base = source_z * ls;
        self.with_layer_mut(source_z, |lz| {
            for i in 0..ls {
                let pre_i = pre[i] as f32;
                if pre_i == 0.0 {
                    continue;
                }
                let sg = (base + i) as u32;
                for kk in 0..count {
                    let j = target_of(seed, sg, i as u32, level, kk as u32, radius, size) as usize;
                    let pf = if use_psi { psi[j] as f32 } else { 1.0 };
                    lz.out_shadow[i * total_slots + slot_base + kk] += -lr * signal[j] * pre_i * pf;
                }
            }
            for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                *wq = s.round().clamp(-127.0, 127.0) as i8;
            }
        });
    }

    /// Windowed per-neuron eligibility for every layer: pre-trace + ψ accumulated over `waves` after a
    /// `warmup` transient (difference of the running accumulators, so the boots-hot transient is excluded).
    pub fn windowed_eligibility(&mut self, warmup: usize, waves: usize, input: &impl Fn(usize) -> Vec<u32>) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
        let l = self.layer_count();
        self.reset_state();
        for w in 0..warmup {
            self.wave(&input(w));
        }
        let pre0: Vec<Vec<i32>> = (0..l).map(|z| self.with_layer(z, |x| x.elig_pre.clone())).collect();
        let psi0: Vec<Vec<i32>> = (0..l).map(|z| self.with_layer(z, |x| x.elig_post.clone())).collect();
        for w in 0..waves {
            self.wave(&input(warmup + w));
        }
        let pre = (0..l).map(|z| self.with_layer(z, |x| x.elig_pre.iter().zip(&pre0[z]).map(|(a, b)| a - b).collect())).collect();
        let psi = (0..l).map(|z| self.with_layer(z, |x| x.elig_post.iter().zip(&psi0[z]).map(|(a, b)| a - b).collect())).collect();
        (pre, psi)
    }
}
```

Note: `eprop_update` needs the private `seed`. Add a `pub(crate) fn seed_val(&self) -> u64 { self.seed }` accessor to `network.rs` (or read the field directly if `eprop.rs` is a submodule of `wave_net` with field access — it is *not* in `network.rs`, so add the accessor). `with_layer`/`with_layer_mut` are `pub(crate)` — accessible from sibling `wave_net` modules.

- [ ] **Step 4: Register + run**

Add `pub mod eprop;` to `src/wave_net/mod.rs`. Run: `cargo test --lib eprop_update_applies_expected_delta 2>&1 | tail -3` → PASS; then `cargo test 2>&1 | grep "test result:" | head -1` → `165 passed`; `cargo build 2>&1 | grep -c warning` → `0`.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/eprop.rs src/wave_net/mod.rs src/wave_net/network.rs
git commit -m "feat: official e-prop update primitive + windowed_eligibility in wave_net/eprop"
```

---

### Task 3: Move the σ-init + σ diagnostic + `random_l0_input` into `wave_net/critical_init.rs`

**Files:**
- Create: `src/wave_net/critical_init.rs`
- Modify: `src/wave_net/mod.rs` (`pub mod critical_init;`), `src/bench/critical_init.rs` (delete moved code; keep experiments; import engine API), `src/wave_net/calibrate.rs` (remove `random_l0_input` — Task 5 removes the file, but this task moves the fn out first)

**Interfaces:**
- Produces:
  - `pub fn wave_net::critical_init::random_l0_input(seed, size, fraction_q16) -> impl Fn(usize)->Vec<u32>`
  - `pub fn wave_net::critical_init::forward_avalanche(net, drive_seed, frac, warmup, n_perturb, burst) -> Vec<f64>`
  - `pub struct CriticalInitParams` + `pub fn Network::critical_init(&mut self, drive_seed, frac, &CriticalInitParams)`

- [ ] **Step 1: Create `wave_net/critical_init.rs`** by moving from `bench/critical_init.rs`: `random_l0_input` (from `wave_net/calibrate.rs`), `forward_avalanche`, `CriticalInitParams` (the `SigmaEprop` knobs — rename to `CriticalInitParams`), and `sigma_eprop_init` **rewritten as `Network::critical_init`** calling `self.windowed_eligibility` + `self.eprop_update(src, 0, &pre[src], &psi[z], &vec![sig_err; ls], lr, false)` (the σ-error is a uniform per-target signal). Key body:

```rust
pub fn critical_init(&mut self, drive_seed: u64, frac_q16: u32, params: &CriticalInitParams) {
    let l = self.layer_count();
    let ls = (self.size() * self.size()) as usize;
    let drive = random_l0_input(drive_seed, self.size(), frac_q16);
    for z in 1..l {
        let src = z - 1;
        for _ in 0..params.rounds {
            let fp = forward_avalanche(self, drive_seed, frac_q16, params.warmup, params.n_perturb, params.burst);
            let denom = fp[z - 1];
            let sigma = if denom > 0.0 { fp[z] / denom } else { 0.0 };
            if denom > 0.0 && (sigma - 1.0).abs() <= params.tol as f64 {
                break;
            }
            let sig_err = (sigma - 1.0) as f32;
            let (pre, psi) = self.windowed_eligibility(params.warmup, params.waves, &drive);
            self.eprop_update(src, 0, &pre[src], &psi[z], &vec![sig_err; ls], params.lr, false);
        }
    }
}
```

(`forward_avalanche` moves verbatim; it uses `Arc`/`Mutex`/`HashSet` + `random_l0_input` + `mix`/`key` — all engine-available.)

- [ ] **Step 2: Delete the moved code from `bench/critical_init.rs`.** Keep only the experiments (`rate_init_*`, `sigma_*`, `computation_*`, the task helpers). They now `use wave_net::wave_net::critical_init::{random_l0_input, forward_avalanche, CriticalInitParams};` (internal path `crate::wave_net::critical_init::…`) and call `net.critical_init(...)` where they called `sigma_eprop_init(...)`. **Note (log it):** the stepping-stone `rate_reg_init` / `sigma_gain_init` and their experiments are dropped here unless kept for the record — per the spec, only σ-eprop is promoted. If kept, they must still compile against the moved helpers.

- [ ] **Step 3: Register + verify**

Add `pub mod critical_init;` to `src/wave_net/mod.rs`. Run: `cargo build 2>&1 | grep -c warning` → `0`; `cargo test 2>&1 | grep "test result:" | head -1` → still green; `cargo test --release critical_init:: -- --ignored --nocapture 2>&1 | grep -E "test result|panic" | head` runs one experiment to confirm the engine API path works.

- [ ] **Step 4: Commit**

```bash
git add src/wave_net/critical_init.rs src/wave_net/mod.rs src/bench/critical_init.rs src/wave_net/calibrate.rs
git commit -m "feat: critical_init (rate-free σ≈1) + forward_avalanche + random_l0_input into wave_net"
```

---

### Task 4: Add the feed-forward `train_ff` driver (`wave_net/eprop.rs`)

**Files:**
- Modify: `src/wave_net/eprop.rs`

**Interfaces:**
- Produces: `pub fn Network::train_ff(&mut self, trials, present, drive: impl Fn(usize,usize)->Vec<u32>, signal: impl Fn(&Network, usize)->Vec<Vec<f32>>, lr: f32) -> Vec<()>` — runs the FF e-prop loop.

- [ ] **Step 1: Write the failing test** — `train_ff` on a 2-layer net with a constant negative signal must *raise* the summed `out_shadow` of the trained layer (drives weights up):

```rust
#[test]
fn train_ff_moves_weights_by_signal() {
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::TopologyLevel;
    let lc = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
        leak: (3,5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 4, adapt_bump: 5, adapt_decay: 6 };
    let mut net = Network::new(Config { seed: 7, size: 8, layers: vec![lc; 2] });
    let before: f32 = net.with_layer(0, |l| l.out_shadow.iter().sum());
    let all: Vec<u32> = (0..64).collect();
    net.train_ff(30, 8, |_t, _w| all.clone(), |_net, _t| vec![vec![-1.0f32; 64]], 0.02);
    let after: f32 = net.with_layer(0, |l| l.out_shadow.iter().sum());
    assert!(after > before + 1.0, "signal<0 should raise weights: {before} -> {after}");
}
```

- [ ] **Step 2: Run — verify it fails.** Run: `cargo test --lib train_ff_moves_weights_by_signal 2>&1 | tail -3` → FAIL.

- [ ] **Step 3: Implement `train_ff`:**

```rust
impl Network {
    /// Feed-forward e-prop training driver. Per trial: reset, drive `drive(trial, wave)` for `present`
    /// waves (eligibility accrues), read each source layer's pre-trace + its target's ψ, get the per-
    /// (computational-source-layer) learning `signal(&net, trial)`, and apply `eprop_update` to each.
    /// The caller's `signal` owns all task logic (readout error, DFA, rate, …). FF (single up entry).
    pub fn train_ff(&mut self, trials: usize, present: usize,
        drive: impl Fn(usize, usize) -> Vec<u32>,
        signal: impl Fn(&Network, usize) -> Vec<Vec<f32>>, lr: f32) -> Vec<()> {
        let l = self.layer_count();
        for t in 0..trials {
            self.reset_state();
            for w in 0..present {
                self.wave(&drive(t, w));
            }
            let sig = signal(self, t); // sig[src_z-1] = learning signal for target layer src_z
            let pre: Vec<Vec<i32>> = (0..l).map(|z| self.with_layer(z, |x| x.elig_pre.clone())).collect();
            let psi: Vec<Vec<i32>> = (0..l).map(|z| self.with_layer(z, |x| x.elig_post.clone())).collect();
            for z in 1..l {
                let src = z - 1;
                self.eprop_update(src, 0, &pre[src], &psi[z], &sig[src], lr, true);
            }
        }
        Vec::new()
    }
}
```

(Return type is a placeholder `Vec<()>`; the caller reads accuracy via its own readout after training. Keep it minimal — the driver owns the loop, not the metric.)

- [ ] **Step 4: Run — verify pass + suite.** `cargo test --lib train_ff_moves_weights_by_signal` → PASS; `cargo test 2>&1 | grep "test result:" | head -1` → `166 passed`; `cargo build` warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/eprop.rs
git commit -m "feat: feed-forward train_ff e-prop driver (loop + callbacks) in wave_net/eprop"
```

---

### Task 5: Downgrade calibration — `wave_net/calibrate.rs` → `bench/calibrate.rs`

**Files:**
- Create: `src/bench/calibrate.rs`
- Delete: `src/wave_net/calibrate.rs`
- Modify: `src/wave_net/mod.rs` (remove `pub mod calibrate;`), `src/bench/mod.rs` (`pub mod calibrate;`), `src/bench/rsnn.rs` (call-site updates), `src/bench/critical_init.rs` + `benches/throughput.rs` (call-site updates)

**Interfaces:**
- Produces: `pub fn bench::calibrate::calibrate(net: &mut Network, params: &CalibrateParams, input: &impl Fn(usize)->Vec<u32>)`, `pub struct bench::calibrate::CalibrateParams`.

- [ ] **Step 1: Create `bench/calibrate.rs`** — the body of `Network::calibrate` becomes a free `fn calibrate(net, params, input)` (uses `net.measure_layer_rates`/`net.with_layer_mut` — `pub(crate)`, reachable from bench). Move `CalibrateParams` and the calibrate tests. (Its `#[cfg(test)]` tests move too; adjust to call `calibrate(&mut net, …)`.)

- [ ] **Step 2: Delete `src/wave_net/calibrate.rs`** and remove `pub mod calibrate;` from `src/wave_net/mod.rs`. Add `pub mod calibrate;` to `src/bench/mod.rs`.

- [ ] **Step 3: Update call sites.** In `src/bench/rsnn.rs`, replace each `net.calibrate(&cfg.calib, &input)` with `crate::bench::calibrate::calibrate(&mut net, &cfg.calib, &input)` (import `use crate::bench::calibrate::{calibrate, CalibrateParams};`; `RsnnConfig.calib: CalibrateParams` now refers to the bench type). Update `src/bench/critical_init.rs` and `benches/throughput.rs` (`net.calibrate(...)` → `calibrate(&mut net, ...)`, importing `wave_net::bench::calibrate::{calibrate, CalibrateParams}` in the external bench file).

- [ ] **Step 4: Verify**

Run: `cargo build 2>&1 | grep -c warning` → `0`; `cargo test 2>&1 | grep "test result:" | head -1` → green; `cargo bench --bench throughput -- --test 2>&1 | tail -3` → runs (guard passes).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: downgrade calibration to a bench tool (bench/calibrate.rs); update call sites"
```

---

### Task 6: Validation gate — end-to-end FF training, `critical_init` vs `calibrate`

**Files:**
- Modify: `src/bench/critical_init.rs` (add the gate experiment) or `src/bench/rsnn.rs` (reuse task/readout infra)

**Interfaces:**
- Consumes: `Network::critical_init`, `Network::train_ff`, `bench::calibrate::calibrate`, an existing FF task + trained readout from `rsnn.rs` (temporal-XOR or parity via `train_ff`).

- [ ] **Step 1: Write the gate experiment** (`#[test] #[ignore]`). For **≥3 seeds × up_count ∈ {16, 32} × the 5-layer / size-32 FF config**, build the net; init path A `bench::calibrate::calibrate(&mut net, …)`, path B `net.critical_init(seed, frac, &CriticalInitParams::default())`; then **train the same FF task end-to-end** through `train_ff` with a trained readout (reuse the readout + learning-signal helpers from `rsnn.rs`, or a focused temporal-XOR); report held-out accuracy per (init, up_count), averaged over seeds. Print a table.

```
// pseudocode of the printed comparison:
// up_count=16: calibration acc=___‰   critical_init acc=___‰
// up_count=32: calibration acc=___‰   critical_init acc=___‰   (mean over N seeds)
```

- [ ] **Step 2: Run it.** Run: `cargo test --release ff_init_validation_gate -- --ignored --nocapture 2>&1 | grep -E "up_count|acc"`. **GATE:** record the numbers. Proceed to Task 7 **only if** `critical_init ≥ calibration` on trained accuracy across settings and it **wins at up_count 16 depth** (the brittle regime). If it does not, STOP: record the result in `docs/experiments_results.md`, do **not** flip the default, and report back.

- [ ] **Step 3: Commit** (the experiment + the recorded result either way)

```bash
git add -A
git commit -m "feat: FF end-to-end validation gate (critical_init vs calibration, deep/width-32/density sweep, multi-seed)"
```

---

### Task 7: Flip the FF default + side-car uses `eprop_update` + AGENTS.md rewrite

**Files:**
- Modify: `src/bench/rsnn.rs` (FF `train_*` use `critical_init`; side-car e-prop updates call `net.eprop_update`), `AGENTS.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Flip the FF default.** In `rsnn.rs`, the **feed-forward** `train_*` paths (the ones on `engine_config`/`engine_config_xor` — not the side-car/recurrent ones) replace `calibrate(&mut net, &cfg.calib, &input)` with `net.critical_init(cfg.seed ^ 0xE9, cfg.calib_fraction_q16, &CriticalInitParams::default())`. The **side-car/recurrent** `train_*` keep `calibrate(...)` (the fallback, per spec).

- [ ] **Step 2: Side-car uses the primitive.** Replace the inline e-prop weight updates in the side-car/multilayer trainers with `net.eprop_update(z, entry_idx, &pre[z], &psi[tgt], &l_sig, cfg.hidden_lr, true)` calls (per trained edge), passing the same eligibility/signal they compute now. **This must be byte-identical** — verify the side-car recurrence tests / results are unchanged (run the relevant `#[ignore]`d recurrence test before and after; the numbers must match).

- [ ] **Step 3: Verify green + unchanged results.** `cargo test 2>&1 | grep "test result:"` → green; run one recurrence benchmark test `--ignored` before/after Step 2 to confirm identical output.

- [ ] **Step 4: Rewrite AGENTS.md.** Update "The three modules" and "Architecture map": `wave_net` now owns the e-prop update primitive + `train_ff` + `critical_init` (default FF init) + `forward_avalanche`; `bench` owns tasks/DFA/side-car/sequence learning signals + benchmarks + the downgraded `calibrate` (recurrent fallback). Update the "Calibration" section to describe `critical_init` as the default and calibration as the deprecated fallback. Note recurrent-σ is the pending extension.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: critical_init is the default FF init; side-car uses eprop_update; AGENTS.md rewrite"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-11-eprop-official-init-design.md`):
- Module moves (target_of, eprop.rs, critical_init.rs, calibrate downgrade, random_l0_input home) → Tasks 1–3, 5.
- Engine API (eprop_update, forward_avalanche, critical_init, train_ff) → Tasks 2–4.
- Default/fallback story → Task 7 Step 1.
- Validation gate (deep ≥4 layers, width 32, up_count sweep, multi-seed, gate condition) → Task 6.
- Constraints (stdlib, determinism, warning-free, tests green, no wave_state_machine) → every task's verify + Global Constraints.
- AGENTS.md rewrite → Task 7 Step 4.
- Non-goal (recurrent deferred; side-car keeps calibrate fallback) → Task 7 Step 1.

**Placeholder scan:** the only deliberate open decision is "drop vs keep the stepping-stone rate/gain-scaling experiments" (Task 3 Step 2) — flagged for the implementer, not a blocker. `train_ff` returns `Vec<()>` intentionally (metrics are the caller's readout). No TBDs.

**Type consistency:** `eprop_update(source_z, entry_idx, pre, psi, signal, lr, use_psi)` is used identically in Tasks 2 (def), 3 (critical_init), 4 (train_ff), 7 (side-car). `critical_init`, `forward_avalanche`, `windowed_eligibility`, `random_l0_input`, `calibrate`, `CalibrateParams`, `CriticalInitParams` signatures match across tasks.
