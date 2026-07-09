# Firing-rate regularization in the e-prop learning signal — Implementation Plan

> **For agentic workers:** Per this repo's `AGENTS.md`, execute this plan **inline and autonomously**
> (REQUIRED SUB-SKILL: superpowers:executing-plans) — **never** the subagent-driven option. Steps use
> checkbox (`- [ ]`) syntax. One commit per task.

**Goal:** Fold a per-neuron firing-rate regularization term into `train_eprop`'s learning signal so training
keeps every layer alive (near a target rate), and test whether that pushes the depth-20 ceiling.

**Architecture:** Add `L_j^reg = c_reg·(r_j − r_target)` to the existing per-target-neuron learning signal,
through the same eligibility `e_ij = pre_i·ψ_j`. `r_j = elig_pre_tgt[j] / n_waves` comes free from engine
state — **no engine change**. A small refactor first extracts the trained net so the revive-probe can read
per-layer rates. Guarded by `rate_reg != 0.0`, so the default path is byte-identical.

**Tech Stack:** Rust edition 2024, std-only, deterministic. No new dependencies. All work in
`src/bench/rsnn.rs`.

## Global Constraints

- **std-only**, **no `unsafe`**, **warning-free build**.
- **Determinism** — every result a pure function of `(seed, task_seed, config)`.
- **`wave_state_machine` frozen**; **no engine change** in this plan.
- **`rate_reg = 0.0` (default) must be byte-identical to current `train_eprop`** — the existing suite stays
  green (that is the regression check).
- Inline `#[cfg(test)]` tests, TDD. Expensive experiments are **`#[ignore]`d** ("run manually in
  `--release`").
- **One commit per task**, conventional-commit messages. **NEVER add a `Co-Authored-By` trailer.**
- Branch `eprop-rate-regularization` (already created). **Never push.**

## File Structure

- `src/bench/rsnn.rs` — everything:
  - Task 1: extract `train_eprop_inner` (trained net + readout); `train_eprop` calls it.
  - Task 2: `RsnnConfig.rate_reg` / `rate_target_permille` + the guarded reg term in the training loop.
  - Task 3: two `#[ignore]`d experiments (depth wall, revive probe).

---

### Task 1: Refactor — extract `train_eprop_inner` returning the trained net

**Files:**
- Modify: `src/bench/rsnn.rs` (`train_eprop`, new `train_eprop_inner`)

**Interfaces:**
- Produces: `fn train_eprop_inner(cfg: &RsnnConfig) -> (Network, Vec<Vec<f32>>)` — build + calibrate + run
  the full training loop; return the trained `Network` and the `k×ls` readout weights.
- Consumes: unchanged internals (`reservoir_activity`, `dfa_weight`, `target_of`, `softmax`, `pick_class`).

This is a **pure refactor** — no behaviour change. The existing `train_eprop` tests are the identity check.

- [ ] **Step 1: Run the existing tests to capture the green baseline**

Run: `cargo test --lib bench::rsnn::tests::multilayer_is_deterministic bench::rsnn::tests::eprop_hidden_learns_reliably`
Expected: PASS (these pin `train_eprop`'s behaviour; they must still pass after the refactor).

- [ ] **Step 2: Add `train_eprop_inner` and make `train_eprop` call it**

Replace the entire current `train_eprop` function with the two functions below. The training loop is moved
verbatim into `train_eprop_inner`; only the holdout-eval loop stays in `train_eprop`.

```rust
/// Build + calibrate + train (readout + hidden e-prop weights); return the trained net and the top-layer
/// readout. Split out so callers can both evaluate held-out accuracy and probe the trained net's per-layer
/// firing rates. `hidden_lr = 0` leaves the reservoir fixed (readout-only baseline).
fn train_eprop_inner(cfg: &RsnnConfig) -> (Network, Vec<Vec<f32>>) {
    let mut net = Network::new(cfg.engine_config());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let top = l - 1;
    let up = cfg.up_count as usize;
    let mut w = vec![vec![0f32; ls]; cfg.k]; // readout on the TOP layer only
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..w.len()).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    for t in 0..cfg.trials {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a_full = reservoir_activity(&mut net, cfg, class, t);
        let a_top = a_full[(l - 2) * ls..].to_vec(); // top-layer chunk
        let p = softmax(&score(&w, &a_top));
        let err: Vec<f32> = (0..cfg.k).map(|c| p[c] - if c == class { 1.0 } else { 0.0 }).collect();
        for c in 0..cfg.k {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * a_top[j];
            }
        }
        if cfg.hidden_lr != 0.0 {
            let trained: Vec<usize> = if cfg.multi_layer { (0..top).collect() } else { vec![top - 1] };
            for z in trained {
                let tgt = z + 1;
                let l_sig: Vec<f32> = (0..ls)
                    .map(|j| {
                        (0..cfg.k)
                            .map(|c| {
                                let b = if tgt == top { w[c][j] } else { dfa_weight(cfg.seed, (tgt * ls + j) as u32, c) };
                                b * err[c]
                            })
                            .sum()
                    })
                    .collect();
                let pre = net.with_layer_mut(z, |x| x.elig_pre.clone());
                let psi = net.with_layer_mut(tgt, |x| x.elig_post.clone());
                net.with_layer_mut(z, |lz| {
                    for i in 0..ls {
                        let pre_i = pre[i] as f32;
                        if pre_i == 0.0 {
                            continue;
                        }
                        let sg = (z * ls + i) as u32;
                        for kk in 0..up {
                            let j = target_of(cfg.seed, sg, i as u32, 1, kk as u32, cfg.up_radius, cfg.size) as usize;
                            lz.out_shadow[i * up + kk] += -cfg.hidden_lr * l_sig[j] * pre_i * psi[j] as f32;
                        }
                    }
                    for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                        *wq = s.round().clamp(-127.0, 127.0) as i8;
                    }
                });
            }
        }
    }
    (net, w)
}

/// Train a TOP-layer readout AND (via e-prop) the hidden layers' weights; return held-out permille.
pub fn train_eprop(cfg: &RsnnConfig) -> u64 {
    let (mut net, w) = train_eprop_inner(cfg);
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..w.len()).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a_full = reservoir_activity(&mut net, cfg, class, t);
        let a_top = a_full[(l - 2) * ls..].to_vec();
        let scores = score(&w, &a_top);
        let pred = (0..cfg.k).max_by(|&x, &y| scores[x].total_cmp(&scores[y])).unwrap();
        if pred == class {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}
```

- [ ] **Step 3: Run the identity tests**

Run: `cargo test --lib bench::rsnn::tests::multilayer_is_deterministic bench::rsnn::tests::eprop_hidden_learns_reliably bench::rsnn::tests::multilayer_beats_single_layer_at_depth`
Expected: PASS — the refactor is behaviour-preserving.

- [ ] **Step 4: Full suite + build**

Run: `cargo test --lib 2>&1 | tail -3 && cargo build 2>&1 | grep -E "warning|error" || echo "build clean"`
Expected: all tests pass; no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "refactor: extract train_eprop_inner returning the trained net"
```

---

### Task 2: Rate-regularization config + wiring

**Files:**
- Modify: `src/bench/rsnn.rs` (`RsnnConfig` + `demo()`, `train_eprop_inner`, tests)

**Interfaces:**
- Consumes: `train_eprop_inner` (Task 1), `Layer.elig_pre` (via `with_layer_mut`).
- Produces: `RsnnConfig.rate_reg: f32` (0 = off), `RsnnConfig.rate_target_permille: u32`; the guarded reg
  term added to `l_sig[j]` in the hidden-update loop.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `src/bench/rsnn.rs`:

```rust
#[test]
fn rate_reg_path_is_deterministic() {
    let mut cfg = RsnnConfig::demo();
    cfg.layers = 4;
    cfg.multi_layer = true;
    cfg.trials = 200;
    cfg.rate_reg = 0.5;
    cfg.rate_target_permille = 100;
    assert_eq!(train_eprop(&cfg), train_eprop(&cfg));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib bench::rsnn::tests::rate_reg_path_is_deterministic`
Expected: FAIL — `no field rate_reg`.

- [ ] **Step 3: Add the config fields**

In `struct RsnnConfig`, after `pub calib_fraction_q16: u32,`:

```rust
    pub rate_reg: f32,             // firing-rate regularization coefficient c_reg (0.0 = off)
    pub rate_target_permille: u32, // target per-neuron firing rate r_target, permille (e.g. 100 = 10%)
```

In `RsnnConfig::demo()`, after `calib_fraction_q16: 20000,`:

```rust
            rate_reg: 0.0,
            rate_target_permille: 100,
```

- [ ] **Step 4: Wire the reg term into `train_eprop_inner`**

In `train_eprop_inner`, replace the `l_sig` binding (the `let l_sig: Vec<f32> = (0..ls).map(...).collect();`
block) with a mutable one followed by the guarded reg term:

```rust
                let mut l_sig: Vec<f32> = (0..ls)
                    .map(|j| {
                        (0..cfg.k)
                            .map(|c| {
                                let b = if tgt == top { w[c][j] } else { dfa_weight(cfg.seed, (tgt * ls + j) as u32, c) };
                                b * err[c]
                            })
                            .sum()
                    })
                    .collect();
                // Firing-rate regularization (LSNN-style): keep each target neuron near r_target by adding
                // c_reg·(r_j − r_target) to its learning signal, carried by the SAME eligibility. A too-quiet
                // neuron (r_j < r_target) gets a negative signal → its incoming weights rise → it fires more.
                // Guarded, so rate_reg = 0 is byte-identical.
                if cfg.rate_reg != 0.0 {
                    let n_waves = (cfg.present_waves + cfg.delay + cfg.read_waves) as f32;
                    let r_target = cfg.rate_target_permille as f32 / 1000.0;
                    let post_pre = net.with_layer_mut(tgt, |x| x.elig_pre.clone());
                    for j in 0..ls {
                        let r_j = post_pre[j] as f32 / n_waves;
                        l_sig[j] += cfg.rate_reg * (r_j - r_target);
                    }
                }
```

(The `pre`/`psi` fetch and the `net.with_layer_mut(z, ...)` update block below it are unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib bench::rsnn`
Expected: PASS — the new determinism test **and** every existing `train_eprop` test (confirming
`rate_reg = 0.0` is byte-identical: `multilayer_beats_single_layer_at_depth`, `eprop_hidden_learns_reliably`,
`multilayer_is_deterministic`).

- [ ] **Step 6: Build + commit**

Run: `cargo build 2>&1 | grep -E "warning|error" || echo "build clean"`

```bash
git add src/bench/rsnn.rs
git commit -m "feat: add firing-rate regularization to the e-prop learning signal"
```

---

### Task 3: Experiments — depth wall + revive probe

**Files:**
- Modify: `src/bench/rsnn.rs` (tests)

**Interfaces:**
- Consumes: `train_eprop` (Task 2), `train_eprop_inner` (Task 1), `Network::measure_layer_rates`
  (`pub(crate)`, same crate), `random_l0_input`.

- [ ] **Step 1: Add the two `#[ignore]`d experiments**

Add to the `tests` module in `src/bench/rsnn.rs`:

```rust
#[test]
#[ignore] // expensive; run manually in --release
fn rate_reg_depth_wall() {
    // Does keeping every layer alive push the depth-20 wall (doc's ceiling ~485)? Worst-seed held-out,
    // multi-layer, trial length scaled to depth, rate_reg off vs a c_reg sweep.
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    let depth = 20usize;
    for reg in [0.0f32, 0.1, 0.5, 2.0] {
        let mut worst = 1000u64;
        for &s in &seeds {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 16;
            c.layers = depth;
            c.multi_layer = true;
            c.trials = 1500;
            c.present_waves = depth; // scale trial length to depth
            c.read_waves = depth;
            c.delay = 4;
            c.rate_reg = reg;
            c.rate_target_permille = 100;
            let acc = train_eprop(&c);
            eprintln!("depth {depth} rate_reg {reg} seed {s:#x}  {acc}");
            worst = worst.min(acc);
        }
        eprintln!("depth {depth} rate_reg {reg}: WORST {worst}");
    }
}

#[test]
#[ignore] // expensive; run manually in --release
fn rate_reg_revives_dead_layers() {
    // Per-layer firing rate of a TRAINED deep net, rate_reg off vs on. Off: deep layers dead (~0).
    // On: they should fire near the target through the full depth (liveness climbed the stack).
    for reg in [0.0f32, 0.5] {
        let mut c = RsnnConfig::demo();
        c.seed = 0xE9_0B_0A17;
        c.task_seed = 0xE9_0B_0A17;
        c.size = 16;
        c.layers = 16;
        c.multi_layer = true;
        c.trials = 800;
        c.present_waves = 16;
        c.read_waves = 16;
        c.delay = 4;
        c.rate_reg = reg;
        c.rate_target_permille = 100;
        let (mut net, _w) = train_eprop_inner(&c);
        let rates = net.measure_layer_rates(
            c.calib.warmup,
            c.calib.waves,
            &random_l0_input(c.seed ^ 0xE9, c.size, c.calib_fraction_q16),
        );
        let r2: Vec<f64> = rates.iter().map(|x| (x * 100.0).round() / 100.0).collect();
        eprintln!("rate_reg {reg}: per-layer rates {r2:?}");
    }
}
```

- [ ] **Step 2: Verify they compile and are registered**

Run: `cargo test --lib --no-run 2>&1 | grep -E "warning|error" || echo "compile clean"`
Then: `cargo test --lib -- --ignored --list 2>&1 | grep -E "rate_reg_depth_wall|rate_reg_revives"`
Expected: compile clean; both experiments listed.

- [ ] **Step 3: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "test: rate-reg depth-wall and dead-layer-revival experiments"
```

---

## Final verification (after Task 3)

- [ ] `cargo test --lib` — whole suite green (fast; ignored experiments skipped).
- [ ] `cargo build` — warning-free.
- [ ] `cargo test --lib -- --ignored --list` includes `rate_reg_depth_wall` and `rate_reg_revives_dead_layers`.
- [ ] The pre-existing `train_eprop` headline tests still pass unchanged (identity of the `rate_reg = 0`
      default path): `multilayer_beats_single_layer_at_depth`, `eprop_hidden_learns_reliably`,
      `multilayer_is_deterministic`.

## Running the experiments (manual, after the plan)

```bash
cargo test --release --lib rate_reg_revives_dead_layers -- --ignored --nocapture   # mechanism: do layers revive?
cargo test --release --lib rate_reg_depth_wall          -- --ignored --nocapture   # payoff: does depth-20 lift?
```

Interpret per the spec's honesty gate: layers revive but accuracy stays chance ⇒ the wall is the **credit
rule** (→ BPTT); accuracy collapses under large `c_reg` ⇒ the reg is **homogenizing** activity (find the
usable window). Consolidate whatever we find — plus the abandoned-criticality findings — into
`docs/experiments_results.md`.
