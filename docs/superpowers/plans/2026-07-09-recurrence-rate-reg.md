# Rate-reg-sustained recurrence — Implementation Plan

> **For agentic workers:** Per this repo's `AGENTS.md`, execute this plan **inline and autonomously**
> (REQUIRED SUB-SKILL: superpowers:executing-plans) — **never** the subagent-driven option. Steps use
> checkbox (`- [ ]`) syntax. One commit per task.

**Goal:** Wire firing-rate regularization into the two recurrent trainers (`recurrent_update` for level-0
lateral, `train_recurrent` for backward) so the recurrent loop stays alive through the silent gap, and test
whether that lets temporal e-prop sustain the cue — on temporal XOR (LIF, delay 20), for both substrates.

**Architecture:** Add `c_reg·(r_j − r_target)` to each recurrent trainer's per-neuron learning signal,
carried by the temporal eligibility they already build (`r_j` from the per-wave fired-sets they record).
Guarded by `rate_reg != 0.0` → the default path is byte-identical. A small refactor exposes the trained
`train_xor` net so the gap-survival probe can read per-wave recurrent activity. Reuses the merged
`RsnnConfig.rate_reg` / `rate_target_permille`; **no engine change**.

**Tech Stack:** Rust edition 2024, std-only, deterministic. All work in `src/bench/rsnn.rs`.

## Global Constraints

- **std-only**, **no `unsafe`**, **warning-free build**.
- **Determinism** — pure function of `(seed, task_seed, config)`.
- **`wave_state_machine` frozen**; **no engine change**.
- **`rate_reg = 0.0` (default) must be byte-identical** to current `train_xor` / `train_recurrent` — the
  existing recurrence tests stay green (that is the regression check).
- Inline `#[cfg(test)]` tests, TDD. Expensive experiments are **`#[ignore]`d** ("run manually in
  `--release`").
- **One commit per task**, conventional-commit messages. **NEVER add a `Co-Authored-By` trailer.**
- Branch `recurrence-rate-reg` (already created). **Never push.**

## File Structure

- `src/bench/rsnn.rs` — everything:
  - Task 1: extract `train_xor_inner` (trained net + readout); `train_xor` calls it.
  - Task 2: rate-reg term in `recurrent_update` (level-0 lateral).
  - Task 3: rate-reg term in `train_recurrent` (backward).
  - Task 4: three `#[ignore]`d experiments (gap survival, lateral vs FF, backward vs FF).

---

### Task 1: Refactor — extract `train_xor_inner` returning the trained net

**Files:**
- Modify: `src/bench/rsnn.rs` (`train_xor`, new `train_xor_inner`)

**Interfaces:**
- Produces: `fn train_xor_inner(cfg: &RsnnConfig) -> (Network, Vec<Vec<f32>>)` — build + calibrate + train
  (readout + level-0 recurrent weights); return the trained net and the L1 readout.

Pure refactor — the existing `train_xor` tests are the identity check.

- [ ] **Step 1: Capture the green baseline**

Run: `cargo test --lib -- bench::rsnn::tests::temporal_xor_ff_is_near_chance bench::rsnn::tests::recurrence_does_not_yet_beat_ff_on_temporal_xor`
Expected: PASS (these pin `train_xor`).

- [ ] **Step 2: Replace `train_xor` with `train_xor_inner` + a thin `train_xor`**

Replace the entire current `train_xor` function (from its doc comment through its closing `}`) with:

```rust
/// Build + calibrate + train (readout + level-0 recurrent weights) for temporal XOR; return the trained net
/// and the L1 readout. Split out so callers can probe the trained net's per-wave recurrent activity.
fn train_xor_inner(cfg: &RsnnConfig) -> (Network, Vec<Vec<f32>>) {
    let mut net = Network::new(cfg.engine_config_xor());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    if cfg.rec_count > 0 && cfg.rec_init != 0 {
        // bootstrap self-excitation so recurrent activity persists through the gap (else no eligibility)
        net.with_layer_mut(1, |l1| {
            for wq in l1.out_weights.iter_mut() {
                *wq = cfg.rec_init;
            }
            for (s, wq) in l1.out_shadow.iter_mut().zip(&l1.out_weights) {
                *s = *wq as f32;
            }
        });
    }
    let ls = (cfg.size * cfg.size) as usize;
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    for t in 0..cfg.trials {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let label = a ^ b;
        let (act, waves) = xor_trial(&mut net, cfg, a, b, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.rec_count > 0 {
            recurrent_update(&mut net, cfg, &w, &err, &waves);
        }
    }
    (net, w)
}

/// Train a readout on L1's read-window activity for temporal XOR; with `rec_count > 0` also trains the L1
/// level-0 recurrent weights. Returns held-out test accuracy permille.
pub fn train_xor(cfg: &RsnnConfig) -> u64 {
    let (mut net, w) = train_xor_inner(cfg);
    let ls = (cfg.size * cfg.size) as usize;
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let (act, _) = xor_trial(&mut net, cfg, a, b, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == (a ^ b) {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}
```

- [ ] **Step 3: Run the identity tests + build**

Run: `cargo test --lib -- bench::rsnn::tests::temporal_xor_ff_is_near_chance bench::rsnn::tests::recurrence_does_not_yet_beat_ff_on_temporal_xor`
Then: `cargo build 2>&1 | grep -E "warning|error" || echo "build clean"`
Expected: PASS; build clean.

- [ ] **Step 4: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "refactor: extract train_xor_inner returning the trained net"
```

---

### Task 2: Rate-reg term in `recurrent_update` (level-0 lateral)

**Files:**
- Modify: `src/bench/rsnn.rs` (`recurrent_update`, tests)

**Interfaces:**
- Consumes: `RsnnConfig.rate_reg`, `rate_target_permille` (merged); the `fired` table `recurrent_update`
  already builds.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn rate_reg_lateral_is_deterministic() {
    let mut cfg = RsnnConfig::demo();
    cfg.adapt_bump = 0;
    cfg.delay = 20;
    cfg.rec_count = 8;
    cfg.rec_init = 0;
    cfg.rate_reg = 5.0;
    cfg.rate_target_permille = 100;
    cfg.trials = 150;
    assert_eq!(train_xor(&cfg), train_xor(&cfg));
}
```

- [ ] **Step 2: Run — expect PASS already (guarded no-op not yet added, but rate_reg has no effect yet)**

Run: `cargo test --lib bench::rsnn::tests::rate_reg_lateral_is_deterministic`
Expected: PASS but **meaningless** — `rate_reg` is not yet read in `recurrent_update`, so this only checks
determinism of the unchanged path. It becomes a real test after Step 3 wires `rate_reg` in. (If it FAILs to
compile, the config fields are missing — they should already exist from the merged work.)

- [ ] **Step 3: Add the guarded reg term in `recurrent_update`**

In `recurrent_update`, replace the `l_sig` binding:

```rust
    let l_sig: Vec<f32> = (0..ls).map(|j| (0..2).map(|c| w[c][j] * err[c]).sum()).collect();
```

with a mutable binding followed by the guarded reg term:

```rust
    let mut l_sig: Vec<f32> = (0..ls).map(|j| (0..2).map(|c| w[c][j] * err[c]).sum()).collect();
    // Firing-rate regularization: keep the recurrent layer near r_target so the loop stays alive through
    // the gap (carried by the same temporal eligibility). Guarded — rate_reg = 0 is byte-identical.
    if cfg.rate_reg != 0.0 {
        let r_target = cfg.rate_target_permille as f32 / 1000.0;
        for j in 0..ls {
            let r_j = (0..ttot).map(|t| fired[t][j]).sum::<f32>() / ttot.max(1) as f32;
            l_sig[j] += cfg.rate_reg * (r_j - r_target);
        }
    }
```

- [ ] **Step 4: Run tests + build**

Run: `cargo test --lib -- bench::rsnn::tests::rate_reg_lateral_is_deterministic bench::rsnn::tests::temporal_xor_ff_is_near_chance bench::rsnn::tests::recurrence_does_not_yet_beat_ff_on_temporal_xor`
Then: `cargo build 2>&1 | grep -E "warning|error" || echo "build clean"`
Expected: PASS (determinism + the existing `train_xor` tests confirming `rate_reg = 0` identity); build clean.

- [ ] **Step 5: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: rate regularization in recurrent_update (level-0 lateral)"
```

---

### Task 3: Rate-reg term in `train_recurrent` (backward)

**Files:**
- Modify: `src/bench/rsnn.rs` (`train_recurrent`, tests)

**Interfaces:**
- Consumes: `RsnnConfig.rate_reg`, `rate_target_permille`; the `fired` table `train_recurrent` already
  builds per layer.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn rate_reg_backward_is_deterministic() {
    let mut cfg = RsnnConfig::demo();
    cfg.size = 16;
    cfg.layers = 4;
    cfg.back_count = 8;
    cfg.adapt_bump = 0;
    cfg.delay = 20;
    cfg.rate_reg = 5.0;
    cfg.rate_target_permille = 100;
    cfg.trials = 150;
    assert_eq!(train_recurrent(&cfg), train_recurrent(&cfg));
}
```

- [ ] **Step 2: Run — determinism of the (as-yet unchanged) path**

Run: `cargo test --lib bench::rsnn::tests::rate_reg_backward_is_deterministic`
Expected: PASS (meaningless until Step 3, like Task 2 Step 2).

- [ ] **Step 3: Add a per-layer rate table and the reg term in the `l_sig` closure**

In `train_recurrent`, the `l_sig` closure currently reads:

```rust
        let l_sig = |tz: usize, j: usize| -> f32 {
            (0..2)
                .map(|c| {
                    let bb = if tz == top { w[c][j] } else { dfa_weight(cfg.seed, (tz * ls + j) as u32, c) };
                    bb * err[c]
                })
                .sum()
        };
```

Replace it with a per-layer rate table (built from the `fired` table already in scope) plus the guarded reg
term in the closure:

```rust
        // per-(layer, neuron) firing rate for the rate regularizer (fraction of recorded waves j fired)
        let rate: Vec<Vec<f32>> = (0..l)
            .map(|z| {
                (0..ls)
                    .map(|j| (0..ttot).map(|tt| fired[z][tt][j]).sum::<f32>() / ttot.max(1) as f32)
                    .collect()
            })
            .collect();
        let r_target = cfg.rate_target_permille as f32 / 1000.0;
        // Firing-rate regularization: keep every recurrent layer near r_target so the loop stays alive
        // through the gap, carried by the same temporal eligibility. Guarded — rate_reg = 0 is byte-identical.
        let l_sig = |tz: usize, j: usize| -> f32 {
            let task: f32 = (0..2)
                .map(|c| {
                    let bb = if tz == top { w[c][j] } else { dfa_weight(cfg.seed, (tz * ls + j) as u32, c) };
                    bb * err[c]
                })
                .sum();
            task + if cfg.rate_reg != 0.0 { cfg.rate_reg * (rate[tz][j] - r_target) } else { 0.0 }
        };
```

- [ ] **Step 4: Run tests + build**

Run: `cargo test --lib -- bench::rsnn::tests::rate_reg_backward_is_deterministic bench::rsnn::tests::subthreshold_psi_is_deterministic`
Then: `cargo build 2>&1 | grep -E "warning|error" || echo "build clean"`
Expected: PASS (determinism + `subthreshold_psi_is_deterministic` confirming `rate_reg = 0` identity for
`train_recurrent`); build clean.

- [ ] **Step 5: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: rate regularization in train_recurrent (backward)"
```

---

### Task 4: Experiments — gap survival, lateral vs FF, backward vs FF

**Files:**
- Modify: `src/bench/rsnn.rs` (tests)

**Interfaces:**
- Consumes: `train_xor_inner` (Task 1), `xor_trial`, `train_xor`, `train_recurrent`.

- [ ] **Step 1: Add the three `#[ignore]`d experiments**

Add to the `tests` module:

```rust
#[test]
#[ignore] // expensive; run manually in --release
fn lateral_gap_survival() {
    // Does rate reg keep the level-0 recurrent loop (L1) alive through the 20-wave gap? Per-wave L1 spike
    // counts on a TRAINED net, reg off vs on. Off: activity dies in ~6 waves. On: should persist. Trial
    // phases (present 6, delay 20, present 6, read 6): present-A 0..6, GAP 6..26, present-B 26..32, read 32..38.
    for reg in [0.0f32, 5.0] {
        let mut c = RsnnConfig::demo();
        c.seed = 0xE9_0B_0A17;
        c.task_seed = 0xE9_0B_0A17;
        c.adapt_bump = 0;
        c.delay = 20;
        c.rec_count = 24;
        c.rec_radius = 2;
        c.rec_tau = 20.0;
        c.rec_init = 0;
        c.trials = 800;
        c.rate_reg = reg;
        c.rate_target_permille = 100;
        let (mut net, _w) = train_xor_inner(&c);
        let (_, waves) = xor_trial(&mut net, &c, 1, 0, 0);
        let per_wave: Vec<usize> = waves.iter().map(|wv| wv.len()).collect();
        eprintln!("rate_reg {reg}: L1 spikes/wave {per_wave:?}");
    }
}

#[test]
#[ignore] // expensive; run manually in --release
fn lateral_recurrence_vs_ff() {
    // Level-0 lateral recurrence + rate reg vs feed-forward on temporal XOR (LIF, delay 20), multi-seed.
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    let (mut best_ff, mut best_rec) = (0u64, 0u64);
    for &s in &seeds {
        let mut ff = RsnnConfig::demo();
        ff.seed = s;
        ff.task_seed = s;
        ff.adapt_bump = 0;
        ff.delay = 20;
        ff.trials = 1500;
        let mut rec = ff.clone();
        rec.rec_count = 24;
        rec.rec_radius = 2;
        rec.rec_tau = 20.0;
        rec.rec_init = 0;
        rec.rate_reg = 5.0;
        rec.rate_target_permille = 100;
        let fa = train_xor(&ff);
        let ra = train_xor(&rec);
        eprintln!("lateral seed {s:#x}  FF {fa}  +rec+reg {ra}");
        best_ff = best_ff.max(fa);
        best_rec = best_rec.max(ra);
    }
    eprintln!("lateral: best FF {best_ff}  best +rec+reg {best_rec}");
}

#[test]
#[ignore] // expensive; run manually in --release
fn backward_recurrence_vs_ff() {
    // Backward recurrence (level −1/−2) + rate reg vs feed-forward on temporal XOR (LIF, delay 20), on the
    // alive-LIF deep config (size 16, depth 4, up_count 32). Multi-seed. Compare to the lateral result.
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    let (mut best_ff, mut best_bw) = (0u64, 0u64);
    for &s in &seeds {
        let mut ff = RsnnConfig::demo();
        ff.seed = s;
        ff.task_seed = s;
        ff.size = 16;
        ff.layers = 4;
        ff.adapt_bump = 0;
        ff.delay = 20;
        ff.trials = 1500;
        ff.up_count = 32; // alive-LIF drive for the deep net
        ff.present_waves = 12;
        ff.base_q16 = 30000;
        let mut bw = ff.clone();
        bw.back_count = 8;
        bw.rate_reg = 5.0;
        bw.rate_target_permille = 100;
        let fa = train_recurrent(&ff);
        let ba = train_recurrent(&bw);
        eprintln!("backward seed {s:#x}  FF {fa}  +back+reg {ba}");
        best_ff = best_ff.max(fa);
        best_bw = best_bw.max(ba);
    }
    eprintln!("backward: best FF {best_ff}  best +back+reg {best_bw}");
}
```

- [ ] **Step 2: Verify they compile and are registered**

Run: `cargo test --lib --no-run 2>&1 | grep -E "warning|error" || echo "compile clean"`
Then: `cargo test --lib -- --ignored --list 2>&1 | grep -E "lateral_gap_survival|lateral_recurrence_vs_ff|backward_recurrence_vs_ff"`
Expected: compile clean; all three listed.

- [ ] **Step 3: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "test: rate-reg recurrence experiments (gap survival, lateral/backward vs FF)"
```

---

## Final verification (after Task 4)

- [ ] `cargo test --lib` — whole suite green (fast; ignored experiments skipped).
- [ ] `cargo build` — warning-free.
- [ ] `cargo test --lib -- --ignored --list` includes the three new experiments.
- [ ] Pre-existing recurrence tests still pass unchanged (identity of the `rate_reg = 0` default path):
      `temporal_xor_ff_is_near_chance`, `recurrence_does_not_yet_beat_ff_on_temporal_xor`,
      `subthreshold_psi_is_deterministic`.

## Running the experiments (manual, after the plan)

```bash
cargo test --release --lib lateral_gap_survival        -- --ignored --nocapture  # mechanism: loop alive in the gap?
cargo test --release --lib lateral_recurrence_vs_ff    -- --ignored --nocapture  # payoff: lateral beats FF?
cargo test --release --lib backward_recurrence_vs_ff   -- --ignored --nocapture  # backward, and vs lateral
```

Interpret per the spec's honesty gate: loop alive + XOR lifts ⇒ recurrence earns its keep; loop alive but
nulls ⇒ the **temporal credit rule** is the wall (→ BPTT), confound removed; loop dies even with reg ⇒
Approach A's eligibility gating starves the gap (→ Approach B). Consolidate into
`docs/experiments_results.md`.
