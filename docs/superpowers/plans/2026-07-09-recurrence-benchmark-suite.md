# Recurrence Benchmark Suite — Implementation Plan

> **For agentic workers:** Per this repo's `AGENTS.md`, execute this plan **inline and autonomously**
> (REQUIRED SUB-SKILL: superpowers:executing-plans) — **never** the subagent-driven option. Steps use
> checkbox (`- [ ]`) syntax. One commit per task.

**Goal:** Add three benchmarks that need recurrent *computation* (parity, distractor-XOR, flip-flop) via a
general N-cue sequence runner + a task-parameterized trainer, so the recurrence conclusion rests on more
than temporal XOR.

**Architecture:** `sequence_trial` generalizes `xor_trial` to an N-cue sequence (gap between cues, seed
`trial·n + pos` so the 2-cue case is byte-identical to `xor_trial`). Task generators map `(seed, trial)` to
`(classes, label)`. `train_sequence` reuses the `train_xor` readout + `recurrent_update` loop, parameterized
by a task closure. ALIF on, rate reg off, FF (`rec_count 0`) vs +lateral-recurrence. **No engine change.**

**Tech Stack:** Rust edition 2024, std-only, deterministic. All work in `src/bench/rsnn.rs`.

## Global Constraints

- **std-only**, **no `unsafe`**, **warning-free build**.
- **Determinism** — pure function of `(seed, task_seed, config)`.
- **`wave_state_machine` frozen**; **no engine change**. `xor_trial` and the existing XOR/recurrence
  experiments **stay byte-identical** (new code paths only) — the full suite stays green.
- Inline `#[cfg(test)]` tests, TDD. Expensive experiments are **`#[ignore]`d** ("run manually in `--release`").
- **One commit per task**, conventional commits. **NEVER add a `Co-Authored-By` trailer.**
- Branch `recurrence-benchmarks` (already created). **Never push.**

## File Structure

- `src/bench/rsnn.rs` — everything:
  - Task 1: `sequence_trial` (generalizes `xor_trial`).
  - Task 2: `task_parity` / `task_distractor` / `task_flipflop`.
  - Task 3: `train_sequence` (task-parameterized trainer).
  - Task 4: the three `#[ignore]`d experiments.

---

### Task 1: `sequence_trial` — general N-cue sequence runner

**Files:** Modify `src/bench/rsnn.rs` (new `sequence_trial`, test).

**Interfaces:**
- Produces: `fn sequence_trial(net: &mut Network, cfg: &RsnnConfig, classes: &[usize], trial: usize) ->
  (Vec<f32>, Vec<Vec<u32>>)` — reset → for each class present `cue_realization` for `present_waves` with a
  `delay` gap *between* cues → `read_waves` silent → (read-window L1 counts, per-wave L1 fired-sets). The
  `classes = [a, b]` case is byte-identical to `xor_trial(a, b)`.
- Consumes: `cue_realization`, `Network`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn sequence_trial_matches_xor_on_two_cues() {
    // The 2-cue sequence must reproduce xor_trial exactly (same gap structure, same per-cue seed scheme).
    let mut cfg = RsnnConfig::demo();
    cfg.delay = 20;
    cfg.rec_count = 0;
    let build = || {
        let mut net = Network::new(cfg.engine_config_xor());
        net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
        net
    };
    let mut n1 = build();
    let mut n2 = build();
    let (a1, w1) = xor_trial(&mut n1, &cfg, 1, 0, 3);
    let (a2, w2) = sequence_trial(&mut n2, &cfg, &[1, 0], 3);
    assert_eq!(a1, a2, "read-window activity matches xor_trial");
    assert_eq!(w1, w2, "per-wave fired-sets match xor_trial");
}
```

- [ ] **Step 2: Run — expect compile failure**

Run: `cargo test --lib bench::rsnn::tests::sequence_trial_matches_xor_on_two_cues`
Expected: FAIL — `cannot find function sequence_trial`.

- [ ] **Step 3: Implement `sequence_trial`** (place it right after `xor_trial`)

```rust
/// reset → for each class in `classes`: (a `delay` gap before every cue except the first) present
/// cue(class) for `present_waves` → `read_waves` silent. Records L1 per-wave fired-sets; returns
/// (read-window L1 spike counts, per-wave fired-sets). Generalizes `xor_trial`: `classes = [a, b]` with the
/// per-cue seed `trial·n + pos` reproduces `xor_trial(a, b)` exactly.
fn sequence_trial(net: &mut Network, cfg: &RsnnConfig, classes: &[usize], trial: usize) -> (Vec<f32>, Vec<Vec<u32>>) {
    let ls = (cfg.size * cfg.size) as usize;
    let n = classes.len();
    let rec: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let r = rec.clone();
        net.on_layer(1, Box::new(move |_w, fired: &[u32]| r.lock().unwrap().push(fired.to_vec())));
    }
    net.reset_state();
    for (pos, &class) in classes.iter().enumerate() {
        if pos > 0 {
            for _ in 0..cfg.delay {
                net.wave(&[]);
            }
        }
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * n + pos, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
        }
    }
    let read_start = rec.lock().unwrap().len();
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
    }
    net.clear_listeners();
    let waves = rec.lock().unwrap().clone();
    let mut act = vec![0f32; ls];
    for wave in waves.iter().skip(read_start) {
        for &loc in wave {
            act[loc as usize] += 1.0;
        }
    }
    (act, waves)
}
```

Note: `xor_trial` uses `trial * 2 + phase`; with `n = 2` this equals `trial * n + pos`, and the gap sits
between the two cues in both — so the outputs match.

- [ ] **Step 4: Run test — expect PASS**

Run: `cargo test --lib bench::rsnn::tests::sequence_trial_matches_xor_on_two_cues`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: sequence_trial — general N-cue sequence runner (xor_trial generalized)"
```

---

### Task 2: Task generators — parity, distractor-XOR, flip-flop

**Files:** Modify `src/bench/rsnn.rs` (three fns, tests).

**Interfaces:**
- Produces: `fn task_parity(seed: u64, trial: usize, n: usize) -> (Vec<usize>, usize)`;
  `fn task_distractor(seed: u64, trial: usize) -> (Vec<usize>, usize)`;
  `fn task_flipflop(seed: u64, trial: usize, n_ops: usize) -> (Vec<usize>, usize)`.
- Consumes: `mix`, `key` (already imported).

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module:

```rust
#[test]
fn task_labels_are_correct() {
    for trial in 0..25 {
        let (bits, label) = task_parity(42, trial, 4);
        assert_eq!(bits.len(), 4);
        assert!(bits.iter().all(|&b| b <= 1));
        assert_eq!(label, bits.iter().fold(0, |a, &b| a ^ b), "parity label is the XOR of the bits");

        let (classes, dlabel) = task_distractor(42, trial);
        assert_eq!(classes.len(), 3);
        assert_eq!(classes[1], 2, "middle cue is the label-irrelevant distractor class 2");
        assert_eq!(dlabel, classes[0] ^ classes[2], "distractor label is a XOR b, ignoring D");

        let (ops, flabel) = task_flipflop(42, trial, 3);
        assert_eq!(ops.len(), 3);
        assert!(ops.iter().all(|&o| o <= 1));
        let last = *ops.last().unwrap();
        assert_eq!(flabel, if last == 0 { 1 } else { 0 }, "state = set(0)->on(1), reset(1)->off(0)");
    }
}
```

- [ ] **Step 2: Run — expect compile failure**

Run: `cargo test --lib bench::rsnn::tests::task_labels_are_correct`
Expected: FAIL — `cannot find function task_parity`.

- [ ] **Step 3: Implement the three generators** (place them near `pick_ab`)

```rust
/// `n` deterministic bits from `(seed, trial)`; label = their XOR (parity — non-monotone, needs recurrence).
fn task_parity(seed: u64, trial: usize, n: usize) -> (Vec<usize>, usize) {
    let bits: Vec<usize> = (0..n).map(|i| (mix(key(seed, trial as u32, 0, i as u32, 51)) & 1) as usize).collect();
    let label = bits.iter().fold(0usize, |acc, &b| acc ^ b);
    (bits, label)
}

/// `[a, distractor, b]` where the middle is a label-irrelevant cue (class 2); label = a XOR b (ignore D).
fn task_distractor(seed: u64, trial: usize) -> (Vec<usize>, usize) {
    let a = (mix(key(seed, trial as u32, 0, 0, 51)) & 1) as usize;
    let b = (mix(key(seed, trial as u32, 0, 0, 53)) & 1) as usize;
    (vec![a, 2, b], a ^ b)
}

/// `n_ops` set(class 0)/reset(class 1) ops; label = final state (set -> on 1, reset -> off 0).
fn task_flipflop(seed: u64, trial: usize, n_ops: usize) -> (Vec<usize>, usize) {
    let ops: Vec<usize> = (0..n_ops).map(|i| (mix(key(seed, trial as u32, 0, i as u32, 57)) & 1) as usize).collect();
    let last = *ops.last().unwrap();
    (ops.clone(), if last == 0 { 1 } else { 0 })
}
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cargo test --lib bench::rsnn::tests::task_labels_are_correct`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: parity / distractor-XOR / flip-flop task generators"
```

---

### Task 3: `train_sequence` — task-parameterized trainer

**Files:** Modify `src/bench/rsnn.rs` (new `train_sequence`, test).

**Interfaces:**
- Produces: `pub fn train_sequence(cfg: &RsnnConfig, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) ->
  u64` — build `engine_config_xor` (level-0 recurrence when `rec_count > 0`), calibrate once as a sensible
  init, train (readout delta-rule on the read window + `recurrent_update` when `rec_count > 0`), held-out
  over disjoint trials. FF = `rec_count 0`; recurrence = `rec_count > 0`.
- Consumes: `sequence_trial` (Task 1), `recurrent_update`, `softmax`, `engine_config_xor`.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module:

```rust
#[test]
fn train_sequence_is_deterministic() {
    let mut cfg = RsnnConfig::demo();
    cfg.delay = 8;
    cfg.rec_count = 8;
    cfg.rec_init = 0;
    cfg.trials = 120;
    let run = || train_sequence(&cfg, |seed, t| task_parity(seed, t, 3));
    assert_eq!(run(), run());
}
```

- [ ] **Step 2: Run — expect compile failure**

Run: `cargo test --lib bench::rsnn::tests::train_sequence_is_deterministic`
Expected: FAIL — `cannot find function train_sequence`.

- [ ] **Step 3: Implement `train_sequence`** (place it after `train_xor`)

```rust
/// Train a readout (+ level-0 recurrent weights when `rec_count > 0`) on an arbitrary sequence task, given
/// by a `task(task_seed, trial) -> (cue-class sequence, binary label)` closure. Returns held-out permille.
/// Calibration is a one-time sensible init; ALIF and rate reg come from `cfg`.
pub fn train_sequence(cfg: &RsnnConfig, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) -> u64 {
    let mut net = Network::new(cfg.engine_config_xor());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let ls = (cfg.size * cfg.size) as usize;
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    for t in 0..cfg.trials {
        let (classes, label) = task(cfg.task_seed, t);
        let (act, waves) = sequence_trial(&mut net, cfg, &classes, t);
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
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (classes, label) = task(cfg.task_seed, t);
        let (act, _) = sequence_trial(&mut net, cfg, &classes, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == label {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}
```

- [ ] **Step 4: Run test + build**

Run: `cargo test --lib bench::rsnn::tests::train_sequence_is_deterministic`
Then: `cargo build 2>&1 | grep -E "warning|error" || echo "build clean"`
Expected: PASS; build clean.

- [ ] **Step 5: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: train_sequence — task-parameterized recurrence trainer"
```

---

### Task 4: The three benchmark experiments

**Files:** Modify `src/bench/rsnn.rs` (tests).

**Interfaces:** Consumes `train_sequence`, `task_parity`, `task_distractor`, `task_flipflop`.

- [ ] **Step 1: Add the three `#[ignore]`d experiments**

Add to the `tests` module:

```rust
#[test]
#[ignore] // expensive; run manually in --release
fn parity_recurrence_sweep() {
    // Sequential parity (non-monotone -> adaptation can't fake it). ALIF, FF vs +lateral-recurrence, over N.
    // Expect FF to solve N=2 (that's XOR) and fail N>=3; does recurrence rescue it?
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    for n in [2usize, 3, 4, 5] {
        let (mut worst_ff, mut worst_rec) = (1000u64, 1000u64);
        for &s in &seeds {
            let mut ff = RsnnConfig::demo(); // ALIF on (adapt_bump 20), rate_reg off
            ff.seed = s;
            ff.task_seed = s;
            ff.delay = 8;
            ff.trials = 1500;
            let mut rec = ff.clone();
            rec.rec_count = 24;
            rec.rec_radius = 2;
            rec.rec_tau = 20.0;
            rec.rec_init = 0;
            let fa = train_sequence(&ff, |seed, t| task_parity(seed, t, n));
            let ra = train_sequence(&rec, |seed, t| task_parity(seed, t, n));
            eprintln!("parity N={n} seed {s:#x}  FF {fa}  +rec {ra}");
            worst_ff = worst_ff.min(fa);
            worst_rec = worst_rec.min(ra);
        }
        eprintln!("parity N={n}: WORST FF {worst_ff}  +rec {worst_rec}");
    }
}

#[test]
#[ignore] // expensive; run manually in --release
fn distractor_xor_recurrence() {
    // Delayed XOR with an irrelevant distractor cue between A and B. ALIF, FF vs +recurrence, multi-seed.
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    for &s in &seeds {
        let mut ff = RsnnConfig::demo();
        ff.seed = s;
        ff.task_seed = s;
        ff.delay = 20;
        ff.trials = 1500;
        let mut rec = ff.clone();
        rec.rec_count = 24;
        rec.rec_radius = 2;
        rec.rec_tau = 20.0;
        rec.rec_init = 0;
        let fa = train_sequence(&ff, task_distractor);
        let ra = train_sequence(&rec, task_distractor);
        eprintln!("distractor-XOR seed {s:#x}  FF {fa}  +rec {ra}");
    }
}

#[test]
#[ignore] // expensive; run manually in --release
fn flipflop_recurrence() {
    // Set/reset flip-flop, state held across the read gap. ALIF, FF vs +recurrence, multi-seed.
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    for &s in &seeds {
        let mut ff = RsnnConfig::demo();
        ff.seed = s;
        ff.task_seed = s;
        ff.delay = 12;
        ff.read_waves = 12; // read after a gap so the state must be held
        ff.trials = 1500;
        let mut rec = ff.clone();
        rec.rec_count = 24;
        rec.rec_radius = 2;
        rec.rec_tau = 20.0;
        rec.rec_init = 0;
        let fa = train_sequence(&ff, |seed, t| task_flipflop(seed, t, 3));
        let ra = train_sequence(&rec, |seed, t| task_flipflop(seed, t, 3));
        eprintln!("flip-flop seed {s:#x}  FF {fa}  +rec {ra}");
    }
}
```

- [ ] **Step 2: Verify compile + registration**

Run: `cargo test --lib --no-run 2>&1 | grep -E "warning|error" || echo "compile clean"`
Then: `cargo test --lib -- --ignored --list 2>&1 | grep -E "parity_recurrence_sweep|distractor_xor_recurrence|flipflop_recurrence"`
Expected: compile clean; all three listed.

- [ ] **Step 3: Commit**

```bash
git add src/bench/rsnn.rs
git commit -m "test: recurrence benchmark experiments (parity sweep, distractor-XOR, flip-flop)"
```

---

## Final verification (after Task 4)

- [ ] `cargo test --lib` — whole suite green (fast; ignored experiments skipped).
- [ ] `cargo build` — warning-free.
- [ ] `cargo test --lib -- --ignored --list` includes the three new experiments.
- [ ] `sequence_trial_matches_xor_on_two_cues` passes — the generalization is faithful, so the existing XOR
      experiments (which still call `xor_trial`) are unaffected.

## Running the experiments (manual, after the plan)

```bash
cargo test --release --lib parity_recurrence_sweep    -- --ignored --nocapture
cargo test --release --lib distractor_xor_recurrence  -- --ignored --nocapture
cargo test --release --lib flipflop_recurrence        -- --ignored --nocapture
```

Interpret per the spec's honesty gate, across tasks and N: FF fails + rec lifts ⇒ recurrence earns its keep;
FF already solves ⇒ task didn't need recurrence; both fail ⇒ temporal credit rule is the wall (→ BPTT) on a
task that genuinely needs recurrence. Consolidate the cross-task pattern into `docs/experiments_results.md`.
