# wave_driven recurrence-validation suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (this repo mandates **inline** execution — never subagent-driven; see AGENTS.md). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Confirm — multi-seed, all-benchmark, matched-FF-baseline — that wave_driven's spike-ψ `εᵃ` side-car recurrence beats feed-forward, by porting the two missing tasks (distractor-XOR, flip-flop) and running a focused confirmation at the Phase-2b operating point.

**Architecture:** All test-only, in `src/bench/wave_driven_bench.rs`. Recover `task_distractor`/`task_flipflop` (git `9a39048`), gate them with a cheap `task_labels_correct` unit test, then add an `#[ignore]` experiment that trains FF vs side-car (size 32, rec 8, 3 seeds) across four tasks and reports worst+mean per task. Record the outcome in `docs/experiments_results.md`. No engine changes.

**Tech Stack:** Rust edition 2024, std only. Inline `#[cfg(test)]` tests; the confirmation is `#[ignore]` (`--release`).

## Global Constraints

- **Standard library only**; **warning-free** `cargo build`; `cargo test` green and fast (the confirmation stays `#[ignore]`). (AGENTS.md)
- **NEVER add a `Co-Authored-By` trailer.** Conventional commits. **One commit per task.** **NEVER push.**
- Fixed operating point: **size 32, rec_count 8, 3 seeds**; FF baseline is **5-layer** (depth-matched), membrane (`elig_beta 0`), best-checkpointed to its ceiling with a budget equal to the side-car's.
- Tasks + delays: temporal XOR `parity(2)` delay 8; parity N=4 `parity(4)` delay 8; distractor-XOR delay 20; flip-flop `n_ops 4` delay 12. `present 6, read 8` throughout.
- **No hard pass/fail on the FF-vs-side-car comparison** (a task the side-car doesn't win is a real result); only sanity gate = side-car clears chance on temporal XOR (worst > 700).
- Branch: `feat/wave-driven-recurrence-validation` (already checked out).

---

### Task 1: Recover the two task generators + a correctness unit test

**Files:**
- Modify: `src/bench/wave_driven_bench.rs` (add to the `#[cfg(test)] mod tests`)

**Interfaces:**
- Produces: `task_distractor(seed: u64, t: usize) -> (Vec<usize>, usize)`, `task_flipflop(seed: u64, t: usize, n_ops: usize) -> (Vec<usize>, usize)`. (`task_parity` already exists.)

- [ ] **Step 1: Write the failing unit test.** Add to the `tests` module (near `task_parity`):

```rust
#[test]
fn task_labels_correct() {
    for trial in 0..25 {
        let (bits, label) = task_parity(42, trial, 4);
        assert_eq!(bits.len(), 4);
        assert!(bits.iter().all(|&b| b <= 1));
        assert_eq!(label, bits.iter().fold(0, |a, &b| a ^ b), "parity label is the XOR of the bits");

        let (classes, dlabel) = task_distractor(42, trial);
        assert_eq!(classes.len(), 3);
        assert_eq!(classes[1], 2, "middle cue is the class-2 distractor");
        assert!(classes[0] <= 1 && classes[2] <= 1);
        assert_eq!(dlabel, classes[0] ^ classes[2], "distractor label ignores the middle cue");

        let (ops, flabel) = task_flipflop(42, trial, 4);
        assert_eq!(ops.len(), 4);
        assert!(ops.iter().all(|&o| o <= 1));
        let last = *ops.last().unwrap();
        assert_eq!(flabel, if last == 0 { 1 } else { 0 }, "flip-flop label is the final state");
    }
}
```

- [ ] **Step 2: Run to verify failure.**

Run: `cargo test wave_driven_bench::tests::task_labels_correct`
Expected: FAIL to compile (`task_distractor` / `task_flipflop` not found).

- [ ] **Step 3: Add the two generators** (verbatim from git `9a39048`), next to `task_parity`:

```rust
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
    (ops, if last == 0 { 1 } else { 0 })
}
```

- [ ] **Step 4: Run the test.**

Run: `cargo test wave_driven_bench::tests::task_labels_correct`
Expected: PASS. `cargo build --tests` warning-free (both generators are used by the test now; the experiment in Task 2 uses them too).

- [ ] **Step 5: Commit.**

```bash
git add src/bench/wave_driven_bench.rs
git commit -m "test(wave_driven): recover distractor-XOR + flip-flop task generators (git 9a39048)"
```

---

### Task 2: The `#[ignore]` recurrence-confirmation experiment

**Files:**
- Modify: `src/bench/wave_driven_bench.rs`

**Interfaces:**
- Consumes: `task_parity`, `task_distractor`, `task_flipflop`, `make_ff`, `make_sidecar`, `ff_cfg`, `train_and_eval_best`, `rate_profile` (all already in the module).

- [ ] **Step 1: Add the experiment** after `wave_driven_sidecar_vs_ff` in the `tests` module:

```rust
#[test]
#[ignore] // validation: multi-seed, all-benchmark, matched-FF-baseline recurrence confirmation (--release, ~tens of min)
fn wave_driven_recurrence_confirmation() {
    let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
    let size = 32u32;
    let rec = 8u32;
    eprintln!("== wave_driven recurrence confirmation — size {size}, rec {rec}, spike-ψ εᵃ (β=0.4), {} seeds ==", seeds.len());
    eprintln!("   {:<15} | FF w/mean | side-car w/mean | σ", "task");

    struct B {
        name: &'static str,
        present: usize,
        delay: usize,
        read: usize,
        task: Box<dyn Fn(u64, usize) -> (Vec<usize>, usize)>,
    }
    let benches: Vec<B> = vec![
        B { name: "temporal-XOR", present: 6, delay: 8, read: 8, task: Box::new(|s, t| task_parity(s, t, 2)) },
        B { name: "parity-4", present: 6, delay: 8, read: 8, task: Box::new(|s, t| task_parity(s, t, 4)) },
        B { name: "distractor-XOR", present: 6, delay: 20, read: 8, task: Box::new(|s, t| task_distractor(s, t)) },
        B { name: "flip-flop", present: 6, delay: 12, read: 8, task: Box::new(|s, t| task_flipflop(s, t, 4)) },
    ];

    let mut sidecar_xor_worst = 0u64;
    let mut beats = 0usize;
    for b in &benches {
        let mkcfg = || {
            let mut c = ff_cfg();
            c.size = size;
            c.present = b.present;
            c.delay = b.delay;
            c.read = b.read;
            c.holdout = 200;
            c
        };
        let (mut ff_bests, mut sc_bests, mut sigmas) = (Vec::new(), Vec::new(), Vec::new());
        for &s in &seeds {
            // FF baseline (5-layer, membrane β=0), best-checkpointed to its ceiling
            let (mut ffn, fe) = make_ff(s, size, 5, 32, 3, 5, 6);
            let (fb, _) = train_and_eval_best(&mut ffn, &fe, s, s, &mkcfg(), b.task.as_ref(), 300, 3, 2400);
            ff_bests.push(fb);
            // side-car (rec 8, spike-ψ εᵃ β=0.4), same budget
            let (mut scn, se) = make_sidecar(s, size, 32, 3, rec, 4, 5, 6);
            let (sb, _) = train_and_eval_best(&mut scn, &se, s, s, &mkcfg(), b.task.as_ref(), 300, 3, 2400);
            sc_bests.push(sb);
            let (_p, sigma) = rate_profile(&mut scn, size, s, 0, 16, 64);
            sigmas.push(sigma);
        }
        let ffw = *ff_bests.iter().min().unwrap();
        let ffm = ff_bests.iter().sum::<u64>() / ff_bests.len() as u64;
        let scw = *sc_bests.iter().min().unwrap();
        let scm = sc_bests.iter().sum::<u64>() / sc_bests.len() as u64;
        let sig = sigmas.iter().sum::<f64>() / sigmas.len() as f64;
        eprintln!("   {:<15} | {ffw:>4}/{ffm:<4} | {scw:>4}/{scm:<4}      | {sig:.2}", b.name);
        if scw >= ffw {
            beats += 1;
        }
        if b.name == "temporal-XOR" {
            sidecar_xor_worst = scw;
        }
    }
    eprintln!("== recurrence beats FF (worst-seed) on {beats}/{} tasks ==", benches.len());
    // plumbing sanity gate only: the side-car demonstrably solves temporal XOR
    assert!(sidecar_xor_worst > 700, "side-car should clear chance on temporal XOR (worst {sidecar_xor_worst}); harness broken?");
}
```

> `train_and_eval_best` takes `task: impl Fn(u64,usize)->(Vec<usize>,usize)`; `b.task.as_ref()` is a `&dyn Fn(...)`, which implements `Fn`, so it passes cleanly and is reusable across seeds.

- [ ] **Step 2: Verify it compiles warning-free.**

Run: `cargo build --tests 2>&1 | grep -iE "warning|error" || echo clean`
Expected: `clean`.

- [ ] **Step 3: Commit.**

```bash
git add src/bench/wave_driven_bench.rs
git commit -m "test(wave_driven): recurrence confirmation experiment (4 tasks, 3 seeds, matched FF baseline)"
```

---

### Task 3: Run the confirmation + record the outcome

**Files:**
- Modify: `docs/experiments_results.md`

- [ ] **Step 1: Run the confirmation** (long; `--release`). Capture the output:

Run: `cargo test --release wave_driven_recurrence_confirmation -- --ignored --nocapture`
Expected: prints the per-task `FF worst/mean | side-car worst/mean | σ` table and the `beats FF on N/4` summary; the temporal-XOR sanity assert passes. If it panics on that assert, the harness is broken — fix before recording.

- [ ] **Step 2: Update `docs/experiments_results.md`.** In the "Reproduced on `wave_driven` with spike-ψ εᵃ" subsection (added Phase 2b), replace the *2-seed / 2-task* caveat and its table with the **3-seed, 4-benchmark, matched-baseline** results from Step 1: a table of `task | FF worst/mean | side-car worst/mean`, the `beats FF on N/4` verdict, and an honest one-line conclusion (e.g. "confirmed across all four benchmarks, 3 seeds, matched depth-5 FF baseline at size 32" — or, if a task does not beat FF, state exactly which and note it for the width/bump-ψ follow-up). Keep the σ/operating-point notes from Phase 2b.

- [ ] **Step 3: Commit.**

```bash
git add docs/experiments_results.md
git commit -m "docs(experiments): 3-seed all-benchmark matched-baseline recurrence confirmation results"
```

---

## Self-Review

**Spec coverage:** Task generators (distractor + flip-flop) + `task_labels_correct` gate → Task 1. The `#[ignore]` confirmation experiment (size 32, rec 8, 3 seeds, 4 tasks with historical delays, FF depth-matched + best-checkpointed, worst+mean + σ per task, N/4 summary, temporal-XOR sanity gate) → Task 2. Record outcome in `experiments_results.md` → Task 3. Matched-baseline (5-layer FF, equal budget, best-checkpoint) → Task 2 (`make_ff(.., 5, ..)` + same `train_and_eval_best` args). No hard comparison assert; only the plumbing gate → Task 2. Non-goals (perf suite, bump-ψ, width/rec sweep, engine changes) → none built. **All spec sections map to a task.**

**Placeholder scan:** No "TBD/TODO". Task 3's doc edit describes exact content to write from real Step-1 output (the numbers are produced by the run, not invented) — not a placeholder.

**Type consistency:** `task_distractor(seed,trial)`, `task_flipflop(seed,trial,n_ops)` match their call sites (Tasks 1–2). `B.task: Box<dyn Fn(u64,usize)->(Vec<usize>,usize)>` passed as `b.task.as_ref()` to `train_and_eval_best(..., task: impl Fn(u64,usize)->(Vec<usize>,usize), ...)` — signature verified against the existing `wave_driven_sidecar_vs_ff` call. `make_ff(seed,size,layers,up_count,up_radius,adapt_bump,adapt_decay)` and `make_sidecar(seed,size,uc,ur,n,r,adapt_bump,adapt_decay)` match their definitions. `rate_profile(net,size,task_seed,class,warmup,waves) -> (Vec<f64>, f64)` matches.

**Known follow-ups (out of scope):** if a task doesn't beat FF, the width bump / bump-ψ fast-follow (later session); the performance-benchmark suite (later session).
