# wave_resonate Phase 3 — experiment suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Checkbox steps.
> Experiments are `#[ignore]`d benchmark tests (run in `--release`); findings → `docs/experiments_results.md`.

**Goal:** Build the BRF+HYPR temporal-task experiment harness (tasks + side-car topology), validate it at
size 16, run the size-32 study (FF vs side-car, frozen vs trained ω/b′, ω/δ sweep) in the background, and
consolidate findings.

**Architecture:** Extend `src/bench/wave_resonate_bench.rs` — port the engine-agnostic task generators and
a BRF `make_sidecar`, add a `rate_profile` σ, and the comparison experiments. Reuse the existing
`run_trial` / `build_signal` / `train_and_eval_best` (already wired for `train_omega_b`).

## Global Constraints

- Std only; warning-free; determinism preserved.
- **AGENTS.md benchmark contract:** sweep every axis (task, topology, ω-mode, ω-init×δ), ≥3 seeds, report
  **worst + mean**, read the top spiking layer, report fan-in density / σ / per-layer spiking profile /
  held-out accuracy. **No silent caps** — `log`/`eprintln` anything dropped.
- **BRF operating point:** θ_c 0.1, eps_cut 1e-6, hidden_lr ~2, omega_b_lr ~2.
- One commit per task; conventional commits; **no `Co-Authored-By` trailer**.

---

### Task 1: Port the temporal-task generators + guard test

**Files:** Modify `src/bench/wave_resonate_bench.rs`.

- [ ] **Step 1:** Port `task_parity(seed,t,n)`, `task_distractor(seed,t)`, `task_flipflop(seed,t,n_ops)`
  verbatim from `wave_driven_bench.rs` (they use only `mix`/`key`, already imported).
- [ ] **Step 2:** Port `task_labels_correct` (the generator guard). Run: `cargo test wave_resonate_bench::tests::task_labels_correct`. Expected PASS.
- [ ] **Step 3: Commit** `test(wave_resonate): port temporal-task generators (parity/distractor/flip-flop)`.

---

### Task 2: BRF side-car topology (`make_sidecar`) + σ in `rate_profile`

**Files:** Modify `src/bench/wave_resonate_bench.rs`.

**Interfaces:** `make_sidecar(seed,size,uc,ur,n,r,theta_c,b_off,train_omega_b) -> (Network, Vec<Vec<Edge>>)`
mirroring the wave_driven layout (`L0→L1(+1)`, `L1→L3(+2)`, `L2 self(0)+L2→L3(+1)`, `L3→L2(−1)+L3→L4(+1)`,
`L4` read). Extend `rate_profile` to also return a coarse σ (mean consecutive-layer spike ratio over
`1..l-1`), as in `wave_driven_bench`.

- [ ] **Step 1: Write the failing test** — side-car builds, trains, and is live:

```rust
    #[test]
    #[ignore] // smoke (size 16): side-car builds + is live + trains above chance (--release --nocapture)
    fn wave_resonate_sidecar_smoke_size16() {
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_sidecar(seed, 16, 32, 3, 8, 4, 0.1, (0.0, 0.2), false);
        let (pct, sigma) = rate_profile(&mut net, 16, seed, 0, 16, 48);
        eprintln!("side-car size16 rate% {pct:?} σ≈{sigma:.2}");
        assert!(pct.iter().skip(1).any(|&r| r > 0.5), "side-car compute layers must be live: {pct:?}");
        let mut cfg = ff_cfg();
        cfg.size = 16; cfg.present = 6; cfg.delay = 8; cfg.read = 8;
        let task = |s: u64, t: usize| task_parity(s, t, 2);
        let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, task, 100, 4, 1500);
        eprintln!("side-car size16 temporal-XOR: best {best}@{at}");
        assert!(best > 600, "side-car should clear chance on temporal XOR: {best}");
    }
```

- [ ] **Step 2: Run to verify failure** — FAIL (`make_sidecar` missing).
- [ ] **Step 3: Implement** `make_sidecar` (port `wave_driven_bench::make_sidecar`, swapping the LIF
  `LayerConfig` fields for BRF `inhibitor_ratio/omega_init/b_offset_init/tau_out`, `set_elig_params` with
  `train_omega_b`, `enable_training`, and the matching `entries`). Extend `rate_profile` to return `(Vec<f64>, f64)`
  (add σ); update the existing `wave_resonate_liveness_probe` caller to take `.0`.
- [ ] **Step 4: Run** `cargo test --release wave_resonate_sidecar_smoke_size16 -- --ignored --nocapture`.
  If a compute layer is silent, first bump recurrent/forward fan-in (AGENTS.md liveness rule) or adjust θ_c;
  if it clears chance, proceed.
- [ ] **Step 5: Commit** `test(wave_resonate): BRF side-car topology + σ profile (size-16 smoke)`.

---

### Task 3: The size-16 validation experiment (fast, catches bugs)

**Files:** Modify `src/bench/wave_resonate_bench.rs`.

- [ ] **Step 1:** Add an `#[ignore]` experiment `wave_resonate_temporal_size16` that, for each of the 4
  tasks, at size 16, ≥2 seeds, reports FF (frozen), FF (trained ω/b′), side-car (frozen) — worst+mean +
  side-car σ. No hard assertion beyond a plumbing gate (temporal-XOR side-car worst > chance). Windows per
  task mirror `wave_driven_recurrence_confirmation` (XOR/parity4 present6/delay8/read8; distractor delay20;
  flip-flop delay12).
- [ ] **Step 2: Run** `--release --ignored --nocapture`; capture the table. Sanity-check the numbers
  (nonzero, not all-chance, σ finite). Tune `hidden_lr`/`omega_b_lr`/fan-in if a config is dead.
- [ ] **Step 3: Commit** `test(wave_resonate): size-16 temporal validation experiment (FF/side-car, frozen/trained)`.

---

### Task 4: The size-32 study (background) + consolidate findings

**Files:** Modify `src/bench/wave_resonate_bench.rs`; modify `docs/experiments_results.md`.

- [ ] **Step 1:** Add `wave_resonate_temporal_size32` (`#[ignore]`): the full matrix — 4 tasks × 3 seeds ×
  {FF frozen, FF trained, side-car frozen, side-car trained} at size 32 (`r3/c32` forward, side-car `rec 8`),
  worst+mean + σ + per-layer spiking profile per config. Plus `wave_resonate_omega_delta_sweep`
  (`#[ignore]`): temporal-XOR, sweep `ω`-init ∈ {(3,6),(5,10),(10,20)} × `δ` ∈ {0.02,0.05,0.1} (respecting
  δω≤1), report liveness + best.
- [ ] **Step 2: Launch in background** (`cargo test --release wave_resonate_temporal_size32 -- --ignored
  --nocapture` and the sweep) since each is many minutes. Poll for completion; capture the readouts.
- [ ] **Step 3:** If it does not finish in a reasonable time, record the perf wall explicitly (what size /
  how many seeds completed), fall back to size 16 / 2 seeds, and note it — no silent truncation.
- [ ] **Step 4:** Write a `wave_resonate` section in `docs/experiments_results.md`: the worst+mean tables,
  σ/spiking profiles, the **frozen-vs-trained ω/b′ verdict** and the **FF-vs-side-car verdict**,
  contextualized against the ALIF numbers (temporal XOR, flip-flop, distractor-XOR, parity-4).
- [ ] **Step 5: Commit** `docs(experiments): wave_resonate BRF+HYPR temporal results (FF/side-car, ω/b′)`.

---

### Task 5: Full suite green + docs

- [ ] **Step 1:** `cargo test` green (the non-ignored suite unaffected); `cargo build` warning-free.
- [ ] **Step 2:** AGENTS.md: note Phase 3 experiments landed + point to the results section.
- [ ] **Step 3: Commit** `docs(wave_resonate): register Phase 3 experiments`.

---

## Self-Review

**Spec coverage:** tasks (Task 1); side-car + σ (Task 2); RQ1/RQ2 FF frozen-vs-trained + RQ3 FF-vs-side-car
(Tasks 3–4); RQ4 ω/δ sweep (Task 4); worst+mean/σ/profile reporting + experiments_results.md consolidation
(Task 4); runtime-aware validate-at-16-then-run-32 + perf-wall honesty (Tasks 3–4). ✓
**Placeholder scan:** hyperparameters have start values with an explicit tune step; no shipped TBDs. Task 4
runtimes are genuinely unknown → handled by the background-run + fallback step, not a placeholder. ✓
**Type consistency:** `make_sidecar(...) -> (Network, Vec<Vec<Edge>>)` matches `make_ff`'s shape and the
`train_and_eval_best(net, entries, seed, task_seed, cfg, task, eval_every, patience, max_trials)` signature;
`rate_profile -> (Vec<f64>, f64)` (σ added) with its one existing caller updated. ✓
