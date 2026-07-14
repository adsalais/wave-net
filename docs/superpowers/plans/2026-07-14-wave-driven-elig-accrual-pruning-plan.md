# Plan — wave_driven εᵃ accrual pruning + unchecked indexing

Spec: `docs/superpowers/specs/2026-07-14-wave-driven-elig-accrual-pruning-design.md`.
One commit per task. Keep `cargo test` green and the build warning-free throughout.

## Task 0 — land the profiling harness (the instrumentation that motivated this)

- Commit `examples/profile_driven_train.rs` (already written): tight-loop wave_driven training trial with
  coarse phase timers.
- `commit: test(wave_driven): perf harness for the online training trial`

## Task 1 — spec + plan docs

- This spec + plan. `commit: docs(wave_driven): spec+plan for εᵃ accrual pruning`

## Task 2 — Change 1: prune `elig_active` (TDD)

1. Add `#[cfg(test)] pub(crate) fn elig_active_len(&self, z) -> usize` on `Network` (test scaffolding,
   next to `seed_worksets_test`).
2. **RED** — `elig_active_prunes_dead_rows` in `training.rs` tests: `adapt_decay=2`, drive active then a
   silent tail; assert `elig_active_len(mid) == 0` and that the layer actually fired. Run → watch it fail
   (current monotonic set stays populated).
3. Add `online_equals_dense_with_eps_a_pruning` (bit-exact guard) — same config; assert online.elig ==
   dense.elig. Confirm it passes on current code (correctness baseline).
4. **GREEN** — restructure the εᵃ branch of `accrue_eligibility` to take-and-re-push `elig_active`,
   re-pushing a source iff `pretr[i] != 0.0 || any_ea_alive`. Run both new tests + full suite → green.
5. `commit: perf(wave_driven): prune elig_active when a source's εᵃ row fully decays`

## Task 3 — Change 2: `get_unchecked` in the εᵃ inner loop

1. Replace the four provably-in-bounds indexes with `get_unchecked`/`get_unchecked_mut` + `// SAFETY:`
   comments mirroring `process_layer`.
2. Update AGENTS.md's `unsafe` convention bullet to name both sites.
3. `cargo test` (incl. all equivalence oracles) green; `cargo build` warning-free.
4. `commit: perf(wave_driven): unchecked indexing in εᵃ accrual inner loop`

## Task 4 — measure

- Rebuild `--profile profiling`; run `profile_driven_train 32 2000` before/after (before = main).
- Report ms/trial, `wave` self-time share, and the win. Note whether Change 2's marginal delta justifies
  the expanded `unsafe` surface.
