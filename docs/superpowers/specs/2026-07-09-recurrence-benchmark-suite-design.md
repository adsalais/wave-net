# Recurrence benchmark suite — tasks that need computation, not memory — design

**Date:** 2026-07-09
**Status:** approved (design), pre-plan
**Scope:** build **several benchmarks that specifically test recurrent computation** — tasks a slow
adaptation variable provably cannot fake — so the "does recurrence earn its keep" conclusion rests on more
than one measurement. Three tasks: **sequential parity** (N‑bit, the canonical test), **delayed XOR with a
distractor**, and a **set/reset flip‑flop**. Each run **ALIF, FF vs +lateral‑recurrence, multi‑seed**. A
general N‑cue sequence runner + a task‑parameterized trainer; all in `bench::rsnn`, **no engine change**.

## Why these tasks

The prior recurrence null was on **temporal XOR**, which ALIF adaptation solves feed‑forward (to delay 120+,
recurrence only hurt) — so XOR can't test recurrence: adaptation makes the point moot. We need tasks where a
monotone "how much / how recently did I fire" accumulator (what adaptation provides) is **provably
insufficient**, so any success must come from recurrent state:

- **Sequential parity** (label `= b₁⊕…⊕b_N`) is the cleanest: parity is **non‑monotone** — it flips with
  every bit — so no count of activity yields it. We *know* ALIF‑FF solves N=2 (that is temporal XOR); it
  should **fail N≥3**, and recurrence maintaining a toggling state should rescue it *iff* the crude e-prop
  can train it. One benchmark, a whole sweep.
- **Delayed XOR with a distractor** (`A → D → B`, label `A⊕B`) adds a **gating** demand: hold A *and ignore*
  the irrelevant strong cue D. Tests selective memory + computation, not just holding.
- **Set/reset flip‑flop** (`set`/`reset` ops → gap → query, label = final state) tests **persistent,
  updatable state** held robustly across a gap (an attractor) — recurrence's domain, where a *decaying*
  adaptation trace should struggle. (If ALIF‑FF solves it, that is itself the finding: flip‑flop is memory,
  not computation — the experiment adjudicates.)

## Mechanism

**A general N‑cue sequence runner** `sequence_trial(net, cfg, classes: &[usize], trial) -> (Vec<f32>,
Vec<Vec<u32>>)`: reset → for each class in `classes`, present `cue_realization(class)` for `present_waves`
then `delay` silent gap waves → `read_waves` silent → return the read‑window L1 spike counts and the per‑wave
L1 fired‑sets. This generalizes `xor_trial` (which is the `classes = [a, b]` case); `xor_trial` is left
untouched so the existing XOR experiments stay byte‑identical.

**Task generators** map `(task_seed, trial)` to a `(classes: Vec<usize>, label: usize)` (all binary, `k = 2`,
so the existing 2‑class readout is reused):
- `task_parity(seed, trial, n)` → `n` bits, `label = ⊕ bits`.
- `task_distractor(seed, trial)` → `[a, 2, b]` (class `2` is a label‑irrelevant distractor cue — a distinct
  `cue_realization` pattern), `label = a⊕b`.
- `task_flipflop(seed, trial, n_ops)` → `n_ops` `set`/`reset` ops (as two distinct cue classes), `label =`
  the last op's state.

**A task‑parameterized trainer** `train_sequence(cfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize)) ->
u64`: reuse the `train_xor` structure — build `engine_config_xor` (L0→L1, level‑0 recurrence when
`rec_count > 0`), calibrate **once as a sensible init**, loop trials (readout delta‑rule on the read window +
`recurrent_update` when `rec_count > 0`), held‑out over disjoint trials. FF = `rec_count = 0`; recurrence =
`rec_count > 0`. **ALIF on** (`adapt_bump > 0`, the demo default), rate reg off.

Reuses `cue_realization`, `recurrent_update`, `pick_ab`‑style deterministic draws, and the existing readout
loop. No new config fields — task arity (`n`, `n_ops`) is captured in the experiment's task closure.

## Experiments (`#[ignore]`, release, multi‑seed)

For each, ALIF, FF (`rec_count = 0`) vs +lateral‑recurrence (`rec_count = 24`), 3+ seeds, worst‑ or
best‑seed reported per the honesty gate:

- `parity_recurrence_sweep` — parity at **N = 2, 3, 4, 5**, FF vs +rec. The centerpiece: locate where FF
  fails and whether recurrence rescues it.
- `distractor_xor_recurrence` — delayed‑XOR‑with‑distractor, FF vs +rec.
- `flipflop_recurrence` — set/reset flip‑flop, FF vs +rec.

## Success criterion

- **Per task:** recurrence's held‑out accuracy clearly above the FF baseline where FF fails, multi‑seed —
  recurrence earning its keep on a task that *needs* it.
- Determinism: pure function of `(seed, task_seed, config)`.

**Honesty gate** — report which, per task, never a single seed:
1. **FF fails, +rec lifts it** ⇒ recurrence earns its keep on a genuinely recurrent task — the result the
   whole arc has chased.
2. **FF already solves it** ⇒ that task didn't need recurrence (adaptation suffices) — expected for
   parity N=2, informative if it happens for flip‑flop (⇒ flip‑flop is memory, not computation).
3. **Both FF and +rec fail** (e.g. parity N≥4) ⇒ the task is beyond the current e-prop's **temporal credit**
   — the wall is the credit rule (crude spike‑timing eligibility), now shown on a task that *does* need
   recurrence → surrogate‑gradient BPTT, not the substrate.

The suite's value is the **pattern across tasks and N**: a single null on one task no longer decides it.

## Determinism & constraints

- Engine untouched, integer, deterministic; the sequence runner + trainer are bench‑side `f32`.
  Single‑threaded.
- **`xor_trial` and the existing XOR/recurrence experiments stay byte‑identical** (new code paths only) — the
  full suite stays green.
- Calibration is a **one‑time sensible init** (per AGENTS.md), not a rate target. `wave_state_machine`
  frozen; held‑out + multi‑seed from the start.

## Testing

- `sequence_trial_matches_xor_on_two_cues` — `sequence_trial(&[a, b])` produces the same read‑window activity
  as `xor_trial(a, b)` on a fixed config (the generalization is faithful for the 2‑cue case).
- `task_parity_labels_are_correct` / `task_distractor_labels_are_correct` / `task_flipflop_labels_are_correct`
  — the label functions match their definitions on hand‑checked draws.
- `train_sequence_is_deterministic` — a short parity run is a pure function of `(seed, config)`.
- `parity_recurrence_sweep`, `distractor_xor_recurrence`, `flipflop_recurrence` — the three headline
  experiments (`#[ignore]`, release, multi‑seed).
- Regression: whole suite green; the existing XOR/recurrence tests unchanged.

## Deferred

- **Backward (level −1/−2) recurrence** on the suite, if lateral shows a signal worth the topology sweep.
- **A purpose‑built graded task** (adding problem, N‑back) if binary‑cue tasks under‑discriminate.
- **Surrogate‑gradient BPTT** where the credit rule is the wall (honesty‑gate case 3).
- Consolidate the cross‑task pattern into `docs/experiments_results.md` once the suite has run.
