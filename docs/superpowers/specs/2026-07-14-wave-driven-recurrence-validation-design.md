# wave_driven — recurrence-beats-FF validation suite (spike-ψ εᵃ)

- **Date:** 2026-07-14
- **Status:** design approved; ready for an implementation plan
- **Scope:** A **multi-seed, all-benchmark, matched-FF-baseline** confirmation of the Phase 2b finding
  that spike-ψ `εᵃ` recurrence beats feed-forward. Adds the two missing benchmark tasks (distractor-XOR,
  flip-flop) and runs a focused confirmation at the Phase-2b operating point. Research/validation work —
  **not** the performance-benchmark suite (deferred to a later session).

## Motivation

Phase 2b showed spike-ψ `εᵃ` unlocks recurrence, but only a **2-seed, parity-only** run against an FF
baseline that (at size 16) sat below `wave_bitnet`'s historical FF. The finding was recorded in
`docs/experiments_results.md` with explicit caveats: *2-seed / 2-task; wave_driven FF baseline lower than
historical → cross-engine comparison not airtight.* This suite closes those caveats: **3 seeds**, **all
four benchmarks** the historical bump-ψ result used (temporal XOR, parity N=4, distractor-XOR,
flip-flop), and a **matched FF baseline** (FF trained to its own ceiling, depth-matched to the side-car),
at the Phase-2b operating point where the FF baseline already reaches ~historical.

## Goal (success criterion)

Answer, rigorously: **does spike-ψ `εᵃ` side-car recurrence beat a fairly-trained FF across every
benchmark, multi-seed?** Report FF vs side-car **worst + mean over 3 seeds, per task**, and record the
outcome — confirming or qualifying the finding honestly (per the project rule: conclusive results, don't
overclaim). There is **no hard pass/fail assertion** on "beats FF" (it is a research readout, and
flip-flop/distractor under spike-ψ are genuinely untested); the experiment prints the numbers and a
summary. A cheap **unit test** does gate the new task generators for correctness.

## Components

### 1. Task generators (recovered from git `9a39048`)

Ported into `bench/wave_driven_bench.rs` (each a `Fn(u64, usize) -> (Vec<usize>, usize)` — a cue-class
sequence + a 2-class label; `run_trial` already drives an arbitrary class sequence and the readout is
2-class, so the class-2 distractor cue and the set/reset ops are just input patterns):

- **`task_parity(seed, t, n)`** — already present: n bits, label = their XOR. (N=2 = temporal XOR.)
- **`task_distractor(seed, t)`** — `[a, 2, b]`, label = `a ^ b`. The middle **class-2** cue is
  label-irrelevant; the net must ignore it and remember `a` across the delay.
- **`task_flipflop(seed, t, n_ops)`** — `n_ops` set(class 0)/reset(class 1) ops; label = the final
  state (`last == 0` [set] → 1, else 0). The net must track the most recent op across the delay.

A **`#[cfg(test)]` unit test `task_labels_correct`** (ported from git `9a39048`) asserts the label logic
for all three (bit ranges, XOR label, distractor 3-cue with class-2 middle, flip-flop last-op state). It
runs in `cargo test` (no training — cheap).

### 2. The confirmation experiment (`#[ignore]`, `--release`)

`wave_driven_recurrence_confirmation` — the focused run at a **fixed operating point**:

- **Size 32**, **rec_count 8** (the Phase-2b sweet spot: sparse recurrence + width), **3 seeds**.
- **Side-car**: `make_sidecar` (uc 32 / ur 3 forward, rec 8 / r 4 recurrent, adapt_bump 5, adapt_decay 6),
  spike-ψ `εᵃ` (`elig_beta 0.4`, `rec_tau 20`).
- **FF baseline**: `make_ff` — a **5-layer** stack (depth-matched to the side-car), up_count 32, radius 3,
  membrane-only (`elig_beta 0`); this is the "matched baseline".
- **Per-task trial params** (historical delays):

  | task | generator | present | delay | read |
  |---|---|---|---|---|
  | temporal XOR | `task_parity(·, 2)` | 6 | 8 | 8 |
  | parity N=4 | `task_parity(·, 4)` | 6 | 8 | 8 |
  | distractor-XOR | `task_distractor` | 6 | 20 | 8 |
  | flip-flop | `task_flipflop(·, 4)` | 6 | 12 | 8 |

- **Training**: each (task, engine, seed) trains via the existing `train_and_eval_best` (periodic held-out
  eval + **best-checkpoint** → returns the peak) with a **generous, equal budget** for FF and side-car
  (`eval_every 300`, `patience 3`, `max_trials ~2400`, `holdout 200`). Equal budget + best-checkpoint +
  depth-match is what makes the baseline **matched** — the FF-vs-side-car delta reflects the topology, not
  FF under-training.
- **Output**: per task, print FF **worst + mean** and side-car **worst + mean** over the 3 seeds, plus the
  side-car's **σ + per-layer rate profile** (the dynamics diagnostic), and a final summary line
  *"recurrence beats FF (worst-seed) on N/4 tasks"*.

### 3. Record the outcome

After the run, update `docs/experiments_results.md`: replace the current 2-seed/parity-only caveat block
with the **3-seed, 4-benchmark, matched-baseline** table and an honest verdict (confirmed across N/4, or
qualified where it isn't).

## Data flow

`task_*` → `run_trial(net, classes, present, delay, read)` (drives the cue sequence; online `εᵃ`
accrual happens inside `wave()` for the side-car) → readout + `build_signal` → `dfa_update` →
`train_and_eval_best` loops with held-out eval → `(best_permille, at)`. The experiment wraps this per
(task, engine, seed) and aggregates worst+mean. No engine or `run_trial` changes are needed — only the
task generators and the new experiment harness.

## Runtime

~24 training runs (4 tasks × 2 engines × 3 seeds) at the size-32 side-car cost (~20–36 ms/trial,
≤ 2400 trials, often less via early-stop patience) → **tens of minutes** in `--release`. Bounded by
design (the "focused" scope); `#[ignore]` so it never runs in `cargo test`.

## Testing

- **`task_labels_correct`** (unit, runs in `cargo test`): gates the recovered generators.
- **`wave_driven_recurrence_confirmation`** (`#[ignore]`): the research readout; **no hard assertion on
  the FF-vs-side-car comparison** (a task where FF or even the side-car sits at chance is a legitimate
  result to report, not a bug). The only sanity/plumbing gate is that the **side-car clears chance on the
  easy task** (temporal XOR, worst-seed > ~700 permille) — it demonstrably solves that one, so a failure
  there means the harness is broken, not that the finding is negative.
- `cargo build` warning-free; `cargo test` stays green and fast (the confirmation stays ignored).

## Non-goals

Performance/throughput benchmarks (separate session); bump-ψ / decide snapshots; the width × rec_count
sweep (Phase 2b did it — the exploratory `wave_driven_sidecar_vs_ff` stays for reference); new sizes or
rec_counts beyond the fixed operating point; engine changes.

## Risks

- **The finding may not fully confirm** — flip-flop/distractor under spike-ψ are untested; a task where
  the side-car does *not* beat FF is a legitimate outcome to report (not a bug), and per the convergence
  ladder would motivate a width bump or the bump-ψ fast-follow in a later session.
- **Flip-flop `n_ops` / exact delays** are best-effort reconstructions of the historical params (the git
  history preserved the *generators* but not every trial param); the delays chosen (distractor 20,
  flip-flop 12) match the `experiments_results.md` descriptions. If a task trains at chance for a
  plumbing reason, the above-chance sanity assert flags it.

## Appendix — copied vs new

- **Recovered from git `9a39048`:** `task_distractor`, `task_flipflop`, and the `task_labels_correct`
  unit test.
- **Reused as-is:** `run_trial`, `build_signal`, `train_and_eval_best`, `make_ff`, `make_sidecar`,
  `rate_profile`, `task_parity` (all already in `bench/wave_driven_bench.rs`).
- **New:** the `wave_driven_recurrence_confirmation` experiment harness (fixed operating point; task ×
  seed aggregation; worst+mean reporting) and the `experiments_results.md` update.
