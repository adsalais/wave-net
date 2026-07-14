# wave_resonate Phase 3 — experiment suite (BRF vs ALIF, recurrence, ω/δ) design

- **Date:** 2026-07-14
- **Status:** design approved; ready for an implementation plan
- **Scope:** the **research readout** for `wave_resonate`. Port the temporal-task battery to the BRF+HYPR
  engine and answer, multi-seed and matched-baseline: (1) is BRF+HYPR a viable temporal learner vs the
  ALIF `wave_driven` reference; (2) does **trainable ω/b′** (Phase 2b) help; (3) does BRF's **intrinsic
  resonance** change the recurrence-beats-FF story (the repo headline); (4) how sensitive is BRF to the
  **timescale** (`ω`-init × `δ`). All experiments are `#[ignore]`d benchmark tests; findings consolidate
  into `docs/experiments_results.md`.

## Research questions & success criteria

- **RQ1 — viable temporal learner?** BRF FF should clear chance on temporal XOR / flip-flop /
  distractor-XOR / parity-4 (worst-seed > chance), reported next to `wave_driven` ALIF FF numbers.
- **RQ2 — does trainable ω/b′ help?** Frozen-ω/b′ vs trained-ω/b′, same budget. Expect a benefit on the
  *temporal* tasks (static single-cue already showed none — see Phase 2b). A null result is still a finding.
- **RQ3 — resonance vs topological recurrence?** BRF FF vs BRF backward-fed **side-car** (the topology that
  wins for ALIF). Two sub-questions: does the side-car *also* beat FF for BRF, and — the interesting one —
  does BRF's per-neuron oscillatory memory let **FF alone** close the gap that ALIF needed the side-car for?
- **RQ4 — timescale sensitivity.** Sweep `ω`-init range × `δ` on temporal XOR; report liveness + accuracy.

## Method (AGENTS.md benchmark contract)

- **Sweep every axis, multi-seed, worst + mean.** Axes: task, topology (FF / side-car), ω-mode (frozen /
  trained), and (RQ4) `ω`-init × `δ`. ≥ 3 seeds. Report **worst and mean** held-out accuracy, never a
  single point. Best-checkpoint each run (peak of a duration sweep via `train_and_eval_best`).
- **Read the top spiking layer directly** (no dedicated readout layer); report per config: fan-in density,
  the **σ branching ratio**, the **per-layer spiking profile** (`rate_profile`), and held-out accuracy.
- **Matched baseline.** Same trial budget / present-delay-read windows across FF and side-car, mirroring
  `wave_driven_bench::wave_driven_recurrence_confirmation`, so the comparison is apples-to-apples.
- **BRF operating point (from Phase 2 bring-up, load-bearing):** `θ_c ≈ 0.1` (a resonator is silent under
  DC drive at θ_c=1), `eps_cut 1e-6`, `hidden_lr ~2` (δ-scaled ε), `omega_b_lr ~2` (tuned per Task).

## Topologies

- **FF** — `make_ff`, 5 layers at size 32, generous fan-in (`r3/c32`), read the top.
- **Side-car** — port `wave_driven_bench::make_sidecar` to BRF: `L0→L1(+1)`, `L1→L3(+2 skip)`,
  `L2 self(0) + L2→L3(+1)`, `L3→L2(−1 back) + L3→L4(+1)`, `L4` read. Recurrent fan-in (`n`, `r`) swept
  **separately** from the forward path (AGENTS.md). BRF `EligParams`/`entries` mirror the topology.

## Tasks (engine-agnostic generators, ported verbatim)

`task_parity(seed,t,n)` (n=2 temporal XOR, n=4 parity-4), `task_distractor` (a, class-2 distractor, b →
a⊕b), `task_flipflop(seed,t,n_ops)` (set/reset → final state). `cue_sites` already supports the class-2
distractor. A `task_labels_correct` unit test guards the generators (ported).

## Execution plan (runtime-aware)

BRF is f32 + per-synapse 2-state eligibility, and resonators ring (larger eligibility active set), so
size-32 multi-seed sweeps are slow (estimate tens of minutes to a couple of hours). Therefore:

1. **Validate the harness at size 16** (fast `#[ignore]` smoke): side-car builds + is live, tasks train
   above chance, frozen-vs-trained runs. Catch bugs cheaply before the big run.
2. **Run the size-32 study in the background** (`#[ignore]`, `--release`), one experiment per RQ. Capture
   the eprintln readouts.
3. **If the size-32 run hits a perf wall** (too slow to finish multi-seed), that is itself a reportable
   finding (BRF f32 may hit the scaling wall earlier than the integer engine — cf. the standing
   perf-then-scaling note); fall back to size 16 / fewer seeds and say so explicitly (no silent caps).
4. **Consolidate** worst+mean tables + σ/spiking profiles + the frozen-vs-trained and FF-vs-side-car
   verdicts into `docs/experiments_results.md`, contextualized against the ALIF numbers.

## Non-goals / deferred

- Persistence-based best-checkpoint to disk (in-memory `train_and_eval_best` peak suffices here).
- The `q→b` second-order eligibility term and fixed-point port (unchanged from earlier deferrals) — pulled
  in only if a *conclusive* underperformance implicates them (don't dismiss on weak implementation; but let
  the workload prove the cost bites).
- Non-temporal paper benchmarks (SHD/S-MNIST) — out of scope; we compare on this repo's task battery.

## References

`docs/experiments_results.md` (source of truth for findings + the ALIF baselines);
`src/bench/wave_driven_bench.rs` (the harness shape + tasks + `make_sidecar` ported from);
`docs/superpowers/specs/2026-07-14-wave-resonate-brf-hypr-design.md` (engine spec).
