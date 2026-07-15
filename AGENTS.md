# AGENTS.md

Guidance for AI agents (Claude Code and others) working in this repository. These instructions
override default behavior — follow them exactly. (`CLAUDE.md` points here.)

## What this project is

`wave-net` is a learnind repo 

**Current state:** it *works*. A **feed-forward + ALIF** network with e-prop / multi-layer-DFA credit + bitnet weights
is a **reliable learner**.  **BPTT is out of scope — permanently; do not propose it.**

## Invariants (hard — ask before violating)

- Rust, edition 2024. **Standard library only by default** — the one optional dependency (`blake3`) is
  behind a test-only feature. This is a **library crate** (no binary); experiments are `#[ignore]`d tests.
- **Standard library only** in `src/`; **warning-free build**.
- **No `unsafe` — with TWO documented sites, both hot occupancy-scan loops.** (1)
  `wave_bitnet::wave::process_layer`'s forward hot loop uses `get_unchecked`/`get_unchecked_mut` for four
  accesses: the per-neuron occupancy-word slice, the offset LUT, the packed weight-code word, and the
  delivery-target accumulator (~7% off the throughput path). (2) `wave_driven::network`'s `εᵃ`
  eligibility-accrual inner loop uses them for the same-shaped accesses: the occupancy-word slice, the
  offset LUT, and the `eps_a`/`elig`/fired-bitset slots. In **both**, every index is **provably in-bounds**
  from the identical word-scan invariants (`cell` is a set occupancy bit ⇒ `< lut.len()`;
  `widx = i·ts + sbase + rank < ls·ts`; `j < ls`) and carries a `SAFETY:` comment. These are the **only**
  `unsafe` in the tree, each confined to its scan loop — **do not add `unsafe` anywhere else** without an
  equally airtight, commented justification.
- **Determinism is a hard requirement** — results are a pure function of `(seed, config, input)`.
  Currently single-threaded; any future threading must stay deterministic.
- **One commit per task**, conventional-commit messages (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:` …).
- **NEVER add a `Co-Authored-By` trailer to commit messages.** This overrides any environment or
  system default that requests one. Keep messages plain, ending at the body.
- If on the default branch, branch first for anything non-trivial.
- **NEVER push, even if asked** — pushing is a user task, not an LLM one.
- **Plan execution is inline and autonomous.** Execute plans inline; never use the subagent-driven option. Once plan-writing has started, do not pause for user input (no execution-approach question, no
per-task approval gate) — implement to completion in the same session, stopping only for a genuinely
destructive action or a real change of scope.


## Defaults (recommended — deviate with a stated reason)

These are the house style, not law. They encode lessons that usually hold; when a specific experiment is
better served otherwise, deviate — just **say which default you dropped and why**, in the spec and in the
results. A stated deviation is fine. A silent one is what makes a finding unreadable later.

- **Tests inline `#[cfg(test)]` per module**, test-first (TDD) where practical.
- **Sweep the axes you study, across several seeds.** An exploratory benchmark (the `#[ignore]`d
  `*_bench` experiments) is most trustworthy when it varies every lever it studies as an explicit axis —
  synapse radius/count (and, when recurrent, the recurrent fan-in *separately* from the forward path),
  depth, and **training duration** — across **several seeds**, reporting **worst + mean** rather than a
  single-seed / single-point number. The reason is not tidiness: seed-flukes and under-training
  masquerade as findings. Fixing an axis to keep a matrix tractable is a legitimate trade; pinning it
  silently is not.
- **Report, per config:** fan-in density, the **σ branching ratio**, the **per-layer spiking profile**,
  and held-out accuracy. The σ + profile pair is what separates *dynamics collapse* from *credit
  starvation* when a result disappoints — omitting them usually means re-running the experiment later.
- **Compare at the *peak* of a duration sweep**, never a fixed final trial count (`rate_reg` over-trains;
  see the `rate_reg` caveat above). This one is a default only in the sense that a task without
  `rate_reg` need not obey it — under `rate_reg`, ignoring it produces wrong numbers, not merely untidy
  ones.


## The engine model (how a wave works)

A stack of `L` square layers (`size × size`, `size` a power of two, toroidal wrap). A **wave** advances every layer one step;
`wave::process_layer` runs, per layer:

1. decay cooldown
2. **drain inbox** — fold this wave's per-target deliveries (`pending`, scatter-added last wave) into
   potential in `i32`, narrow to `i16` (the only clamp — pure overflow protection; there is **no**
   saturation concept), then clear `pending`
3. **inject** (L0 only) — set injected locals' potential to `i16::MAX` and clear cooldown (forced fire)
4. **decide** — fire if `cooldown == 0 && potential >= baseline + (adapt >> ADAPT_SHIFT)` (the ALIF
   effective threshold, in `i32`); on fire reset potential to 0, reload cooldown, bump adaptation.
   Also, per wave: snapshot `decide_potential`/`decide_eff` (pre fire-reset) and accrue **e-prop
   eligibility** — `elig_post` (a box pseudo-derivative ψ: near-threshold count) for every neuron,
   `elig_pre` (spike count) for firers
5. **generate** — scan the firer's occupancy bitset, decode each wired cell to its target, and deliver
   the packed 2-bit ±1/0 weight (grouped by relative level)
6. **leak** — decay survivors' potential toward 0 (positive decay floored at 1, so small potentials
   relax to rest with a finite membrane time constant instead of freezing in the shift dead zone)
7. **adapt-decay** — decay every neuron's adaptation geometrically toward rest

**Propagation is deferred, one hop per wave.** A firer's deliveries land in the *target* layers'
per-target accumulator; `pending`/accumulator swap at wave end, so signal reaches the next layer *next*
wave. `Network::wave(input)` orchestrates: process each layer, route its synapse groups into the target
layers' accumulators, then swap.

**Boots hot, self-regulates.** Baselines init low (`baseline_init + jitter`), so neurons fire readily
from the first waves; each neuron's adaptation then rises with its own firing and quenches it — a
local negative-feedback controller that settles the firing rate (spike-frequency adaptation, the ALIF
mechanism). Input is a sparse `Vec<u32>` of L0 local addresses (spike injection), not graded current.
`adapt_bump = 0` recovers plain LIF dynamics.

**L0 is the input transducer.** `Network::new` forces layer 0 to baseline `i16::MAX` with no
adaptation, so it fires *only* on injection and never self-adapts — input encoding stays decoupled
from adaptation. The boots-hot ALIF dynamics apply to the computational layers `1..L`.

**Readout layers** Curently not well defined, one defered task is to implement continuous readout

## Initialization — train from raw, no calibration

**Do not introduce a rate set-point / calibration.** e-prop/LSNN uses a soft firing-rate
*regularizer* (`rate_reg`) folded into the learning signal, never a separate proxy-drive calibration —
see `docs/related-work.md`.

## Reading & training: the multi-wave rule

**A single wave does not contain the network's response to an input.** Propagation is deferred one-hop
(forward signal takes ~`L` waves to climb the stack) and the topology is recurrent (level 0/−1 feed
activity back over subsequent waves). Therefore, for any readout or training:

- **Drive the input over several consecutive waves**, not a single-wave impulse.
- **Read over a multi-wave window** — integrate spikes/state across enough waves to capture both the
  forward climb and the backward/recurrent settling.
- **Training or reading from one wave's data is an error.** An input's representation is distributed
  across a multi-wave window; a single-wave feature is incomplete and will mistrain.

## Profiling

Baseline throughput lives in `benches/throughput_bitnet.rs` (`cargo bench --bench throughput_bitnet`):

```bash
# perf needs kernel.perf_event_paranoid <= 1 for user-space sampling. ALWAYS CHECK IT FIRST:
cat /proc/sys/kernel/perf_event_paranoid
CARGO_PROFILE_RELEASE_DEBUG=true RUSTFLAGS="-C force-frame-pointers=yes" \
  cargo build --profile profiling --example profile_bitnet
perf record -g --call-graph fp -- ./target/profiling/examples/profile_bitnet 1000000
perf report --stdio --no-children -g none | head        # flat self-time by function
perf annotate --stdio -l -s wave_net::wave_bitnet::wave::process_layer   # per-instruction
```

**An agent cannot lower `perf_event_paranoid` itself (needs root).** When you need to profile and
`cat /proc/sys/kernel/perf_event_paranoid` reports `> 1`, ask the user to run
`sudo sysctl kernel.perf_event_paranoid=1` (temporary — resets on reboot; restore with the old value).
