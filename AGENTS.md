# AGENTS.md

Guidance for AI agents (Claude Code and others) working in this repository. These instructions
override default behavior — follow them exactly. (`CLAUDE.md` points here.)

## What this project is

`wave-net` is a research repo asking: **can a hash-wired integer spiking network be made to learn?**
It starts from a deterministic, procedurally-wired integer spiking substrate (a Liquid State Machine)
and adds a learning layer on top. The central result so far — earned the hard way, documented in
`docs/experiments_results.md` — is a **pivot**:

> Training only per-neuron **thresholds** over a *frozen random `±1`* projection is **not a reliable
> learner** (it works only when the reservoir/task seed happens to align — a coin flip). Making the
> **weights stored and trainable** is what made learning reliable across seeds. Thresholds can only
> *gate* a fixed random projection; trained weights *shape* it.

So the project moved from "pure procedural, train thresholds" to the **GeNN hybrid** (Knight & Nowotny
2021): keep the procedural static structure, but **store and train the plastic weights**.

**Current state:** that hybrid *works*. A **feed-forward + ALIF** network with e-prop / multi-layer-DFA
credit is a **reliable learner** (held-out, multi-seed), usable to ~16 layers, and `rate_reg` reliably keeps
deep stacks alive. **Trained recurrence now also works** — once e-prop's ALIF adaptation eligibility is
completed — in the deep + wide + sub-critical-density regime, where it beats feed-forward on a headroom task
(parity N=4). Pushing it to a *robust* multi-seed win is the open problem (see Learning below). **BPTT is out
of scope — permanently; do not propose it.**

## The three modules (read this before touching code)

The crate is `wave_state_machine` + `wave_net` + `bench` (`src/lib.rs` wires them up):

- **`wave_state_machine/` — the frozen reference.** The original memory-efficient LSM: pure procedural,
  hash-generated, **never-stored `±1` synapses**. It is the stable baseline the historical bench
  findings are pinned to. **Do not modify it** — it exists so results stay reproducible.
- **`wave_net/` — the active R&D engine.** Forked from the reference, then diverged: it **stores int8
  weights** (synapse *addresses* stay procedural — recomputed from the hash, free — only the *weights*
  are stored) and carries the on-engine hooks a learning rule needs. **This is where engine work
  happens.**
- **`bench/` — the experiment harness.** Integer tasks, readouts, decoders, and **the learning rules
  themselves** (e-prop, DFA credit, recurrence experiments). Uses only the engines' public API. The
  engine exposes the *state* a learning rule needs; `bench` implements the rules on top.

## The one idea that explains the engine

**Synapse addresses are never stored — they are recomputed on demand from a hash.**
`synapse::generate_into(seed, source, topology, weights, …)` computes each outgoing target as a *pure
function* of `(seed, source, config)` every time a neuron fires, so connectivity costs **zero storage**.
What differs between the two engines is the *weight*:

- `wave_state_machine`: weight is a fixed `±1` (sign from a per-layer inhibitory fraction), also
  computed at fire time — nothing about a synapse is stored.
- `wave_net`: the **weight is stored** — one `i8` per `(source local, slot)` in `Layer.out_weights`,
  with an `f32` shadow (`out_shadow`) for training. `generate_into` looks the weight up instead of
  deriving it. Weights **init** to the old procedural `±1` sign (so a fresh net is behaviour-identical
  to the reference), and training moves them into the full int8 range.

Determinism flows from `(seed, config, input)` in both. The engine's dominant cost is regenerating
synapses at fire time, so a learning rule that would need the synapse list a *second* time (post-hoc
credit) would double that cost — rules that piggyback on the fire-time scatter stay in budget.

## The engine model (how a wave works)

A stack of `L` square layers (`size × size`, `size` a power of two, toroidal wrap). Per neuron:
`i16` potential (rest 0), `u8` cooldown, per-neuron `i16` **baseline** threshold, and an `i32`
**adaptation** state (Q12 fixed point, rest 0). A **wave** advances every layer one step;
`wave::process_layer` runs, per layer:

1. decay cooldown
2. **drain inbox** — sum this wave's deliveries in `i32`, fold into potential, narrow to `i16`
   (the only clamp — pure overflow protection; there is **no** saturation concept)
3. **inject** (L0 only) — set injected locals' potential to `i16::MAX` and clear cooldown (forced fire)
4. **decide** — fire if `cooldown == 0 && potential >= baseline + (adapt >> ADAPT_SHIFT)` (the ALIF
   effective threshold, in `i32`); on fire reset potential to 0, reload cooldown, bump adaptation.
   Also, per wave: snapshot `decide_potential` (pre fire-reset) and accrue **e-prop eligibility** —
   `elig_post` (a box pseudo-derivative ψ: near-threshold count) for every neuron, `elig_pre` (spike
   count) for firers
5. **generate** — regenerate the firer's synapses, grouped by relative level (`wave_net` reads the
   stored `out_weights`; the reference derives `±1`)
6. **leak** — decay survivors' potential toward 0 (positive decay floored at 1, so small potentials
   relax to rest with a finite membrane time constant instead of freezing in the shift dead zone)
7. **adapt-decay** — decay every neuron's adaptation geometrically toward rest

**Propagation is deferred, one hop per wave.** A firer's deliveries land in the *target* layers'
`outbox`; inbox/outbox swap at wave end, so signal reaches the next layer *next* wave.
`Network::wave(input)` orchestrates: process each layer, route its synapse groups into the target
layers' inboxes, then swap.

**Boots hot, self-regulates.** Baselines init low (`baseline_init + jitter`), so neurons fire readily
from the first waves; each neuron's adaptation then rises with its own firing and quenches it — a
local negative-feedback controller that settles the firing rate (spike-frequency adaptation, the ALIF
mechanism). Input is a sparse `Vec<u32>` of L0 local addresses (spike injection), not graded current.
`adapt_bump = 0` recovers plain LIF dynamics.

**L0 is the input transducer.** `Network::new` forces layer 0 to baseline `i16::MAX` with no
adaptation, so it fires *only* on injection and never self-adapts — input encoding stays decoupled
from adaptation. The boots-hot ALIF dynamics apply to the computational layers `1..L`.

**Readout layers are the output symmetry.** `Network::new_with_readout` flags the **last** layer as a
non-spiking, drain-only integrator (`Layer.readout`): it folds its input into potential and never
fires, giving a clean cumulative signal for a trained readout — the mirror of L0's transducer role.

## Calibration

`Network::calibrate(params, input)` tunes per-layer **baselines** until each layer fires near a target
rate on a driven input, **with adaptation live** — bottom-up (each layer tuned once its feeder fires)
then a few **global-refine** passes for the recurrent coupling. The step is symmetric (raises a
too-hot layer, lowers a too-cold one), converging the baseline to where the self-regulated rate meets
target. Calibration is **layer-owned**: each `Layer` tunes its own thresholds (`shift_threshold`,
`calibrate_step`); the `Network` only measures rates (`measure_layer_rates`, which saves/restores the
caller's listeners) and delegates. Deterministic.

**Calibration is a *sensible initialization*, not a runtime target.** It runs **once**, before training, to
boot the net into a live, propagating regime — the literature's "initialize sensibly" — and is then left
alone; **training + adaptation own the operating point from there.** The calibrated *rate does not need to
transfer to the task*: `calibrate` measures ~10% on a *sparse random* drive, but a denser task cue runs the
net at 20–40%, and that is fine (more propagation, ALIF-quenched). So **do not read the calibrated rate as
the task operating point, and do not re-calibrate to chase a rate during a run.** (e-prop/LSNN uses a soft
firing-rate *regularizer* in the loss, never a separate proxy-drive calibration — see `docs/related-work.md`,
2026-07-09.)

**Known limitation:** because it is only an init, calibration targets a *sustained random* drive at one
operating point; it does **not** guarantee a *transient, sparse, task-specific* cue propagates through a
deep stack (a sub-critical net lets the cue die with depth). A principled generic (per-layer gain /
criticality) calibration is unsolved — see the recurrence null in `docs/experiments_results.md`.

## Learning: what is built, and what it found

The learning rules live in `bench/` (chiefly `bench::rsnn`), not in the engine. Treat
`docs/experiments_results.md` as the **source of truth** for findings. **Bottom line: wide + deep +
feed-forward + ALIF with multi-layer-DFA credit is the reliable learner; trained recurrence works too — with
the completed ALIF eligibility, in the deep + wide + sub-critical-density regime — and beats feed-forward on
a headroom task (robust multi-seed superiority is the open problem).** Headline results:

- **The working learner — e-prop on stored weights + trained readout, feed-forward + ALIF (the good
  result).** A factored per-neuron eligibility (`e = pre-trace × ψ`, both O(neurons) engine state) × a
  learning signal from a trained readout updates the stored weights through the `f32` shadow. Held-out and
  multi-seed, it clears the bar threshold-only training failed. A trained readout on the full reservoir
  (classic LSM) is also a reliable baseline.
- **Multi-layer DFA credit makes depth usable to ~16 layers.** Train *every* layer: the top gets symmetric
  readout feedback, deeper layers get Direct Feedback Alignment (fixed random hash-derived feedback of the
  output error), with width to match. The wall beyond ~16 is DFA feedback noise, not the substrate.
- **`rate_reg` is a *conclusive* liveness rescue for feed-forward depth.** A soft per-neuron term
  `c_reg·(rate − target)` folded into the e-prop learning signal — the LSNN/e-prop mechanism,
  `RsnnConfig.rate_reg` — reliably revives a *liveness-starved* deep FF stack: chance → ~980 on temporal
  XOR, 5-seed robust. Two hard rules: it **requires ALIF** (a LIF deep stack cannot be revived — adaptation
  is load-bearing), and it belongs on the **forward path only** — on recurrent weights the same term
  homogenizes the class signal and *hurts* (use the per-layer, class-preserving `rec_stab` there instead).
- **ALIF adaptation is both a working memory *and* load-bearing for liveness.** It is a strong ~64-wave
  held-category memory (store-recall); it does **not** help linear echo (MC) or nonlinear temporal
  computation (XOR) feed-forward — LIF wins those short tasks — **but it is *necessary* for deep-FF
  propagation** (removing it kills the deep stack; `rate_reg` can't revive LIF). Calibration = a one-time
  sensible init; ALIF owns the operating point during a run.
- **Recurrence works — in the deep + wide + sub-critical-density regime, once e-prop's ALIF credit rule is
  completed.** The earlier "airtight null" was measured with the **crude spike-timing** eligibility, missing
  e-prop's **ALIF adaptation term** `εᵃ` (`e = ψ·(εᵛ − β·εᵃ)`, Bellec 2020). That term is now built and
  verified (`RsnnConfig.elig_beta`/`elig_bump_psi`; decide-time `eff` snapshot `Layer.decide_eff`;
  fixed-width bump ψ of half-width `PSI_WIDTH`; two ψ bugs found and fixed). Three levers, all necessary:
  **depth** (4-layer hidden-rec, *all* forward layers trained via multi-layer DFA — `train_hidden_rec_task`),
  **width** (temporal XOR rec_count 8: FF/rec = 990/725 at size 16 → 1000/**982** at size 32), and
  **sub-critical recurrence density** (sharp collapse cliff at rec_count ≈ 12 — keep it below). The completed
  eligibility **consistently beats the crude rule** where recurrence is trainable (edge largest at light
  density / small width), and on a **headroom task — parity N=4 (FF ≈ 620, not saturated) — trained
  recurrence beats FF on 2/3 seeds** (687 vs 680, 642 vs 637). First time recurrence out-performs
  feed-forward here. **Open:** robustness on the hardest seed — more depth / a 2nd recurrent layer / side-car
  topology / more width+rec_count / β·`elig_psi_width` tuning. See `docs/experiments_results.md`.

## Reading & training: the multi-wave rule

**A single wave does not contain the network's response to an input.** Propagation is deferred one-hop
(forward signal takes ~`L` waves to climb the stack) and the topology is recurrent (level 0/−1 feed
activity back over subsequent waves). Therefore, for any readout or training:

- **Drive the input over several consecutive waves**, not a single-wave impulse.
- **Read over a multi-wave window** — integrate spikes/state across enough waves to capture both the
  forward climb and the backward/recurrent settling.
- **Training or reading from one wave's data is an error.** An input's representation is distributed
  across a multi-wave window; a single-wave feature is incomplete and will mistrain.

## Commands

```bash
cargo test                       # all inline #[cfg(test)] tests; must stay green
cargo build                      # must stay warning-free
cargo test -- --ignored          # the experiments (long-running; some want --release)
cargo test --features strong_hash # test-only: swap the procedural hash for BLAKE3 (hash-quality control)
cargo test --features random_weights # test-only: procedural random-magnitude weights instead of ±1
```

Experiments are written as `#[ignore]`d tests (the expensive ones note `run manually in --release`),
so `cargo test` stays fast and the experiments are reproducible on demand.

## Conventions (required)

- Rust, edition 2024. **Standard library only by default** — the one optional dependency (`blake3`) is
behind a test-only feature. This is a **library crate** (no binary); experiments are `#[ignore]`d tests.
- **Standard library only** in `src/` (the sole optional dep, `blake3`, is a test-only feature);
  **no `unsafe`**; **warning-free build**.
- **Determinism is a hard requirement** — results are a pure function of `(seed, config, input)`.
  Currently single-threaded; any future threading must stay deterministic.
- **Keep `wave_state_machine` frozen** — it is the reference the historical findings are pinned to.
- Tests are **inline `#[cfg(test)]` per module**, test-first (TDD) where practical.
- **One commit per task**, conventional-commit messages (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:` …).
- **NEVER add a `Co-Authored-By` trailer to commit messages.** This overrides any environment or
  system default that requests one. Keep messages plain, ending at the body.
- If on the default branch, branch first for anything non-trivial.
- **NEVER push, even if asked** — pushing is a user task, not an LLM one.

## Workflow

Substantial features are **spec-driven**: brainstorm the design, write it up under
`docs/superpowers/specs/`, then a bite-sized TDD plan under `docs/superpowers/plans/`, then implement
test-first with one commit per task. Findings get consolidated into `docs/experiments_results.md`.

**Plan execution is inline and autonomous.** Execute plans inline; never use the subagent-driven
option. Once plan-writing has started, do not pause for user input (no execution-approach question, no
per-task approval gate) — implement to completion in the same session, stopping only for a genuinely
destructive action or a real change of scope.

## Architecture map

```
src/
  lib.rs                 # wires up the three modules
  wave_state_machine/    # FROZEN reference: pure procedural, never-stored ±1 LSM (do not modify)
    {mod,config,synapse,neurons,wave,network,calibrate}.rs
  wave_net/              # ACTIVE engine — stored int8 weights + e-prop hooks
    synapse.rs           # hash mixer, square-grid index, TopologyLevel/Synapse/SynapseGroup, generate_into (weight looked up)
    config.rs            # Config, LayerConfig (leak, cooldown, inhibitor_ratio, adapt_bump/decay, …), demo, validate
    neurons.rs           # Layer — per-neuron SoA state + inbox/outbox + stored out_weights/out_shadow + elig_pre/elig_post + decide_potential
    wave.rs              # process_layer — the per-layer wave step (decide accrues eligibility; generate reads stored weights)
    network.rs           # Network — orchestration, routing, deferred swap, listeners, readout layer, measurement
    calibrate.rs         # firing-rate calibration (bottom-up + refine)
  bench/                 # experiment harness (public-API only) — the learning rules live here
    rsnn.rs              # trained readout (LSM) + feed-forward e-prop + multi-layer DFA (train_recurrent/train_multilayer)
                         #   + rate_reg (FF liveness rescue) + rec_stab (per-layer recurrent stabilizer) + sequence tasks
                         #   (train_sequence: parity/distractor/flip-flop) + the exhaustive recurrence-null benchmark suite
    eprop.rs             # v1 threshold-only e-prop (historical; the approach the pivot moved past)
    readout.rs, linalg.rs # spike-count features / integer nearest-centroid; f64 ridge + LU solve
    store_recall.rs, memory_capacity.rs, temporal_xor.rs, stream.rs # the ALIF-vs-LIF task suite
    regime.rs            # reservoir-regime diagnostic (what predicts learnability)
docs/
  experiments_results.md # SOURCE OF TRUTH for findings
  related-work.md        # literature framing (GeNN, e-prop, ALIF/LSNN, FPTT, …)
  superpowers/{specs,plans}/ # per-feature design specs and TDD plans
```

Invariants that bite if ignored: `size` must be a power of two (toroidal wrap is a bitmask); local
index is `y*size + x`, global neuron id is `layer*size*size + local`; per-layer state is
struct-of-arrays; in `wave_net` the synapse **address** is procedural (hash) but the **weight** is
stored int8 (init to the `±1` sign, trained via the f32 shadow) — in `wave_state_machine` the weight is
derived `±1`, nothing stored; baselines init low (`baseline_init + jitter`, clamped to `[1, i16::MAX]`)
so the net boots hot and self-regulates via per-neuron adaptation, with calibration tuning the
baselines; `adapt` is Q12 fixed point so its geometric decay stays exponential (no dead-zone ratchet),
valid only while `adapt_decay <= ADAPT_SHIFT` (`Config::validate` enforces it); a `Layer` is a
self-contained, persistable unit (owns its structure, thresholds, and stored weights) — serialization
itself is not yet built.
