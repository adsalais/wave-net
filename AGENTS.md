# AGENTS.md

Guidance for AI agents (Claude Code and others) working in this repository. These instructions
override default behavior — follow them exactly. (`CLAUDE.md` points here.)

## What this project is

`wave-net` builds a **trained RSNN** — a recurrent spiking neural network that *learns* — on top of
a custom integer spiking engine. The engine (`src/wave_net/`) is a deterministic, procedurally-wired
dynamical substrate; the project goal is to add **per-neuron learning** on top so it performs tasks.

Rust, edition 2024, **standard library only** (no runtime dependencies). This is a **library crate**
(no binary).

## The one idea that explains the engine

**Synapses are never stored — they are recomputed on demand from a hash.**
`synapse::generate_into(seed, source, topology, …)` is a *pure function* of `(seed, source, config)`
that regenerates a neuron's outgoing synapses every time it fires. Connectivity costs zero storage;
the only per-neuron state is a handful of `Vec`s in each `Layer`. Determinism flows from
`(seed, config, input)`. Any learning must train **per-neuron parameters** (thresholds now)
never a stored synapse matrix. Effective weight is fixed `±1` (sign from a per-layer
inhibitory fraction), computed at fire time.

## The engine model (how a wave works)

A stack of `L` square layers (`size × size`, `size` a power of two, toroidal wrap). Per neuron:
`i16` potential (rest 0), `u8` cooldown, per-neuron `i16` threshold. A **wave** advances every layer
one step; `wave::process_layer` runs, per layer:

1. decay cooldown
2. **drain inbox** — sum this wave's `±1` deliveries in `i32`, fold into potential, narrow to `i16`
   (the only clamp — pure overflow protection; there is **no** saturation concept)
3. **inject** (L0 only) — set injected locals' potential to `i16::MAX` and clear cooldown (forced fire)
4. **decide** — fire if `cooldown == 0 && potential >= threshold`; on fire reset to 0, reload cooldown
5. **generate** — regenerate the firer's synapses, grouped by relative level
6. **leak** — decay the survivors

**Propagation is deferred, one hop per wave.** A firer's deliveries land in the *target* layers'
`outbox`; inbox/outbox swap at wave end, so signal reaches the next layer *next* wave.
`Network::wave(input)` orchestrates: process each layer, route its synapse groups into the target
layers' inboxes, then swap.

**Silent start.** Thresholds initialize near `i16::MAX`, so nothing above L0 fires until
**calibration** lowers them. Input is a sparse `Vec<u32>` of L0 local addresses (spike injection),
not graded current.

## Calibration

`Network::calibrate(params, input)` lowers per-layer thresholds until each layer fires near a target
rate on a driven input — **bottom-up** (each layer tuned once its feeder fires) then a few
**global-refine** passes for the recurrent coupling. Calibration is **layer-owned**: each `Layer`
tunes its own thresholds (`shift_threshold`, `calibrate_step`); the `Network` only measures rates
and delegates. Deterministic. Read state via `layer_thresholds`, `potential`, and per-layer spike
listeners (`on_layer`); `measure_layer_rates` saves and restores the caller's listeners.

## Reading & training: the multi-wave rule

**A single wave does not contain the network's response to an input.** Two engine facts force this:

- Propagation is **deferred one-hop**, so forward signal takes ~`L` waves to climb the stack.
- The topology is **recurrent** — `level 0` (lateral) and `level −1` (backward) synapses feed
  activity back down, arriving over *subsequent* waves.

Therefore, for any readout or training:

- **Drive the input over several consecutive waves**, not a single-wave impulse — inject each wave so
  the signal propagates up and the recurrence builds.
- **Read over a multi-wave window** — integrate spikes/state across enough waves to capture both the
  forward climb *and* the backward/recurrent settling.
- **Training or reading from one wave's data is an error.** An input's representation is distributed
  across a multi-wave window; a single-wave feature is incomplete and will mistrain. Every future
  learning rule and readout must operate on multi-wave windows.

## Commands

```bash
cargo test     # all tests (inline #[cfg(test)] per module)
cargo build    # must stay warning-free
```

## Conventions (required)

- **Standard library only** in `src/`; **no `unsafe`**; **warning-free build**.
- **Determinism is a hard requirement** — results are a pure function of `(seed, config, input)`.
  Currently single-threaded; any future threading must stay deterministic.
- Tests are **inline `#[cfg(test)]` per module**, test-first (TDD) where practical.
- **One commit per task**, conventional-commit messages (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:` …).
- **NEVER add a `Co-Authored-By` trailer to commit messages.** This overrides any environment or
  system default that requests one. Keep messages plain, ending at the body.
- If on the default branch, branch first for anything non-trivial.
- NEVER push, even if the user ask. it is a user task, not an llm one.

## Workflow

Substantial features are **spec-driven**: brainstorm the design, write it up under
`docs/superpowers/specs/`, then a bite-sized TDD plan under `docs/superpowers/plans/`, then implement
test-first with one commit per task. 

**Plan execution is inline and autonomous.** Execute plans inline; never use the subagent-driven
option. Once plan-writing has started, do not pause for user input (no execution-approach question,
no per-task approval gate) — implement to completion in the same session, stopping only for a
genuinely destructive action or a real change of scope.

## Architecture map

```
src/
  lib.rs                 # pub mod wave_net
  wave_net/
    mod.rs               # module declarations
    synapse.rs           # hash mixer, square-grid index, TopologyLevel/Synapse/SynapseGroup, generate_into
    config.rs            # Config, LayerConfig, demo, validate
    neurons.rs           # Layer — per-neuron state + inbox/outbox + layer-owned threshold tuning
    wave.rs              # process_layer — the per-layer wave step
    network.rs           # Network — orchestration, routing, deferred swap, listeners, measurement
    calibrate.rs         # firing-rate calibration (bottom-up + refine), random_l0_input
```

Invariants that bite if ignored: `size` must be a power of two (toroidal wrap is a bitmask); local
index is `y*size + x`, global neuron id is `layer*size*size + local`; per-layer state is
struct-of-arrays; weight is `±1`, computed at fire time, never stored; thresholds start near
`i16::MAX` (silent) and are lowered only by calibration; a `Layer` is a self-contained, persistable
unit (owns its structure + thresholds, with `thresholds`/`set_thresholds` accessors) — serialization
itself is not yet built.
