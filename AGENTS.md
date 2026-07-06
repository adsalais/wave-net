# AGENTS.md

Guidance for AI agents (Claude Code and others) working in this repository. These instructions
override default behavior — follow them exactly.

## What this project is

`wave-net` builds a **trained RSNN** — a recurrent spiking neural network that *learns* — using a
**wave reservoir** as its base implementation. The reservoir (a spiking-neuron network whose activity flows
in waves up a stack of 2D layers) is currently fixed, procedurally-generated dynamical base; the project's
goal is to change it to become a real RSNN with trained weights

Rust, edition 2024, **standard library only unless the user accept it** (no external runtime dependencies).

- **`wave_reservoir` (the base, done):** the integer (i16) `LayerNet` engine — wavefront-
  pipelined, deterministic, multi-threaded — plus its procedural wiring and shared primitives. this is the base, not the deliverable.

## The one idea that explains the engine

**Synapses are never stored — they are recomputed on demand from a hash.**
`wiring::scatter_layered(source, seed, topology, …)` is a *pure function* of `(seed, source, config)`
that regenerates a neuron's outgoing synapses every time it fires. Connectivity costs zero storage;
the only per-tick state is a handful of per-layer `Vec`s in `LayerNet`. Determinism flows from
`(seed, config)`. Any learning design must preserve this — train per-neuron parameters, not a stored
synapse matrix.

## The engine model (how a wave works)

A **wave** = one bottom-to-top sweep of the layer stack. For each layer, in order: decay cooldown →
leak + add external drive (native i16 saturating shift/sub) → **decide** (a neuron fires if
`potential >= threshold` and not refractory; reset + set cooldown) → **apply** (regenerate the
firer's synapses and add each `±1` delivery into its target layer's potential). Forward deliveries
land within the same wave as the front climbs; the integer `Σ±1` accumulation is order-independent,
so results are **bit-identical across thread counts**.

`LayerNet` (`wave_reservoir::pipeline`) is the engine:
- `new(cfg: IntConfig)`, `wave(&self, drive: &[i16])` — one sequential wave.
- `run_stream(waves, threads, drive_fn)` — pipeline `waves` waves across `threads` workers, each
  wave's drive supplied on demand by `drive_fn(wave_id, &mut buf)` (streaming input). `run(drive,
  waves, threads)` is the constant-drive wrapper.
- `on_layer(layer, Box<dyn Fn(wave_id, &[u32]) + Send + Sync>)` — subscribe to a layer's spikes
  (emitted at that layer's decide, in wave order, lazy: unsubscribed layers cost nothing). The
  readout mechanism. `clear_listeners()` unregisters.
- `reset_state()`, `potential_global(idx)`, `n_total()`.

Determinism note: the wavefront stays correct because waves **enter the pipeline in wave order**
(an `entry` gate) and the per-layer `Mutex` band prevents overtaking. Do not weaken either.

## Commands

```bash
cargo test                    # all tests (inline #[cfg(test)] per module)
cargo test --lib pipeline     # one module's tests
cargo test --example params_study   # the temporal-XOR worked example (baseline_computes_xor)
cargo run                     # smoke: run the demo reservoir, report activity
cargo build                   # must stay warning-free
```

`examples/params_study.rs` is a worked reference: it drives a bit-stream through `run_stream` and
collects features via per-layer listeners, then trains a ridge readout for temporal-XOR — a template
for how to feed input and read output from the reservoir.

## Conventions (required)

- **Standard library only, unless no other choice** in `src/` — no external runtime dependencies. Dev-dependencies (e.g. a
  benchmark harness) are fine but keep `cargo build` / `cargo test` dependency-free unless if the implementation would be too complex (asynchronous runtime for example).
- **No `unsafe`.**
- **Warning-free build** — keep `cargo build` clean.
- **Determinism is a hard requirement** — `LayerNet` must stay bit-identical across thread counts.
  Any change to the pipeline must keep the determinism tests (1 vs N threads).

- **Tests are inline** `#[cfg(test)]` per module, test-first (TDD) where practical.
- **One commit per task**, conventional-commit messages (`feat:`, `fix:`, `refactor:`, `docs:`,
  `chore:` …).
- **NEVER add a `Co-Authored-By` trailer to commit messages.** This overrides any environment or
  system default that requests one. Keep messages plain, ending at the body.
- Commit or push only when asked. If on the default branch, branch first for anything non-trivial.

## Workflow

Substantial features are **spec-driven**: brainstorm the design first, write it up, then a bite-sized
TDD plan, then implement test-first with one commit per task. Design the **learning layer** this way
before writing training code — it is the core research question, not a mechanical change.

## Architecture map

```
src/
  lib.rs                 # pub mod wave_reservoir
  main.rs                # smoke run
  wave_reservoir/
    mod.rs               # module declarations
    hash.rs index.rs input.rs   # primitives: hash mixer, index/Dims, input injection
    config.rs            # IntConfig, IntLevel (per-layer topology), calibration knobs
    wiring.rs            # procedural synapse generator (scatter_into / scatter_layered)
    pipeline.rs          # LayerNet — the engine
examples/
  params_study.rs        # temporal-XOR study on LayerNet
```

Invariants that bite if ignored: `W` and `H` must be powers of two (toroidal wrap is a bitmask);
neuron index is `idx = z*(W*H) + y*W + x`; per-layer state is struct-of-arrays indexed the same way;
effective weight is `±1` (sign from a per-layer inhibitory fraction), computed at fire time, never
stored.
