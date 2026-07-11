# wave_net throughput benchmark — baseline

**Date:** 2026-07-11
**Status:** approved design
**Scope:** a criterion benchmark that measures the `wave_net` engine's raw wave throughput
(waves/second) at a fixed 32×32 × 5 feed-forward configuration, calibrated to ~10% firing. This is
the **baseline** the upcoming engine-performance work optimizes against.

## Motivation

The single-threaded integer engine is too slow to run the side-car scaling sweeps multi-seed at
size ≥ 64 (`docs/experiments_results.md`, and `AGENTS.md` "Blocker"). Before optimizing, we need a
stable, reproducible throughput number. This spec defines that measurement. **The optimization
itself, and any training benchmarks, are out of scope here** — they come later against this baseline.

## Goal

Report **waves/second** for `wave_net` at:

- `size = 32` (32×32 = 1024 neurons/layer), `layers = 5` (5120 neurons total)
- **uniform feed-forward** topology (the same `LayerConfig` on every layer)
- calibrated to a **~10% per-layer firing rate**, tuned in setup and **excluded from the measured
  region**
- driven by a **random-noise** L0 input

## Non-goals

- Optimizing the engine (next effort; this only establishes the baseline).
- Training / learning-rule benchmarks (later).
- Side-car or `wave_state_machine`-reference benchmark groups (can be added later; a uniform FF
  baseline is the clean starting point).
- Cross-machine absolute comparability — the workload is fixed and deterministic, but wall-time is
  machine-dependent (as with any benchmark).

## Wiring

- New file `benches/throughput.rs`, compiled as a separate crate against `wave-net`'s **public API
  only**.
- `Cargo.toml`:
  - `[dev-dependencies]` → `criterion = { version = "0.5", features = ["html_reports"] }`
  - `[[bench]]` → `name = "throughput"`, `harness = false`
- **Convention:** criterion is a **dev-dependency**. It never enters `src/`, so the "standard
  library only in `src/`, warning-free library build" rule is preserved — the same spirit as
  `blake3` being a test-only feature. The bench touches no private engine internals.

## Config under test

`size = 32`, `layers = 5`, fixed `seed`. Every layer uses the current feed-forward `LayerConfig`
values (inlined in the bench, so the benchmark is self-documenting and does not depend on a private
`bench::rsnn` helper):

```
LayerConfig {
    topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
    leak: (3, 5),
    cooldown_base: 2,
    inhibitor_ratio: 0,
    threshold_jitter: 32,
    baseline_init: 6,
    adapt_bump: 5,
    adapt_decay: 6,
}
```

Rationale: the dominant engine cost is regenerating a firer's synapses at fire time, so the per-layer
synapse count (`count: 32`) is the primary throughput driver. A uniform FF stack makes that load
homogeneous and easy to reason about — a pure per-neuron forward-scatter baseline.

**Departure from the literal FF `engine_config` (decided during implementation, 2026-07-11).** The
literal defaults (`up_count 16, adapt_bump 20`) are **sub-critical at depth 5**: the cue dies with
depth (measured per-layer rates ≈ `30, 3.7, 0.6, 0.6, 0.7 %`), so calibration **cannot** reach a 10%
operating point — the documented "cue dies with depth" limitation (`AGENTS.md`; training's `rate_reg`
revives a deep FF stack, calibration cannot). To make the spec's own "~10% firing" requirement
achievable, the config uses the scaling study's **forward-drive threshold `up_count = 32`** and
**softened adaptation `adapt_bump = 5`**. At those values the uniform pure-FF stack genuinely
propagates and calibrates to ~10% through all layers (measured `30.5, 11.5, 8.0, 6.3, 5.3 %`), so the
throughput number reflects a real spiking load rather than a mostly-cold deep stack.

## Named constants (bench-local)

- `SIZE: u32 = 32`
- `LAYERS: usize = 5`
- `SEED: u64` — a fixed value (`0xC0FFEE_1234_5678`).
- `NOISE_FRACTION_Q16: u32 = 20000` — L0 injection density for the random-noise drive (~30% of
  65536), the sustained density the training code calibrates against. The same value drives both
  calibration and measurement. (A ~10% sparse drive starves the hidden layers on top of the
  criticality decay, so the denser sustained drive is used.)
- `WAVES_PER_ITER: u64 = 256` — waves run per measured criterion iteration; also the
  `Throughput::Elements` count so criterion reports waves/second directly.

## Setup (outside the measured region)

1. Build `Network::new(config)`.
2. Build the input closure `input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16)` — the
   deterministic per-wave sparse random L0 drive already provided by the engine.
3. Calibrate: `net.calibrate(&CalibrateParams { target_permille: 100, ..default }, &input)`. This
   tunes per-layer baselines so each layer fires near 10% on this drive. **The same `input`
   closure is used for both calibration and measurement**, so the calibrated operating point
   transfers to the measured run.
4. **Operating-point guard.** Reproduce the private `measure_layer_rates` through the public
   listener API: install a spike-count listener per layer via `on_layer`, `reset_state`, run a
   warmup window (discarded), then a counted window; compute per-layer firing rate. **Print the
   rates** (so the bench output records the regime) and **assert** each computational layer is
   within a tolerance band of 10% (e.g. `[3%, 25%]`). This guarantees we are benchmarking a *live,
   propagating* net — not a dead or saturated one — and fails loudly if calibration drifts. Clear
   the listeners afterward so they add no cost to the measured region.

## Measurement

- **Pre-generate** a fixed ring of `WAVES_PER_ITER` noise vectors (`Vec<Vec<u32>>`) by calling the
  `input` closure for waves `0..WAVES_PER_ITER`, **outside** the timed loop. This measures the pure
  engine, not the input-hash RNG.
- `group.throughput(Throughput::Elements(WAVES_PER_ITER))` → criterion reports elem/s = **waves/s**.
- Timed closure: `b.iter(|| { for v in &noise { net.wave(v); } })`. `Network::wave` takes `&self`
  (interior mutability via per-layer `Mutex` + atomic wave id), so the closure borrows `net`
  immutably; no per-iteration setup.

### State across iterations: continuous steady-state (decided)

Build + calibrate + warm the net **once**; then let criterion measure back-to-back batches of
`WAVES_PER_ITER` waves with **no per-iteration `reset_state`**. ALIF adaptation settles into its
self-regulated ~10% regime during criterion's own warmup iterations and stays there, so every
measured batch reflects the real steady operating point.

Rejected alternative: `iter_batched` with a per-batch `reset_state` + warmup. That would repeatedly
measure the boots-hot transient (unrepresentative of steady operation) and pay reset/warmup cost
inside the harness. The step-4 guard already confirms the regime is live before the long run.

Determinism note: state carries across iterations, but the per-wave input is the fixed pre-generated
ring, so the engine's trajectory is a pure function of `(seed, config, ring)`; criterion only
measures wall-time over that fixed trajectory.

## Verification

- `cargo build` stays warning-free.
- `cargo bench --no-run` compiles the bench.
- One real `cargo bench` run confirms it emits a waves/second figure and the step-4 firing-rate
  guard passes (printed rates near 10%).
- `cargo test` is unaffected (benches do not run under it), so the suite stays fast.

## File / structure summary

```
Cargo.toml            # + [dev-dependencies] criterion, + [[bench]] throughput (harness = false)
benches/
  throughput.rs       # build → calibrate → guard(print+assert ~10%) → pre-gen noise ring →
                      #   criterion group with Throughput::Elements(WAVES_PER_ITER), continuous iter
```

## Baseline result (2026-07-11)

First measured baseline, `cargo bench --bench throughput` (release/`bench` profile):

- per-layer firing rate: `30.5, 11.5, 8.0, 6.3, 5.3 %` (L0 = injection density; layers 1–4 ~10%)
- **throughput ≈ 7.07 Kelem/s ≈ 7,070 waves/second** (36.2 ms per 256-wave batch)

This is the number the upcoming engine-performance work optimizes against. (Machine-dependent
absolute wall-time; the workload is fixed and deterministic.)
