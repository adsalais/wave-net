# wave_net Throughput Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline, per this repo's AGENTS.md — never the subagent-driven option). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a criterion benchmark that reports the `wave_net` engine's raw wave throughput (waves/second) on a 32×32 × 5 uniform feed-forward network calibrated to ~10% firing — the baseline for the upcoming engine-performance work.

**Architecture:** A new `benches/throughput.rs` compiled as a separate crate against `wave-net`'s public API. Setup (build → calibrate → assert live operating point) runs outside the measured region; the measured region replays a fixed, pre-generated ring of random-noise L0 inputs through `Network::wave` continuously (no per-iteration reset), so criterion's `Throughput::Elements` reports waves/second at the steady ~10% regime.

**Tech Stack:** Rust (edition 2024), criterion 0.5 (dev-dependency only), the existing `wave_net` engine public API.

## Global Constraints

- **Determinism:** results are a pure function of `(seed, config, input)`. Fixed `SEED`; pre-generated deterministic noise; no wall-clock in the engine path.
- **`src/` stays standard-library-only and warning-free.** criterion is a **dev-dependency** — it never enters `src/`. `cargo build` must stay warning-free.
- **Do not modify the engine** (`src/wave_net/**`) or `wave_state_machine`. This task is purely additive: `Cargo.toml` + `benches/throughput.rs`. The bench uses only public API.
- **Conventional-commit messages** (`feat:`/`chore:` …). **One commit per task.**
- **NEVER add a `Co-Authored-By` trailer.** Keep messages plain.
- Branch `perf/throughput-bench` is already checked out. Do not push.
- **External import paths** (crate `wave_net` → inner module `wave_net`, so paths double up):
  `wave_net::wave_net::network::Network`, `::config::{Config, LayerConfig}`,
  `::calibrate::{random_l0_input, CalibrateParams}`, `::synapse::TopologyLevel`.

## File Structure

```
Cargo.toml            # MODIFY: + [dev-dependencies] criterion, + [[bench]] throughput (harness = false)
benches/
  throughput.rs       # CREATE: constants, build_config(), setup_net(), measure_rates(),
                      #   assert_operating_point(), bench_throughput() + criterion_main
```

One file owns the whole bench: config construction, calibration setup, the operating-point guard, and the measured loop. It touches no engine internals.

## Verification note for benches

Benches do **not** run under `cargo test`. The run-check at each task uses criterion's test mode:
`cargo bench --bench throughput -- --test` runs the benchmark **once** (setup + guard + a single
measured pass, no timing) — this executes `assert_operating_point`, so it is the real behavioural
check. A full `cargo bench` produces the waves/second figure.

---

### Task 1: Wire criterion + compiling scaffold

Add the criterion dev-dependency and `[[bench]]`, and a `benches/throughput.rs` that builds the
32×32×5 FF net and runs `WAVES_PER_ITER` waves per iteration with a `Throughput::Elements`
annotation. No calibration or guard yet — this proves the wiring and public-API path compile and run.

**Files:**
- Modify: `Cargo.toml` (add `[dev-dependencies]` and `[[bench]]`)
- Create: `benches/throughput.rs`

**Interfaces:**
- Consumes (from engine public API): `Network::new(Config) -> Network`, `Network::wave(&self, &[u32])`,
  `random_l0_input(u64, u32, u32) -> impl Fn(usize) -> Vec<u32>`, `Config`, `LayerConfig`, `TopologyLevel`.
- Produces (used by later tasks): `const SIZE`, `const LAYERS`, `const SEED`, `const NOISE_FRACTION_Q16`,
  `const WAVES_PER_ITER`, `fn build_config() -> Config`, `fn bench_throughput(&mut Criterion)`.

- [ ] **Step 1: Add criterion dev-dependency and bench target to `Cargo.toml`**

Append after the existing `[features]` block:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "throughput"
harness = false
```

- [ ] **Step 2: Create `benches/throughput.rs` scaffold**

```rust
//! Baseline throughput benchmark for the `wave_net` engine.
//!
//! Reports waves/second on a 32×32 × 5 uniform feed-forward network. Calibration and the
//! operating-point guard run in setup (outside the measured region, added in later tasks); the
//! measured region runs random-noise L0 input through `Network::wave`.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use wave_net::wave_net::calibrate::random_l0_input;
use wave_net::wave_net::config::{Config, LayerConfig};
use wave_net::wave_net::network::Network;
use wave_net::wave_net::synapse::TopologyLevel;

const SIZE: u32 = 32;
const LAYERS: usize = 5;
const SEED: u64 = 0xC0FFEE_1234_5678;
/// L0 injection density for the random-noise drive, Q16 (~10% of 65536). Same value for
/// calibration and measurement so the calibrated operating point transfers.
const NOISE_FRACTION_Q16: u32 = 6554;
/// Waves per measured iteration; also the `Throughput::Elements` count → criterion reports waves/s.
const WAVES_PER_ITER: u64 = 256;

/// The 32×32 × 5 uniform feed-forward config under test. Values are the current engine FF defaults,
/// inlined so the benchmark is self-documenting.
fn build_config() -> Config {
    let layer = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: 3, count: 16 }],
        leak: (3, 5),
        cooldown_base: 2,
        inhibitor_ratio: 0,
        threshold_jitter: 32,
        baseline_init: 6,
        adapt_bump: 20,
        adapt_decay: 6,
    };
    Config { seed: SEED, size: SIZE, layers: vec![layer; LAYERS] }
}

fn bench_throughput(c: &mut Criterion) {
    let net = Network::new(build_config());
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    group.bench_function("ff_32x32x5", |b| {
        b.iter(|| {
            for w in 0..WAVES_PER_ITER as usize {
                net.wave(&input(w));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
```

- [ ] **Step 3: Verify the build stays warning-free**

Run: `cargo build 2>&1 | tail -5`
Expected: builds clean, no warnings (the library is unchanged; the bench is not built by `cargo build`).

- [ ] **Step 4: Verify the bench compiles and runs once in test mode**

Run: `cargo bench --bench throughput -- --test 2>&1 | tail -20`
Expected: compiles; runs the `throughput/ff_32x32x5` benchmark once with no panic (exit 0).

- [ ] **Step 5: Sanity-run the full bench once (scaffold number)**

Run: `cargo bench --bench throughput 2>&1 | tail -20`
Expected: criterion prints a `throughput/ff_32x32x5` result with a `thrpt: [... Kelem/s]` (=waves/s)
line. (This is an *uncalibrated* scaffold number, not the baseline — calibration lands in Task 2.)

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock benches/throughput.rs
git commit -m "chore: criterion throughput bench scaffold (32x32x5 FF, waves/s)"
```

---

### Task 2: Calibrate + operating-point guard

Move construction into a `setup_net()` that calibrates to 10% firing, and add a `measure_rates()`
helper (reproducing the engine's private `measure_layer_rates` through the public listener API) plus
`assert_operating_point()` that prints per-layer rates and asserts the computational layers are in a
live band. Wire both into `bench_throughput` so the net is calibrated and verified live before
measurement.

**Files:**
- Modify: `benches/throughput.rs`

**Interfaces:**
- Consumes (engine public API): `Network::calibrate(&mut self, &CalibrateParams, &impl Fn(usize)->Vec<u32>)`,
  `CalibrateParams`, `Network::on_layer(&mut self, usize, Box<dyn Fn(usize,&[u32])+Send+Sync>)`,
  `Network::clear_listeners(&mut self)`, `Network::reset_state(&self)`, `Network::layer_count(&self) -> usize`,
  `Network::size(&self) -> u32`.
- Produces (used by Task 3): `fn setup_net() -> Network` (returns a calibrated net),
  `fn assert_operating_point(&mut Network)`.

- [ ] **Step 1: Add the `std::sync` and `CalibrateParams` imports**

Add to the imports at the top of `benches/throughput.rs`:

```rust
use std::sync::{Arc, Mutex};

use wave_net::wave_net::calibrate::CalibrateParams;
```

(The existing `use wave_net::wave_net::calibrate::random_l0_input;` line stays; you may merge them
into `use wave_net::wave_net::calibrate::{random_l0_input, CalibrateParams};`.)

- [ ] **Step 2: Add `setup_net()` — construct + calibrate to 10%**

Add below `build_config()`:

```rust
/// Build and calibrate the net to ~10% per-layer firing on the random-noise drive. Calibration is
/// setup, not measured. The same drive is used for measurement so the operating point transfers.
fn setup_net() -> Network {
    let mut net = Network::new(build_config());
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let params = CalibrateParams { target_permille: 100, ..CalibrateParams::default() };
    net.calibrate(&params, &input);
    net
}
```

- [ ] **Step 3: Add `measure_rates()` — per-layer firing rate via public listeners**

Add below `setup_net()`. This mirrors the engine's private `measure_layer_rates`, using only the
public listener API:

```rust
/// Per-layer firing rate (fraction of neurons firing per wave) over a counted window, measured
/// through the public listener API. Warmup waves are discarded. Leaves the net in a warmed state.
fn measure_rates(
    net: &mut Network,
    warmup: usize,
    waves: usize,
    input: &impl Fn(usize) -> Vec<u32>,
) -> Vec<f64> {
    let l = net.layer_count();
    let counts = Arc::new(Mutex::new(vec![0u64; l]));
    for z in 0..l {
        let c = counts.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| {
            c.lock().unwrap()[z] += fired.len() as u64;
        }));
    }
    net.reset_state();
    for w in 0..warmup {
        net.wave(&input(w));
    }
    counts.lock().unwrap().iter_mut().for_each(|x| *x = 0); // discard warmup
    for w in 0..waves {
        net.wave(&input(warmup + w));
    }
    net.clear_listeners();
    let counts = std::mem::take(&mut *counts.lock().unwrap());
    let ls = (net.size() as u64) * (net.size() as u64);
    let denom = (ls * waves as u64) as f64;
    counts.iter().map(|&s| s as f64 / denom).collect()
}
```

- [ ] **Step 4: Add `assert_operating_point()` — print + guard the live regime**

Add below `measure_rates()`:

```rust
/// Confirm the calibrated net is in a live, propagating regime before the timed run: print the
/// per-layer firing rates and assert every computational layer (1..L) is within a generous live
/// band around 10%. Fails loudly if calibration drifted (dead or saturated net).
fn assert_operating_point(net: &mut Network) {
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let rates = measure_rates(net, 32, 128, &input);
    let pct: Vec<f64> = rates.iter().map(|r| (r * 1000.0).round() / 10.0).collect();
    println!("wave_net 32x32x5 FF per-layer firing rate (%): {pct:?}");
    for z in 1..net.layer_count() {
        assert!(
            (0.03..=0.25).contains(&rates[z]),
            "layer {z} firing rate {:.3} outside live band [0.03, 0.25] — calibration drifted",
            rates[z]
        );
    }
}
```

- [ ] **Step 5: Wire calibration + guard into `bench_throughput`**

Replace the first two lines of `bench_throughput` (the `Network::new(...)` and `random_l0_input`
lines) so it calibrates and asserts the operating point before measuring:

```rust
fn bench_throughput(c: &mut Criterion) {
    let mut net = setup_net();
    assert_operating_point(&mut net);
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    group.bench_function("ff_32x32x5", |b| {
        b.iter(|| {
            for w in 0..WAVES_PER_ITER as usize {
                net.wave(&input(w));
            }
        })
    });
    group.finish();
}
```

- [ ] **Step 6: Verify the guard runs and passes in test mode**

Run: `cargo bench --bench throughput -- --test 2>&1 | tail -20`
Expected: prints `wave_net 32x32x5 FF per-layer firing rate (%): [...]` with the computational
layers near 10%, no assertion panic, exit 0.

Contingency: if a **deep** layer's rate falls **below 0.03** (starved cue with depth), the drive is
too sparse — raise `NOISE_FRACTION_Q16` toward `20000` (≈30%, the density the training code uses;
still random noise) and re-run. If a layer is **above 0.25** (saturated), lower it. Do **not** widen
the band to hide a dead/saturated regime.

- [ ] **Step 7: Verify the build stays warning-free**

Run: `cargo build 2>&1 | tail -5`
Expected: no warnings.

- [ ] **Step 8: Commit**

```bash
git add benches/throughput.rs
git commit -m "feat: calibrate throughput bench to 10% + operating-point guard"
```

---

### Task 3: Pre-generated noise ring + continuous steady-state measurement

Move noise generation out of the timed loop (pre-generate a fixed ring of `WAVES_PER_ITER` input
vectors) so the measured region is the pure engine, and replay the ring continuously with no
per-iteration reset (the decided steady-state methodology).

**Files:**
- Modify: `benches/throughput.rs`

**Interfaces:**
- Consumes: `setup_net()`, `assert_operating_point()` (Task 2); `Network::wave` (Task 1).
- Produces: final `bench_throughput` reporting steady-state waves/second.

- [ ] **Step 1: Pre-generate the noise ring and replay it in the timed loop**

Replace `bench_throughput` with:

```rust
fn bench_throughput(c: &mut Criterion) {
    // Setup (not measured): build, calibrate to 10%, assert the regime is live.
    let mut net = setup_net();
    assert_operating_point(&mut net);

    // Pre-generate the fixed noise ring OUTSIDE the timed loop so we measure the pure engine,
    // not the input-hash RNG.
    let input = random_l0_input(SEED, SIZE, NOISE_FRACTION_Q16);
    let noise: Vec<Vec<u32>> = (0..WAVES_PER_ITER as usize).map(&input).collect();

    // Measured region: replay the ring continuously (no per-iteration reset). ALIF adaptation
    // stays in its self-regulated ~10% regime across iterations, so each batch reflects steady state.
    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(WAVES_PER_ITER));
    group.bench_function("ff_32x32x5", |b| {
        b.iter(|| {
            for v in &noise {
                net.wave(v);
            }
        })
    });
    group.finish();
}
```

- [ ] **Step 2: Verify test mode still passes (guard + single measured pass)**

Run: `cargo bench --bench throughput -- --test 2>&1 | tail -20`
Expected: prints the firing-rate line, no panic, exit 0.

- [ ] **Step 3: Run the full benchmark and capture the baseline number**

Run: `cargo bench --bench throughput 2>&1 | tail -25`
Expected: criterion reports `throughput/ff_32x32x5` with `time:` and `thrpt: [... Kelem/s]` — the
Kelem/s figure is the baseline **waves/second**. Note it (this is the number the optimization work
will improve).

- [ ] **Step 4: Verify the build stays warning-free**

Run: `cargo build 2>&1 | tail -5`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add benches/throughput.rs
git commit -m "feat: steady-state waves/s measurement over pre-generated noise ring"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-11-wavenet-throughput-bench-design.md`):
- Wiring (dev-dep + `[[bench]]`, public-API only) → Task 1.
- Config under test (32×32×5 FF, exact `LayerConfig`) → Task 1 `build_config()`.
- Named constants (`SIZE`, `LAYERS`, `SEED`, `NOISE_FRACTION_Q16`, `WAVES_PER_ITER`) → Task 1.
- Setup: construct + calibrate to 10% (same drive) → Task 2 `setup_net()`.
- Operating-point guard (reproduce `measure_layer_rates` via listeners, print + assert) → Task 2.
- Measurement: pre-generated ring, `Throughput::Elements`, continuous steady-state → Task 3.
- Verification (`cargo build` clean, `--test` runs, full `cargo bench` emits waves/s) → each task's
  verify steps.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; the `SEED`/fraction/waves
values are concrete.

**Type consistency:** `build_config`/`setup_net`/`measure_rates`/`assert_operating_point`/
`bench_throughput` names and signatures are used identically across tasks; constants declared once in
Task 1 and reused; import paths match the `wave_net::wave_net::…` convention throughout.
