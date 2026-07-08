# Temporal XOR Bench Implementation Plan (Spec 2b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure temporal XOR `y(t) = u(t) ⊕ u(t−τ)` accuracy vs `τ` for {recurrent, feed-forward} × {ALIF, LIF} — the direct test of whether ALIF's nonlinearity buys nonlinear temporal computation.

**Architecture:** Extract the MC bit-stream + state collection + engine-config into a shared `src/bench/stream.rs` (behavior-preserving refactor), then add `src/bench/temporal_xor.rs` that reuses it plus `RidgeReadout` (thresholded at 0.5 as a linear classifier).

**Tech Stack:** Rust edition 2024, std only, no deps. Inline `#[cfg(test)]` tests.

## Global Constraints

- **Std only**; **no `unsafe`**; **warning-free** build.
- **`f64` allowed in the bench**, engine untouched. Single-threaded, fixed reduction order → deterministic.
- **Determinism is a hard requirement** — pure function of `(seed, config, params)`.
- Tests inline `#[cfg(test)]`, test-first.
- **One commit per task**, conventional commits. **NEVER** a `Co-Authored-By` trailer. **NEVER** push.
- On branch `feat/temporal-xor-bench` (spec committed there).
- Verify each task with `cargo test` + warning-free `cargo build` before committing.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/bench/mod.rs` | module decls | add `pub mod stream;` (T1), `pub mod temporal_xor;` (T2) |
| `src/bench/stream.rs` | shared bit stream, state collection, engine config | new (T1, moved from memory_capacity) |
| `src/bench/memory_capacity.rs` | MC (refactored to use `stream`) | modify (T1) |
| `src/bench/temporal_xor.rs` | XOR task + experiment | new (T2) |

---

### Task 1: Extract shared streaming into `stream.rs` (behavior-preserving)

**Files:**
- Modify: `src/bench/mod.rs`, `src/bench/memory_capacity.rs`
- Create: `src/bench/stream.rs`

**Interfaces:**
- Produces: `bench::stream::{StreamParams, bit, stream_pattern, collect_states, engine_config}` (all `pub(crate)`), moved verbatim from `memory_capacity` with `collect_states` now taking `&StreamParams` and `engine_config` a free function.

- [ ] **Step 1: Create `stream.rs` and declare it**

Add to `src/bench/mod.rs`:
```rust
pub mod stream;
```

Create `src/bench/stream.rs`:
```rust
//! `stream` — the shared streaming substrate for the temporal bench tasks: a binary i.i.d. bit
//! stream, its L0 injection encoding, per-bin spike-count state collection, and the engine-config
//! builder (recurrent vs feed-forward). Used by `memory_capacity` and `temporal_xor`.

use crate::bench::readout::record_response;
use crate::wave_net::config::{Config, LayerConfig};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix, TopologyLevel};

const P_BIT: u64 = 23; // input bit per timestep
const P_STREAM: u64 = 29; // fixed L0 pattern injected on a "1" bit

/// The i.i.d. input bit for timestep `t`.
pub(crate) fn bit(bit_seed: u64, t: usize) -> bool {
    (mix(key(bit_seed, t as u32, 0, 0, P_BIT)) & 1) == 1
}

/// The fixed L0 pattern injected whenever the bit is 1 (same every timestep).
pub(crate) fn stream_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
    let ls = size * size;
    (0..ls).filter(|&s| ((mix(key(seed, s, 0, 0, P_STREAM)) & 0xFFFF) as u32) < density_q16).collect()
}

/// Engine config for the bench. `recurrent` adds level 0 / -1 coupling; feed-forward is level +1
/// only. Both use the dense drive the floored leak requires.
pub(crate) fn engine_config(
    seed: u64,
    size: u32,
    layers: usize,
    baseline_init: i16,
    adapt_bump: i16,
    adapt_decay: u8,
    recurrent: bool,
) -> Config {
    let mut topology = vec![TopologyLevel { level: 1, radius: 3, count: 16 }];
    if recurrent {
        topology.push(TopologyLevel { level: 0, radius: 1, count: 3 });
        topology.push(TopologyLevel { level: -1, radius: 1, count: 3 });
    }
    let layer = LayerConfig {
        topology,
        leak: (3, 5),
        cooldown_base: 2,
        inhibitor_ratio: 0,
        threshold_jitter: 32,
        baseline_init,
        adapt_bump,
        adapt_decay,
    };
    Config { seed, size, layers: vec![layer; layers] }
}

/// Streaming parameters shared by the temporal tasks.
pub(crate) struct StreamParams {
    pub seed: u64,
    pub size: u32,
    pub stream_density_q16: u32,
    pub bit_seed: u64,
    pub bin_waves: usize,
    pub warmup_bins: usize,
    pub collect_bins: usize,
}

/// Drive the continuous bit stream and collect per-bin state rows (per-neuron spike counts over the
/// bin, layers 1..L, ++ a bias 1.0) and the bit sequence. Warmup bins advance the reservoir but are
/// not collected. No reset between bins.
pub(crate) fn collect_states(net: &mut Network, p: &StreamParams) -> (Vec<Vec<f64>>, Vec<f64>) {
    let pattern = stream_pattern(p.seed, p.size, p.stream_density_q16);
    for t in 0..p.warmup_bins {
        let on = bit(p.bit_seed, t);
        for _ in 0..p.bin_waves {
            net.wave(if on { &pattern } else { &[] });
        }
    }
    let mut xs = Vec::with_capacity(p.collect_bins);
    let mut us = Vec::with_capacity(p.collect_bins);
    for i in 0..p.collect_bins {
        let on = bit(p.bit_seed, p.warmup_bins + i);
        let pat = if on { pattern.clone() } else { Vec::new() };
        let counts = record_response(net, p.bin_waves, move |_w| pat.clone());
        let mut row: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        row.push(1.0); // bias
        xs.push(row);
        us.push(if on { 1.0 } else { 0.0 });
    }
    (xs, us)
}
```

- [ ] **Step 2: Refactor `memory_capacity.rs` to use `stream`**

In `src/bench/memory_capacity.rs`: delete the moved items (`P_BIT`, `P_STREAM`, `bit`, `stream_pattern`, `collect_states`, and the body of `McConfig::engine_config`), and update the imports. The top of the file becomes:
```rust
//! `memory_capacity` — the Tier-1 Memory Capacity metric. A binary i.i.d. bit stream is fed to the
//! reservoir in bins of `B` waves (continuously, no reset); per-bin spike counts form the state
//! `x(t)`, and a ridge readout reconstructs `u(t-k)` for each lag `k`. `MC = Σ_k r²_k`.

use crate::bench::readout::RidgeReadout;
use crate::bench::stream::{self, StreamParams};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::Config;
use crate::wave_net::network::Network;
```

Replace `McConfig::engine_config` with a delegating method and add `stream_params`:
```rust
    pub fn engine_config(&self, adapt_bump: i16, recurrent: bool) -> Config {
        stream::engine_config(self.seed, self.size, self.layers, self.baseline_init, adapt_bump, self.adapt_decay, recurrent)
    }

    fn stream_params(&self) -> StreamParams {
        StreamParams {
            seed: self.seed,
            size: self.size,
            stream_density_q16: self.stream_density_q16,
            bit_seed: self.bit_seed,
            bin_waves: self.bin_waves,
            warmup_bins: self.warmup_bins,
            collect_bins: self.collect_bins,
        }
    }
```
(Keep the `McConfig` struct fields and `demo()` unchanged. Delete the old inline `engine_config` topology body and the standalone `collect_states` fn.)

In `memory_capacity()`, replace the collection call:
```rust
    let (xs, us) = stream::collect_states(&mut net, &cfg.stream_params());
```

Update the two MC tests that referenced the moved items:
- `bit_stream_is_deterministic_and_balanced`: `bit(42, t)` → `stream::bit(42, t)`.
- `collect_states_shape_and_determinism`: replace `collect_states(&mut net, &cfg)` with
  `stream::collect_states(&mut net, &cfg.stream_params())`.

(Add `use crate::bench::stream;` inside `mod tests` via the existing `use super::*;` — `stream` is already imported at module scope, so `super::*` re-exports it; if the test references `stream::bit` it resolves. No extra import needed.)

- [ ] **Step 3: Verify MC behavior is unchanged**

Run: `cargo test bench::memory_capacity bench::stream` and `cargo build`
Expected: all MC tests still pass (the refactor is behavior-preserving); warning-free. If `stream::bit`/`stream_params` are only used by tests at this point, no dead-code warning arises because `collect_states`/`engine_config` are used by non-test `memory_capacity()`.

- [ ] **Step 4: Commit**

```bash
git add src/bench/mod.rs src/bench/stream.rs src/bench/memory_capacity.rs
git commit -m "refactor: extract shared bit-stream/state-collection into bench::stream"
```

---

### Task 2: Temporal XOR task + experiment

**Files:**
- Modify: `src/bench/mod.rs`
- Create: `src/bench/temporal_xor.rs`

**Interfaces:**
- Consumes: `stream::{engine_config, collect_states, StreamParams}`, `RidgeReadout`, `calibrate`/`random_l0_input`/`CalibrateParams`.
- Produces: `bench::temporal_xor::{XorConfig, XorCurve, temporal_xor}`.

- [ ] **Step 1: Declare the module and write the failing tests**

Add to `src/bench/mod.rs`:
```rust
pub mod temporal_xor;
```

Create `src/bench/temporal_xor.rs`:
```rust
//! `temporal_xor` — the Tier-1 temporal XOR task `y(t) = u(t) ⊕ u(t-τ)`, swept over `τ`. A thresholded
//! ridge readout on the reservoir state classifies the (non-linearly-separable) XOR; accuracy vs `τ`
//! tests whether ALIF's nonlinearity buys nonlinear temporal computation. Reuses `bench::stream`.

use crate::bench::readout::RidgeReadout;
use crate::bench::stream::{self, StreamParams};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::network::Network;

/// XOR of two bits held as f64 `0.0`/`1.0`.
fn xor(a: f64, b: f64) -> f64 {
    ((a != 0.0) ^ (b != 0.0)) as u8 as f64
}

/// Configuration for the temporal XOR experiment.
#[derive(Clone, Debug)]
pub struct XorConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub baseline_init: i16,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub bit_seed: u64,
    pub stream_density_q16: u32,
    pub bin_waves: usize,
    pub warmup_bins: usize,
    pub collect_bins: usize,
    pub taus: Vec<usize>,
    pub lambda: f64,
    pub train_frac_permille: u64,
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl XorConfig {
    pub fn demo() -> XorConfig {
        let seed = 0x0A17_C0DE;
        XorConfig {
            seed,
            size: 8,
            layers: 3,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
            bit_seed: seed ^ 0xB17,
            stream_density_q16: 20000,
            bin_waves: 3,
            warmup_bins: 100,
            collect_bins: 700,
            taus: vec![1, 2, 4, 8],
            lambda: 1.0,
            train_frac_permille: 700,
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
        }
    }

    fn stream_params(&self) -> StreamParams {
        StreamParams {
            seed: self.seed,
            size: self.size,
            stream_density_q16: self.stream_density_q16,
            bit_seed: self.bit_seed,
            bin_waves: self.bin_waves,
            warmup_bins: self.warmup_bins,
            collect_bins: self.collect_bins,
        }
    }
}

/// XOR classification accuracy (permille) at each `τ`, for one variant.
#[derive(Clone, Debug, PartialEq)]
pub struct XorCurve {
    pub taus: Vec<usize>,
    pub accuracy_permille: Vec<u64>,
}

/// Build+calibrate one variant, stream, and fit a thresholded ridge classifier per `τ` for
/// `u(t) ⊕ u(t-τ)`. `adapt_bump` selects ALIF (>0) vs LIF (0); `recurrent` selects the topology.
pub fn temporal_xor(cfg: &XorConfig, adapt_bump: i16, recurrent: bool) -> XorCurve {
    let mut net = Network::new(stream::engine_config(
        cfg.seed, cfg.size, cfg.layers, cfg.baseline_init, adapt_bump, cfg.adapt_decay, recurrent,
    ));
    let input = random_l0_input(cfg.seed ^ 0x0AB1, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);
    let (xs, us) = stream::collect_states(&mut net, &cfg.stream_params());

    let n = xs.len();
    let tau_max = *cfg.taus.iter().max().unwrap();
    let split = (n as u64 * cfg.train_frac_permille / 1000) as usize;
    // Same design matrix for every τ; rows [tau_max, split) train, [split, n) test.
    let x_train: Vec<Vec<f64>> = xs[tau_max..split].to_vec();
    let x_test: Vec<Vec<f64>> = xs[split..n].to_vec();
    let ridge = RidgeReadout::fit(&x_train, cfg.lambda);

    let mut accuracy_permille = Vec::with_capacity(cfg.taus.len());
    for &tau in &cfg.taus {
        let y_train: Vec<f64> = (tau_max..split).map(|i| xor(us[i], us[i - tau])).collect();
        let w = ridge.weights(&x_train, &y_train);
        let pred = RidgeReadout::predict(&x_test, &w);
        let mut correct = 0usize;
        for (j, i) in (split..n).enumerate() {
            let phat = if pred[j] >= 0.5 { 1.0 } else { 0.0 };
            if phat == xor(us[i], us[i - tau]) {
                correct += 1;
            }
        }
        let acc = if x_test.is_empty() { 0 } else { (correct as u64 * 1000) / x_test.len() as u64 };
        accuracy_permille.push(acc);
    }
    XorCurve { taus: cfg.taus.clone(), accuracy_permille }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_target_is_correct() {
        assert_eq!(xor(0.0, 0.0), 0.0);
        assert_eq!(xor(1.0, 0.0), 1.0);
        assert_eq!(xor(0.0, 1.0), 1.0);
        assert_eq!(xor(1.0, 1.0), 0.0);
    }

    #[test]
    fn temporal_xor_is_deterministic() {
        let cfg = XorConfig::demo();
        let a = temporal_xor(&cfg, cfg.adapt_bump, true);
        let b = temporal_xor(&cfg, cfg.adapt_bump, true);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    #[test]
    fn xor_solvable_above_chance_at_small_tau() {
        // Sanity: the recurrent reservoir + linear readout separates XOR at the smallest lag, well
        // above chance (500). XOR is not linearly separable in the raw inputs, so this confirms the
        // reservoir provides the nonlinear features and the readout works.
        let cfg = XorConfig::demo();
        let lif = temporal_xor(&cfg, 0, true);
        let alif = temporal_xor(&cfg, cfg.adapt_bump, true);
        eprintln!("rec tau {:?}  LIF {:?}  ALIF {:?}", cfg.taus, lif.accuracy_permille, alif.accuracy_permille);
        assert!(lif.accuracy_permille[0] > 600, "reservoir should solve tau=1 XOR (LIF {})", lif.accuracy_permille[0]);
    }

    #[test]
    fn temporal_xor_alif_vs_lif() {
        // The experiment. Print all four curves; assert the OBSERVED relationship after tuning.
        // Honesty gate: report a null / LIF-favoring result rather than forcing an ALIF win.
        let cfg = XorConfig::demo();
        for &rec in &[true, false] {
            let lif = temporal_xor(&cfg, 0, rec);
            let alif = temporal_xor(&cfg, cfg.adapt_bump, rec);
            eprintln!(
                "{} tau {:?}  LIF {:?}  ALIF {:?}",
                if rec { "rec" } else { "ff " }, cfg.taus, lif.accuracy_permille, alif.accuracy_permille
            );
        }
        // Placeholder assertion — REPLACE in Step 4 with the true observed relationship.
        assert!(true);
    }
}
```

- [ ] **Step 2: Run to verify it compiles and the sanity/determinism tests behave**

Run: `cargo test bench::temporal_xor -- --nocapture`
Expected: `xor_target_is_correct` passes; the printed curves appear. `xor_solvable_above_chance_at_small_tau` may pass or fail depending on tuning — proceed to Step 3/4.

- [ ] **Step 3: Read the curves and TUNE (do not fudge)**

Read the printed `rec`/`ff` LIF/ALIF accuracy-vs-`τ` curves. The sanity gate must hold: recurrent solves `τ=1` XOR well above chance. If it doesn't, tune `XorConfig::demo()` (never the mechanism):
- `stream_density_q16`, `bin_waves` (drive), `baseline_init` (operating point), `collect_bins` (less noisy accuracy), `lambda`, `size`/`layers` (feature richness / capacity), `taus` (curve range).

- [ ] **Step 4: Replace the placeholder assertion with the observed truth**

Based on the tuned curves, replace `assert!(true)` in `temporal_xor_alif_vs_lif` with the real relationship, e.g. one of:
- If ALIF wins: `assert!(alif beats lif at the largest solvable τ)` — the tradeoff confirmed.
- If LIF wins / tie: assert that (e.g. `lif ≥ alif` at each τ), and record the finding.

Keep the `xor_solvable_above_chance_at_small_tau` sanity assertion firm. **Honesty gate:** do not assert an ALIF advantage that the curves don't show; a LIF-favoring or null result is the finding to report (it further characterizes ALIF's memory as held-category only). If even the sanity gate can't be met (reservoir can't solve XOR at all), stop and report — the setup or substrate can't do nonlinear temporal separation.

- [ ] **Step 5: Full suite + warning-free**

Run: `cargo test` and `cargo build`
Expected: all pass; warning-free.

- [ ] **Step 6: Commit**

```bash
git add src/bench/mod.rs src/bench/temporal_xor.rs
git commit -m "feat: temporal XOR experiment (accuracy vs tau, ALIF vs LIF)"
```

---

## Self-review

**Spec coverage:**
- Shared streaming refactor (`stream.rs`; MC unchanged) → Task 1.
- XOR target `u(t) ⊕ u(t−τ)`, swept `τ`, thresholded `RidgeReadout` classifier, four runs → Task 2.
- Accuracy-vs-`τ` curves; firm sanity assertion (`τ=1` above chance) + determinism + observed ALIF-vs-LIF → Task 2 tests.
- `f64` in bench only; engine untouched; reuses harness + ridge → throughout.

**Placeholder scan:** the `assert!(true)` in `temporal_xor_alif_vs_lif` is intentional and explicitly replaced in Task 2 Step 4 with the observed relationship — it must not survive to commit.

**Type consistency:** `stream::{engine_config, collect_states, StreamParams}` signatures match both `memory_capacity` (Task 1) and `temporal_xor` (Task 2) call sites. `XorConfig`/`XorCurve`/`temporal_xor` consistent. `RidgeReadout::{fit,weights,predict}` reused unchanged.

**Note on Task 2:** honesty gate mirrors MC — sanity (XOR solvable at small τ) is firm; the ALIF-vs-LIF direction is the finding, asserted to the truth, never forced.
