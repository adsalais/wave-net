# Memory Capacity Bench Implementation Plan (Spec 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the reusable `f64` ridge readout + `linalg` and measure Memory Capacity (`MC = Σ_k r²_k`) for {recurrent, feed-forward} × {ALIF, LIF}, showing quantitatively how far back the reservoir remembers and that ALIF extends the tail.

**Architecture:** Extend `src/bench/`. `linalg.rs` = `f64` LU solve + `XᵀX`/`Xᵀy`. `readout.rs` gains `RidgeReadout` (factor `XᵀX+λI` once, back-substitute per target). `memory_capacity.rs` = binary bit stream, binned continuous drive, per-bin state via `record_response`, the ridge memory curve, and the four-run experiment. Engine untouched; `f64` only in the bench.

**Tech Stack:** Rust edition 2024, std only, no deps. Inline `#[cfg(test)]` tests.

## Global Constraints

- **Std only**; **no `unsafe`**; **warning-free** build.
- **`f64` allowed in the bench**, integer engine untouched. Single-threaded, fixed reduction order → deterministic.
- **Determinism is a hard requirement** — pure function of `(seed, config, params)`.
- Tests inline `#[cfg(test)]`, test-first.
- **One commit per task**, conventional commits. **NEVER** a `Co-Authored-By` trailer. **NEVER** push.
- On branch `feat/memory-capacity-bench` (spec committed there).
- Verify each task with `cargo test` + warning-free `cargo build` before committing.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/bench/mod.rs` | module decls | add `pub mod linalg;` (T1), `pub mod memory_capacity;` (T3) |
| `src/bench/linalg.rs` | f64 LU solve, `XᵀX`, `Xᵀy` | new (T1) |
| `src/bench/readout.rs` | add `RidgeReadout` | modify (T2) |
| `src/bench/memory_capacity.rs` | bit stream, state collection, MC experiment | new (T3, T4) |

---

### Task 1: `linalg` — f64 LU solve + normal-equation builders

**Files:**
- Modify: `src/bench/mod.rs`
- Create: `src/bench/linalg.rs`

**Interfaces:**
- Produces: `bench::linalg::{Lu, xt_x, xt_y}`. `Lu::factor(Vec<Vec<f64>>) -> Lu`, `Lu::solve(&self, &[f64]) -> Vec<f64>`. `xt_x(&[Vec<f64>]) -> Vec<Vec<f64>>` (d×d), `xt_y(&[Vec<f64>], &[f64]) -> Vec<f64>` (d).

- [ ] **Step 1: Declare the module and write the failing test**

Add to `src/bench/mod.rs`:
```rust
pub mod linalg;
```

Create `src/bench/linalg.rs`:
```rust
//! `linalg` — the small f64 linear algebra the bench readouts need: an LU solve (Gaussian
//! elimination with partial pivoting, factored once and reused across right-hand sides) and the
//! normal-equation builders `XᵀX` and `Xᵀy`. Single-threaded and deterministic.

/// LU factorization with partial pivoting of a square matrix; reusable across right-hand sides.
pub struct Lu {
    lu: Vec<Vec<f64>>, // L (unit diag, below) and U (on/above) packed
    piv: Vec<usize>,   // row permutation
    n: usize,
}

impl Lu {
    /// Factor a square matrix. Panics if it is empty or exactly singular.
    pub fn factor(mut a: Vec<Vec<f64>>) -> Lu {
        let n = a.len();
        assert!(n > 0 && a.iter().all(|r| r.len() == n), "matrix must be square and non-empty");
        let mut piv: Vec<usize> = (0..n).collect();
        for col in 0..n {
            let mut pmax = col;
            let mut vmax = a[col][col].abs();
            for r in (col + 1)..n {
                if a[r][col].abs() > vmax {
                    vmax = a[r][col].abs();
                    pmax = r;
                }
            }
            assert!(vmax > 0.0, "singular matrix");
            if pmax != col {
                a.swap(col, pmax);
                piv.swap(col, pmax);
            }
            let pivot = a[col][col];
            for r in (col + 1)..n {
                let f = a[r][col] / pivot;
                a[r][col] = f; // store multiplier
                for c in (col + 1)..n {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
        Lu { lu: a, piv, n }
    }

    /// Solve `A x = b`.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = b[self.piv[i]];
        }
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.lu[i][j] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.lu[i][j] * x[j];
            }
            x[i] /= self.lu[i][i];
        }
        x
    }
}

/// `Xᵀ X` — square, dimension = number of columns of `x`.
pub fn xt_x(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let mut a = vec![vec![0.0; d]; d];
    for row in x {
        for i in 0..d {
            let ri = row[i];
            for j in 0..d {
                a[i][j] += ri * row[j];
            }
        }
    }
    a
}

/// `Xᵀ y`.
pub fn xt_y(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let mut v = vec![0.0; d];
    for (row, &yi) in x.iter().zip(y) {
        for i in 0..d {
            v[i] += row[i] * yi;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solves_known_system() {
        // [[2,1],[1,3]] x = [3,5] -> x = [0.8, 1.4]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let lu = Lu::factor(a);
        let x = lu.solve(&[3.0, 5.0]);
        assert!((x[0] - 0.8).abs() < 1e-9 && (x[1] - 1.4).abs() < 1e-9, "got {x:?}");
    }
}
```

- [ ] **Step 2: Run to verify it passes**

Run: `cargo test bench::linalg` and `cargo build`
Expected: `lu_solves_known_system` passes; warning-free. (Self-contained; green immediately.)

- [ ] **Step 3: Commit**

```bash
git add src/bench/mod.rs src/bench/linalg.rs
git commit -m "feat: bench f64 linalg (LU solve + normal-equation builders)"
```

---

### Task 2: `RidgeReadout` — ridge regression on the shared LU factor

**Files:**
- Modify: `src/bench/readout.rs`

**Interfaces:**
- Consumes: `bench::linalg::{Lu, xt_x, xt_y}`.
- Produces: `bench::readout::RidgeReadout` with `fit(x_train: &[Vec<f64>], lambda: f64) -> RidgeReadout`, `weights(&self, x_train: &[Vec<f64>], y_train: &[f64]) -> Vec<f64>`, and `predict(x: &[Vec<f64>], w: &[f64]) -> Vec<f64>`.

- [ ] **Step 1: Write the failing test**

Add to `src/bench/readout.rs`, inside `#[cfg(test)] mod tests` (it already has `use super::*;`):
```rust
    use crate::wave_net::synapse::mix;

    fn synth_design(n: usize, d: usize) -> Vec<Vec<f64>> {
        // Deterministic rows in [-1,1) with a trailing bias column of 1.0.
        (0..n)
            .map(|i| {
                let mut row: Vec<f64> = (0..d - 1)
                    .map(|j| {
                        let h = mix(((i as u64) << 20) ^ ((j as u64) << 3) ^ 0x9E37_79B9);
                        ((h & 0xFFFF) as f64 / 65536.0) * 2.0 - 1.0
                    })
                    .collect();
                row.push(1.0);
                row
            })
            .collect()
    }

    #[test]
    fn ridge_recovers_planted_linear_map() {
        let (n, d) = (60usize, 4usize); // 3 features + bias
        let x = synth_design(n, d);
        let w_true = [1.5, -2.0, 0.5, 0.25];
        let y: Vec<f64> = x.iter().map(|r| r.iter().zip(&w_true).map(|(a, b)| a * b).sum()).collect();
        let ridge = RidgeReadout::fit(&x, 1e-6);
        let w = ridge.weights(&x, &y);
        for (got, want) in w.iter().zip(&w_true) {
            assert!((got - want).abs() < 1e-2, "weight {got} != {want}");
        }
        let pred = RidgeReadout::predict(&x, &w);
        let max_err = pred.iter().zip(&y).map(|(p, t)| (p - t).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-2, "prediction error {max_err} too large");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::readout::tests::ridge_recovers_planted_linear_map`
Expected: FAIL to compile — `RidgeReadout` not defined.

- [ ] **Step 3: Implement `RidgeReadout`**

Add to `src/bench/readout.rs` (top, after the existing `use` lines):
```rust
use crate::bench::linalg::{xt_x, xt_y, Lu};

/// Ridge-regression linear readout. Factors `(XᵀX + λI)` once from the training design matrix
/// (which must already include a bias column); each target column is solved by back-substitution.
pub struct RidgeReadout {
    lu: Lu,
}

impl RidgeReadout {
    pub fn fit(x_train: &[Vec<f64>], lambda: f64) -> RidgeReadout {
        let mut a = xt_x(x_train);
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += lambda;
        }
        RidgeReadout { lu: Lu::factor(a) }
    }

    /// Weight vector reconstructing one target column `y_train` from `x_train`.
    pub fn weights(&self, x_train: &[Vec<f64>], y_train: &[f64]) -> Vec<f64> {
        self.lu.solve(&xt_y(x_train, y_train))
    }

    /// Prediction `X · w`.
    pub fn predict(x: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
        x.iter().map(|row| row.iter().zip(w).map(|(a, b)| a * b).sum()).collect()
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test bench::readout` and `cargo build`
Expected: PASS; warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/bench/readout.rs
git commit -m "feat: bench ridge-regression readout on shared LU factor"
```

---

### Task 3: Bit stream + binned state collection + `McConfig`

**Files:**
- Modify: `src/bench/mod.rs`
- Create: `src/bench/memory_capacity.rs`

**Interfaces:**
- Consumes: `record_response` (Spec 1), `Network`, `calibrate`/`random_l0_input`, `Config`/`LayerConfig`/`TopologyLevel`, `synapse::{key, mix}`.
- Produces: `McConfig` (+ `McConfig::demo()`, `engine_config`), module-private `bit`, `stream_pattern`, `collect_states`.

- [ ] **Step 1: Declare the module and write the failing tests**

Add to `src/bench/mod.rs`:
```rust
pub mod memory_capacity;
```

Create `src/bench/memory_capacity.rs`:
```rust
//! `memory_capacity` — the Tier-1 Memory Capacity metric. A binary i.i.d. bit stream is fed to the
//! reservoir in bins of `B` waves (continuously, no reset); per-bin spike counts form the state
//! `x(t)`, and a ridge readout reconstructs `u(t-k)` for each lag `k`. `MC = Σ_k r²_k`.

use crate::bench::readout::record_response;
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::{Config, LayerConfig};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix, TopologyLevel};

const P_BIT: u64 = 23; // input bit per timestep
const P_STREAM: u64 = 29; // fixed L0 pattern injected on a "1" bit

/// The i.i.d. input bit for timestep `t`.
fn bit(bit_seed: u64, t: usize) -> bool {
    (mix(key(bit_seed, t as u32, 0, 0, P_BIT)) & 1) == 1
}

/// The fixed L0 pattern injected whenever the bit is 1 (same every timestep).
fn stream_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
    let ls = size * size;
    (0..ls).filter(|&s| ((mix(key(seed, s, 0, 0, P_STREAM)) & 0xFFFF) as u32) < density_q16).collect()
}

/// Configuration for the Memory Capacity experiment.
#[derive(Clone, Debug)]
pub struct McConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub baseline_init: i16,
    pub adapt_bump: i16, // ALIF value; LIF passes 0
    pub adapt_decay: u8,
    pub bit_seed: u64,
    pub stream_density_q16: u32,
    pub bin_waves: usize,
    pub warmup_bins: usize,
    pub collect_bins: usize,
    pub k_lags: usize,
    pub lambda: f64,
    pub train_frac_permille: u64,
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl McConfig {
    /// Small, fast, deterministic config for the inline test.
    pub fn demo() -> McConfig {
        let seed = 0x3EC0_DE5;
        McConfig {
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
            k_lags: 20,
            lambda: 1.0,
            train_frac_permille: 700,
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
        }
    }

    /// Build the engine config. `recurrent` adds level 0 / -1 coupling; feed-forward is level +1 only.
    /// Both use the dense drive the floored leak requires.
    pub fn engine_config(&self, adapt_bump: i16, recurrent: bool) -> Config {
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
            baseline_init: self.baseline_init,
            adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        Config { seed: self.seed, size: self.size, layers: vec![layer; self.layers] }
    }
}

/// Drive the continuous bit stream and collect per-bin state rows (with a trailing bias 1.0) and the
/// bit sequence. Warmup bins advance the reservoir but are not collected. No reset between bins.
fn collect_states(net: &mut Network, cfg: &McConfig) -> (Vec<Vec<f64>>, Vec<f64>) {
    let pattern = stream_pattern(cfg.seed, cfg.size, cfg.stream_density_q16);
    for t in 0..cfg.warmup_bins {
        let on = bit(cfg.bit_seed, t);
        for _ in 0..cfg.bin_waves {
            net.wave(if on { &pattern } else { &[] });
        }
    }
    let mut xs = Vec::with_capacity(cfg.collect_bins);
    let mut us = Vec::with_capacity(cfg.collect_bins);
    for i in 0..cfg.collect_bins {
        let on = bit(cfg.bit_seed, cfg.warmup_bins + i);
        let pat = if on { pattern.clone() } else { Vec::new() };
        let counts = record_response(net, cfg.bin_waves, move |_w| pat.clone());
        let mut row: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        row.push(1.0); // bias
        xs.push(row);
        us.push(if on { 1.0 } else { 0.0 });
    }
    (xs, us)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_stream_is_deterministic_and_balanced() {
        let n = 2000;
        let ones = (0..n).filter(|&t| bit(42, t)).count();
        assert_eq!((0..n).map(|t| bit(42, t)).collect::<Vec<_>>(), (0..n).map(|t| bit(42, t)).collect::<Vec<_>>());
        let frac = ones as f64 / n as f64;
        assert!((frac - 0.5).abs() < 0.05, "bit stream not ~balanced: {frac}");
    }

    #[test]
    fn collect_states_shape_and_determinism() {
        let cfg = McConfig::demo();
        let build = || {
            let mut net = Network::new(cfg.engine_config(cfg.adapt_bump, true));
            let input = random_l0_input(cfg.seed ^ 0x3EC0, cfg.size, cfg.calib_fraction_q16);
            net.calibrate(&cfg.calib, &input);
            collect_states(&mut net, &cfg)
        };
        let (xs, us) = build();
        assert_eq!(xs.len(), cfg.collect_bins);
        assert_eq!(us.len(), cfg.collect_bins);
        assert_eq!(xs[0].len(), (cfg.layers - 1) * (cfg.size * cfg.size) as usize + 1); // +bias
        let (xs2, _) = build();
        assert_eq!(xs, xs2, "state collection must be deterministic");
    }
}
```

- [ ] **Step 2: Run to verify it passes**

Run: `cargo test bench::memory_capacity` and `cargo build`
Expected: PASS; warning-free. (`RidgeReadout`/`memory_capacity` not referenced yet; `collect_states` is used by the test, so no dead-code warning. `bit`/`stream_pattern` used via `collect_states`.)

- [ ] **Step 3: Commit**

```bash
git add src/bench/mod.rs src/bench/memory_capacity.rs
git commit -m "feat: MC bit stream + binned state collection"
```

---

### Task 4: `memory_capacity` experiment + the four-run ALIF-vs-LIF tests

**Files:**
- Modify: `src/bench/memory_capacity.rs`

**Interfaces:**
- Consumes: `RidgeReadout` (Task 2), `collect_states`/`McConfig` (Task 3).
- Produces: `McCurve`, `r2`, `memory_capacity(cfg: &McConfig, adapt_bump: i16, recurrent: bool) -> McCurve`.

- [ ] **Step 1: Write the failing tests**

Add to `src/bench/memory_capacity.rs` `mod tests`:
```rust
    #[test]
    fn memory_capacity_is_deterministic() {
        let cfg = McConfig::demo();
        let a = memory_capacity(&cfg, cfg.adapt_bump, true);
        let b = memory_capacity(&cfg, cfg.adapt_bump, true);
        assert_eq!(a.r2, b.r2);
        assert_eq!(a.total, b.total);
    }

    #[test]
    fn memory_capacity_feedforward_alif_beats_lif() {
        let cfg = McConfig::demo();
        let alif = memory_capacity(&cfg, cfg.adapt_bump, false);
        let lif = memory_capacity(&cfg, 0, false);
        eprintln!("ff ALIF total {:.3} r2 {:?}", alif.total, alif.r2);
        eprintln!("ff LIF  total {:.3} r2 {:?}", lif.total, lif.r2);
        assert!(alif.total > lif.total + 0.5, "ALIF MC {} should exceed LIF MC {} (feed-forward)", alif.total, lif.total);
        let tail = cfg.k_lags - 1;
        assert!(alif.r2[tail] > lif.r2[tail] + 0.02, "ALIF should retain more memory at the largest lag");
    }

    #[test]
    fn memory_capacity_recurrent_has_memory() {
        let cfg = McConfig::demo();
        let alif = memory_capacity(&cfg, cfg.adapt_bump, true);
        let lif = memory_capacity(&cfg, 0, true);
        eprintln!("rec ALIF total {:.3}", alif.total);
        eprintln!("rec LIF  total {:.3}", lif.total);
        assert!(lif.total > 1.0 && alif.total > 1.0, "recurrent reservoir should hold >1 bit of memory");
        assert!(alif.total >= lif.total - 0.3, "adaptation should not materially reduce recurrent memory");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::memory_capacity::tests::memory_capacity_is_deterministic`
Expected: FAIL to compile — `memory_capacity` / `McCurve` / `r2` not defined.

- [ ] **Step 3: Implement `McCurve`, `r2`, `memory_capacity`**

Add to `src/bench/memory_capacity.rs` (non-test; add `use crate::bench::readout::RidgeReadout;` to the imports):
```rust
/// The memory curve: `r2[k-1]` for lag `k = 1..=K`, and their sum.
#[derive(Clone, Debug, PartialEq)]
pub struct McCurve {
    pub r2: Vec<f64>,
    pub total: f64,
}

/// Squared Pearson correlation between prediction and target, clamped to [0,1].
fn r2(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mp = pred.iter().sum::<f64>() / n;
    let mt = target.iter().sum::<f64>() / n;
    let (mut cov, mut vp, mut vt) = (0.0, 0.0, 0.0);
    for (&p, &t) in pred.iter().zip(target) {
        cov += (p - mp) * (t - mt);
        vp += (p - mp) * (p - mp);
        vt += (t - mt) * (t - mt);
    }
    if vp <= 0.0 || vt <= 0.0 {
        return 0.0;
    }
    let r = cov / (vp.sqrt() * vt.sqrt());
    (r * r).clamp(0.0, 1.0)
}

/// Build+calibrate one variant, stream the reservoir, and fit a ridge readout per lag to reconstruct
/// `u(t-k)`. `adapt_bump` selects ALIF (>0) vs LIF (0); `recurrent` selects the topology.
pub fn memory_capacity(cfg: &McConfig, adapt_bump: i16, recurrent: bool) -> McCurve {
    let mut net = Network::new(cfg.engine_config(adapt_bump, recurrent));
    let input = random_l0_input(cfg.seed ^ 0x3EC0, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);
    let (xs, us) = collect_states(&mut net, cfg);

    let n = xs.len();
    let k = cfg.k_lags;
    let split = (n as u64 * cfg.train_frac_permille / 1000) as usize;
    // Same design matrix for every lag; rows [k, split) train, [split, n) test.
    let x_train: Vec<Vec<f64>> = xs[k..split].to_vec();
    let x_test: Vec<Vec<f64>> = xs[split..n].to_vec();
    let ridge = RidgeReadout::fit(&x_train, cfg.lambda);

    let mut r2s = Vec::with_capacity(k);
    for lag in 1..=k {
        let y_train: Vec<f64> = (k..split).map(|i| us[i - lag]).collect();
        let w = ridge.weights(&x_train, &y_train);
        let pred = RidgeReadout::predict(&x_test, &w);
        let y_test: Vec<f64> = (split..n).map(|i| us[i - lag]).collect();
        r2s.push(r2(&pred, &y_test));
    }
    let total = r2s.iter().sum();
    McCurve { r2: r2s, total }
}
```

- [ ] **Step 4: Run, then TUNE against the printed curves (do not fudge)**

Run: `cargo test bench::memory_capacity -- --nocapture`
Read the printed `ff`/`rec` totals and curves. Expected: feed-forward ALIF total clearly exceeds LIF (LIF tail → ~0 past pipeline depth); recurrent both hold >1 bit; recurrent ALIF ≥ LIF.

If an assertion fails, tune **only `McConfig::demo()` knobs** (never the mechanism/metric):
- `adapt_bump` up, `adapt_decay` up (longer τ) → longer ALIF tail.
- `collect_bins` up (less noisy `r²`), `lambda` up/down (regularization), `k_lags` (curve length).
- `bin_waves`, `stream_density_q16`, `baseline_init` (operating point), `size`/`layers` (feature richness) — mind test runtime.

**Honesty gate:** if recurrent MC shows `ALIF ≈ LIF`, keep the soft recurrent assertion and *report it* (recurrence supplies the memory) — that is a real result. But do **not** weaken the **feed-forward** claim below "ALIF MC clearly exceeds LIF, LIF tail collapses." If ALIF cannot beat LIF even feed-forward after reasonable tuning, stop and report — the adaptation memory is not linearly readable, a substrate finding.

- [ ] **Step 5: Full suite + warning-free**

Run: `cargo test` and `cargo build`
Expected: all pass; warning-free.

- [ ] **Step 6: Commit**

```bash
git add src/bench/memory_capacity.rs
git commit -m "feat: Memory Capacity experiment (recurrent + feed-forward, ALIF vs LIF)"
```

---

## Self-review

**Spec coverage:**
- `f64` linalg (LU solve, `XᵀX`/`Xᵀy`) → Task 1. `RidgeReadout` (factor once, per-lag solve) → Task 2.
- Binary bit stream, binned continuous drive (no reset), per-bin state via `record_response` → Task 3.
- `r²_k` (squared Pearson corr on test split), `MC = Σ r²_k` → Task 4.
- Four runs {recurrent, feed-forward} × {ALIF, LIF} via `memory_capacity(cfg, adapt_bump, recurrent)` → Task 4.
- Assertions: feed-forward ALIF>LIF headline; recurrent both>1 & ALIF≥LIF soft; determinism → Task 4. linalg + ridge unit tests → Tasks 1–2. Bit-stream determinism/balance, state-collection shape/determinism → Task 3.
- Configs: dense recurrent + feed-forward, calibrated per variant → `engine_config` (Task 3), used in Task 4.

**Placeholder scan:** none — every step has concrete code and commands.

**Type consistency:** `Lu::factor(Vec<Vec<f64>>)`/`solve(&[f64])->Vec<f64>`, `xt_x`/`xt_y` used identically in `RidgeReadout`. `RidgeReadout::{fit,weights,predict}` signatures match Task 4 usage. `McConfig` fields match `demo()`/`engine_config`/`collect_states`/`memory_capacity`. `McCurve { r2: Vec<f64>, total: f64 }` consistent.

**Note on Task 4:** the tuning-and-honesty protocol mirrors Spec 1 — the feed-forward ALIF>LIF result is the claim that must hold; a recurrent `ALIF ≈ LIF` is a reportable finding, not a failure to hide.
