# e-prop Learning Implementation Plan (Spec 3, v1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A gradient-free, e-prop-like rule that trains per-neuron thresholds from a global reward × eligibility trace, demonstrated by beating a frozen-threshold control on a `K=2` held-category task.

**Architecture:** New `src/bench/eprop.rs`. Per trial: run a store-recall-style cue/delay/probe, accumulate per-neuron spike-count **eligibility** via listeners, read `K` output neurons (top layer), compute a graded **reward**, and nudge thresholds by `−lr·(R−R̄)·eⱼ`, accumulated in an **f64 shadow** and written back to the integer engine. Reuses store-recall's cue/probe and the shared engine-config builder. No engine change.

**Tech Stack:** Rust edition 2024, std only, no deps. Inline `#[cfg(test)]` tests. `f64` allowed in the bench.

## Global Constraints

- **Std only**; **no `unsafe`**; **warning-free** build.
- **Engine (`src/wave_net/`) untouched** — trainer uses the public / in-crate API (`on_layer`, `with_layer_mut`, `layer_thresholds`, public `Layer.threshold`).
- **`f64` allowed in the bench** (shadow, reward). Single-threaded, fixed reduction order → deterministic.
- **Determinism is a hard requirement** — pure function of `(seed, config, params)`.
- Tests inline `#[cfg(test)]`, test-first.
- **One commit per task**, conventional commits. **NEVER** a `Co-Authored-By` trailer. **NEVER** push.
- On branch `feat/eprop-learning` (spec committed there).
- Verify each task with `cargo test` + warning-free `cargo build` before committing.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/bench/mod.rs` | module decls | add `pub mod eprop;` |
| `src/bench/store_recall.rs` | cue/probe encoding | make `cue_realization` + `probe_pattern` `pub(crate)` |
| `src/bench/eprop.rs` | reward tracker, shadow, trial eligibility, `train` | new |

---

### Task 1: Scaffold + reward-prediction-error tracker

**Files:**
- Modify: `src/bench/mod.rs`, `src/bench/store_recall.rs`
- Create: `src/bench/eprop.rs`

**Interfaces:**
- Produces: `bench::eprop::EpropConfig` (+ `demo()`, `engine_config()`); module-private `RewardTracker { step(r) -> signal }`.
- Consumes: `bench::stream::engine_config`, store-recall `cue_realization`/`probe_pattern` (now `pub(crate)`).

- [ ] **Step 1: Make cue/probe reusable and declare the module**

In `src/bench/store_recall.rs`, change the two encoders from `fn` to `pub(crate) fn`:
```rust
pub(crate) fn cue_realization(
```
```rust
pub(crate) fn probe_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
```

Add to `src/bench/mod.rs`:
```rust
pub mod eprop;
```

- [ ] **Step 2: Write the failing test**

Create `src/bench/eprop.rs`:
```rust
//! `eprop` — a gradient-free, e-prop-like learning rule (v1): per-neuron threshold updates driven by a
//! global reward × a per-neuron eligibility trace, on a K=2 held-category task with spiking output
//! neurons. Trains thresholds to beat a frozen-threshold control. Reuses `store_recall`'s cue/probe.

use crate::bench::store_recall::{cue_realization, probe_pattern};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::Config;
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix};
use std::sync::{Arc, Mutex};

/// Reward-prediction-error tracker: returns `R − R̄` and updates the running mean `R̄` (EMA).
struct RewardTracker {
    mean: f64,
    rate: f64,
}

impl RewardTracker {
    fn new(rate: f64) -> RewardTracker {
        RewardTracker { mean: 0.0, rate }
    }
    /// Signal for this reward (before absorbing it), then update the mean.
    fn step(&mut self, r: f64) -> f64 {
        let s = r - self.mean;
        self.mean += self.rate * s;
        s
    }
}

/// Configuration for the e-prop learning experiment.
#[derive(Clone, Debug)]
pub struct EpropConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub baseline_init: i16,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub k: usize,           // classes = output neurons
    pub present_waves: usize,
    pub delay: usize,
    pub read_waves: usize,
    pub base_q16: u32,
    pub keep_q16: u32,
    pub noise_q16: u32,
    pub probe_q16: u32,
    pub trials: usize,
    pub block: usize,       // accuracy-curve window
    pub reward_rate: f64,   // EMA rate for R̄
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl EpropConfig {
    pub fn demo() -> EpropConfig {
        let seed = 0xE9_0B_0A17;
        EpropConfig {
            seed,
            size: 8,
            layers: 3,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
            k: 2,
            present_waves: 6,
            delay: 4,
            read_waves: 6,
            base_q16: 18000,
            keep_q16: 60000,
            noise_q16: 1500,
            probe_q16: 20000,
            trials: 1200,
            block: 100,
            reward_rate: 0.02,
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
        }
    }

    fn engine_config(&self) -> Config {
        // Dense feed-forward ALIF (held memory needs dense fan-out; feed-forward isolates adaptation).
        crate::bench::stream::engine_config(
            self.seed, self.size, self.layers, self.baseline_init, self.adapt_bump, self.adapt_decay, 0, false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_prediction_error_centers() {
        let mut rt = RewardTracker::new(0.1);
        // A constant reward: the signal should decay toward 0 as R̄ tracks it.
        let first = rt.step(5.0);
        assert!((first - 5.0).abs() < 1e-9, "first signal is R − 0");
        let mut last = first;
        for _ in 0..200 {
            last = rt.step(5.0);
        }
        assert!(last.abs() < 0.05, "constant reward should center to ~0, got {last}");
    }
}
```

- [ ] **Step 3: Run**

Run: `cargo test bench::eprop` and `cargo build`
Expected: `reward_prediction_error_centers` passes; warning-free. (`cue_realization`/`probe_pattern`/`engine_config` are imported but not yet used in non-test code — if that warns, it resolves in Task 2 where they are used; to keep this task warning-free, add `#[allow(unused_imports)]` on the `use` block here and remove it in Task 2. To avoid churn, this plan commits Task 1 and Task 2 together — see Task 2 Step 5.)

- [ ] **Step 4: Do not commit yet** — proceed to Task 2 (the imports become live there; committing together avoids a temporary allow).

---

### Task 2: Shadow read/write + trial eligibility

**Files:**
- Modify: `src/bench/eprop.rs`

**Interfaces:**
- Produces (module-private): `read_shadow(&Network) -> Vec<Vec<f64>>`, `write_thresholds(&Network, &[Vec<f64>])`, `trial_eligibility(&mut Network, &EpropConfig, class, trial) -> Vec<Vec<u32>>` (per-neuron spike counts for computational layers `1..L`).
- Consumes: `Network` (`layer_count`, `size`, `layer_thresholds`, `with_layer_mut`, `on_layer`, `clear_listeners`, `reset_state`, `wave`), cue/probe.

- [ ] **Step 1: Write the failing tests**

Add to `src/bench/eprop.rs` `mod tests` (add `use crate::wave_net::network::Network;` and config imports at the top of `mod tests`):
```rust
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    fn tiny_net() -> Network {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 2, count: 6 }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0,
            baseline_init: 50, adapt_bump: 8, adapt_decay: 6,
        };
        Network::new(Config { seed: 3, size: 4, layers: vec![layer; 3] })
    }

    #[test]
    fn shadow_write_roundtrips_thresholds() {
        let net = tiny_net();
        let mut shadow = read_shadow(&net);
        // layer index 0 in the shadow = engine layer 1; nudge neuron 0 by +0.6 → +1 after rounding.
        let before = net.layer_thresholds(1)[0];
        shadow[0][0] += 0.6;
        write_thresholds(&net, &shadow);
        assert_eq!(net.layer_thresholds(1)[0], before + 1);
        // a +0.4 sub-unit nudge rounds to no change (accumulation must cross the integer boundary).
        shadow[0][1] += 0.4;
        write_thresholds(&net, &shadow);
        assert_eq!(net.layer_thresholds(1)[1], net.layer_thresholds(1)[1]); // unchanged from calibration
    }

    #[test]
    fn trial_eligibility_shape_and_determinism() {
        let cfg = EpropConfig::demo();
        let mut net = Network::new(cfg.engine_config());
        let e1 = trial_eligibility(&mut net, &cfg, 0, 0);
        let e2 = trial_eligibility(&mut net, &cfg, 0, 0);
        assert_eq!(e1.len(), cfg.layers - 1); // computational layers 1..L
        assert_eq!(e1[0].len(), (cfg.size * cfg.size) as usize);
        assert_eq!(e1, e2, "a trial (reset each time) must be deterministic");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::eprop::tests::shadow_write_roundtrips_thresholds`
Expected: FAIL to compile — `read_shadow` / `write_thresholds` / `trial_eligibility` not defined.

- [ ] **Step 3: Implement the helpers**

Add to `src/bench/eprop.rs` (non-test code):
```rust
/// f64 shadow of the computational-layer thresholds (`1..L`), read from the current engine thresholds.
fn read_shadow(net: &Network) -> Vec<Vec<f64>> {
    let l = net.layer_count();
    (1..l).map(|z| net.layer_thresholds(z).iter().map(|&t| t as f64).collect()).collect()
}

/// Write the rounded, clamped shadow back to the engine's integer thresholds (`1..L`).
fn write_thresholds(net: &Network, shadow: &[Vec<f64>]) {
    let l = net.layer_count();
    for z in 1..l {
        let s = &shadow[z - 1];
        net.with_layer_mut(z, |layer| {
            for (i, t) in layer.threshold.iter_mut().enumerate() {
                *t = s[i].round().clamp(1.0, i16::MAX as f64) as i16;
            }
        });
    }
}

/// Run one trial (reset → present cue → delay → probe) accumulating per-neuron spike counts over the
/// whole trial for the computational layers `1..L`. This is the per-neuron eligibility trace.
fn trial_eligibility(net: &mut Network, cfg: &EpropConfig, class: usize, trial: usize) -> Vec<Vec<u32>> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![vec![0u32; ls]; l]));
    for z in 1..l {
        let c = counts.clone();
        net.on_layer(z, Box::new(move |_w: usize, fired: &[u32]| {
            let mut g = c.lock().unwrap();
            for &loc in fired {
                g[z][loc as usize] += 1;
            }
        }));
    }
    net.reset_state();
    for w in 0..cfg.present_waves {
        let sites = cue_realization(cfg.seed, cfg.size, class, trial, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
        net.wave(&sites);
    }
    for _ in 0..cfg.delay {
        net.wave(&[]);
    }
    let probe = probe_pattern(cfg.seed, cfg.size, cfg.probe_q16);
    for _ in 0..cfg.read_waves {
        net.wave(&probe);
    }
    net.clear_listeners();
    let g = counts.lock().unwrap();
    (1..l).map(|z| g[z].clone()).collect()
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test bench::eprop` and `cargo build`
Expected: both new tests pass; warning-free (cue/probe/engine_config are now used in non-test code).

- [ ] **Step 5: Commit Tasks 1 + 2 together**

```bash
git add src/bench/mod.rs src/bench/store_recall.rs src/bench/eprop.rs
git commit -m "feat: e-prop scaffold — reward tracker, threshold shadow, trial eligibility"
```

---

### Task 3: `train` loop + the learning-vs-frozen headline

**Files:**
- Modify: `src/bench/eprop.rs`

**Interfaces:**
- Consumes: Task 1–2 pieces.
- Produces: `bench::eprop::{LearnCurve, train}`. `train(cfg: &EpropConfig, lr: f64) -> LearnCurve` (`lr = 0.0` → frozen control).

- [ ] **Step 1: Write the failing tests**

Add to `src/bench/eprop.rs` `mod tests`:
```rust
    #[test]
    fn eprop_is_deterministic() {
        let cfg = EpropConfig::demo();
        let a = train(&cfg, 0.002);
        let b = train(&cfg, 0.002);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    #[test]
    fn eprop_learns_and_beats_frozen_control() {
        let cfg = EpropConfig::demo();
        let learn = train(&cfg, 0.002);
        let frozen = train(&cfg, 0.0);
        eprintln!("learn  {:?}", learn.accuracy_permille);
        eprintln!("frozen {:?}", frozen.accuracy_permille);
        let lf = *learn.accuracy_permille.last().unwrap();
        let ff = *frozen.accuracy_permille.last().unwrap();
        let chance = 1000 / cfg.k as u64;
        assert!(lf > chance + 80, "learning final accuracy {lf} should be above chance {chance}");
        assert!(lf > ff + 80, "learning {lf} should beat the frozen control {ff}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::eprop::tests::eprop_is_deterministic`
Expected: FAIL to compile — `train` / `LearnCurve` not defined.

- [ ] **Step 3: Implement `train`**

Add to `src/bench/eprop.rs` (non-test):
```rust
/// Windowed accuracy trajectory over training.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnCurve {
    pub accuracy_permille: Vec<u64>,
}

/// Deterministic class for trial `t`.
fn pick_class(seed: u64, t: usize, k: usize) -> usize {
    (mix(key(seed, t as u32, 0, 0, 41)) % k as u64) as usize
}

/// Train per-neuron thresholds by global-reward × eligibility. `lr = 0.0` freezes the thresholds
/// (the control). Returns block-windowed accuracy over training.
pub fn train(cfg: &EpropConfig, lr: f64) -> LearnCurve {
    let mut net = Network::new(cfg.engine_config());
    let input = random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);

    let mut shadow = read_shadow(&net);
    let mut rt = RewardTracker::new(cfg.reward_rate);
    let mut outcomes: Vec<bool> = Vec::with_capacity(cfg.trials);

    for t in 0..cfg.trials {
        let class = pick_class(cfg.seed, t, cfg.k);
        let elig = trial_eligibility(&mut net, cfg, class, t);
        let top = &elig[elig.len() - 1];
        let outs = &top[0..cfg.k];
        let pred = (0..cfg.k).max_by_key(|&i| outs[i]).unwrap();
        outcomes.push(pred == class);

        let correct = outs[class] as f64;
        let best_rival = (0..cfg.k).filter(|&i| i != class).map(|i| outs[i]).max().unwrap_or(0) as f64;
        let signal = rt.step(correct - best_rival);

        if lr != 0.0 {
            for (zi, layer_e) in elig.iter().enumerate() {
                for (i, &e) in layer_e.iter().enumerate() {
                    shadow[zi][i] += -lr * signal * e as f64;
                }
            }
            write_thresholds(&net, &shadow);
        }
    }

    let block = cfg.block.max(1);
    let accuracy_permille = outcomes
        .chunks(block)
        .map(|c| (c.iter().filter(|&&b| b).count() as u64 * 1000) / c.len() as u64)
        .collect();
    LearnCurve { accuracy_permille }
}
```

- [ ] **Step 4: Run, then TUNE against the printed curves (do not fudge)**

Run: `cargo test bench::eprop::tests::eprop_learns_and_beats_frozen_control -- --nocapture`
Read the `learn` and `frozen` trajectories. Expected: the learning curve rises over blocks and ends clearly above both chance and the frozen control.

If it does not, tune **only `EpropConfig::demo()` / the test `lr`** (never the rule):
- `lr` (too small → no movement; too large → unstable/oscillates), `reward_rate` (R̄ tracking speed).
- `delay` **down toward 0** — isolates the *learning* mechanism from the memory demand (the spec's main lever).
- `base_q16`/`noise_q16` for more separable cues; `trials` up (more learning); `size`/`layers` (capacity);
  `present_waves`/`read_waves`.

**Honesty gate:** global-reward × eligibility is the crudest credit assignment; the output-neuron↔class
assignment is fixed, so the reservoir may not be shapeable to it by thresholds alone. If, after reasonable
tuning (including `delay = 0`), the learning run cannot beat the frozen control, **stop and report** — that
is the finding (points to the deferred upgrades: non-spiking potential readout / broadcast-error alignment).
Do not weaken the assertion below "learning beats frozen, above chance." Keep the margin modest but real.

- [ ] **Step 5: Full suite + warning-free**

Run: `cargo test` and `cargo build`
Expected: all pass; warning-free.

- [ ] **Step 6: Commit**

```bash
git add src/bench/eprop.rs
git commit -m "feat: e-prop threshold training loop (learning beats frozen control)"
```

---

## Self-review

**Spec coverage:**
- Three-factor rule `Δθ = −lr·(R−R̄)·e`, all thresholds → Task 3. Eligibility trace (per-neuron trial spike counts) → Task 2. Reward-prediction-error → Task 1. f64 shadow (integer engine) → Task 2/3.
- Task: K=2 store-recall cue/delay/probe, output neurons = top-layer first K, graded margin reward → Tasks 2–3.
- Success: learning beats frozen control + above chance; determinism → Task 3. Reward-PE unit + shadow-write unit → Tasks 1–2.
- Engine untouched (listeners + `with_layer_mut` + public `threshold`); reuse cue/probe (`pub(crate)`) → throughout.

**Placeholder scan:** none — every step has concrete code and commands.

**Type consistency:** `RewardTracker::step(f64)->f64`, `read_shadow(&Network)->Vec<Vec<f64>>`, `write_thresholds(&Network,&[Vec<f64>])`, `trial_eligibility(&mut Network,&EpropConfig,usize,usize)->Vec<Vec<u32>>`, `train(&EpropConfig,f64)->LearnCurve` used consistently. Shadow index `z-1` ↔ engine layer `z` throughout. `EpropConfig` fields match `demo()` and all consumers.

**Note on Task 3:** the tuning-and-honesty protocol is load-bearing — this is the first learning rule and it may not learn; a reported null (pointing to v2/broadcast) is a valid outcome, never a faked curve.
