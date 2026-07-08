# Store-recall Bench Implementation Plan (Spec 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable integer bench module (`src/bench/`) and a Tier-0 store-recall experiment that produces an ALIF-vs-LIF memory-horizon curve, validating that ALIF's adaptive threshold holds a cue across a delay that plain LIF forgets.

**Architecture:** A new `src/bench/` module using only the engine's public API. `readout.rs` = per-neuron spike-count feature extraction + an integer nearest-centroid classifier. `store_recall.rs` = deterministic cue/probe encoding, a present→delay→probe trial runner, and the `memory_horizon` sweep. LIF is the same network with `adapt_bump = 0`; both variants are calibrated to the same rate for a fair comparison. Integer-only, no floats.

**Tech Stack:** Rust edition 2024, standard library only, no deps. Inline `#[cfg(test)]` tests per module.

## Global Constraints

- **Standard library only**; **no `unsafe`**; **warning-free** `cargo build`.
- **Integer-only in this spec** — nearest-centroid decoder, accuracy as permille `u64`. No `f64` (that is Spec 2).
- **Determinism is a hard requirement** — every result is a pure function of `(seed, config, task params)`; single-threaded.
- **Engine (`src/wave_net/`) is not modified** — the bench only uses the public API. (If a needed accessor is genuinely missing, add it minimally — but none is expected.)
- Tests are **inline `#[cfg(test)]` per module**, test-first (TDD).
- **One commit per task**, conventional-commit messages. **NEVER** a `Co-Authored-By` trailer. **NEVER** push.
- Already on branch `feat/store-recall-bench` (spec committed there).
- Verify each task with `cargo test` and `cargo build` (warning-free) before committing.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/lib.rs` | crate root | add `pub mod bench;` |
| `src/bench/mod.rs` | bench module decls | new |
| `src/bench/readout.rs` | spike-count features + `NearestCentroid` | new |
| `src/bench/store_recall.rs` | cue/probe encoding, trial runner, `memory_horizon`, `BenchConfig` | new |

---

### Task 1: Bench scaffold + integer NearestCentroid decoder

**Files:**
- Modify: `src/lib.rs`
- Create: `src/bench/mod.rs`, `src/bench/readout.rs`

**Interfaces:**
- Produces: `bench::readout::NearestCentroid` with `fit(features: &[Vec<u32>], labels: &[usize], k: usize) -> NearestCentroid` and `predict(&self, feature: &[u32]) -> usize`.

- [ ] **Step 1: Wire the module and write the failing decoder test**

Add to `src/lib.rs` (after `pub mod wave_net;`):
```rust
pub mod bench;
```

Create `src/bench/mod.rs`:
```rust
//! `bench` — an integer test bench for the RSNN: spike-count readouts, decoders, and temporal
//! tasks that validate the substrate (and, later, training). Uses only the engine's public API.

pub mod readout;
pub mod store_recall;
```

Create `src/bench/store_recall.rs` as an empty stub so `mod.rs` compiles (filled in Task 3):
```rust
//! `store_recall` — the Tier-0 delayed-match task and the ALIF-vs-LIF memory-horizon experiment.
```

Create `src/bench/readout.rs` with the decoder and its test:
```rust
//! `readout` — per-neuron spike-count features over a multi-wave window, and an integer
//! nearest-centroid classifier. No floats: centroids are integer means, distances are i64.

/// Integer nearest-centroid classifier over fixed-length `u32` feature vectors.
pub struct NearestCentroid {
    centroids: Vec<Vec<i64>>, // one centroid (integer mean) per class
}

impl NearestCentroid {
    /// Fit `k` class centroids (integer means) from labelled feature vectors (labels in `0..k`).
    pub fn fit(features: &[Vec<u32>], labels: &[usize], k: usize) -> NearestCentroid {
        let dim = features.first().map(|f| f.len()).unwrap_or(0);
        let mut sums = vec![vec![0i64; dim]; k];
        let mut counts = vec![0i64; k];
        for (f, &lab) in features.iter().zip(labels) {
            counts[lab] += 1;
            for (acc, &v) in sums[lab].iter_mut().zip(f) {
                *acc += v as i64;
            }
        }
        let centroids = sums
            .iter()
            .zip(&counts)
            .map(|(sum, &c)| {
                let denom = c.max(1);
                sum.iter().map(|&s| s / denom).collect()
            })
            .collect();
        NearestCentroid { centroids }
    }

    /// Index of the class whose centroid is nearest in squared L2 distance (i64).
    pub fn predict(&self, feature: &[u32]) -> usize {
        let mut best = 0usize;
        let mut best_dist = i64::MAX;
        for (c, centroid) in self.centroids.iter().enumerate() {
            let mut dist = 0i64;
            for (&f, &m) in feature.iter().zip(centroid) {
                let d = f as i64 - m;
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_centroid_separates_clusters() {
        // Class 0 clusters near (10,0), class 1 near (0,10).
        let features = vec![
            vec![10u32, 0], vec![9, 1], vec![11, 0],
            vec![0, 10], vec![1, 9], vec![0, 11],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let clf = NearestCentroid::fit(&features, &labels, 2);
        assert_eq!(clf.predict(&[10, 1]), 0);
        assert_eq!(clf.predict(&[1, 10]), 1);
    }
}
```

- [ ] **Step 2: Run the test to verify it passes and the crate builds**

Run: `cargo test bench::readout` and `cargo build`
Expected: `nearest_centroid_separates_clusters` passes; warning-free build. (The test is self-contained, so it goes green immediately — this task is scaffold + a correct-by-construction unit.)

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs src/bench/mod.rs src/bench/readout.rs src/bench/store_recall.rs
git commit -m "feat: bench scaffold + integer nearest-centroid decoder"
```

---

### Task 2: Spike-count feature extraction (`record_response`)

**Files:**
- Modify: `src/bench/readout.rs`

**Interfaces:**
- Consumes: `crate::wave_net::network::Network` public API (`layer_count`, `size`, `on_layer`, `clear_listeners`, `wave`).
- Produces: `bench::readout::record_response(net: &mut Network, waves: usize, input: impl Fn(usize) -> Vec<u32>) -> Vec<u32>` — per-neuron spike counts over `waves` waves, concatenated across layers `1..L` (L0 excluded). Length `(L-1) * size * size`.

- [ ] **Step 1: Write the failing test**

Add to `src/bench/readout.rs` above the existing `#[cfg(test)] mod tests` block, add the import and function stub location later; first add the test inside `mod tests`:
```rust
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    fn two_layer_low() -> Config {
        // L0 -> L1 straight up; L1 baseline low so L0 injection makes it fire.
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
            leak: (3, 5),
            cooldown_base: 1,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 2,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let l1 = LayerConfig { topology: vec![], ..l0.clone() };
        Config { seed: 1, size: 4, layers: vec![l0, l1] }
    }

    #[test]
    fn record_response_counts_spikes() {
        let mut net = Network::new(two_layer_low());
        let ls = 16;
        // silent run -> all zero, correct length ((L-1)*ls = 1*16)
        net.reset_state();
        let silent = record_response(&mut net, 4, |_w| Vec::new());
        assert_eq!(silent.len(), ls);
        assert!(silent.iter().all(|&c| c == 0), "silent run must record no spikes");
        // drive all L0 every wave -> L1 should fire, so some counts are non-zero
        net.reset_state();
        let all_l0: Vec<u32> = (0..ls as u32).collect();
        let driven = record_response(&mut net, 6, move |_w| all_l0.clone());
        assert_eq!(driven.len(), ls);
        assert!(driven.iter().any(|&c| c > 0), "driven run must record L1 spikes");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test bench::readout::tests::record_response_counts_spikes`
Expected: FAIL to compile — `record_response` does not exist yet.

- [ ] **Step 3: Implement `record_response`**

Add to `src/bench/readout.rs` (top of file, after the doc comment):
```rust
use std::sync::{Arc, Mutex};

use crate::wave_net::network::Network;

/// Run `waves` waves feeding `input(w)` each wave, returning per-neuron spike counts over the
/// computational layers `1..L` concatenated (layer 0, the transducer, excluded). Installs counting
/// listeners, runs, then clears listeners. Does not reset state — the caller sets up the run.
pub fn record_response(net: &mut Network, waves: usize, input: impl Fn(usize) -> Vec<u32>) -> Vec<u32> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![0u32; l.saturating_sub(1) * ls]));
    for z in 1..l {
        let c = counts.clone();
        net.on_layer(
            z,
            Box::new(move |_w: usize, fired: &[u32]| {
                let mut g = c.lock().unwrap();
                let base = (z - 1) * ls;
                for &local in fired {
                    g[base + local as usize] += 1;
                }
            }),
        );
    }
    for w in 0..waves {
        net.wave(&input(w));
    }
    net.clear_listeners();
    std::mem::take(&mut *counts.lock().unwrap())
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test bench::readout` and `cargo build`
Expected: PASS; warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/bench/readout.rs
git commit -m "feat: bench spike-count feature extraction (record_response)"
```

---

### Task 3: Deterministic cue and probe encoding

**Files:**
- Modify: `src/bench/store_recall.rs`

**Interfaces:**
- Consumes: `crate::wave_net::synapse::{key, mix}`.
- Produces (module-private helpers, exercised by tests):
  - `cue_realization(seed, size, class, trial, wave, base_q16, keep_q16, noise_q16) -> Vec<u32>`
  - `probe_pattern(seed, size, density_q16) -> Vec<u32>`

- [ ] **Step 1: Write the failing test**

Replace the contents of `src/bench/store_recall.rs` with the doc comment, imports, purpose constants, the two helpers' signatures pending, and the test module:
```rust
//! `store_recall` — the Tier-0 delayed-match task and the ALIF-vs-LIF memory-horizon experiment.

use crate::wave_net::synapse::{key, mix};

const P_CUE: u64 = 7; // base cue membership per class
const P_TRIAL: u64 = 11; // per-trial keep of base sites
const P_NOISE: u64 = 13; // per-trial noise additions
const P_PROBE: u64 = 17; // fixed probe pattern

/// True iff the low 16 bits of the mixed key fall under `thresh_q16` (Q16 probability).
fn selected(seed: u64, site: u32, class_slot: i32, wave_slot: u32, purpose: u64, thresh_q16: u32) -> bool {
    ((mix(key(seed, site, class_slot, wave_slot, purpose)) & 0xFFFF) as u32) < thresh_q16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cue_encoding_is_deterministic_and_distinct() {
        let (seed, size) = (42u64, 8u32);
        let a = cue_realization(seed, size, 0, 3, 1, 20000, 60000, 2000);
        let b = cue_realization(seed, size, 0, 3, 1, 20000, 60000, 2000);
        assert_eq!(a, b, "same args must reproduce the same injection set");
        let other = cue_realization(seed, size, 1, 3, 1, 20000, 60000, 2000);
        assert_ne!(a, other, "different classes must differ");
        // probe is fixed and reproducible
        let p1 = probe_pattern(seed, size, 20000);
        let p2 = probe_pattern(seed, size, 20000);
        assert_eq!(p1, p2);
        assert!(!p1.is_empty(), "probe should select some sites at ~30% density");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test bench::store_recall`
Expected: FAIL to compile — `cue_realization` / `probe_pattern` are not defined.

- [ ] **Step 3: Implement the encoders**

Add to `src/bench/store_recall.rs` (after `selected`):
```rust
/// L0 injection set for one wave of one trial: the class's base sites (kept with prob `keep_q16`)
/// plus a few noise sites (non-base, added with prob `noise_q16`). Base membership per class is
/// fixed by `base_q16`; per-trial variability comes from `(trial, wave)` folded into the slot.
fn cue_realization(
    seed: u64,
    size: u32,
    class: usize,
    trial: usize,
    wave: usize,
    base_q16: u32,
    keep_q16: u32,
    noise_q16: u32,
) -> Vec<u32> {
    let ls = size * size;
    let slot = (trial as u32).wrapping_mul(1009).wrapping_add(wave as u32);
    let mut v = Vec::new();
    for s in 0..ls {
        let base = selected(seed, s, class as i32, 0, P_CUE, base_q16);
        let hit = if base {
            selected(seed, s, class as i32, slot, P_TRIAL, keep_q16)
        } else {
            selected(seed, s, class as i32, slot, P_NOISE, noise_q16)
        };
        if hit {
            v.push(s);
        }
    }
    v
}

/// Fixed probe pattern (same for every cue and trial): L0 sites selected at `density_q16`.
fn probe_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
    let ls = size * size;
    (0..ls).filter(|&s| selected(seed, s, 0, 0, P_PROBE, density_q16)).collect()
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test bench::store_recall` and `cargo build`
Expected: PASS; warning-free. (`cue_realization`/`probe_pattern` are used only by tests for now — but they will be used by Task 4 in non-test code, so no dead-code warning once Task 4 lands. If Task 3 is committed alone, add `#[allow(dead_code)]` to the two fns and remove it in Task 4. To avoid churn, this plan commits Task 3 and Task 4 together — see Task 4 Step 5.)

- [ ] **Step 5: Do not commit yet** — proceed to Task 4 (the encoders become non-test code there; committing together avoids a temporary `dead_code` allow).

---

### Task 4: Trial runner (present → delay → probe + read)

**Files:**
- Modify: `src/bench/store_recall.rs`

**Interfaces:**
- Consumes: `record_response` (Task 2), the encoders (Task 3), `Network` (`reset_state`, `wave`).
- Produces: `TaskParams` (encoding/timing knobs) and `run_trial(net: &mut Network, tp: &TaskParams, class: usize, trial: usize, delay: usize) -> Vec<u32>` returning the read-window feature vector.

- [ ] **Step 1: Write the failing test**

Add to `src/bench/store_recall.rs` `mod tests` (add imports at top of `mod tests`):
```rust
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    fn small_net(adapt_bump: i16) -> Network {
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
                TopologyLevel { level: -1, radius: 1, count: 2 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay: 5,
        };
        Network::new(Config { seed: 7, size: 8, layers: vec![layer; 4] })
    }

    fn task_params() -> TaskParams {
        TaskParams {
            seed: 7,
            size: 8,
            present_waves: 6,
            read_waves: 6,
            base_q16: 20000,
            keep_q16: 60000,
            noise_q16: 2000,
            probe_q16: 20000,
        }
    }

    #[test]
    fn run_trial_shape_and_determinism() {
        let mut net = small_net(16);
        let tp = task_params();
        let f1 = run_trial(&mut net, &tp, 0, 0, 8);
        let f2 = run_trial(&mut net, &tp, 0, 0, 8);
        // length = (L-1)*size*size = 3*64 = 192
        assert_eq!(f1.len(), 3 * 64);
        assert_eq!(f1, f2, "trial must be deterministic (reset each time)");
        // a probe response should produce some spikes for an ALIF net at a short delay
        assert!(f1.iter().any(|&c| c > 0), "probe should elicit a response");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test bench::store_recall::tests::run_trial_shape_and_determinism`
Expected: FAIL to compile — `TaskParams` / `run_trial` not defined.

- [ ] **Step 3: Implement `TaskParams` and `run_trial`**

Add to `src/bench/store_recall.rs` (non-test code; add `use crate::wave_net::network::Network;` and `use crate::bench::readout::record_response;` at the top):
```rust
/// Encoding + timing knobs for one store-recall trial.
#[derive(Clone, Debug)]
pub struct TaskParams {
    pub seed: u64,
    pub size: u32,
    pub present_waves: usize,
    pub read_waves: usize,
    pub base_q16: u32,  // base cue density per class
    pub keep_q16: u32,  // prob a base site is injected on a given trial/wave
    pub noise_q16: u32, // prob a non-base site is injected (noise)
    pub probe_q16: u32, // probe density
}

/// One trial: reset, present the noisy cue for `present_waves`, stay silent for `delay` waves, then
/// inject the fixed probe for `read_waves` and return the per-neuron spike-count feature vector.
pub fn run_trial(net: &mut Network, tp: &TaskParams, class: usize, trial: usize, delay: usize) -> Vec<u32> {
    net.reset_state();
    for w in 0..tp.present_waves {
        let sites = cue_realization(tp.seed, tp.size, class, trial, w, tp.base_q16, tp.keep_q16, tp.noise_q16);
        net.wave(&sites);
    }
    for _ in 0..delay {
        net.wave(&[]);
    }
    let probe = probe_pattern(tp.seed, tp.size, tp.probe_q16);
    record_response(net, tp.read_waves, move |_w| probe.clone())
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test bench::store_recall` and `cargo build`
Expected: PASS; warning-free (the encoders are now used by `run_trial`, so no dead-code warning).

- [ ] **Step 5: Commit Tasks 3 + 4 together**

```bash
git add src/bench/store_recall.rs
git commit -m "feat: store-recall encoding + trial runner (present/delay/probe)"
```

---

### Task 5: `memory_horizon` experiment + the ALIF-vs-LIF headline test

**Files:**
- Modify: `src/bench/store_recall.rs`

**Interfaces:**
- Consumes: `run_trial`, `NearestCentroid` (Task 1), `calibrate` / `random_l0_input` / `CalibrateParams`, `Config`/`LayerConfig`/`TopologyLevel`.
- Produces: `BenchConfig`, `BenchConfig::demo()`, `HorizonCurve`, and `memory_horizon(cfg: &BenchConfig, adapt_bump: i16) -> HorizonCurve`.

- [ ] **Step 1: Write the failing tests**

Add to `src/bench/store_recall.rs` `mod tests`:
```rust
    #[test]
    fn memory_horizon_is_deterministic() {
        let cfg = BenchConfig::demo();
        let a = memory_horizon(&cfg, cfg.adapt_bump);
        let b = memory_horizon(&cfg, cfg.adapt_bump);
        assert_eq!(a.delays, b.delays);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    #[test]
    fn store_recall_alif_beats_lif_at_long_delay() {
        let cfg = BenchConfig::demo();
        let alif = memory_horizon(&cfg, cfg.adapt_bump);
        let lif = memory_horizon(&cfg, 0);
        eprintln!("delays {:?}", alif.delays);
        eprintln!("ALIF   {:?}", alif.accuracy_permille);
        eprintln!("LIF    {:?}", lif.accuracy_permille);
        let chance = 1000 / cfg.k as u64;
        let last = cfg.delays.len() - 1;
        // (1) encodable: both decode well above chance at the shortest delay
        assert!(alif.accuracy_permille[0] > 650, "ALIF should decode at short delay");
        assert!(lif.accuracy_permille[0] > 650, "LIF should decode at short delay");
        // (2) ALIF holds, LIF forgets, at the longest delay
        assert!(
            alif.accuracy_permille[last] > lif.accuracy_permille[last] + 100,
            "ALIF should beat LIF at long delay (ALIF {} vs LIF {})",
            alif.accuracy_permille[last], lif.accuracy_permille[last]
        );
        assert!(alif.accuracy_permille[last] > chance + 80, "ALIF should stay above chance at long delay");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::store_recall::tests::memory_horizon_is_deterministic`
Expected: FAIL to compile — `BenchConfig` / `HorizonCurve` / `memory_horizon` not defined.

- [ ] **Step 3: Implement `BenchConfig`, `HorizonCurve`, `memory_horizon`**

Add to `src/bench/store_recall.rs` (add imports: `use crate::wave_net::config::{Config, LayerConfig}; use crate::wave_net::synapse::TopologyLevel; use crate::wave_net::calibrate::{CalibrateParams, random_l0_input}; use crate::bench::readout::NearestCentroid;`):
```rust
/// Full configuration for the store-recall memory-horizon experiment.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub k: usize,               // number of cue classes (chance = 1/k)
    pub baseline_init: i16,
    pub adapt_bump: i16,        // ALIF value; LIF variant passes 0 to memory_horizon
    pub adapt_decay: u8,
    pub trials_per_class: usize,
    pub delays: Vec<usize>,     // swept, ascending
    pub task: TaskParams,
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl BenchConfig {
    /// Small, fast, deterministic config tuned for the inline test. adapt_decay 6 -> tau ~64 waves,
    /// well past the leak horizon (~15-20 waves for leak (3,5)); the longest delay sits between them.
    pub fn demo() -> BenchConfig {
        let seed = 0xB0A7_57ED;
        let size = 8;
        BenchConfig {
            seed,
            size,
            layers: 4,
            k: 4,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
            trials_per_class: 10,
            delays: vec![0, 8, 24],
            task: TaskParams {
                seed,
                size,
                present_waves: 6,
                read_waves: 6,
                base_q16: 18000,
                keep_q16: 60000,
                noise_q16: 1500,
                probe_q16: 20000,
            },
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
        }
    }

    fn to_engine_config(&self, adapt_bump: i16) -> Config {
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
                TopologyLevel { level: -1, radius: 1, count: 2 },
            ],
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

/// Accuracy (permille) at each swept delay, for one variant.
#[derive(Clone, Debug)]
pub struct HorizonCurve {
    pub delays: Vec<usize>,
    pub accuracy_permille: Vec<u64>,
}

/// Run the store-recall sweep for one variant. `adapt_bump` selects the variant (0 = plain LIF).
/// The net is built + calibrated once, then trials reuse the calibrated baselines (reset per trial).
pub fn memory_horizon(cfg: &BenchConfig, adapt_bump: i16) -> HorizonCurve {
    let mut net = crate::wave_net::network::Network::new(cfg.to_engine_config(adapt_bump));
    let input = random_l0_input(cfg.seed ^ 0xCA11B, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);

    let mut accuracy_permille = Vec::with_capacity(cfg.delays.len());
    for &delay in &cfg.delays {
        let mut feats: Vec<Vec<u32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for t in 0..cfg.trials_per_class {
            for c in 0..cfg.k {
                feats.push(run_trial(&mut net, &cfg.task, c, t, delay));
                labels.push(c);
            }
        }
        // Deterministic split: even trial index -> train, odd -> test (balanced across classes).
        let (mut tr_f, mut tr_l, mut te_f, mut te_l) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, (f, l)) in feats.into_iter().zip(labels).enumerate() {
            if (i / cfg.k) % 2 == 0 {
                tr_f.push(f);
                tr_l.push(l);
            } else {
                te_f.push(f);
                te_l.push(l);
            }
        }
        let clf = NearestCentroid::fit(&tr_f, &tr_l, cfg.k);
        let correct = te_f.iter().zip(&te_l).filter(|(f, &l)| clf.predict(f) == l).count();
        let acc = if te_f.is_empty() { 0 } else { (correct as u64 * 1000) / te_f.len() as u64 };
        accuracy_permille.push(acc);
    }
    HorizonCurve { delays: cfg.delays.clone(), accuracy_permille }
}
```

- [ ] **Step 4: Run, then TUNE against real output (do not fudge)**

Run: `cargo test bench::store_recall -- --nocapture`
Read the printed `ALIF` / `LIF` curves. Expected outcome: at delay 0 both are high; at delay 24 ALIF > LIF and ALIF above chance (250‰).

If the assertion fails, tune **only the `BenchConfig::demo()` / `TaskParams` knobs** (never the mechanism) and re-run — the levers, in order of leverage:
- `adapt_bump` up (stronger footprint → stronger probe gating), `adapt_decay` up (longer τ so memory survives the long delay).
- `present_waves` up (more adaptation buildup), `base_q16` down / `k` effect (more distinct class footprints).
- `trials_per_class` up (less noisy accuracy), `keep_q16` / `noise_q16` (cleaner within-class signal).
- longest `delay` down toward the leak horizon if τ can't be pushed far enough.

**Honesty gate:** if, after reasonable tuning, ALIF cannot be made to beat LIF at a delay past the leak horizon, **stop and report it** — that is a real finding about the substrate (the adaptation memory is too weak to be read this way), not a test to force green. Capture the observed curves and hypotheses; do not weaken the assertion below "ALIF > LIF past the leak horizon, ALIF above chance."

- [ ] **Step 5: Run the full suite and confirm warning-free**

Run: `cargo test` and `cargo build`
Expected: all tests pass; warning-free.

- [ ] **Step 6: Commit**

```bash
git add src/bench/store_recall.rs
git commit -m "feat: store-recall memory-horizon experiment (ALIF vs LIF)"
```

---

## Self-review

**Spec coverage** (each spec section → task):
- Bench module `src/bench/` (readout + store_recall) → Tasks 1–5.
- Integer nearest-centroid readout → Task 1. Spike-count features over layers 1..L → Task 2.
- Cue encoding + within-class noise + fixed probe → Task 3. Present→delay→probe trial → Task 4.
- Fair comparison (same config, `adapt_bump` on/off, each calibrated) → Task 5 (`memory_horizon` builds+calibrates per variant).
- Memory-horizon curve + assertions (encodable at small N; ALIF > LIF and > chance at large N) → Task 5 headline test.
- Determinism → `memory_horizon_is_deterministic` (Task 5) + deterministic trial (Task 4).
- Integer-only / permille accuracy / no floats → enforced throughout; `HorizonCurve.accuracy_permille: Vec<u64>`.
- YAGNI (no MC/ridge/f64, no e-prop, no datasets) → none added.

**Placeholder scan:** none — every step has concrete code and commands.

**Type consistency:** `NearestCentroid::fit(&[Vec<u32>], &[usize], usize)` / `predict(&[u32]) -> usize` used identically in Task 5. `record_response(&mut Network, usize, impl Fn) -> Vec<u32>` used by `run_trial`. `TaskParams` fields match between Task 4 def and `BenchConfig::demo()` in Task 5. `run_trial(&mut Network, &TaskParams, usize, usize, usize) -> Vec<u32>` used consistently.

**Note on Task 5:** it is the tuning-heavy task and the actual experiment. The plan explicitly forbids weakening the assertion to force a pass — a genuine ALIF-≤-LIF result is a substrate finding to report, not a bug to hide.
