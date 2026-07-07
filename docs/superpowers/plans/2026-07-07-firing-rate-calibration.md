# Firing-rate calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline, per AGENTS.md — no subagent option). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Lower per-layer thresholds so each layer fires near a target rate on a driven input, making the silent-by-default `wave_net` non-inert.

**Architecture:** Calibration is **layer-owned** — each `Layer` tunes its own thresholds (`shift_threshold`, `calibrate_step`); the `Network` only orchestrates measurement (`measure_layer_rates`, which saves/restores the caller's listeners) and delegates per-layer adjustment via `with_layer_mut`. Hybrid convergence: bottom-up (each layer tuned once its feeder fires) then global-refine passes.

**Tech Stack:** Rust (edition 2024), std only, no `unsafe`, inline `#[cfg(test)]`.

**Spec:** `docs/superpowers/specs/2026-07-07-firing-rate-calibration-design.md`

## Global Constraints

- Standard library only; no `unsafe`; `cargo build` warning-free.
- Deterministic: results are a pure function of `(net seed/config, params, input)`; single-threaded.
- Layer stays a self-contained persistable unit (owns `topology`/`leak`/`cooldown_base`/`inhibitor_ratio`/`threshold` + threshold get/set). No serialization code in this plan.
- Conventional commits, one per task, **no `Co-Authored-By` trailer** (AGENTS.md). Do not push unless asked.

## File Structure

- `src/wave_net/neurons.rs` — add `Layer::{shift_threshold, calibrate_step, thresholds, set_thresholds}` (the layer-owned tuning).
- `src/wave_net/network.rs` — add `Network::{with_layer_mut, measure_layer_rates, layer_thresholds}` (orchestration + listener save/restore).
- `src/wave_net/synapse.rs` — add `P_INPUT`.
- `src/wave_net/calibrate.rs` — replace stub: `CalibrateParams`, `random_l0_input`, `impl Network { pub fn calibrate }`.

---

### Task 1: Layer-owned tuning methods

**Files:**
- Modify: `src/wave_net/neurons.rs`

**Interfaces:**
- Consumes: `Layer` (existing), `Layer::max_threshold` (existing).
- Produces: `Layer::shift_threshold(&mut self, delta: i32)`; `Layer::calibrate_step(&mut self, rate: f64, target: f64, tol: f64, step_shift: u32) -> bool`; `Layer::thresholds(&self) -> &[i16]`; `Layer::set_thresholds(&mut self, t: Vec<i16>)`.

- [ ] **Step 1: Write failing tests** (append to `neurons.rs` `mod tests`)

```rust
    #[test]
    fn shift_threshold_clamps_and_preserves_jitter() {
        let mut l = Layer::new(&lc(128), 1, 0, 8);
        let before = l.thresholds().to_vec();
        l.shift_threshold(1000);
        for (a, b) in before.iter().zip(l.thresholds()) {
            assert_eq!(*b as i32, (*a as i32 - 1000).max(1));
        }
        l.shift_threshold(i16::MAX as i32); // drive well past the floor
        assert!(l.thresholds().iter().all(|&t| t == 1));
        l.shift_threshold(-(i16::MAX as i32)); // raise past the cap
        assert!(l.thresholds().iter().all(|&t| t == i16::MAX));
    }

    #[test]
    fn calibrate_step_lowers_cold_raises_hot_holds_in_band() {
        let mut l = Layer::new(&lc(0), 1, 0, 8); // jitter 0 -> all i16::MAX
        let m0 = l.max_threshold();
        assert!(l.calibrate_step(0.0, 0.1, 0.02, 2)); // cold -> lower
        assert!(l.max_threshold() < m0);
        let m1 = l.max_threshold();
        assert!(!l.calibrate_step(0.1, 0.1, 0.02, 2)); // in band -> no change
        assert_eq!(l.max_threshold(), m1);
        assert!(l.calibrate_step(0.5, 0.1, 0.02, 2)); // hot -> raise
        assert!(l.max_threshold() > m1);
    }

    #[test]
    fn thresholds_round_trip() {
        let mut l = Layer::new(&lc(128), 1, 0, 8);
        let snap = l.thresholds().to_vec();
        l.shift_threshold(500);
        l.set_thresholds(snap.clone());
        assert_eq!(l.thresholds(), snap.as_slice());
    }
```

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::neurons` → FAIL (methods missing).

- [ ] **Step 3: Implement** (add inside `impl Layer`, after `max_threshold`)

```rust
    /// Subtract `delta` from every threshold (delta>0 lowers), clamped to [1, i16::MAX].
    /// Uniform shift, so per-neuron jitter is preserved.
    pub fn shift_threshold(&mut self, delta: i32) {
        for t in self.threshold.iter_mut() {
            *t = ((*t as i32) - delta).clamp(1, i16::MAX as i32) as i16;
        }
    }

    /// One measure-informed tuning step toward `target` (fractions in 0..1). Returns whether it
    /// adjusted. Geometric step `max_threshold >> step_shift`; lower when too cold, raise when hot,
    /// no-op inside the tolerance band.
    pub fn calibrate_step(&mut self, rate: f64, target: f64, tol: f64, step_shift: u32) -> bool {
        if (rate - target).abs() <= tol {
            return false;
        }
        let step = ((self.max_threshold() as i32) >> step_shift).max(1);
        let delta = if rate < target { step } else { -step };
        self.shift_threshold(delta);
        true
    }

    pub fn thresholds(&self) -> &[i16] {
        &self.threshold
    }

    pub fn set_thresholds(&mut self, t: Vec<i16>) {
        assert_eq!(t.len(), self.threshold.len(), "threshold length mismatch");
        self.threshold = t;
    }
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::neurons` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wave_net): layer-owned threshold tuning (shift/calibrate_step/accessors)"`

---

### Task 2: Network measurement + per-layer access (listener save/restore)

**Files:**
- Modify: `src/wave_net/network.rs`

**Interfaces:**
- Consumes: `Layer`, `Layer::thresholds`, `Network::{wave, reset_state, on_layer, size, layers}`.
- Produces: `Network::with_layer_mut<R>(&self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R` (pub(crate)); `Network::measure_layer_rates(&mut self, warmup: usize, waves: usize, input: &impl Fn(usize) -> Vec<u32>) -> Vec<f64>` (pub(crate)); `Network::layer_thresholds(&self, z: usize) -> Vec<i16>` (pub).

- [ ] **Step 1: Write failing tests** (append to `network.rs` `mod tests`)

```rust
    #[test]
    fn layer_thresholds_reads_layer() {
        let net = Network::new(two_layer()); // jitter 0 -> all i16::MAX
        let t = net.layer_thresholds(1);
        assert_eq!(t.len(), 16); // size 4 -> 16
        assert!(t.iter().all(|&x| x == i16::MAX));
    }

    #[test]
    fn measure_rates_reflects_l0_injection() {
        // inject L0 locals 0..8 every wave -> rates[0] ≈ 8/64 = 0.125; L1 silent (near-max threshold)
        let mut net = Network::new(two_layer());
        let input = |_w: usize| (0..8u32).collect::<Vec<u32>>();
        let rates = net.measure_layer_rates(4, 32, &input);
        assert!((rates[0] - 8.0 / 64.0).abs() < 0.02, "L0 rate {} != ~0.125", rates[0]);
        assert!(rates[1] < 0.01, "L1 should be silent, got {}", rates[1]);
    }

    #[test]
    fn measure_preserves_listeners() {
        let mut net = Network::new(two_layer());
        let hits = Arc::new(Mutex::new(0usize));
        {
            let h = hits.clone();
            net.on_layer(0, Box::new(move |_w, _f| *h.lock().unwrap() += 1));
        }
        let input = |_w: usize| vec![0u32];
        net.measure_layer_rates(2, 8, &input);
        *hits.lock().unwrap() = 0; // reset, then one wave must still hit the user listener
        net.wave(&[0]);
        assert_eq!(*hits.lock().unwrap(), 1, "user listener must survive measurement");
    }
```

(The `network.rs` test module already imports `std::sync::{Arc, Mutex}`.)

- [ ] **Step 2: Run — expect fail** — `cargo test --lib wave_net::network` → FAIL (methods missing).

- [ ] **Step 3: Implement.** First ensure the import line is `use std::sync::{Arc, Mutex};` (it is currently `use std::sync::Mutex;` — change it). Then add inside `impl Network`:

```rust
    /// Locked mutable access to one layer (how calibration reaches Layer methods).
    pub(crate) fn with_layer_mut<R>(&self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R {
        let mut g = self.layers[z].lock().unwrap();
        f(&mut g)
    }

    /// A copy of a layer's per-neuron thresholds (introspection / determinism tests).
    pub fn layer_thresholds(&self, z: usize) -> Vec<i16> {
        self.with_layer_mut(z, |l| l.thresholds().to_vec())
    }

    /// Reset, run `warmup` waves (discarded), then `waves` counted; per-layer firing rate =
    /// spikes / (layer_size * waves). Saves and restores the caller's listeners around the run.
    pub(crate) fn measure_layer_rates(
        &mut self,
        warmup: usize,
        waves: usize,
        input: &impl Fn(usize) -> Vec<u32>,
    ) -> Vec<f64> {
        let l = self.layers.len();
        // Move the caller's listeners aside (boxed Fn is not Clone), install counters.
        let saved = std::mem::replace(&mut self.listeners, (0..l).map(|_| None).collect());
        let counts = Arc::new(Mutex::new(vec![0u64; l]));
        for z in 0..l {
            let c = counts.clone();
            self.listeners[z] = Some(Box::new(move |_w: usize, fired: &[u32]| {
                c.lock().unwrap()[z] += fired.len() as u64;
            }));
        }
        self.reset_state();
        for w in 0..warmup {
            self.wave(&input(w));
        }
        counts.lock().unwrap().iter_mut().for_each(|c| *c = 0); // discard warmup
        for w in 0..waves {
            self.wave(&input(warmup + w));
        }
        self.listeners = saved; // restore caller's listeners; counters dropped
        let counts = std::mem::take(&mut *counts.lock().unwrap());
        let ls = (self.size as u64) * (self.size as u64);
        let denom = (ls * waves as u64) as f64;
        counts.iter().map(|&s| s as f64 / denom).collect()
    }
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::network` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(wave_net): layer measurement (listener save/restore) + per-layer access"`

---

### Task 3: `calibrate` entry + `random_l0_input` + `P_INPUT`

**Files:**
- Modify: `src/wave_net/synapse.rs` (add `P_INPUT`)
- Modify: `src/wave_net/calibrate.rs` (replace stub)

**Interfaces:**
- Consumes: `Network::{layer_count, measure_layer_rates, with_layer_mut, layer_thresholds}`, `Layer::calibrate_step`, `mix`, `key`, `P_INPUT`.
- Produces: `pub const P_INPUT: u64 = 5;`; `CalibrateParams` (+ `Default`); `pub fn random_l0_input(seed: u64, size: u32, fraction_q16: u32) -> impl Fn(usize) -> Vec<u32>`; `impl Network { pub fn calibrate(&mut self, params: &CalibrateParams, input: &impl Fn(usize) -> Vec<u32>) }`.

- [ ] **Step 1: Add `P_INPUT`** to `synapse.rs`, next to `P_TARGET`/`P_THRESHOLD`:

```rust
pub const P_INPUT: u64 = 5;
```

- [ ] **Step 2: Write failing tests** — replace `calibrate.rs` entirely with the implementation + tests below (do the impl and tests together; new symbols make the tests uncompilable until the impl exists, so this is the red→green in one file).

- [ ] **Step 3: Implement `calibrate.rs`**

```rust
//! Firing-rate calibration: lower per-layer thresholds so each layer fires near a target rate on a
//! driven input. Bottom-up (each layer tuned once its feeder fires) then a few global refine passes
//! to absorb downward (level 0/-1) coupling. The Network orchestrates measurement; each `Layer`
//! owns tuning its own thresholds. Deterministic and single-threaded.

use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix, P_INPUT};

#[derive(Clone, Copy, Debug)]
pub struct CalibrateParams {
    pub target_permille: u64, // desired per-layer firing rate, e.g. 100 = 10%
    pub tol_permille: u64,    // stop a layer when |rate - target| <= tol
    pub warmup: usize,        // waves discarded per measurement
    pub waves: usize,         // waves counted per measurement
    pub max_steps: usize,     // max adjust steps per layer (bottom-up)
    pub refine_passes: usize, // global all-layers passes after bottom-up
    pub step_shift: u32,      // geometric step = max_threshold >> step_shift
}

impl Default for CalibrateParams {
    fn default() -> CalibrateParams {
        CalibrateParams {
            target_permille: 100,
            tol_permille: 20,
            warmup: 32,
            waves: 128,
            max_steps: 48,
            refine_passes: 4,
            step_shift: 2,
        }
    }
}

/// A deterministic per-wave input: injects each L0 local with probability `fraction_q16 / 2^16`.
pub fn random_l0_input(seed: u64, size: u32, fraction_q16: u32) -> impl Fn(usize) -> Vec<u32> {
    let ls = size * size;
    move |wave: usize| {
        let mut v = Vec::new();
        for local in 0..ls {
            let h = mix(key(seed, local, 0, wave as u32, P_INPUT));
            if ((h & 0xFFFF) as u32) < fraction_q16 {
                v.push(local);
            }
        }
        v
    }
}

impl Network {
    /// Lower per-layer thresholds (layers 1..L; L0 is the input surface, left as-is) so each fires
    /// near target on `input`. Mutates in place; preserves the caller's listeners.
    pub fn calibrate(&mut self, params: &CalibrateParams, input: &impl Fn(usize) -> Vec<u32>) {
        let l = self.layer_count();
        let target = params.target_permille as f64 / 1000.0;
        let tol = params.tol_permille as f64 / 1000.0;

        // Phase 1 — bottom-up: fix each layer before moving up (its feeder is now firing).
        for z in 1..l {
            for _ in 0..params.max_steps {
                let rates = self.measure_layer_rates(params.warmup, params.waves, input);
                let adjusted = self.with_layer_mut(z, |layer| {
                    layer.calibrate_step(rates[z], target, tol, params.step_shift)
                });
                if !adjusted {
                    break;
                }
            }
        }

        // Phase 2 — global refine: absorb the downward (level 0/-1) coupling.
        for _ in 0..params.refine_passes {
            let rates = self.measure_layer_rates(params.warmup, params.waves, input);
            let mut moved = false;
            for z in 1..l {
                moved |= self.with_layer_mut(z, |layer| {
                    layer.calibrate_step(rates[z], target, tol, params.step_shift)
                });
            }
            if !moved {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::TopologyLevel;

    fn test_config() -> Config {
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 64,
        };
        Config { seed: 0x00C0_FFEE, size: 8, layers: vec![layer; 4] }
    }

    #[test]
    fn random_input_hits_expected_fraction() {
        let input = random_l0_input(1, 8, 32768); // ~50%
        let total: usize = (0..200).map(|w| input(w).len()).sum();
        let frac = total as f64 / (200 * 64) as f64;
        assert!((frac - 0.5).abs() < 0.05, "fraction {frac} != ~0.5");
    }

    #[test]
    fn calibrate_warms_silent_upper_layers() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(0xABC, 8, 20000); // ~30% of L0 driven
        let params = CalibrateParams::default();
        let top = net.layer_count() - 1;

        let before = net.measure_layer_rates(params.warmup, params.waves, &input)[top];
        assert!(before < 0.01, "precondition: top silent, got {before}");

        net.calibrate(&params, &input);

        let after = net.measure_layer_rates(params.warmup, params.waves, &input)[top];
        let target = params.target_permille as f64 / 1000.0;
        assert!(after > 0.0, "top should fire after calibration");
        assert!(after > target / 2.0 && after < target * 2.0, "top rate {after} not near {target}");
        let max_t = net.layer_thresholds(top).into_iter().max().unwrap();
        assert!(max_t < i16::MAX, "top threshold should have dropped, is {max_t}");
    }

    #[test]
    fn calibrate_lowers_every_upper_layer() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(7, 8, 20000);
        net.calibrate(&CalibrateParams::default(), &input);
        for z in 1..net.layer_count() {
            let max_t = net.layer_thresholds(z).into_iter().max().unwrap();
            assert!(max_t < i16::MAX, "layer {z} threshold should have dropped");
        }
    }

    #[test]
    fn calibrate_is_deterministic() {
        let input = random_l0_input(42, 8, 20000);
        let params = CalibrateParams::default();
        let run = || {
            let mut net = Network::new(test_config());
            net.calibrate(&params, &input);
            (0..net.layer_count()).map(|z| net.layer_thresholds(z)).collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn calibrate_preserves_listeners() {
        let mut net = Network::new(test_config());
        let hits = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        {
            let h = hits.clone();
            net.on_layer(0, Box::new(move |_w, _f| *h.lock().unwrap() += 1));
        }
        let input = random_l0_input(3, 8, 20000);
        net.calibrate(&CalibrateParams::default(), &input);
        *hits.lock().unwrap() = 0;
        net.wave(&input(0));
        assert!(*hits.lock().unwrap() >= 1, "user listener must survive calibration");
    }
}
```

- [ ] **Step 4: Run — expect pass** — `cargo test --lib wave_net::calibrate` → PASS. If `calibrate_warms_silent_upper_layers` lands outside the `(target/2, target*2)` band, tune `max_steps`/`waves` upward (descent headroom / estimate stability) — do not loosen the assertion below "fires and dropped."

- [ ] **Step 5: Full suite + warning check** — `cargo test` all pass; `cargo build --all-targets 2>&1` no warnings.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(wave_net): firing-rate calibration (hybrid bottom-up + global refine)"`

---

## Self-review notes

- **Spec coverage:** Layer-owned tuning (T1), listener save/restore + measurement + per-layer access (T2), hybrid `calibrate` + `random_l0_input` + `P_INPUT` + determinism/warm/preserve tests (T3). Persistence hooks = `thresholds`/`set_thresholds` (T1), no serializer (non-goal). ✓
- **Types:** `calibrate_step(rate,target,tol,step_shift)->bool` consistent across T1 def / T3 call; `measure_layer_rates(warmup,waves,input)->Vec<f64>` consistent T2/T3; `with_layer_mut` closure returns the `bool` from `calibrate_step`. ✓
- **Determinism:** hash-based `random_l0_input`, `reset_state` per measurement, single-threaded. ✓
