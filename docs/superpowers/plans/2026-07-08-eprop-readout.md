# Non-spiking Potential Readout Implementation Plan (Spec 3, V2a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-spiking drain-only readout layer to the engine, and an e-prop learning variant that reads graded membrane **potential** (population sums) as its output — beating a frozen control on the K=2 held-category task.

**Architecture:** Engine gets a `Layer.readout` flag (default off), a `Network::new_with_readout` constructor flagging the last layer, and a `process_layer` early-return that makes a readout layer a drain-only integrator (no fire, no leak). The bench extends `eprop.rs`: when `cfg.readout`, append a readout layer, build with `new_with_readout`, and score classes from readout potentials — reusing V1's reward/eligibility/shadow/loop.

**Tech Stack:** Rust edition 2024, std only, no deps. Inline `#[cfg(test)]` tests. `f64` in bench only.

## Global Constraints

- **Std only**; **no `unsafe`**; **warning-free** build.
- **Determinism is a hard requirement**; single-threaded.
- Engine change must **not alter non-readout behavior** — every existing test stays green.
- `f64` only in the bench. Tests inline `#[cfg(test)]`, test-first.
- **One commit per task**, conventional commits. **NEVER** a `Co-Authored-By` trailer. **NEVER** push.
- On branch `feat/eprop-readout`.
- Verify each task with `cargo test` + warning-free `cargo build` before committing.

## File structure

| File | Change |
|---|---|
| `src/wave_net/neurons.rs` | add `pub readout: bool` to `Layer`; init `false` in `Layer::new` |
| `src/wave_net/network.rs` | `new` → private `build(config, readout_last)`; add `new_with_readout` |
| `src/wave_net/wave.rs` | readout early-return in `process_layer`; engine test |
| `src/bench/eprop.rs` | `EpropConfig.readout`, readout engine config, readout scoring, `train` branch, tests |

---

### Task 1: Engine — non-spiking readout layer

**Files:**
- Modify: `src/wave_net/neurons.rs`, `src/wave_net/network.rs`, `src/wave_net/wave.rs`

**Interfaces:**
- Produces: `Layer.readout: bool` (public); `Network::new_with_readout(config) -> Network` (flags the last layer readout). A readout layer runs drain-only in `process_layer` (never fires, no leak).

- [ ] **Step 1: Add the `Layer.readout` field**

In `src/wave_net/neurons.rs`, add the field to the struct (after `adapt_decay`):
```rust
    pub adapt_bump: i16,   // added to adapt on each fire (0 = plain LIF)
    pub adapt_decay: u8,   // right-shift decay of adapt per wave
    pub readout: bool,     // non-spiking drain-only output layer: integrates input, never fires
```
And initialize it in the `Layer::new` returned struct (after `adapt_decay: cfg.adapt_decay,`):
```rust
            adapt_bump: cfg.adapt_bump,
            adapt_decay: cfg.adapt_decay,
            readout: false,
```

- [ ] **Step 2: Refactor `Network::new` and add `new_with_readout`**

In `src/wave_net/network.rs`, replace the `pub fn new(config: Config) -> Network { ... }` body with a delegation, and add the constructor + a private builder:
```rust
    pub fn new(config: Config) -> Network {
        Network::build(config, false)
    }

    /// Like `new`, but flags the **last** layer as a non-spiking drain-only readout (output sink).
    pub fn new_with_readout(config: Config) -> Network {
        Network::build(config, true)
    }

    fn build(config: Config, readout_last: bool) -> Network {
        config.validate().expect("invalid config");
        let size = config.size;
        let l = config.layers.len();
        let mut layers = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let mut layer = Layer::new(lc, config.seed, z as u32, size);
            if z == 0 {
                // L0 is the input transducer (fires only on injection, never adapts).
                layer.threshold.iter_mut().for_each(|t| *t = i16::MAX);
                layer.adapt_bump = 0;
            }
            if readout_last && z == l - 1 {
                layer.readout = true;
            }
            layers.push(Mutex::new(layer));
        }
        Network {
            seed: config.seed,
            size,
            layers,
            wave_id: AtomicUsize::new(0),
            listeners: (0..l).map(|_| None).collect(),
        }
    }
```

- [ ] **Step 3: Write the failing engine test**

Add to `src/wave_net/wave.rs` `#[cfg(test)] mod tests` (it has helpers already; add imports it needs):
```rust
    #[test]
    fn readout_layer_integrates_and_never_fires() {
        use crate::wave_net::config::{Config, LayerConfig};
        use crate::wave_net::network::Network;
        use std::sync::{Arc, Mutex};
        let build = |readout: bool| -> (usize, i16) {
            let l0 = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 0, count: 1 }],
                leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0,
                baseline_init: 2, adapt_bump: 0, adapt_decay: 5,
            };
            let l1 = LayerConfig { topology: vec![], ..l0.clone() };
            let cfg = Config { seed: 1, size: 4, layers: vec![l0, l1] };
            let mut net = if readout { Network::new_with_readout(cfg) } else { Network::new(cfg) };
            let fires = Arc::new(Mutex::new(0usize));
            {
                let f = fires.clone();
                net.on_layer(1, Box::new(move |_w, fired: &[u32]| *f.lock().unwrap() += fired.len()));
            }
            let all: Vec<u32> = (0..16).collect();
            for _ in 0..8 {
                net.wave(&all);
            }
            (*fires.lock().unwrap(), net.potential(1, 0))
        };
        let (normal_fires, _) = build(false);
        let (readout_fires, readout_pot) = build(true);
        assert!(normal_fires > 0, "control: a normal L1 fires under the drive");
        assert_eq!(readout_fires, 0, "readout L1 must never fire");
        assert!(readout_pot > 1, "readout L1 must integrate its input (potential {readout_pot})");
    }
```

- [ ] **Step 4: Run to verify it fails**

Run: `cargo test wave::tests::readout_layer_integrates_and_never_fires`
Expected: FAIL — `new_with_readout` exists, but `process_layer` still fires the readout layer (`readout_fires > 0`).

- [ ] **Step 5: Implement the readout branch in `process_layer`**

In `src/wave_net/wave.rs`, wrap the decide loop so a readout layer skips it (and returns before generate/leak/adapt). Replace the decide block:
```rust
    // 4. decide (ALIF effective threshold = baseline + adapt, in i32; fire bumps adapt)
    fired.clear();
    for i in 0..ls {
        let eff = layer.threshold[i] as i32 + (layer.adapt[i] >> ADAPT_SHIFT);
        if layer.cooldown[i] == 0 && (layer.potential[i] as i32) >= eff {
            layer.potential[i] = 0;
            layer.cooldown[i] = layer.cooldown_base;
            let bumped = layer.adapt[i] + ((layer.adapt_bump as i32) << ADAPT_SHIFT);
            layer.adapt[i] = bumped.clamp(0, ADAPT_MAX);
            fired.push(i as u32);
        }
    }
```
with:
```rust
    // A readout layer is a non-spiking drain-only integrator: its potential (folded above) is the
    // clean cumulative ±1 input for the trial. No fire, no generate, no leak, no adapt — return now.
    fired.clear();
    if layer.readout {
        return;
    }

    // 4. decide (ALIF effective threshold = baseline + adapt, in i32; fire bumps adapt)
    for i in 0..ls {
        let eff = layer.threshold[i] as i32 + (layer.adapt[i] >> ADAPT_SHIFT);
        if layer.cooldown[i] == 0 && (layer.potential[i] as i32) >= eff {
            layer.potential[i] = 0;
            layer.cooldown[i] = layer.cooldown_base;
            let bumped = layer.adapt[i] + ((layer.adapt_bump as i32) << ADAPT_SHIFT);
            layer.adapt[i] = bumped.clamp(0, ADAPT_MAX);
            fired.push(i as u32);
        }
    }
```

- [ ] **Step 6: Run the full suite (regression) + warning-free**

Run: `cargo test` and `cargo build`
Expected: the engine test passes; **all existing tests still pass** (readout is off by default; non-readout behavior is unchanged); warning-free.

- [ ] **Step 7: Commit**

```bash
git add src/wave_net/neurons.rs src/wave_net/network.rs src/wave_net/wave.rs
git commit -m "feat: non-spiking drain-only readout layer (Layer.readout + new_with_readout)"
```

---

### Task 2: Bench — potential-readout learning variant

**Files:**
- Modify: `src/bench/eprop.rs`

**Interfaces:**
- Consumes: `Network::new_with_readout`, `Network::potential`, V1's `RewardTracker`/`read_shadow`/`write_thresholds`/`trial_eligibility`.
- Produces: `EpropConfig.readout: bool`; `train` scores from readout potentials when set.

- [ ] **Step 1: Write the failing tests**

Add to `src/bench/eprop.rs` `mod tests`:
```rust
    #[test]
    fn eprop_readout_is_deterministic() {
        let mut cfg = EpropConfig::demo();
        cfg.readout = true;
        let a = train(&cfg, 0.3);
        let b = train(&cfg, 0.3);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    #[test]
    fn eprop_readout_learns_and_beats_frozen() {
        let mut cfg = EpropConfig::demo();
        cfg.readout = true;
        let learn = train(&cfg, 0.3);
        let frozen = train(&cfg, 0.0);
        eprintln!("readout learn  {:?}", learn.accuracy_permille);
        eprintln!("readout frozen {:?}", frozen.accuracy_permille);
        // V1-vs-V2a comparison (spiking-population output, same config sans readout):
        let mut v1 = cfg.clone();
        v1.readout = false;
        eprintln!("v1 learn       {:?}", train(&v1, 0.3).accuracy_permille);
        let ll = late_mean(&learn.accuracy_permille);
        let lf = late_mean(&frozen.accuracy_permille);
        let chance = 1000 / cfg.k as u64;
        assert!(ll > chance + 80, "readout learning {ll} should be above chance {chance}");
        assert!(ll > lf + 150, "readout learning {ll} should beat frozen {lf}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::eprop::tests::eprop_readout_is_deterministic`
Expected: FAIL to compile — `EpropConfig` has no `readout` field.

- [ ] **Step 3: Add the readout field, config, scoring, and `train` branch**

In `EpropConfig`, add the field (and set it in `demo()`):
```rust
    pub calib_fraction_q16: u32,
    pub readout: bool, // V2a: append a non-spiking readout layer and score from its potentials
}
```
```rust
            calib_fraction_q16: 20000,
            readout: false,
        }
```

Add a readout engine config + scorer (non-test code, near `engine_config`):
```rust
    fn engine_config_readout(&self) -> Config {
        use crate::wave_net::config::LayerConfig;
        use crate::wave_net::synapse::TopologyLevel;
        let comp = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: 16 }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32,
            baseline_init: self.baseline_init, adapt_bump: self.adapt_bump, adapt_decay: self.adapt_decay,
        };
        let readout = LayerConfig { topology: vec![], ..comp.clone() };
        let mut layers = vec![comp; self.layers];
        layers.push(readout);
        Config { seed: self.seed, size: self.size, layers }
    }
```
```rust
/// Class scores from a readout layer's integrated potentials: K contiguous population sums.
fn readout_scores(net: &Network, readout_z: usize, k: usize) -> Vec<i64> {
    let ls = (net.size() * net.size()) as usize;
    let group = (ls / k).max(1);
    (0..k)
        .map(|c| ((c * group)..((c + 1) * group).min(ls)).map(|i| net.potential(readout_z, i) as i64).sum())
        .collect()
}
```

In `train`, branch the constructor and the scoring (`outs` becomes `Vec<i64>` so both paths share the reward/argmax logic):
```rust
pub fn train(cfg: &EpropConfig, lr: f64) -> LearnCurve {
    let mut net = if cfg.readout {
        Network::new_with_readout(cfg.engine_config_readout())
    } else {
        Network::new(cfg.engine_config())
    };
    let input = random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);

    let mut shadow = read_shadow(&net);
    let mut rt = RewardTracker::new(cfg.reward_rate);
    let mut outcomes: Vec<bool> = Vec::with_capacity(cfg.trials);

    for t in 0..cfg.trials {
        let class = pick_class(cfg.seed, t, cfg.k);
        let elig = trial_eligibility(&mut net, cfg, class, t);
        let outs: Vec<i64> = if cfg.readout {
            readout_scores(&net, net.layer_count() - 1, cfg.k)
        } else {
            let top = &elig[elig.len() - 1];
            let group = (top.len() / cfg.k).max(1);
            (0..cfg.k).map(|c| top[c * group..(c + 1) * group].iter().map(|&x| x as i64).sum()).collect()
        };
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
(The readout layer's spike eligibility is all zeros, so the V1 shadow loop leaves its thresholds untouched — no separate exclusion needed.)

- [ ] **Step 4: Run, then TUNE against the printed curves (do not fudge)**

Run: `cargo test bench::eprop::tests::eprop_readout_learns_and_beats_frozen -- --nocapture`
Read `readout learn` / `readout frozen` / `v1 learn`. Expected: readout learning rises above chance and beats its frozen control. The graded potential margin has a *different scale* than V1's spike margin, so **re-tune `lr` and `reward_rate`** first (the readout scores are larger, so a smaller `lr` may be needed).

If, after reasonable tuning, readout learning cannot beat frozen, **stop and report** — that is the finding (all-internal / feedback-alignment learning can't be driven by thresholds alone here; points to broadcast-error alignment or potential-based internal eligibility). Do not weaken the assertion. Whether V2a beats or trails V1 (~770) is itself a reported result, not something to force.

- [ ] **Step 5: Full suite + warning-free**

Run: `cargo test` and `cargo build`
Expected: all pass (V1 tests included); warning-free.

- [ ] **Step 6: Commit**

```bash
git add src/bench/eprop.rs
git commit -m "feat: potential-readout e-prop variant (graded output from a non-spiking readout)"
```

---

## Self-review

**Spec coverage:**
- Engine: `Layer.readout` + `new_with_readout` + drain-only `process_layer` branch → Task 1. Engine unit (`readout` integrates, never fires) + full-suite regression → Task 1.
- Bench: append readout layer, `new_with_readout`, score from potentials (`readout_scores`, K population sums via existing `potential()`), graded margin reward, internal-only training (readout auto-excluded via zero eligibility) → Task 2.
- Success: readout learns + beats frozen; V1-vs-V2a printed; determinism → Task 2. Honesty gate on all-internal learning → Task 2 Step 4.
- No config-struct churn (`Layer.readout` defaults false); `f64` bench-only; determinism preserved → throughout.

**Placeholder scan:** none — concrete code and commands throughout.

**Type consistency:** `outs: Vec<i64>` unifies spike-population and potential-population scoring; `readout_scores(&Network, usize, usize) -> Vec<i64>`; `new_with_readout(Config) -> Network`; `EpropConfig.readout` set in `demo()` and read in `train`/config. V1's `read_shadow`/`write_thresholds`/`trial_eligibility`/`RewardTracker`/`late_mean` reused unchanged.

**Note:** the drain-only (not leaky) readout is a deliberate refinement over the spec's original "leaky" wording (the floored leak would eat weak input); the spec was amended to match.
