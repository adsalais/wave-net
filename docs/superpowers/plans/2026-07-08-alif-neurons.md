# ALIF Neurons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every neuron an adaptive leaky integrate-and-fire (ALIF) neuron — the firing threshold becomes `baseline + adapt`, where `adapt` is a per-neuron slow variable bumped on fire and decayed each wave — with a low baseline init and calibration retargeted to tune the baseline with adaptation live.

**Architecture:** One new per-neuron `i16` state `adapt` (rest 0) lives beside `potential`. Firing bumps it (saturating, ≥ 0); every wave decays it geometrically like `leak`. The decide step compares `potential` against `baseline + adapt` in `i32`. Three new per-layer config params (`baseline_init`, `adapt_bump`, `adapt_decay`) drive it. `adapt_bump = 0` recovers plain LIF dynamics exactly. Calibration is unchanged code — its symmetric step now tunes the low baseline with adaptation self-regulation already active.

**Tech Stack:** Rust, edition 2024, standard library only, no external deps. Inline `#[cfg(test)]` tests per module.

## Global Constraints

- **Standard library only** in `src/`; **no `unsafe`**; **warning-free build** (`cargo build`).
- **Determinism is a hard requirement** — results are a pure function of `(seed, config, input)`. Single-threaded.
- Tests are **inline `#[cfg(test)]` per module**, test-first (TDD).
- **One commit per task**, conventional-commit messages (`feat:`/`fix:`/`refactor:`/`docs:`/`test:`/`chore:`).
- **NEVER add a `Co-Authored-By` trailer** to commit messages. Keep messages plain, ending at the body.
- **NEVER push.**
- Already on branch `feat/alif-neurons` (created during brainstorming; the design spec is committed there).
- Verify each task with `cargo test` (all modules) and `cargo build` (warning-free) before committing.

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/wave_net/config.rs` | `LayerConfig`/`Config`, `demo`, `validate` | Add 3 params, demo values, validate `adapt_decay >= 1` |
| `src/wave_net/neurons.rs` | `Layer` state + params + threshold tuning | Add `adapt`/`adapt_bump`/`adapt_decay` fields; low-baseline init; doc |
| `src/wave_net/wave.rs` | `process_layer` per-wave step | Effective-threshold fire test; bump on fire; decay adapt; doc |
| `src/wave_net/network.rs` | orchestration, introspection | `reset_state` zeros `adapt`; add `adaptation()`; tests |
| `src/wave_net/calibrate.rs` | firing-rate calibration | No code change; retarget test config, rewrite 2 tests, add 1 |
| `AGENTS.md` | agent guidance | Update silent-start language to boots-hot / self-regulating+calibrated |

---

### Task 1: Config — ALIF parameters + validation

**Files:**
- Modify: `src/wave_net/config.rs` (struct `LayerConfig`, `demo`, `validate`, tests)
- Modify (literal updates only): `src/wave_net/neurons.rs:90`, `src/wave_net/network.rs:171`, `src/wave_net/calibrate.rs:93`, `src/wave_net/wave.rs:87`

**Interfaces:**
- Produces: `LayerConfig` gains public fields `baseline_init: i16`, `adapt_bump: i16`, `adapt_decay: u8`. `Config::validate` returns `Err` when any layer has `adapt_decay == 0`.
- Consumes: nothing new.

- [ ] **Step 1: Add the three fields and update every construction site so the crate compiles**

In `src/wave_net/config.rs`, add fields to `LayerConfig` (after `threshold_jitter`):

```rust
#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub topology: Vec<TopologyLevel>,
    pub leak: (u8, u8),        // right-shift amounts a, b in `p -= (p>>a) + (p>>b)`
    pub cooldown_base: u8,     // refractory reload on fire
    pub inhibitor_ratio: u32,  // Q16: inhibitory iff (hash & 0xFFFF) < inhibitor_ratio
    pub threshold_jitter: u16, // baseline = baseline_init + rand(0..threshold_jitter)
    pub baseline_init: i16,    // construction center for the baseline threshold (low, not i16::MAX)
    pub adapt_bump: i16,       // added to `adapt` on each fire (β); 0 = plain LIF dynamics
    pub adapt_decay: u8,       // right-shift decay of `adapt` per wave: adapt -= adapt >> adapt_decay (>= 1)
}
```

Update `demo()` in `config.rs` — the `layer` literal gains the three fields (ALIF, low baseline):

```rust
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 2, radius: 1, count: 2 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
                TopologyLevel { level: -1, radius: 0, count: 1 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 9830, // ~0.15 * 65536
            threshold_jitter: THRESHOLD_JITTER_DEFAULT,
            baseline_init: 12,
            adapt_bump: 16,
            adapt_decay: 5,
        };
```

Update the four test literals (add the three fields; choose `baseline_init` to preserve each test's current intent). These layers keep `adapt_bump: 0` (LIF) and a valid `adapt_decay: 5`:

`src/wave_net/neurons.rs` `lc()` (keep the high baseline so existing neuron-mechanics tests are stable):
```rust
    fn lc(jitter: u16) -> LayerConfig {
        LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: jitter,
            baseline_init: i16::MAX,
            adapt_bump: 0,
            adapt_decay: 5,
        }
    }
```

`src/wave_net/wave.rs` `low_layer()` (threshold is overwritten after construction, so `baseline_init` is irrelevant here):
```rust
        let cfg = LayerConfig {
            topology: topo,
            leak: (3, 5),
            cooldown_base,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 0,
            adapt_bump: 0,
            adapt_decay: 5,
        };
```

`src/wave_net/network.rs` `two_layer()` `l0` (keep L1 silent for the routing/determinism tests → high baseline):
```rust
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 0, count: 1 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: i16::MAX,
            adapt_bump: 0,
            adapt_decay: 5,
        };
```

`src/wave_net/calibrate.rs` `test_config()` `layer` (keep silent-start so the calibration tests are stable until Task 5 retargets them → high baseline):
```rust
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 64,
            baseline_init: i16::MAX,
            adapt_bump: 0,
            adapt_decay: 5,
        };
```

- [ ] **Step 2: Write the failing test for validation**

Add to the `#[cfg(test)] mod tests` in `src/wave_net/config.rs`:

```rust
    #[test]
    fn rejects_zero_adapt_decay() {
        let mut c = Config::demo();
        c.layers[0].adapt_decay = 0;
        assert!(c.validate().is_err());
    }
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p wave-net rejects_zero_adapt_decay`
Expected: FAIL — `validate()` currently returns `Ok`, so `assert!(...is_err())` fails. (If the crate name differs, `cargo test rejects_zero_adapt_decay`.)

- [ ] **Step 4: Implement the validation check**

In `Config::validate` in `src/wave_net/config.rs`, inside the `for (z, lc) in self.layers.iter().enumerate()` loop, after the `cooldown_base` check, add:

```rust
            if lc.adapt_decay == 0 {
                return Err(format!("layer {z}: adapt_decay must be >= 1"));
            }
```

- [ ] **Step 5: Run tests to verify pass, and the whole suite is green**

Run: `cargo test` and `cargo build`
Expected: PASS — `rejects_zero_adapt_decay` passes; all existing tests still pass; build is warning-free.

- [ ] **Step 6: Commit**

```bash
git add src/wave_net/config.rs src/wave_net/neurons.rs src/wave_net/wave.rs src/wave_net/network.rs src/wave_net/calibrate.rs
git commit -m "feat: add ALIF config params (baseline_init, adapt_bump, adapt_decay)"
```

---

### Task 2: Layer — adaptation state + low-baseline init

**Files:**
- Modify: `src/wave_net/neurons.rs` (`Layer` struct, `Layer::new`, module doc, tests)

**Interfaces:**
- Consumes: `LayerConfig.baseline_init`, `LayerConfig.adapt_bump`, `LayerConfig.adapt_decay` (Task 1).
- Produces: `Layer` gains public fields `adapt: Vec<i16>` (len = layer size, all 0 at construction), `adapt_bump: i16`, `adapt_decay: u8`. Baseline threshold is initialized to `(baseline_init + rand(0..threshold_jitter)).clamp(1, i16::MAX)`.

- [ ] **Step 1: Write the failing tests**

Add to `#[cfg(test)] mod tests` in `src/wave_net/neurons.rs`. First a low-baseline config helper, then two tests:

```rust
    fn lc_baseline(jitter: u16, baseline: i16) -> LayerConfig {
        LayerConfig { baseline_init: baseline, ..lc(jitter) }
    }

    #[test]
    fn thresholds_near_baseline_within_jitter() {
        let l = Layer::new(&lc_baseline(128, 12), 1, 0, 8);
        for &t in &l.threshold {
            assert!((12..12 + 128).contains(&t), "threshold {t} out of [12, 140) band");
        }
    }

    #[test]
    fn new_zeroes_adaptation() {
        let l = Layer::new(&lc_baseline(128, 12), 1, 0, 8);
        assert_eq!(l.adapt.len(), 64);
        assert!(l.adapt.iter().all(|&a| a == 0));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p wave-net thresholds_near_baseline_within_jitter new_zeroes_adaptation`
Expected: FAIL to compile — `l.adapt` does not exist yet, and the current init uses `i16::MAX - jitter` (so the band assertion would also fail). A compile error is the expected failing state.

- [ ] **Step 3: Add the fields and rewrite `Layer::new`**

In `src/wave_net/neurons.rs`, update the module doc comment (top of file) to describe the new model:

```rust
//! `neurons` — a `Layer`'s per-neuron state, its delivery inbox/outbox pair, and its
//! per-layer parameters. The `threshold` field is the ALIF **baseline**: it inits low
//! (`baseline_init + jitter`, clamped to [1, i16::MAX]) and is tuned by calibration. Each
//! neuron also carries `adapt`, a slow variable bumped on fire and decayed each wave; the
//! effective firing threshold is `threshold + adapt`.
```

Add fields to `struct Layer` (after `threshold`, and note `adapt` belongs with the wave-mutable hot state):

```rust
pub struct Layer {
    // wave-mutable hot state
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub adapt: Vec<i16>,      // ALIF adaptation: rest 0, >= 0; bumped on fire, decayed each wave
    pub inbox: Vec<Synapse>,  // drained THIS wave (filled last wave)
    pub outbox: Vec<Synapse>, // filled for NEXT wave; swapped with inbox at wave end

    // tunable params (calibration/training will rewrite these between phases)
    pub threshold: Vec<i16>, // ALIF baseline; effective threshold is threshold + adapt

    // fixed structure
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<TopologyLevel>,
    pub inhibitor_ratio: u32,
    pub adapt_bump: i16,   // added to adapt on each fire (0 = plain LIF)
    pub adapt_decay: u8,   // right-shift decay of adapt per wave
}
```

Rewrite the init loop and the returned struct in `Layer::new`:

```rust
    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;
        let mut threshold = vec![0i16; ls];
        for (local, th) in threshold.iter_mut().enumerate() {
            let global = (base + local) as u32;
            let h = mix(key(seed, global, 0, 0, P_THRESHOLD));
            let jitter = map_range(h as u32, cfg.threshold_jitter as u32) as i32; // [0, jitter)
            *th = (cfg.baseline_init as i32 + jitter).clamp(1, i16::MAX as i32) as i16;
        }
        Layer {
            potential: vec![0; ls],
            cooldown: vec![0; ls],
            adapt: vec![0; ls],
            inbox: Vec::new(),
            outbox: Vec::new(),
            threshold,
            leak: cfg.leak,
            cooldown_base: cfg.cooldown_base,
            topology: cfg.topology.clone(),
            inhibitor_ratio: cfg.inhibitor_ratio,
            adapt_bump: cfg.adapt_bump,
            adapt_decay: cfg.adapt_decay,
        }
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test` and `cargo build`
Expected: PASS — new tests pass; existing neuron tests still pass (`lc()` uses `baseline_init: i16::MAX`, so `thresholds_near_i16_max_within_jitter` still holds — every threshold clamps to `i16::MAX`); build warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/neurons.rs
git commit -m "feat: add per-neuron adaptation state and low-baseline init"
```

---

### Task 3: Wave step — effective threshold, bump on fire, decay

**Files:**
- Modify: `src/wave_net/wave.rs` (`process_layer`, module doc, tests)

**Interfaces:**
- Consumes: `Layer.adapt`, `Layer.adapt_bump`, `Layer.adapt_decay`, `Layer.threshold` (Task 2).
- Produces: `process_layer` now fires when `cooldown == 0 && potential >= threshold + adapt` (compared in `i32`); on fire sets `adapt = (adapt + adapt_bump).clamp(0, i16::MAX)`; after leak, decays every neuron's `adapt` by `adapt -= adapt >> adapt_decay`. No signature change.

- [ ] **Step 1: Write the failing tests**

Add to `#[cfg(test)] mod tests` in `src/wave_net/wave.rs` (reuse the existing `low_layer`/`groups_for` helpers; `Layer` fields are public, so mutate `adapt_bump`/`adapt_decay`/`adapt` directly):

```rust
    #[test]
    fn fire_bumps_adaptation() {
        let mut l = low_layer(4, 3, 2, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        l.adapt_bump = 8;
        l.adapt_decay = 7; // decay of 8 is 8>>7 == 0, so the bump is observable this wave
        for _ in 0..3 {
            l.inbox.push(Synapse { target: 0, inhibitory: false });
        }
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
        assert_eq!(fired, vec![0]);
        assert_eq!(l.adapt[0], 8, "fire should bump adapt by adapt_bump");
    }

    #[test]
    fn adaptation_decays_each_wave() {
        let mut l = low_layer(1, 20_000, 2, vec![]); // threshold high -> no firing
        l.adapt_decay = 3;
        l.adapt[0] = 100;
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        // 100 - (100 >> 3) = 100 - 12 = 88
        assert_eq!(l.adapt[0], 88);
        assert!(fired.is_empty());
    }

    #[test]
    fn high_adaptation_blocks_firing() {
        // potential clears the baseline but not baseline + adapt.
        let drive = |adapt0: i16| {
            let mut l = low_layer(1, 5, 2, vec![]);
            l.adapt[0] = adapt0;
            for _ in 0..10 {
                l.inbox.push(Synapse { target: 0, inhibitory: false });
            }
            let mut acc = vec![0i32; 1];
            let mut out: Vec<SynapseGroup> = Vec::new();
            let mut fired = Vec::new();
            process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
            fired
        };
        assert_eq!(drive(0), vec![0], "baseline 5, potential 10 -> fires with no adaptation");
        assert!(drive(100).is_empty(), "effective threshold 105 blocks potential 10");
    }

    #[test]
    fn bump_zero_leaves_adaptation_at_rest() {
        let mut l = low_layer(4, 3, 1, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        l.adapt_bump = 0; // plain LIF
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        for _ in 0..3 {
            l.inbox.clear();
            for _ in 0..3 {
                l.inbox.push(Synapse { target: 0, inhibitory: false });
            }
            process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
            assert_eq!(l.adapt[0], 0, "adapt must stay 0 when adapt_bump is 0");
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p wave-net fire_bumps_adaptation adaptation_decays_each_wave high_adaptation_blocks_firing bump_zero_leaves_adaptation_at_rest`
Expected: FAIL — `process_layer` does not bump, decay, or gate on `adapt` yet (e.g. `fire_bumps_adaptation` sees `adapt[0] == 0`; `high_adaptation_blocks_firing` fires in both cases).

- [ ] **Step 3: Implement the wave-step changes**

In `src/wave_net/wave.rs`, update the module doc comment (top of file) — replace the final sentence about the clamp with:

```rust
//! `wave` — one layer's per-wave step: integrate (drain inbox) → inject → decide →
//! generate outgoing synapses → leak, then decay adaptation. Touches only this layer; the
//! Network routes the generated synapses into other layers' inboxes for the next wave. Firing
//! uses the ALIF effective threshold `threshold + adapt` (computed in i32); a fire bumps `adapt`
//! (saturating, >= 0) and every neuron's `adapt` decays geometrically each wave, like the leak.
```

Replace the decide loop (step 4):

```rust
    // 4. decide (ALIF effective threshold = baseline + adapt, in i32; fire bumps adapt)
    fired.clear();
    for i in 0..ls {
        let eff = layer.threshold[i] as i32 + layer.adapt[i] as i32;
        if layer.cooldown[i] == 0 && (layer.potential[i] as i32) >= eff {
            layer.potential[i] = 0;
            layer.cooldown[i] = layer.cooldown_base;
            layer.adapt[i] = (layer.adapt[i] as i32 + layer.adapt_bump as i32).clamp(0, i16::MAX as i32) as i16;
            fired.push(i as u32);
        }
    }
```

After the leak loop (step 6), add adaptation decay:

```rust
    // 7. decay adaptation toward rest (geometric, like the potential leak)
    let d = layer.adapt_decay;
    for a in layer.adapt.iter_mut() {
        *a -= *a >> d;
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test` and `cargo build`
Expected: PASS — the four new tests pass; existing wave tests still pass (they run with `adapt_bump = 0` and `adapt` resting at 0, so `eff == threshold` and dynamics are identical); build warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/wave.rs
git commit -m "feat: ALIF wave dynamics — effective threshold, bump on fire, decay"
```

---

### Task 4: Network — reset/introspect adaptation, determinism, self-limiting

**Files:**
- Modify: `src/wave_net/network.rs` (`reset_state`, new `adaptation` accessor, tests)

**Interfaces:**
- Consumes: `Layer.adapt` (Task 2), ALIF dynamics (Task 3).
- Produces: `Network::reset_state` also zeros every layer's `adapt`. New `Network::adaptation(&self, layer: usize, local: usize) -> i16` reads live adaptation state (mirrors `potential`).

- [ ] **Step 1: Write the failing tests**

Add a helper and tests to `#[cfg(test)] mod tests` in `src/wave_net/network.rs`. The helper builds a small low-baseline ALIF net whose upper layer visibly self-limits:

```rust
    // 2 layers, L0 -> L1 (level+1, radius 1, 4 targets), low baseline + strong adaptation on L1.
    fn alif_two_layer() -> Config {
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
            leak: (3, 5),
            cooldown_base: 1,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 2,
            adapt_bump: 200,
            adapt_decay: 4,
        };
        let l1 = LayerConfig { topology: vec![], ..l0.clone() };
        Config { seed: 5, size: 4, layers: vec![l0, l1] }
    }

    #[test]
    fn adaptation_accessor_and_reset() {
        let net = Network::new(alif_two_layer());
        let all_l0 = (0..16u32).collect::<Vec<u32>>();
        for _ in 0..3 {
            net.wave(&all_l0); // injection forces L0 to fire -> bumps L0 adapt
        }
        let any_nonzero = (0..16).any(|i| net.adaptation(0, i) > 0);
        assert!(any_nonzero, "L0 adaptation should be >0 after repeated firing");
        net.reset_state();
        for z in 0..net.layer_count() {
            for i in 0..16 {
                assert_eq!(net.adaptation(z, i), 0, "reset must zero adaptation");
            }
        }
    }

    #[test]
    fn determinism_includes_adaptation() {
        let inputs: [&[u32]; 4] = [&[0, 1, 2, 3], &[4, 5], &[], &[6, 7, 8]];
        let run = || {
            let net = Network::new(Config::demo());
            for _ in 0..6 {
                for inp in inputs {
                    net.wave(inp);
                }
            }
            (0..net.layer_count())
                .flat_map(|z| (0..(net.size() * net.size()) as usize).map(move |i| (z, i)))
                .map(|(z, i)| net.adaptation(z, i))
                .collect::<Vec<i16>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn adaptation_self_limits_rate() {
        let mut net = Network::new(alif_two_layer());
        let counts = Arc::new(Mutex::new(vec![0usize; 40]));
        {
            let c = counts.clone();
            net.on_layer(1, Box::new(move |w: usize, fired: &[u32]| {
                if w < 40 {
                    c.lock().unwrap()[w] += fired.len();
                }
            }));
        }
        let all_l0 = (0..16u32).collect::<Vec<u32>>();
        for _ in 0..40 {
            net.wave(&all_l0); // constant maximal drive into L1
        }
        let c = counts.lock().unwrap();
        let early: usize = c[2..8].iter().sum();  // L1 firing during the initial hot phase
        let late: usize = c[30..36].iter().sum(); // after adaptation has built up
        assert!(early > 0, "L1 should fire during the hot phase, got {early}");
        assert!(late < early, "adaptation should suppress L1 rate over time: early {early} vs late {late}");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p wave-net adaptation_accessor_and_reset determinism_includes_adaptation adaptation_self_limits_rate`
Expected: FAIL to compile — `Network::adaptation` does not exist yet.

- [ ] **Step 3: Implement `adaptation` and extend `reset_state`**

In `src/wave_net/network.rs`, add the adaptation zeroing to `reset_state` (alongside the `potential`/`cooldown` zeroing):

```rust
    pub fn reset_state(&self) {
        for layer in self.layers.iter() {
            let mut g = layer.lock().unwrap();
            g.potential.iter_mut().for_each(|p| *p = 0);
            g.cooldown.iter_mut().for_each(|c| *c = 0);
            g.adapt.iter_mut().for_each(|a| *a = 0);
            g.inbox.clear();
            g.outbox.clear();
        }
        self.wave_id.store(0, Ordering::Relaxed);
    }
```

Add the accessor next to `potential`:

```rust
    pub fn adaptation(&self, layer: usize, local: usize) -> i16 {
        self.layers[layer].lock().unwrap().adapt[local]
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test` and `cargo build`
Expected: PASS — all three new tests pass; existing network tests still pass; build warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/network.rs
git commit -m "feat: reset/introspect adaptation; self-limiting + determinism tests"
```

---

### Task 5: Calibration — retarget for low-baseline ALIF

**Files:**
- Modify: `src/wave_net/calibrate.rs` (test `test_config`, rewrite two tests, add one). No production-code change to `calibrate`/`calibrate_step`.

**Interfaces:**
- Consumes: ALIF dynamics (Task 3), `Network::calibrate` / `measure_layer_rates` (unchanged), `random_l0_input`.
- Produces: no new public API. Confirms calibration converges the baseline so the adaptation-live firing rate lands near target.

- [ ] **Step 1: Retarget the test config to low-baseline ALIF**

In `src/wave_net/calibrate.rs`, change `test_config()`'s `layer` literal to a low baseline with active adaptation (this makes upper layers boot hot, exercising the self-regulation path):

```rust
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 64,
            baseline_init: 8,
            adapt_bump: 24,
            adapt_decay: 5,
        };
```

- [ ] **Step 2: Rewrite the silent-start test and relax the lowering test; add the target test**

In `src/wave_net/calibrate.rs`, replace `calibrate_warms_silent_upper_layers` with `calibrate_settles_upper_layers` (the top now boots hot; calibration should pull it toward target), replace the body of `calibrate_lowers_every_upper_layer` with a move-toward-target assertion, and add `calibrate_hits_target_with_adaptation_live`:

```rust
    #[test]
    fn calibrate_settles_upper_layers() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(0xABC, 8, 20000); // ~30% of L0 driven
        let params = CalibrateParams::default();
        let top = net.layer_count() - 1;
        let target = params.target_permille as f64 / 1000.0;

        net.calibrate(&params, &input);

        let after = net.measure_layer_rates(params.warmup, params.waves, &input)[top];
        assert!(after > 0.0, "top should fire after calibration");
        assert!(after > target / 2.0 && after < target * 2.0, "top rate {after} not near {target}");
    }

    #[test]
    fn calibrate_moves_every_upper_layer_toward_target() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(7, 8, 20000);
        let params = CalibrateParams::default();
        let target = params.target_permille as f64 / 1000.0;

        let before: Vec<f64> = net.measure_layer_rates(params.warmup, params.waves, &input);
        net.calibrate(&params, &input);
        let after: Vec<f64> = net.measure_layer_rates(params.warmup, params.waves, &input);

        for z in 1..net.layer_count() {
            let improved = (after[z] - target).abs() <= (before[z] - target).abs() + 1e-9;
            assert!(improved, "layer {z}: rate moved away from target ({} -> {})", before[z], after[z]);
        }
    }

    #[test]
    fn calibrate_hits_target_with_adaptation_live() {
        let mut net = Network::new(test_config());
        let input = random_l0_input(42, 8, 20000);
        let params = CalibrateParams::default();
        let target = params.target_permille as f64 / 1000.0;

        net.calibrate(&params, &input);

        let rates = net.measure_layer_rates(params.warmup, params.waves, &input);
        for z in 1..net.layer_count() {
            assert!(
                rates[z] > target / 3.0 && rates[z] < target * 3.0,
                "layer {z} self-regulated rate {} not near target {target}",
                rates[z]
            );
        }
    }
```

- [ ] **Step 3: Run the tests to verify they fail (or pass) as expected first**

Run: `cargo test -p wave-net calibrate_settles_upper_layers calibrate_moves_every_upper_layer_toward_target calibrate_hits_target_with_adaptation_live`
Expected: the new tests compile and exercise the retargeted config. If any assertion fails, it indicates the chosen `test_config` ALIF params can't reach the default target — tune `adapt_bump` (lower → higher reachable rate) and/or `baseline_init` in Step 1 and re-run. Do **not** change `calibrate`/`calibrate_step` production code — the algorithm is intended to be unchanged; only the config/params are tuned.

- [ ] **Step 4: Run the whole suite green**

Run: `cargo test` and `cargo build`
Expected: PASS — all calibrate tests pass (including the unchanged `random_input_hits_expected_fraction`, `calibrate_is_deterministic`, `calibrate_preserves_listeners`); build warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/calibrate.rs
git commit -m "test: retarget calibration for low-baseline ALIF"
```

---

### Task 6: Docs — update AGENTS.md silent-start language

**Files:**
- Modify: `AGENTS.md` (§"The engine model", §"Calibration", and the Invariants paragraph)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update the engine-model and silent-start prose**

In `AGENTS.md`, in the "The engine model (how a wave works)" section, update the per-neuron description and step 4, and replace the "Silent start" paragraph.

Change the per-neuron line to mention adaptation:
```markdown
A stack of `L` square layers (`size × size`, `size` a power of two, toroidal wrap). Per neuron:
`i16` potential (rest 0), `u8` cooldown, per-neuron `i16` **baseline** threshold, and an `i16`
**adaptation** state (rest 0). A **wave** advances every layer one step; `wave::process_layer` runs,
per layer:
```

Change step 4 (decide) and add the adaptation-decay step:
```markdown
4. **decide** — fire if `cooldown == 0 && potential >= baseline + adaptation` (the ALIF effective
   threshold, in `i32`); on fire reset potential to 0, reload cooldown, and bump adaptation
6. **leak** — decay the survivors' potential
7. **adapt-decay** — decay every neuron's adaptation geometrically toward rest
```

Replace the **Silent start** paragraph with:
```markdown
**Boots hot, self-regulates.** Baselines initialize low (`baseline_init + jitter`), so neurons fire
readily from the first waves; each neuron's adaptation then rises with its own firing and quenches it,
a local negative-feedback controller that settles the firing rate (spike-frequency adaptation, the
ALIF mechanism). Input is a sparse `Vec<u32>` of L0 local addresses (spike injection), not graded
current. `adapt_bump = 0` recovers plain LIF dynamics.
```

- [ ] **Step 2: Update the Calibration section**

In the "Calibration" section of `AGENTS.md`, adjust the opening sentence so it no longer implies warming from silence:
```markdown
`Network::calibrate(params, input)` tunes per-layer **baselines** until each layer fires near a target
rate on a driven input, **with adaptation live** — bottom-up (each layer tuned once its feeder fires)
then a few **global-refine** passes for the recurrent coupling. The calibration step is symmetric
(raises a too-hot layer's baseline, lowers a too-cold one), so it converges the baseline to the point
where the self-regulated rate matches target. Calibration is **layer-owned** ...
```
(keep the remainder of the paragraph — `shift_threshold`, `calibrate_step`, measurement/listeners — unchanged.)

- [ ] **Step 3: Update the Invariants paragraph**

In the "Architecture map" invariants paragraph, replace the clause about thresholds starting near `i16::MAX`:
```markdown
... per-layer state is struct-of-arrays; weight is `±1`, computed at fire time, never stored;
**baselines init low (`baseline_init + jitter`, clamped to [1, i16::MAX]) so the net boots hot and
self-regulates via per-neuron adaptation**, with calibration tuning the baselines; a `Layer` is a
self-contained, persistable unit ...
```

- [ ] **Step 4: Verify build/tests unaffected and commit**

Run: `cargo test` and `cargo build`
Expected: PASS — docs-only change; everything still green.

```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md silent-start language for ALIF"
```

---

## Self-review

**Spec coverage** (each spec section → task):
- §1 Neuron state (`adapt` field, i16, saturating bump) → Task 2 (field) + Task 3 (saturating bump via `.clamp(0, i16::MAX)`).
- §2 Config (`adapt_bump`, `adapt_decay`, `baseline_init`, validate `adapt_decay >= 1`, demo values) → Task 1.
- §3 Init (low baseline + jitter, clamp floor 1, adapt zeroed) → Task 2.
- §4 Wave step (effective threshold in i32, bump on fire, decay next to leak, ordering) → Task 3.
- §5 Calibration reconciliation (no code change; measurement with adaptation live; test churn) → Task 5.
- §6 Network plumbing (`reset_state` zeros adapt, `adaptation()` accessor, determinism) → Task 4.
- §7 Docs (AGENTS.md silent-start; module docs) → module docs in Tasks 2 & 3, AGENTS.md in Task 6.
- Testing section: `adapt_decays_toward_zero`→Task 3 `adaptation_decays_each_wave`; `fire_bumps_adaptation_and_raises_effective_threshold`→Task 3 `fire_bumps_adaptation` + `high_adaptation_blocks_firing`; `adaptation_self_limits_rate`→Task 4; `bump_zero_is_plain_lif`→Task 3 `bump_zero_leaves_adaptation_at_rest` (plus all existing wave tests run at bump 0 as the regression); `determinism_includes_adaptation`→Task 4; `calibrate_hits_target_with_adaptation_live`→Task 5. Existing-test changes: neurons (Task 2, kept the max-init test as a high-baseline parametric guard and added the low-baseline test — a small, coverage-increasing deviation from the spec's rename), calibrate (Task 5).

**Placeholder scan:** none — every step has concrete code and exact commands.

**Type consistency:** `adapt: Vec<i16>`, `adapt_bump: i16`, `adapt_decay: u8`, `baseline_init: i16` used identically across config/neurons/wave/network. `adaptation(layer, local) -> i16` matches the `potential` accessor shape. `calibrate`/`calibrate_step`/`shift_threshold` signatures untouched.

**Note on Task 5 tuning:** the calibrate assertions depend on the ALIF params in `test_config` being able to reach the default target rate. Step 3 explicitly allows tuning `baseline_init`/`adapt_bump` (config only, never the calibration algorithm) if an assertion can't be met.
