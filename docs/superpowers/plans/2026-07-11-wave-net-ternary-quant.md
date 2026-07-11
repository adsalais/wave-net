# Ternary (BitNet) weight-quant mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a ternary (BitNet-style, pure ±1/0, per-row γ) weight-quantization mode to `wave_net` and an int8-vs-ternary A/B benchmark, to test whether low-precision weights still train.

**Architecture:** A quantization *mode* on `Layer` (`weight_quant`, default `Int8`), flipped via `Network::set_weight_quant`. The only hot-path change is `eprop_update_synaptic`, which becomes per-touched-row requantize (byte-identical for int8, ternary-capable). `Config`, `eprop_update`, and `rsnn.rs` are untouched.

**Tech Stack:** Rust, edition 2024, std-only, `cargo test`.

## Global Constraints

- Rust edition 2024; **std-only** in `src/`; **no `unsafe`**; **warning-free `cargo build`**.
- **Determinism** — every result a pure function of `(seed, config, input)`; per-row γ is a fixed-order sum.
- Untouched: `Config`, `eprop_update` (factored path), `rsnn.rs`, `wave_state_machine`.
- TDD; **one commit per task**; conventional commits; **NO `Co-Authored-By`**; never push. Branch `feat/wave-net-ternary-quant`.
- Benchmarks: sweep every axis + several seeds (worst+mean); read the top spiking layer; report density/σ/per-layer-spiking/accuracy; compare at the duration **peak** (over-training collapse is documented).
- Spec: `docs/superpowers/specs/2026-07-11-wave-net-ternary-quant-design.md`.

---

### Task 1: `WeightQuant` + `Layer.requantize_row`

**Files:**
- Modify: `src/wave_net/neurons.rs` (enum, `Layer` field, `Layer::new` default, `requantize_row`, test)

**Interfaces:**
- Produces: `pub enum WeightQuant { Int8, Ternary }`; `Layer.weight_quant: WeightQuant`; `Layer::requantize_row(&mut self, i: usize)`.

- [ ] **Step 1: Add the enum, field, and method**

In `src/wave_net/neurons.rs`, add the enum just above `pub struct Layer`:

```rust
/// Weight quantizer for the shadow→weight requantize step. `Int8`: per-weight round/clamp to [-127,127]
/// (the default). `Ternary`: BitNet-style {−1,0,+1} with a **per-row** (per source neuron) absmean γ that
/// sets which weights prune to 0; delivered magnitude stays ±1 (pure ternary, no delivery scale).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WeightQuant {
    Int8,
    Ternary,
}
```

Add the field to `struct Layer` (after `pub out_shadow: Vec<f32>,`):

```rust
    pub weight_quant: WeightQuant, // shadow→weight quantizer (default Int8)
```

Set it in `Layer::new`'s returned struct literal (after `out_shadow,`):

```rust
            weight_quant: WeightQuant::Int8,
```

Add the method inside `impl Layer` (after `shift_threshold`):

```rust
    /// Requantise source neuron `i`'s row (`out_{weights,shadow}[i*total_slots .. +total_slots]`) from the
    /// shadow, per `weight_quant`. Int8: per-weight round/clamp. Ternary: per-row absmean γ sets zeros
    /// (|shadow| < 0.5γ → 0), delivery ±1. No-op for a no-outgoing layer (`total_slots == 0`).
    pub fn requantize_row(&mut self, i: usize) {
        let ts = self.total_slots;
        if ts == 0 {
            return;
        }
        let base = i * ts;
        match self.weight_quant {
            WeightQuant::Int8 => {
                for s in 0..ts {
                    self.out_weights[base + s] = self.out_shadow[base + s].round().clamp(-127.0, 127.0) as i8;
                }
            }
            WeightQuant::Ternary => {
                let mut sum = 0.0f32;
                for s in 0..ts {
                    sum += self.out_shadow[base + s].abs();
                }
                let gamma = sum / ts as f32;
                for s in 0..ts {
                    self.out_weights[base + s] = if gamma <= 0.0 {
                        0
                    } else {
                        (self.out_shadow[base + s] / gamma).round().clamp(-1.0, 1.0) as i8
                    };
                }
            }
        }
    }
```

- [ ] **Step 2: Write the failing test**

Find or add a `#[cfg(test)] mod tests` at the bottom of `src/wave_net/neurons.rs`. Add:

```rust
    #[test]
    fn requantize_row_int8_and_ternary() {
        use crate::wave_net::config::LayerConfig;
        use crate::wave_net::synapse::TopologyLevel;
        let cfg = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 0, count: 4 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 6,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let mut layer = Layer::new(&cfg, 1, 0, 2); // size 2 → ls 4, total_slots 4
        assert_eq!(layer.total_slots, 4);
        assert_eq!(layer.weight_quant, WeightQuant::Int8);
        // Int8: per-weight round/clamp
        layer.out_shadow[0..4].copy_from_slice(&[3.7, -50.0, 0.4, 200.0]);
        layer.requantize_row(0);
        assert_eq!(&layer.out_weights[0..4], &[4i8, -50, 0, 127]);
        // Ternary: γ = mean|shadow| = (2+2+0.1+0.1)/4 = 1.05; 2/1.05→1, 0.1/1.05→0
        layer.weight_quant = WeightQuant::Ternary;
        layer.out_shadow[0..4].copy_from_slice(&[2.0, 2.0, 0.1, 0.1]);
        layer.requantize_row(0);
        assert_eq!(&layer.out_weights[0..4], &[1i8, 1, 0, 0]);
    }
```

(If a `mod tests` already exists, add the test there and ensure `use super::*;` is present so `Layer`/`WeightQuant` resolve.)

- [ ] **Step 3: Run test**

Run: `cargo test requantize_row_int8_and_ternary`
Expected: PASS.

- [ ] **Step 4: Warning-free build**

Run: `cargo build 2>&1 | grep -ci warning`
Expected: `0`.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/neurons.rs
git commit -m "feat(wave_net): WeightQuant enum + Layer::requantize_row (int8/ternary per-row)"
```

---

### Task 2: `Network::set_weight_quant`

**Files:**
- Modify: `src/wave_net/network.rs` (method + test)

**Interfaces:**
- Consumes: `Layer.weight_quant`, `Layer::requantize_row`, `WeightQuant` (Task 1).
- Produces: `Network::set_weight_quant(&mut self, q: WeightQuant)`.

- [ ] **Step 1: Add the method**

At the top of `src/wave_net/network.rs`, extend the neurons import:

```rust
use crate::wave_net::neurons::{Layer, WeightQuant, ADAPT_SHIFT};
```

Add inside `impl Network` (near the other `pub fn`s):

```rust
    /// Set the shadow→weight quantizer on every layer and requantise once. On a fresh ±1 net → `Ternary`
    /// this is a no-op (per-row γ = 1 → ±1); on a trained net it re-derives `out_weights` under the new mode.
    pub fn set_weight_quant(&mut self, q: WeightQuant) {
        for layer in self.layers.iter_mut() {
            layer.weight_quant = q;
            if layer.total_slots == 0 {
                continue;
            }
            let rows = layer.out_shadow.len() / layer.total_slots;
            for i in 0..rows {
                layer.requantize_row(i);
            }
        }
    }
```

- [ ] **Step 2: Write the failing test**

In `src/wave_net/network.rs`'s `#[cfg(test)] mod tests`, add:

```rust
    #[test]
    fn set_weight_quant_flips_mode_and_is_noop_on_fresh_net() {
        use crate::wave_net::config::{Config, LayerConfig};
        use crate::wave_net::neurons::WeightQuant;
        use crate::wave_net::synapse::TopologyLevel;
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 6,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let mut net = Network::new(Config { seed: 5, size: 4, layers: vec![lc; 2] });
        let before = net.with_layer(0, |l| l.out_weights.clone()); // fresh ±1
        net.set_weight_quant(WeightQuant::Ternary);
        let after = net.with_layer(0, |l| l.out_weights.clone());
        assert_eq!(before, after, "fresh ±1 net stays ±1 under ternary (γ=1)");
        assert_eq!(net.with_layer(0, |l| l.weight_quant), WeightQuant::Ternary);
    }
```

- [ ] **Step 3: Run test**

Run: `cargo test set_weight_quant_flips_mode_and_is_noop_on_fresh_net`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/wave_net/network.rs
git commit -m "feat(wave_net): Network::set_weight_quant (flip layers + requantize)"
```

---

### Task 3: `eprop_update_synaptic` per-touched-row requantize

**Files:**
- Modify: `src/wave_net/eprop.rs` (`eprop_update_synaptic` only)

**Interfaces:**
- Consumes: `Layer::requantize_row` (Task 1).
- Produces: `eprop_update_synaptic` requantizes per touched row; respects `Layer.weight_quant`.

- [ ] **Step 1: Refactor the update body**

In `src/wave_net/eprop.rs`, in `eprop_update_synaptic`, replace the `with_layer_mut` closure body (the `for i in 0..ls { … }` update loop **and** the trailing `for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) { … }` full pass) with:

```rust
        self.with_layer_mut(source_z, |lz| {
            for i in 0..ls {
                let sg = (base + i) as u32;
                let mut touched = false;
                for kk in 0..count {
                    let e = elig[i * count + kk];
                    if e == 0.0 {
                        continue;
                    }
                    touched = true;
                    let j = target_of(seed, sg, i as u32, level, kk as u32, radius, size) as usize;
                    lz.out_shadow[i * total_slots + slot_base + kk] += -lr * signal[j] * e;
                }
                if touched {
                    lz.requantize_row(i);
                }
            }
        });
```

Do NOT touch `eprop_update` (the factored path, the other requantize site) — it stays a full pass.

- [ ] **Step 2: Write the ternary-invariant test**

In `src/wave_net/eprop.rs`'s `mod tests`, add:

```rust
    #[test]
    fn eprop_update_synaptic_ternary_keeps_weights_ternary() {
        use crate::wave_net::neurons::WeightQuant;
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 6,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let mut net = Network::new(Config { seed: 9, size: 8, layers: vec![lc; 2] });
        net.set_weight_quant(WeightQuant::Ternary);
        let ls = 64;
        // large mixed eligibility + signal → shadows move a lot; weights must stay in {−1,0,+1}
        let elig: Vec<f32> = (0..ls * 4).map(|k| if k % 3 == 0 { 5.0 } else { 0.2 }).collect();
        let signal = vec![0.5f32; ls];
        net.eprop_update_synaptic(0, 0, &elig, &signal, 0.5);
        let ok = net.with_layer(0, |l| l.out_weights.iter().all(|&w| w == -1 || w == 0 || w == 1));
        assert!(ok, "ternary mode must keep every out_weight in {{-1,0,1}}");
    }
```

- [ ] **Step 3: Run the new test AND the full multilayer_dfa suite (byte-identity guard)**

Run: `cargo test eprop_update_synaptic_ternary_keeps_weights_ternary`
Expected: PASS.

Run: `cargo test multilayer_dfa && cargo test eprop_update_synaptic_applies_expected_delta`
Expected: all PASS — the per-row refactor is byte-identical for int8, so the existing `multilayer_dfa` unit tests (determinism, learns-2class, recurrent-edge, step) and the primitive-delta test stay green.

- [ ] **Step 4: Warning-free build**

Run: `cargo build 2>&1 | grep -ci warning`
Expected: `0`.

- [ ] **Step 5: Commit**

```bash
git add src/wave_net/eprop.rs
git commit -m "feat(wave_net): eprop_update_synaptic per-touched-row requantize (ternary-capable, int8 byte-identical)"
```

---

### Task 4: `harness::weight_sparsity` + "ternary trains" test

**Files:**
- Modify: `src/bench/multilayer_dfa.rs` (harness helper + a unit test)

**Interfaces:**
- Consumes: `make_ff`, `ff_cfg`, `train_and_eval`, `single_task` (harness); `Network::set_weight_quant`, `WeightQuant`.
- Produces: `harness::weight_sparsity(net: &Network) -> f64`.

- [ ] **Step 1: Add the sparsity helper**

In `src/bench/multilayer_dfa.rs`, inside `pub(crate) mod harness`, add:

```rust
    /// Fraction of stored weights that are 0, over the computational layers `1..L` (BitNet sparsity).
    pub(crate) fn weight_sparsity(net: &Network) -> f64 {
        let l = net.layer_count();
        let (mut zeros, mut total) = (0usize, 0usize);
        for z in 1..l {
            net.with_layer(z, |lz| {
                zeros += lz.out_weights.iter().filter(|&&w| w == 0).count();
                total += lz.out_weights.len();
            });
        }
        if total == 0 { 0.0 } else { zeros as f64 / total as f64 }
    }
```

- [ ] **Step 2: Write the failing test**

In `src/bench/multilayer_dfa.rs`'s `#[cfg(test)] mod tests`, add:

```rust
    #[test]
    fn ternary_net_trains_above_chance() {
        use crate::wave_net::neurons::WeightQuant;
        // Ternary weights must still train a separable 2-class task above chance (generous fan-in).
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 20, 6);
        net.set_weight_quant(WeightQuant::Ternary);
        let mut cfg = ff_cfg(400, 0.004, 0.0);
        cfg.size = 16;
        let acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, single_task);
        // ternary keeps weights in {−1,0,+1} the whole run
        let ternary = net.with_layer(1, |l| l.out_weights.iter().all(|&w| w == -1 || w == 0 || w == 1));
        assert!(ternary, "weights must stay ternary during training");
        assert!(acc > 600, "ternary net should train above chance: {acc}");
    }
```

- [ ] **Step 3: Run test**

Run: `cargo test ternary_net_trains_above_chance`
Expected: PASS. If `acc <= 600`, ternary may not train this config — first raise `up_count`/`trials`; if it still can't clear chance on a generous FF config, that is a *finding* (record it) — do not silently lower the bar. Re-run `cargo test multilayer_dfa` to confirm nothing else regressed.

- [ ] **Step 4: Commit**

```bash
git add src/bench/multilayer_dfa.rs
git commit -m "test(bench): harness weight_sparsity + ternary-trains-above-chance"
```

---

### Task 5: A/B benchmark file

**Files:**
- Create: `src/bench/multilayer_dfa_bitnet_bench.rs`
- Modify: `src/bench/mod.rs` (`pub mod multilayer_dfa_bitnet_bench;`)

**Interfaces:**
- Consumes: `harness::*` (`make_ff`, `ff_cfg`, `train_and_eval_curve`, `xor_task`, `single_task`, `per_layer_rates`, `sigma_ratio`, `weight_sparsity`), `Network::set_weight_quant`, `WeightQuant`.

- [ ] **Step 1: Create the benchmark file**

Create `src/bench/multilayer_dfa_bitnet_bench.rs`:

```rust
//! Int8-vs-ternary (BitNet) A/B benchmarks for `multilayer_dfa` — `#[ignore]`d, run manually in `--release`.
//! Same net/task/seeds trained under Int8 vs Ternary (per-row γ, pure ±1/0). Per the benchmark convention:
//! sweep fan-in × duration × seeds, read the top spiking layer, report σ / per-layer spiking / accuracy, and
//! (for ternary) the weight-sparsity. Compare at the PEAK of the duration curve (the rate_reg over-training
//! collapse is documented). Question: does per-row pure-ternary reach the int8 baseline, and at what sparsity?

#[cfg(test)]
mod tests {
    use crate::bench::multilayer_dfa::harness::*;
    use crate::wave_net::neurons::WeightQuant;

    const SEEDS: [u64; 3] = [0xE9_0B_0A17, 0x1234_5678, 0xDEAD_BEEF];

    fn fmt_accs(a: &[u64]) -> String {
        a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("/")
    }
    fn mean_curve(c: &[Vec<u64>]) -> Vec<u64> {
        (0..c[0].len()).map(|k| c.iter().map(|r| r[k]).sum::<u64>() / c.len() as u64).collect()
    }
    fn worst_curve(c: &[Vec<u64>]) -> Vec<u64> {
        (0..c[0].len()).map(|k| c.iter().map(|r| r[k]).min().unwrap()).collect()
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_ff_xor() {
        // FF temporal XOR (4 layers, size 32), Int8 vs Ternary, fan-in × duration × seeds. Reports each
        // quantizer's mean/worst acc curve; for ternary also the final weight-sparsity (% pruned to 0).
        let ckpts = [800usize, 2500];
        eprintln!("== BitNet A/B: FF temporal XOR (4L, size 32, read top; {} seeds) ==", SEEDS.len());
        eprintln!("r/c     int8 mean|worst @{ckpts:?}    ternary mean|worst @{ckpts:?}   tern.sparsity%");
        for &(ur, uc) in &[(4u32, 32u32), (4, 64)] {
            let run = |ternary: bool| -> (Vec<Vec<u64>>, f64) {
                let (mut curves, mut sparsity0) = (Vec::new(), 0.0);
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_ff(s, 32, 4, uc, ur, 20, 6);
                    if ternary {
                        net.set_weight_quant(WeightQuant::Ternary);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.4);
                    cfg.size = 32;
                    cfg.delay = 8;
                    cfg.holdout = 300;
                    let accs = train_and_eval_curve(&mut net, &entries, s, s, &cfg, xor_task, &ckpts);
                    if si == 0 {
                        sparsity0 = weight_sparsity(&net);
                    }
                    curves.push(accs);
                }
                (curves, sparsity0)
            };
            let (i8c, _) = run(false);
            let (tc, tsp) = run(true);
            eprintln!(
                "{ur}/{uc:<3}   {:>9} | {:<9}   {:>9} | {:<9}   {:.1}",
                fmt_accs(&mean_curve(&i8c)), fmt_accs(&worst_curve(&i8c)),
                fmt_accs(&mean_curve(&tc)), fmt_accs(&worst_curve(&tc)), tsp * 100.0
            );
        }
    }
}
```

Add to `src/bench/mod.rs`:

```rust
pub mod multilayer_dfa_bitnet_bench;
```

- [ ] **Step 2: Verify it compiles + warning-free (it is `#[ignore]`d, not run in CI)**

Run: `cargo test --no-run 2>&1 | grep -ci warning`
Expected: `0`.

Run: `cargo test bitnet_ff_xor -- --list 2>&1 | grep bitnet_ff_xor`
Expected: the test is listed (confirms it compiles and is registered).

- [ ] **Step 3: Smoke-run once in release to confirm the A/B report**

Run: `cargo test --release bitnet_ff_xor -- --ignored --nocapture 2>&1 | tail -8`
Expected: prints the int8-vs-ternary table with a sparsity column, exit 0. (Interpretation is the experiment's job, not a pass/fail gate.)

- [ ] **Step 4: Commit**

```bash
git add src/bench/multilayer_dfa_bitnet_bench.rs src/bench/mod.rs
git commit -m "test(bench): int8-vs-ternary (BitNet) A/B benchmark"
```

---

## Self-Review

**Spec coverage:** WeightQuant + per-row requantize_row (spec A) → Task 1; set_weight_quant (spec C) → Task 2; eprop_update_synaptic per-touched-row, int8 byte-identical, eprop_update untouched (spec B) → Task 3; weight_sparsity + ternary-trains (spec E tests) → Task 4; A/B benchmark, new file, multi-seed + duration + sparsity (spec D) → Task 5. Config/eprop_update/rsnn untouched → held across all tasks. ✓

**Placeholder scan:** No TBD/TODO; full code in every step; the one empirical bar (Task 4 `acc > 600`) has explicit tuning guidance (raise fan-in/trials; a genuine failure is a finding, not a lowered bar).

**Type consistency:** `WeightQuant{Int8,Ternary}`, `Layer.weight_quant`, `Layer::requantize_row(usize)`, `Network::set_weight_quant(WeightQuant)`, `harness::weight_sparsity(&Network)->f64` — used identically across Tasks 1–5. `requantize_row` reads `total_slots`/`weight_quant`, writes `out_weights` from `out_shadow` — all existing `pub` `Layer` fields. ✓

## Notes for the executor

- Byte-identity (Task 3) rests on: unchanged rows keep already-correct `out_weights` (their shadow didn't move; init `±1` == `round(±1)`), and per-row int8 requantize == full-pass int8 (per-weight, order-independent). If any existing `multilayer_dfa`/`eprop` test regresses, the refactor broke identity — do not "fix" the test; re-examine the refactor.
- Determinism: keep the per-row γ sum in slot order; do not parallelize.
- The benchmark is single-checkpoint-agnostic — read results at the **peak** of the curve (rate_reg over-training).
