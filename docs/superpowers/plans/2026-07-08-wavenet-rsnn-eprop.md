# wave_net RSNN via e-prop on stored int8 weights — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `wave_net` a trained SNN: synapse weights become **stored int8** (addresses stay procedural), a **trained readout** learns reliably on the reservoir (LSM), and **e-prop** trains the feed-forward hidden weights — evaluated held-out + multi-seed.

**Architecture:** Engine changes are `wave_net`-only: stored `out_weights`/`out_shadow` per layer, per-neuron eligibility accumulators, `generate_into` reads stored weights. Training lives in a new `src/bench/rsnn.rs` targeting `wave_net`. `wave_state_machine` + its pinned bench are untouched.

**Tech Stack:** Rust edition 2024, std only. `f32` shadow allowed in the bench/training path. Inline `#[cfg(test)]` tests.

## Global Constraints

- Std only in the engine; **no `unsafe`**; **warning-free**.
- Engine stays integer + deterministic; the f32 shadow lives in the training path.
- Determinism: pure function of `(seed, task_seed, config)`; single-threaded.
- **One commit per task**, conventional commits, **no `Co-Authored-By`**, **never push**.
- On branch `feat/rsnn-eprop`. Verify each task with `cargo test` + warning-free `cargo build`.
- The whole existing suite (incl. `wave_state_machine` + bench) stays green.

## File structure

| File | Change |
|---|---|
| `src/wave_net/neurons.rs` | `Layer`: `total_slots`, `out_weights: Vec<i8>`, `out_shadow: Vec<f32>`, `elig_pre/elig_post: Vec<i32>`; init in `Layer::new` |
| `src/wave_net/synapse.rs` | `generate_into` reads stored weights (drop sign/`inhibitor_ratio`/`random_weights` branch); update its unit tests |
| `src/wave_net/wave.rs` | pass stored weights to `generate_into`; accumulate `elig_pre/elig_post` in decide |
| `src/wave_net/network.rs` | zero eligibility in `reset_state` |
| `src/bench/mod.rs` | `pub mod rsnn;` |
| `src/bench/rsnn.rs` | new — trained readout (LSM) + e-prop hidden training + held-out/multi-seed harness |

---

### Task 1: Substrate — stored int8 weights (behaviour-identical)

**Files:** Modify `src/wave_net/{neurons,synapse,wave}.rs`.

**Interfaces:** Produces `Layer.{total_slots, out_weights, out_shadow}` (public); `generate_into(seed, source_global, src_local, size, topology, weights: &[i8], total_slots, groups)`.

- [ ] **Step 1: Add fields to `Layer` and initialise them**

In `src/wave_net/neurons.rs`, add to the struct (after `readout`):
```rust
    pub readout: bool,
    pub total_slots: usize,   // Σ topology counts — the stride for out_weights[local·total_slots + slot]
    pub out_weights: Vec<i8>, // stored plastic weight per (source local, slot); addresses stay procedural
    pub out_shadow: Vec<f32>, // higher-precision training accumulator, quantised into out_weights
```
Add `P_TARGET` to the import: `use crate::wave_net::synapse::{key, map_range, mix, Synapse, TopologyLevel, P_TARGET, P_THRESHOLD};`.
In `Layer::new`, before the returned struct, build the weights (sign = the old procedural inhibitory rule, magnitude 1 → behaviour-identical):
```rust
        let total_slots: usize = cfg.topology.iter().map(|e| e.count as usize).sum();
        let mut out_weights = vec![0i8; ls * total_slots];
        for local in 0..ls {
            let source_global = (base + local) as u32;
            let mut slot = 0usize;
            for entry in &cfg.topology {
                for k in 0..entry.count {
                    let h = mix(key(seed, source_global, entry.level, k, P_TARGET));
                    out_weights[local * total_slots + slot] =
                        if ((h & 0xFFFF) as u32) < cfg.inhibitor_ratio { -1 } else { 1 };
                    slot += 1;
                }
            }
        }
        let out_shadow: Vec<f32> = out_weights.iter().map(|&w| w as f32).collect();
```
Add the three fields to the returned `Layer { .. }`:
```rust
            readout: false,
            total_slots,
            out_weights,
            out_shadow,
```

- [ ] **Step 2: `generate_into` reads stored weights**

In `src/wave_net/synapse.rs`, replace the signature and body sign/magnitude logic. New signature:
```rust
pub fn generate_into(
    seed: u64,
    source_global: u32,
    src_local: u32,
    size: u32,
    topology: &[TopologyLevel],
    weights: &[i8],
    total_slots: usize,
    groups: &mut [SynapseGroup],
) {
    let (sx, sy) = xy_of(src_local, size);
    let mut slot = 0usize;
    for (entry, group) in topology.iter().zip(groups.iter_mut()) {
        let span = 2 * entry.radius + 1;
        for k in 0..entry.count {
            let h = mix(key(seed, source_global, entry.level, k, P_TARGET));
            let dx = map_range24((h >> 40) as u32, span) as i32 - entry.radius as i32;
            let dy = map_range24(((h >> 16) as u32) & 0x00FF_FFFF, span) as i32 - entry.radius as i32;
            let tx = wrap(sx, dx, size);
            let ty = wrap(sy, dy, size);
            let w = weights[src_local as usize * total_slots + slot] as i16;
            group.synapses.push(Synapse { target: local_of(tx, ty, size), weight: w });
            slot += 1;
        }
    }
}
```
(The `#[cfg(feature = "random_weights")]` branch and the `inhibitor_ratio` param are gone — the stored weight carries the sign. `P_WEIGHT` may be left unused.)

- [ ] **Step 3: Pass stored weights from `wave.rs`**

In `src/wave_net/wave.rs` step 5, update the call:
```rust
        generate_into(
            seed,
            (base + local as usize) as u32,
            local,
            size,
            &layer.topology,
            &layer.out_weights,
            layer.total_slots,
            out,
        );
```

- [ ] **Step 4: Fix `generate_into`'s unit tests + write the roundtrip/drive test**

In `src/wave_net/synapse.rs` tests, update the three `generate_into(..)` calls to pass `&weights, total_slots` (weights all `1`), e.g. add a helper and replace `..., inhibitor_ratio, &mut g)` with `..., &w, tot, &mut g)` where `let tot: usize = t.iter().map(|e| e.count as usize).sum(); let w = vec![1i8; 64 * tot];`. The determinism test's `(x.target, x.weight)` still holds (weights all 1). Add to `src/wave_net/wave.rs` tests:
```rust
    #[test]
    fn stored_weight_sets_delivered_value() {
        // A source neuron's stored out_weight is what its synapse delivers.
        let mut l = low_layer(4, 1, 2, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        l.out_weights[0] = 7; // neuron 0, slot 0 -> radius-0 target is neuron 0 itself
        for c in l.cooldown.iter_mut() { *c = 0; }
        l.potential[0] = 1; // fires (threshold 1)
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
        assert_eq!(fired, vec![0]);
        assert_eq!(out[0].synapses[0].weight, 7, "delivered weight = stored out_weight");
    }
```

- [ ] **Step 5: Run**

Run: `cargo test` and `cargo build`
Expected: **all existing tests still pass** (init to ±1 → behaviour-identical); the new test passes; warning-free. If `P_WEIGHT`/`random_weights` warns in `wave_net`, delete the now-dead `P_WEIGHT` const from `wave_net/synapse.rs` (leave `wave_state_machine` untouched).

- [ ] **Step 6: Commit** — `git commit -m "feat: wave_net stores int8 synapse weights (addresses stay procedural)"`

---

### Task 2: Per-neuron eligibility accumulators

**Files:** Modify `src/wave_net/{neurons,wave,network}.rs`.

**Interfaces:** Produces `Layer.{elig_pre, elig_post}: Vec<i32>`, accumulated per wave, zeroed by `reset_state`.

- [ ] **Step 1: Add the accumulators**

In `neurons.rs` `Layer`, add after `out_shadow`:
```rust
    pub elig_pre: Vec<i32>,  // presynaptic trace: count of this neuron's spikes this trial
    pub elig_post: Vec<i32>, // postsynaptic pseudo-derivative accumulated this trial
```
Init in `Layer::new` returned struct: `elig_pre: vec![0; ls], elig_post: vec![0; ls],`.

- [ ] **Step 2: Accumulate in `process_layer` (decide step), zero in `reset_state`**

In `wave.rs`, inside the decide loop (step 4), after computing `eff` and before/with the fire check, accumulate the pseudo-derivative for every neuron and the spike for firers. Replace the decide loop body:
```rust
    for i in 0..ls {
        let eff = layer.threshold[i] as i32 + (layer.adapt[i] >> ADAPT_SHIFT);
        // pseudo-derivative ψ: a box surrogate — 1 when the potential is within `PSI_BAND` of the
        // effective threshold, else 0. Accumulated over the trial as the post-factor of eligibility.
        const PSI_BAND: i32 = 8;
        if ((layer.potential[i] as i32) - eff).abs() <= PSI_BAND {
            layer.elig_post[i] += 1;
        }
        if layer.cooldown[i] == 0 && (layer.potential[i] as i32) >= eff {
            layer.potential[i] = 0;
            layer.cooldown[i] = layer.cooldown_base;
            let bumped = layer.adapt[i] + ((layer.adapt_bump as i32) << ADAPT_SHIFT);
            layer.adapt[i] = bumped.clamp(0, ADAPT_MAX);
            layer.elig_pre[i] += 1;
            fired.push(i as u32);
        }
    }
```
In `network.rs` `reset_state`, zero them alongside potential/cooldown/adapt — inside its per-layer closure add:
```rust
                for e in layer.elig_pre.iter_mut() { *e = 0; }
                for e in layer.elig_post.iter_mut() { *e = 0; }
```

- [ ] **Step 3: Write the test**

Add to `wave.rs` tests:
```rust
    #[test]
    fn eligibility_accumulates_pre_and_post() {
        let mut l = low_layer(4, 1, 2, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        for c in l.cooldown.iter_mut() { *c = 0; }
        l.potential[0] = 1; // at threshold 1 -> within the ψ band AND fires
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
        assert_eq!(l.elig_pre[0], 1, "a spike bumps the pre-trace");
        assert!(l.elig_post[0] >= 1, "near-threshold bumps the post pseudo-derivative");
    }
```

- [ ] **Step 4: Run + commit**

Run: `cargo test` and `cargo build` (expect all green, warning-free).
`git commit -m "feat: per-neuron e-prop eligibility accumulators (pre-trace + pseudo-derivative)"`

---

### Task 3: Trained readout on the reservoir (reliable LSM) + multi-seed harness

**Files:** Create `src/bench/rsnn.rs`; modify `src/bench/mod.rs`.

**Interfaces:** Produces `bench::rsnn::{RsnnConfig, train_readout}` and the held-out/multi-seed test harness. Consumes `wave_net::network::Network`, `bench::store_recall::{cue_realization, probe_pattern}`, `bench::eprop::pick_class`.

- [ ] **Step 1: Scaffold + the failing headline test**

Add `pub mod rsnn;` to `src/bench/mod.rs`. Create `src/bench/rsnn.rs`:
```rust
//! `rsnn` — training on the LIVE `wave_net` engine. Stage 1: a trained linear readout on the reservoir's
//! top-layer activity (a reliable Liquid State Machine). Stage 2 (Task 4): e-prop on the hidden weights.
//! Evaluated held-out + multi-seed — the bar the threshold-only approach failed.

use crate::bench::eprop::pick_class;
use crate::bench::store_recall::{cue_realization, probe_pattern};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::{Config, LayerConfig};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::TopologyLevel;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct RsnnConfig {
    pub seed: u64,
    pub task_seed: u64,
    pub size: u32,
    pub layers: usize,
    pub k: usize,
    pub present_waves: usize,
    pub delay: usize,
    pub read_waves: usize,
    pub base_q16: u32,
    pub keep_q16: u32,
    pub noise_q16: u32,
    pub probe_q16: u32,
    pub up_count: u32,
    pub up_radius: u32,
    pub trials: usize,
    pub readout_lr: f32,
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl RsnnConfig {
    pub fn demo() -> RsnnConfig {
        let seed = 0xE9_0B_0A17;
        RsnnConfig {
            seed, task_seed: seed, size: 8, layers: 3, k: 2,
            present_waves: 6, delay: 4, read_waves: 6,
            base_q16: 18000, keep_q16: 60000, noise_q16: 1500, probe_q16: 20000,
            up_count: 16, up_radius: 3, trials: 1500, readout_lr: 0.02,
            calib: CalibrateParams { warmup: 16, waves: 48, max_steps: 24, refine_passes: 3, ..CalibrateParams::default() },
            calib_fraction_q16: 20000,
        }
    }
    fn engine_config(&self) -> Config {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32,
            baseline_init: 6, adapt_bump: 20, adapt_decay: 6,
        };
        Config { seed: self.seed, size: self.size, layers: vec![layer; self.layers] }
    }
}

/// Run one trial (reset → cue → delay → probe); return the top computational layer's per-neuron spike
/// counts — the reservoir feature vector the readout reads.
fn top_activity(net: &mut Network, cfg: &RsnnConfig, class: usize, trial: usize) -> Vec<f32> {
    let top = net.layer_count() - 1;
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![0u32; ls]));
    {
        let c = counts.clone();
        net.on_layer(top, Box::new(move |_w, fired: &[u32]| {
            let mut g = c.lock().unwrap();
            for &loc in fired { g[loc as usize] += 1; }
        }));
    }
    net.reset_state();
    for w in 0..cfg.present_waves {
        let sites = cue_realization(cfg.task_seed, cfg.size, class, trial, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
        net.wave(&sites);
    }
    for _ in 0..cfg.delay { net.wave(&[]); }
    let probe = probe_pattern(cfg.task_seed, cfg.size, cfg.probe_q16);
    for _ in 0..cfg.read_waves { net.wave(&probe); }
    net.clear_listeners();
    let g = counts.lock().unwrap();
    g.iter().map(|&x| x as f32).collect()
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum::<f32>().max(1e-30);
    e.iter().map(|v| v / s).collect()
}

/// Train a K×N linear readout (delta rule) on the reservoir; return held-out test accuracy ‰.
pub fn train_readout(cfg: &RsnnConfig) -> u64 {
    let mut net = Network::new(cfg.engine_config());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let n = (cfg.size * cfg.size) as usize;
    let mut w = vec![vec![0f32; n]; cfg.k]; // readout weights (bench-side f32; int8 later)
    for t in 0..cfg.trials {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a = top_activity(&mut net, cfg, class, t);
        let scores: Vec<f32> = (0..cfg.k).map(|c| w[c].iter().zip(&a).map(|(wi, ai)| wi * ai).sum()).collect();
        let p = softmax(&scores);
        for c in 0..cfg.k {
            let err = p[c] - if c == class { 1.0 } else { 0.0 };
            for j in 0..n { w[c][j] -= cfg.readout_lr * err * a[j]; }
        }
    }
    // held-out: frozen readout, disjoint trial indices (unseen cue realisations)
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let class = pick_class(cfg.task_seed, t, cfg.k);
        let a = top_activity(&mut net, cfg, class, t);
        let scores: Vec<f32> = (0..cfg.k).map(|c| w[c].iter().zip(&a).map(|(wi, ai)| wi * ai).sum()).collect();
        let pred = (0..cfg.k).max_by(|&x, &y| scores[x].total_cmp(&scores[y])).unwrap();
        if pred == class { correct += 1; }
    }
    (correct as u64 * 1000) / holdout as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readout_learns_and_generalizes() {
        let test = train_readout(&RsnnConfig::demo());
        eprintln!("readout held-out {test}");
        assert!(test > 650, "trained readout on the reservoir generalizes: {test}");
    }

    #[test]
    fn readout_is_seed_robust() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111];
        let mut worst = 1000u64;
        for &s in &seeds {
            let mut cfg = RsnnConfig::demo();
            cfg.seed = s; cfg.task_seed = s;
            let test = train_readout(&cfg);
            eprintln!("seed {s:#x} held-out {test}");
            worst = worst.min(test);
        }
        assert!(worst > 600, "worst seed still learns (reliable, unlike threshold-only): {worst}");
    }
}
```

- [ ] **Step 2: Run + tune**

Run: `cargo test bench::rsnn -- --nocapture`. Expected: the readout generalizes (>650) and — the whole point — is **seed-robust** (worst seed >600), unlike threshold-only training. Tune only `RsnnConfig::demo()` (`readout_lr`, `trials`) if needed.

**Honesty gate:** if even a *trained readout* isn't seed-robust, stop and report — that would mean the reservoir itself doesn't reliably encode the classes (a deeper problem than the learning rule).

- [ ] **Step 3: Commit** — `git commit -m "feat: rsnn trained readout on wave_net reservoir (reliable, seed-robust LSM)"`

---

### Task 4: e-prop on the hidden feed-forward weights

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Produces `bench::rsnn::train_eprop(cfg) -> u64` (held-out test ‰) — trains the top computational layer's **incoming** weights (the layer-below's `out_weights`) via e-prop, on top of the trained readout.

- [ ] **Step 1: Write the failing test**

Add to `rsnn.rs` tests:
```rust
    #[test]
    fn eprop_hidden_matches_or_beats_readout_and_is_seed_robust() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mut worst = 1000u64;
        for &s in &seeds {
            let mut cfg = RsnnConfig::demo();
            cfg.seed = s; cfg.task_seed = s;
            let test = train_eprop(&cfg);
            eprintln!("seed {s:#x} eprop held-out {test}");
            worst = worst.min(test);
        }
        assert!(worst > 600, "e-prop weight training stays reliable across seeds: {worst}");
    }
```

- [ ] **Step 2: Implement `train_eprop`**

Add to `rsnn.rs`. It mirrors `train_readout` but, each trial, after computing `err`, also updates the top layer's **incoming** hidden weights (stored in the *layer below*, `z_below = top-1`, level+1 slots) via e-prop:
- Read the below-layer activity `pre_i = elig_pre[i]` (source `i` in `z_below`) and the top-layer `psi_j = elig_post[j]` (target `j` in `top`) from the engine (`with_layer_mut` to read the vectors after the trial).
- Learning signal `L_j = Σ_c w[c][j] · err_c` (symmetric feedback from the readout weights).
- For each stored weight `out_weights[z_below][i·total_slots + slot]` whose procedural address targets `j`: `out_shadow += -hidden_lr · L_j · pre_i · psi_j`; then quantise the whole `out_shadow` → `out_weights` (`round().clamp(-127,127)`).
- To map (i, slot) → target `j`, regenerate the address with the same hash used in `generate_into` (a small local helper `target_of(task hash inputs)` reusing `synapse::{key,mix,map_range24,xy_of,wrap,local_of}` — expose these `pub(crate)` from `wave_net::synapse` if not already `pub`).

Concretely add a config field `hidden_lr: f32` (demo `0.01`) and a `RsnnConfig::demo` value; keep the readout update from Task 3; add the hidden update block + a `quantise_shadow(net, z)` helper using `with_layer_mut`. (The below-layer is `z_below = net.layer_count() - 2`; its level+1 slots target the top layer.)

- [ ] **Step 3: Run + tune (do not fudge)**

Run: `cargo test bench::rsnn::tests::eprop_hidden_matches_or_beats_readout_and_is_seed_robust -- --nocapture`.
Tune only `hidden_lr`, `PSI_BAND` (Task 2), `trials`. Expected: e-prop training stays seed-robust (worst >600) — ideally matching/beating the fixed-reservoir readout, proving weight-e-prop learns *reliably* (the headline result).

**Honesty gate:** if e-prop hidden training is *not* seed-robust while the readout (Task 3) is, report that — it localizes the difficulty to hidden-weight credit assignment (the surrogate/eligibility), not the substrate. A reliable readout + an honest e-prop result is still a real, publishable outcome. Do **not** revert to prequential/single-seed.

- [ ] **Step 4: Full suite + commit**

Run: `cargo test` and `cargo build` (all green, warning-free).
`git commit -m "feat: e-prop trains wave_net hidden feed-forward weights (held-out, multi-seed)"`

---

## Self-review

**Spec coverage:** stored int8 weights + procedural addresses (Task 1); per-neuron factored eligibility (Task 2); trained readout + symmetric feedback source (Task 3); e-prop hidden weight update (Task 4); held-out + multi-seed eval + honesty gate (Tasks 3–4); engine-only-`wave_net`, training in new `rsnn.rs`, `wave_state_machine` untouched (throughout). Recurrence explicitly deferred.

**Placeholder scan:** Task 4 Step 2 describes the update at prose+formula level (it is the tuning-heavy, address-regeneration part); everything else is concrete code. The formula, indices (`z_below`, slot→target regen), and learning signal are fully specified.

**Type consistency:** `out_weights: Vec<i8>`, `out_shadow: Vec<f32>`, `elig_pre/elig_post: Vec<i32>`, `total_slots: usize` used consistently; `generate_into(.., &[i8], usize, ..)` matches its call in `wave.rs`; readout `w: Vec<Vec<f32>>` and `train_readout/train_eprop -> u64` consistent. `pick_class`/`cue_realization`/`probe_pattern` reused as engine-agnostic helpers.

**Sequencing note:** Task 3 (reliable readout) is the de-risked win banked before the harder Task 4 (hidden e-prop). Both are judged by the same held-out + multi-seed bar the threshold approach failed.
