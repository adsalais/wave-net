# Multi-layer credit via DFA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train every feed-forward layer of `wave_net` (input projection through top) via e-prop — deeper layers credited by Direct Feedback Alignment — and show it beats single-layer training on a deep net.

**Architecture:** Generalize `train_eprop` to loop over trained layers with a per-layer learning signal (symmetric for the top, random-DFA for deeper). All in `bench::rsnn`; no engine change (`elig_pre`/`elig_post` per layer already exist).

**Tech Stack:** Rust edition 2024, std only. `f32` in the bench. Inline `#[cfg(test)]` tests.

## Global Constraints

- Std only in the engine; **no `unsafe`**; **warning-free**.
- Determinism: pure function of `(seed, task_seed, config)`; single-threaded.
- **One commit per task**, conventional commits, **no `Co-Authored-By`**, **never push**.
- On branch `feat/rsnn-multilayer`. Verify each task with `cargo test` + warning-free `cargo build`.
- Whole existing suite (incl. `wave_state_machine`) stays green; the single-layer path is unchanged.

## File structure

| File | Change |
|---|---|
| `src/bench/rsnn.rs` | `RsnnConfig.multi_layer`; `dfa_weight` (+ `P_DFA`); generalize `train_eprop`'s hidden update to loop over trained layers; tests |

---

### Task 1: DFA feedback + multi-layer `train_eprop`

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Produces `RsnnConfig.multi_layer: bool`; `dfa_weight(seed, neuron_global, class) -> f32`; `train_eprop` trains all layers when `multi_layer` is set.

- [ ] **Step 1: Config flag + `dfa_weight`**

Add to `RsnnConfig` (after `rec_init`): `pub multi_layer: bool,` and set `multi_layer: false,` in `demo()`.
Add (non-test), near `target_of`:
```rust
const P_DFA: u64 = 61; // fixed random Direct-Feedback-Alignment weights

/// Fixed random ±1 DFA feedback weight for (target neuron `neuron_global`, output class `class`) —
/// deterministic, hash-derived, stored-free. Broadcasts the output error to a deep layer.
fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
    if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
}
```

- [ ] **Step 2: Write the unit tests (they compile against the new items)**

Add to `rsnn.rs` `mod tests`:
```rust
    #[test]
    fn dfa_weights_are_deterministic_and_signed() {
        let f = |g, c| dfa_weight(7, g, c);
        assert_eq!(f(10, 0), f(10, 0));
        assert!([-1.0, 1.0].contains(&f(10, 0)) && [-1.0, 1.0].contains(&f(3, 1)));
        let vals: Vec<f32> = (0..20).map(|g| f(g, 0)).collect();
        assert!(vals.iter().any(|&v| v > 0.0) && vals.iter().any(|&v| v < 0.0), "both signs occur");
    }

    #[test]
    fn multilayer_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.layers = 4;
        cfg.multi_layer = true;
        cfg.trials = 600;
        assert_eq!(train_eprop(&cfg), train_eprop(&cfg));
    }
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test bench::rsnn::tests::dfa_weights_are_deterministic_and_signed`
Expected: FAIL to compile — `dfa_weight` / `multi_layer` not yet used by `train_eprop` (dfa_weight unused warning is fine here; resolved in Step 4).

- [ ] **Step 4: Generalize the hidden update in `train_eprop`**

Replace the single-layer `if cfg.hidden_lr != 0.0 { … }` block in `train_eprop` with a loop over trained
layers, each with its own learning signal (symmetric for the top, DFA for deeper):
```rust
        if cfg.hidden_lr != 0.0 {
            let trained: Vec<usize> = if cfg.multi_layer { (0..top).collect() } else { vec![top - 1] };
            for z in trained {
                let tgt = z + 1;
                // learning signal L_j for each target-layer neuron j: symmetric readout feedback for the
                // top layer, random DFA feedback for deeper layers.
                let l_sig: Vec<f32> = (0..ls)
                    .map(|j| {
                        (0..cfg.k)
                            .map(|c| {
                                let b = if tgt == top { w[c][j] } else { dfa_weight(cfg.seed, (tgt * ls + j) as u32, c) };
                                b * err[c]
                            })
                            .sum()
                    })
                    .collect();
                let pre = net.with_layer_mut(z, |x| x.elig_pre.clone());
                let psi = net.with_layer_mut(tgt, |x| x.elig_post.clone());
                net.with_layer_mut(z, |lz| {
                    for i in 0..ls {
                        let pre_i = pre[i] as f32;
                        if pre_i == 0.0 {
                            continue;
                        }
                        let sg = (z * ls + i) as u32;
                        for kk in 0..up {
                            let j = target_of(cfg.seed, sg, i as u32, 1, kk as u32, cfg.up_radius, cfg.size) as usize;
                            lz.out_shadow[i * up + kk] += -cfg.hidden_lr * l_sig[j] * pre_i * psi[j] as f32;
                        }
                    }
                    for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                        *wq = s.round().clamp(-127.0, 127.0) as i8;
                    }
                });
            }
        }
```
(`multi_layer = false` reduces the loop to `{top-1}` — byte-identical to the old single-layer path, so the
existing `eprop_hidden_learns_reliably` test still passes.)

- [ ] **Step 5: Run + commit**

Run: `cargo test` and `cargo build`
Expected: new unit tests pass; **`eprop_hidden_learns_reliably` (single-layer) still passes unchanged**;
warning-free.
`git commit -m "feat: multi-layer e-prop credit via DFA feedback (train all feed-forward layers)"`

---

### Task 2: Deep-net headline — multi-layer vs single-layer

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Consumes `train_eprop` (single/multi). Produces `multilayer_beats_single_layer_at_depth`.

- [ ] **Step 1: Write the headline test (single vs multi at depth)**

```rust
    #[test]
    fn multilayer_beats_single_layer_at_depth() {
        // Separation erodes with depth: training only the last layer should weaken on a deep net, while
        // training every layer (multi-layer DFA credit) keeps it reliable.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let depth = 5usize;
        let mut worst_single = 1000u64;
        let mut worst_multi = 1000u64;
        for &s in &seeds {
            let mut single = RsnnConfig::demo();
            single.seed = s;
            single.task_seed = s;
            single.layers = depth;
            single.trials = 1500;
            let mut multi = single.clone();
            multi.multi_layer = true;
            let sa = train_eprop(&single);
            let ma = train_eprop(&multi);
            eprintln!("depth {depth} seed {s:#x}  single {sa}  multi {ma}");
            worst_single = worst_single.min(sa);
            worst_multi = worst_multi.min(ma);
        }
        eprintln!("worst single {worst_single}  worst multi {worst_multi}");
        assert!(worst_multi > 640, "multi-layer learns reliably at depth (worst {worst_multi})");
        assert!(worst_multi >= worst_single, "multi-layer is at least as good as single-layer at depth");
    }
```

- [ ] **Step 2: Run + read (do not fudge)**

Run: `cargo test bench::rsnn::tests::multilayer_beats_single_layer_at_depth -- --nocapture`
Read single vs multi per seed. Expected: at depth 5, multi-layer is reliable (worst > 640) and ≥ single;
ideally single-layer is visibly weaker (the intended contrast). Tune only `hidden_lr`, `trials`, `depth`,
`up_count`/`up_radius` if needed — never the rule.

**Honesty gate:** if single-layer already matches multi-layer at depth 5, **strengthen the contrast** (deeper
net / harder config) or **report that one trainable layer suffices even deep** — a real finding. If DFA is
*unreliable* (multi-seed spread wide, worst < 640), report the spread; don't cherry-pick a seed. Keep the
`worst_multi >= worst_single` bar honest.

- [ ] **Step 3: Full suite + commit**

Run: `cargo test` and `cargo build` (all green, warning-free).
`git commit -m "feat: multi-layer credit beats single-layer at depth (deep-net e-prop)"`

---

## Self-review

**Spec coverage:** train every layer via per-layer learning signal — symmetric top + DFA deeper (Task 1);
factored per-layer eligibility `elig_pre[z]·elig_post[z+1]` reused (Task 1); hash-derived `dfa_weight`
(`P_DFA`) (Task 1); deep-net single-vs-multi, held-out + multi-seed + honesty gate (Task 2). Input projection
`L0→L1` trained in the `(0..top)` loop. No engine change. Deferred items untouched.

**Placeholder scan:** none — full code and commands throughout.

**Type consistency:** `dfa_weight(u64, u32, usize) -> f32`; `RsnnConfig.multi_layer: bool` set in `demo()`;
`train_eprop` loop uses `top`/`ls`/`up`/`err`/`w` already in scope; `target_of(.., 1, ..)` (level+1) and
neuron-global id `tgt*ls + j` consistent with the engine's `layer·size² + local`. Single-layer path
(`multi_layer = false → {top-1}`) is byte-identical to today, so `eprop_hidden_learns_reliably` regresses
cleanly.
