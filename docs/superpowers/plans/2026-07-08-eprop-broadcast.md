# Broadcast-error Alignment Implementation Plan (Spec 3, V2b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the non-spiking potential readout learn by replacing the global scalar reward with a per-output broadcast error (softmax error × fixed random feedback weights), and show it beats a frozen control.

**Architecture:** Extend `src/bench/eprop.rs`. Two pure helpers (`softmax`, `feedback_weight`), an `EpropConfig { broadcast, softmax_temp }` mode, and a broadcast branch in `train`'s update that computes `Δθⱼ = −lr·Lⱼ·eⱼ` with `Lⱼ = Σᵢ B(j,i)·(targetᵢ − pᵢ)`. Reuses V2a's readout layer, scoring, eligibility, and shadow. No engine change.

**Tech Stack:** Rust edition 2024, std only, no deps. Inline `#[cfg(test)]` tests. `f64` in bench only.

## Global Constraints

- **Std only**; **no `unsafe`**; **warning-free** build.
- **Determinism** — pure function of `(seed, config, params)`; single-threaded.
- Engine untouched. `f64` bench-only. Tests inline, test-first.
- **One commit per task**, conventional commits. **NEVER** a `Co-Authored-By` trailer. **NEVER** push.
- On branch `feat/eprop-broadcast`.
- Verify each task with `cargo test` + warning-free `cargo build` before committing.
- V1 + V2a `eprop` tests must stay green.

## File structure

| File | Change |
|---|---|
| `src/bench/eprop.rs` | `softmax` + `feedback_weight` helpers; `EpropConfig { broadcast, softmax_temp }`; broadcast branch in `train`; tests |

---

### Task 1: `softmax` + `feedback_weight` helpers

**Files:**
- Modify: `src/bench/eprop.rs`

**Interfaces:**
- Produces (module-private): `softmax(&[i64], f64) -> Vec<f64>`; `feedback_weight(seed: u64, global_id: u32, output: usize) -> f64` (`±1`).

- [ ] **Step 1: Write the failing tests**

Add to `src/bench/eprop.rs` `mod tests`:
```rust
    #[test]
    fn softmax_is_a_distribution() {
        let p = softmax(&[3, 1, 1], 1.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9, "sums to 1");
        assert!(p[0] > p[1] && (p[1] - p[2]).abs() < 1e-12, "monotone in the input");
        // overflow-safe on huge scores:
        let q = softmax(&[1_000_000, 0], 100.0);
        assert!(q[0].is_finite() && q[1].is_finite() && q[0] > 0.99, "no overflow; peaks on the max");
    }

    #[test]
    fn feedback_weights_are_deterministic_and_signed() {
        let w = |g, o| feedback_weight(7, g, o);
        assert_eq!(w(10, 0), w(10, 0), "deterministic");
        assert!([-1.0, 1.0].contains(&w(10, 0)) && [-1.0, 1.0].contains(&w(3, 1)), "values are +/-1");
        // not all the same across neurons/outputs (decorrelated):
        let vals: Vec<f64> = (0..20).map(|g| w(g, 0)).collect();
        assert!(vals.iter().any(|&v| v > 0.0) && vals.iter().any(|&v| v < 0.0), "both signs occur");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::eprop::tests::softmax_is_a_distribution`
Expected: FAIL to compile — `softmax` / `feedback_weight` not defined.

- [ ] **Step 3: Implement the helpers**

Add to `src/bench/eprop.rs` (non-test; the `key`/`mix` imports already exist):
```rust
const P_FEEDBACK: u64 = 43; // fixed random feedback weights (broadcast alignment)

/// Softmax over `scores / temp`, with subtract-max for overflow safety (scores are large potential sums).
fn softmax(scores: &[i64], temp: f64) -> Vec<f64> {
    let max = *scores.iter().max().unwrap_or(&0) as f64;
    let t = temp.max(1e-9);
    let exps: Vec<f64> = scores.iter().map(|&s| ((s as f64 - max) / t).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-300);
    exps.iter().map(|&e| e / sum).collect()
}

/// Fixed random feedback weight `±1` for (neuron `global_id`, output `output`) — deterministic,
/// hash-derived, stored-free. Feedback *alignment*: random and fixed, not the forward readout weights.
fn feedback_weight(seed: u64, global_id: u32, output: usize) -> f64 {
    if mix(key(seed, global_id, output as i32, 0, P_FEEDBACK)) & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test bench::eprop` and `cargo build`
Expected: both new tests pass; warning-free (used by tests now, and by `train` in Task 2).

- [ ] **Step 5: Do not commit yet** — proceed to Task 2 (the helpers become used in non-test `train` there; commit together to stay warning-free).

---

### Task 2: Broadcast update path in `train`

**Files:**
- Modify: `src/bench/eprop.rs`

**Interfaces:**
- Consumes: Task 1 helpers; V2a scoring/eligibility/shadow.
- Produces: `EpropConfig { broadcast: bool, softmax_temp: f64 }`; `train` uses the broadcast update when `broadcast` is set.

- [ ] **Step 1: Write the failing tests**

Add to `src/bench/eprop.rs` `mod tests`:
```rust
    fn broadcast_cfg() -> EpropConfig {
        let mut cfg = EpropConfig::demo();
        cfg.readout = true;
        cfg.broadcast = true;
        cfg
    }

    #[test]
    fn eprop_broadcast_is_deterministic() {
        let cfg = broadcast_cfg();
        assert_eq!(train(&cfg, 0.1).accuracy_permille, train(&cfg, 0.1).accuracy_permille);
    }

    #[test]
    fn eprop_broadcast_learns_and_beats_frozen() {
        let cfg = broadcast_cfg();
        let learn = train(&cfg, 0.1);
        let frozen = train(&cfg, 0.0);
        eprintln!("v2b broadcast learn  {:?}", learn.accuracy_permille);
        eprintln!("v2b broadcast frozen {:?}", frozen.accuracy_permille);
        let ll = late_mean(&learn.accuracy_permille);
        let lf = late_mean(&frozen.accuracy_permille);
        let chance = 1000 / cfg.k as u64;
        assert!(ll > chance + 80, "broadcast learning {ll} should be above chance {chance}");
        assert!(ll > lf + 150, "broadcast learning {ll} should beat frozen {lf}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::eprop::tests::eprop_broadcast_is_deterministic`
Expected: FAIL to compile — `EpropConfig` has no `broadcast` / `softmax_temp`.

- [ ] **Step 3: Add config fields and the broadcast update branch**

Add fields to `EpropConfig` (after `readout`), and set them in `demo()`:
```rust
    pub readout: bool,
    pub broadcast: bool,     // V2b: per-output broadcast-error credit instead of global reward
    pub softmax_temp: f64,   // temperature for the readout-score softmax
}
```
```rust
            readout: false,
            broadcast: false,
            softmax_temp: 100.0,
        }
```

In `train`, replace the update block. Currently:
```rust
        if lr != 0.0 {
            for (zi, layer_e) in elig.iter().enumerate() {
                for (i, &e) in layer_e.iter().enumerate() {
                    shadow[zi][i] += -lr * signal * e as f64;
                }
            }
            write_thresholds(&net, &shadow);
        }
```
with:
```rust
        if lr != 0.0 {
            if cfg.broadcast {
                // Per-output broadcast error: softmax the scores, err_i = target_i - p_i (target =
                // one-hot(class)), and project it to each internal neuron via fixed random weights.
                let ls = (cfg.size * cfg.size) as usize;
                let p = softmax(&outs, cfg.softmax_temp);
                let err: Vec<f64> = (0..cfg.k).map(|i| (if i == class { 1.0 } else { 0.0 }) - p[i]).collect();
                for (zi, layer_e) in elig.iter().enumerate() {
                    let z = zi + 1; // eligibility index zi ↔ engine layer z = zi + 1
                    for (i, &e) in layer_e.iter().enumerate() {
                        if e == 0 {
                            continue;
                        }
                        let gid = (z * ls + i) as u32;
                        let l_j: f64 = (0..cfg.k).map(|o| feedback_weight(cfg.seed, gid, o) * err[o]).sum();
                        shadow[zi][i] += -lr * l_j * e as f64;
                    }
                }
            } else {
                for (zi, layer_e) in elig.iter().enumerate() {
                    for (i, &e) in layer_e.iter().enumerate() {
                        shadow[zi][i] += -lr * signal * e as f64;
                    }
                }
            }
            write_thresholds(&net, &shadow);
        }
```
(`signal` is still computed by `rt.step` each trial — used only in the global branch; harmless in broadcast mode.)

- [ ] **Step 4: Run, then TUNE against the printed curves (do not fudge)**

Run: `cargo test bench::eprop::tests::eprop_broadcast_learns_and_beats_frozen -- --nocapture`
Read `v2b broadcast learn` / `frozen`. Expected: broadcast learning rises above chance and beats frozen.
Tune **only** `EpropConfig::demo()` knobs and the test `lr` (never the rule):
- `lr` (the signal is now O(1)·spike-count, so a *larger* `lr` than V2a — start ~0.1, sweep 0.03–0.5).
- `softmax_temp` (too small → saturated `p`, zero error; too large → flat `p`, weak error; start ~100, try 30–300).
- **The update sign** — if learning goes *down*, the feedback-alignment sign is inverted: flip to `+lr`.
- `trials`, `block`, cue separability as before.

**Honesty gate:** feedback-alignment with spike-count eligibility (can't wake silent neurons) may still be
weak. If, after reasonable tuning (`lr`, `softmax_temp`, sign), V2b can't beat frozen, **stop and report** —
that is the finding (points to symmetric feedback or potential-based eligibility). Do not weaken the
assertion or fake a curve. Whether V2b matches/beats V1 (~770) is itself a reported result.

- [ ] **Step 5: Full suite + warning-free**

Run: `cargo test` and `cargo build`
Expected: all pass (V1 + V2a included); warning-free.

- [ ] **Step 6: Commit Tasks 1 + 2 together**

```bash
git add src/bench/eprop.rs
git commit -m "feat: broadcast-error alignment — per-output error trains the potential readout"
```

---

## Self-review

**Spec coverage:**
- Softmax error + hash-derived `±1` feedback weights → Task 1. Per-neuron `Lⱼ = Σ B·err`, `Δθ = −lr·Lⱼ·eⱼ`,
  internal-only (readout `e=0` skipped) → Task 2. `err = target − p`, no `R̄` → Task 2.
- `EpropConfig { broadcast, softmax_temp }`; branch on `broadcast`; V1/V2a paths intact → Task 2.
- Success: learns + beats frozen; V1/V2a/V2b printed; determinism → Task 2. Helper units → Task 1.
- No engine change; `f64` bench-only; determinism → throughout. Global id `z·ls + i` matches the engine.

**Placeholder scan:** none — concrete code and commands throughout.

**Type consistency:** `softmax(&[i64], f64) -> Vec<f64>`, `feedback_weight(u64, u32, usize) -> f64`,
`outs: Vec<i64>` (from V2a) feeds `softmax`, `EpropConfig` fields set in `demo()` and read in `train`. Global
id `(zi+1)·ls + i` uses the engine's `layer·size² + local` convention.

**Note on Task 2:** the update **sign** is the subtle bit — verified empirically in Step 4 (flip if learning
descends); and the honesty gate governs a possible null.
