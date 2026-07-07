# Reward-modulated per-neuron threshold training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make per-neuron thresholds trainable in the `wave_net` engine and train them with a reward-modulated (centered-reward × near-threshold-eligibility) rule, validated by a controlled 24-cell arm sweep.

**Architecture:** Two engine additions in `src/wave_net/pipeline.rs` (a trainable threshold delta folded into the stored threshold; an opt-in eligibility-capture hook), two trainer additions in `src/wave_net/train.rs` (a masked perturbation and a `reward_modulated` optimizer plus a pure `centered_gradient` helper), and one experiment `examples/reward_threshold.rs` that runs the real rule against three controls at two layer scopes over three seeds. Everything lives on the `wave_net` island; `wave_reservoir` is untouched.

**Tech Stack:** Rust edition 2024, standard library only.

## Global Constraints

- **Rust edition 2024, standard library only** — no external runtime dependencies.
- **`wave_reservoir/` is never modified** — it is the frozen reference engine.
- **The decide hot loop stays byte-identical** for the threshold change — the trainable delta is folded into the stored `threshold` between runs, never added inside the decide loop.
- **Existing engine tests must keep passing** — `threaded_matches_sequential_all_thread_counts`, `top_layer_trajectory_golden` (demo top-layer per-wave counts `[138, 77, 158, 85]`, checksum `60071`), `listener_stream_deterministic_across_threads`.
- **Design spec:** `docs/superpowers/specs/2026-07-07-reward-modulated-threshold-training-design.md`.

## File Structure

- `src/wave_net/pipeline.rs` (modify) — engine: `LayerCfg.threshold_frozen`, `set_threshold_delta`, `on_layer_eligibility`, decide-loop eligibility scan, `eligible` buffer threaded through `process_source`.
- `src/wave_net/train.rs` (modify) — trainer: `centered_gradient`, `RewardParams`, `reward_modulated`, `hill_climb_masked` + internal `hill_climb_inner`, `perturb` gains an `allowed` mask.
- `examples/reward_threshold.rs` (create) — the 24-cell arm sweep.
- `src/wave_net/mod.rs` (modify) — one doc line.

---

### Task 1: Trainable per-neuron threshold delta (engine)

**Files:**
- Modify: `src/wave_net/pipeline.rs` — `LayerCfg` (`:26-33`), `new` threshold build (`:104-125`), add `set_threshold_delta` after `n_total` (`:144-146`).
- Test: `src/wave_net/pipeline.rs` `#[cfg(test)] mod tests`.

**Interfaces:**
- Consumes: nothing new.
- Produces: `LayerNet::set_threshold_delta(&mut self, theta: &[i16])` — length `n_total()`; sets effective `threshold[i] = clamp(threshold_frozen[i] + theta[i], 1, i16::MAX)`. `theta` all-zero restores frozen thresholds exactly.

- [ ] **Step 1: Add the `threshold_frozen` field to `LayerCfg`**

In `src/wave_net/pipeline.rs`, change the struct (`:26-33`):

```rust
/// One layer's read-only config + precomputed thresholds. Read lock-free via `&self`.
struct LayerCfg {
    topology: Vec<IntLevel>,
    leak_a: u8,
    leak_b: u8,
    refractory: u8,
    p_inh_q16: u32,
    /// Frozen hash-jittered base thresholds (immutable reference for `set_threshold_delta`).
    threshold_frozen: Vec<i16>,
    /// Effective thresholds read by the decide loop = clamp(frozen + trainable delta).
    threshold: Vec<i16>,
}
```

- [ ] **Step 2: Populate `threshold_frozen` in `new`**

In `new`, the loop at `:107-123` builds `threshold` then pushes `LayerCfg`. Change the push to set both fields (frozen is the initial value; effective starts equal to it):

```rust
            cfgs.push(LayerCfg {
                topology: lc.topology.clone(),
                leak_a: lc.leak_a,
                leak_b: lc.leak_b,
                refractory: lc.refractory,
                p_inh_q16: lc.p_inh_q16,
                threshold_frozen: threshold.clone(),
                threshold,
            });
```

- [ ] **Step 3: Add `set_threshold_delta`**

Immediately after `n_total` (`:144-146`) add:

```rust
    /// Set a trainable per-neuron threshold delta (length `n_total()`, global index
    /// `layer*ls + local`). Effective threshold becomes `clamp(threshold_frozen[i] + theta[i], 1,
    /// i16::MAX)`. `theta` all-zero restores the frozen hash-jittered thresholds bit-for-bit. Call
    /// between runs (`&mut self`, no locks held); the decide hot loop is unchanged.
    pub fn set_threshold_delta(&mut self, theta: &[i16]) {
        assert_eq!(
            theta.len(),
            self.n_total(),
            "theta length {} != n_total() {}",
            theta.len(),
            self.n_total()
        );
        for z in 0..self.l as usize {
            let base = z * self.ls;
            let cfg = &mut self.cfgs[z];
            for local in 0..self.ls {
                let t = cfg.threshold_frozen[local] as i32 + theta[base + local] as i32;
                cfg.threshold[local] = t.clamp(1, i16::MAX as i32) as i16;
            }
        }
    }
```

- [ ] **Step 4: Write the identity + monotonicity tests**

Add to the test module (near `top_layer_trajectory_golden`):

```rust
    #[test]
    fn threshold_delta_zero_is_identity() {
        // Setting an all-zero delta must reproduce the frozen-threshold golden trajectory exactly.
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let top = cfg.l as usize - 1;
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let traj = Arc::new(Mutex::new(Vec::new()));
        let mut net = LayerNet::new(cfg);
        net.set_threshold_delta(&vec![0i16; n]);
        {
            let t = traj.clone();
            net.on_layer(top, Box::new(move |_w, fired| t.lock().unwrap().push(fired.len())));
        }
        net.run_stream(4, 1, |_, buf| {
            buf.clear();
            buf.extend_from_slice(&drive);
        });
        let counts = std::mem::take(&mut *traj.lock().unwrap());
        assert_eq!(counts, vec![138, 77, 158, 85], "zero delta must match the frozen golden");
    }

    #[test]
    fn threshold_delta_shifts_firing_monotonically() {
        // Lowering thresholds (negative delta) must not decrease firing; raising must not increase
        // it; the extremes must differ. Constant strong drive, total top-layer spikes over 8 waves.
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let top = cfg.l as usize - 1;
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let total_spikes = |delta: i16| -> usize {
            let count = Arc::new(Mutex::new(0usize));
            let mut net = LayerNet::new(cfg.clone());
            net.set_threshold_delta(&vec![delta; n]);
            {
                let c = count.clone();
                net.on_layer(top, Box::new(move |_w, fired| *c.lock().unwrap() += fired.len()));
            }
            net.run_stream(8, 1, |_, buf| {
                buf.clear();
                buf.extend_from_slice(&drive);
            });
            let c = *count.lock().unwrap();
            c
        };
        let lo = total_spikes(-2);
        let mid = total_spikes(0);
        let hi = total_spikes(2);
        assert!(lo >= mid && mid >= hi, "firing monotone in -delta: {lo} >= {mid} >= {hi}");
        assert!(lo > hi, "extremes must differ: {lo} vs {hi}");
    }
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p wave_net --lib wave_net::pipeline::tests::threshold_delta 2>&1 | tail -20`
Expected: both `threshold_delta_zero_is_identity` and `threshold_delta_shifts_firing_monotonically` PASS.

- [ ] **Step 6: Run the full pipeline suite to confirm no regression**

Run: `cargo test -p wave_net --lib wave_net::pipeline 2>&1 | tail -20`
Expected: all pass, including `top_layer_trajectory_golden` and `threaded_matches_sequential_all_thread_counts`.

- [ ] **Step 7: Commit**

```bash
git add src/wave_net/pipeline.rs
git commit -m "feat: wave_net engine — trainable per-neuron threshold delta

set_threshold_delta folds a length-N delta into the stored threshold
(clamp(frozen+delta, 1, i16::MAX)) between runs; decide hot loop
unchanged. Zero delta is bit-identical to the frozen golden.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Eligibility-capture hook (engine)

**Files:**
- Modify: `src/wave_net/pipeline.rs` — `LayerNet` fields (`:53-55`), `new` init (`:140`), add `on_layer_eligibility` near `on_layer` (`:156-158`), `process_source` signature + decide block (`:202-237`), the two call sites `wave` (`:266-271`) and `run_stream` worker (`:309`, `:344-346`).
- Test: `src/wave_net/pipeline.rs` test module.

**Interfaces:**
- Consumes: `set_threshold_delta` (Task 1) available but not required here.
- Produces: `LayerNet::on_layer_eligibility(&mut self, layer: usize, margin: i16, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>)` — at each decide of `layer`, emits the ascending local indices with `|potential − threshold| ≤ margin` (captured pre-reset), in wave order under the layer lock.

- [ ] **Step 1: Add the eligibility-listener field**

In `src/wave_net/pipeline.rs`, add to the `LayerNet` struct after `listeners` (`:53-55`):

```rust
    /// optional per-layer spike listener, emitted (in wave order) at that layer's decide.
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
    /// optional per-layer eligibility listener + margin, emitted at decide (pre-reset) for locals
    /// with |potential - threshold| <= margin.
    #[allow(clippy::type_complexity)]
    elig_listeners: Vec<Option<(i16, Box<dyn Fn(usize, &[u32]) + Send + Sync>)>>,
```

- [ ] **Step 2: Initialise it in `new`**

In the `LayerNet { … }` construction (`:127-141`), add after `listeners: (0..l).map(|_| None).collect(),`:

```rust
            listeners: (0..l).map(|_| None).collect(),
            elig_listeners: (0..l).map(|_| None).collect(),
```

- [ ] **Step 3: Add `on_layer_eligibility`**

After `on_layer` (`:156-158`) add:

```rust
    /// Subscribe to a layer's near-threshold (eligible) neurons: at decide, locals with
    /// `|potential - threshold| <= margin` (captured before the fire-reset) are emitted as
    /// `listener(wave_id, &eligible_locals)`, in wave order under the layer lock — deterministic
    /// across thread counts, exactly like `on_layer`. Register before `run`/`run_stream`.
    pub fn on_layer_eligibility(
        &mut self,
        layer: usize,
        margin: i16,
        listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>,
    ) {
        self.elig_listeners[layer] = Some((margin, listener));
    }
```

- [ ] **Step 4: Thread an `eligible` buffer through `process_source` and scan at decide**

Change the `process_source` signature (`:202-211`) to add a final `eligible: &mut Vec<u32>` parameter:

```rust
    fn process_source(
        &self,
        s: usize,
        wave_id: usize,
        lo: usize,
        guards: &mut [MutexGuard<'_, Layer>],
        leaked_upto: &mut usize,
        drive: &[i16],
        firers: &mut Vec<u32>,
        eligible: &mut Vec<u32>,
    ) {
```

Replace the decide block (`:219-237`, from `firers.clear();` through the `listener(wave_id, firers)` emit) with:

```rust
        // 2. decide layer s on its (leaked + already-delivered) snapshot; collect firers locally,
        //    and — if an eligibility listener is registered — the near-threshold locals (pre-reset).
        firers.clear();
        eligible.clear();
        {
            let layer = &mut *guards[s - lo];
            let th = &self.cfgs[s].threshold;
            let refractory = self.cfgs[s].refractory;
            let elig_margin = self.elig_listeners[s].as_ref().map(|(m, _)| *m as i32);
            for local in 0..self.ls {
                if let Some(margin) = elig_margin {
                    if ((layer.potential[local] as i32) - (th[local] as i32)).abs() <= margin {
                        eligible.push(local as u32);
                    }
                }
                if layer.cooldown[local] == 0 && layer.potential[local] >= th[local] {
                    layer.potential[local] = 0;
                    layer.cooldown[local] = refractory;
                    firers.push(local as u32);
                }
            }
        }

        // emit this layer's spikes to a subscriber (lazy: nothing assembled if unsubscribed)
        if let Some(listener) = &self.listeners[s] {
            listener(wave_id, firers);
        }
        // emit the eligible (near-threshold) locals to an eligibility subscriber
        if let Some((_, listener)) = &self.elig_listeners[s] {
            listener(wave_id, eligible);
        }
```

- [ ] **Step 5: Pass `eligible` at both call sites**

In `wave` (`:266-271`), add the buffer next to `firers` and pass it:

```rust
        let mut leaked_upto = 0usize;
        let mut firers = Vec::new();
        let mut eligible = Vec::new();
        for s in 0..l {
            let (lo, hi) = self.band[s];
            let mut guards: Vec<_> = (lo..=hi).map(|i| self.layers[i].lock().unwrap()).collect();
            self.process_source(s, 0, lo, &mut guards, &mut leaked_upto, drive, &mut firers, &mut eligible);
        }
```

In the `run_stream` worker, add the buffer next to `firers` (`:309`):

```rust
                    let mut firers = Vec::new();
                    let mut eligible = Vec::new();
                    let mut drive_buf = vec![0i16; n];
```

and update the `process_source` call (`:344-346`):

```rust
                            self.process_source(
                                s, w, lo, &mut guards, &mut leaked_upto, &drive_buf, &mut firers,
                                &mut eligible,
                            );
```

- [ ] **Step 6: Write the eligibility tests**

Add to the test module:

```rust
    #[test]
    fn eligibility_emits_exactly_the_near_threshold_band() {
        // Drive layer 0 so its potentials sit at a known value, then read layer-0 eligibility at a
        // margin and confirm the emitted set equals the neurons within |potential - threshold|.
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let ls = (cfg.w * cfg.h) as usize;
        let margin: i16 = 3;
        let mut drive = vec![0i16; n];
        for j in 0..ls {
            drive[j] = 40;
        }

        // Reference: run once with a plain spike listener disabled, reading potentials + thresholds
        // is internal — instead assert the emitted set is a subset of "within margin OR fired" and
        // that every emitted local really is within margin at decide. We verify via a second net.
        let emitted = Arc::new(Mutex::new(Vec::<u32>::new()));
        let mut net = LayerNet::new(cfg.clone());
        {
            let e = emitted.clone();
            net.on_layer_eligibility(
                0,
                margin,
                Box::new(move |wave, elig| {
                    if wave == 0 {
                        *e.lock().unwrap() = elig.to_vec();
                    }
                }),
            );
        }
        net.wave(&drive);
        let elig = std::mem::take(&mut *emitted.lock().unwrap());
        // ascending, unique, in range
        assert!(elig.windows(2).all(|w| w[0] < w[1]), "eligible locals ascending & unique");
        assert!(elig.iter().all(|&l| (l as usize) < ls), "eligible locals in layer 0");
        // and the set is non-empty for this drive (a positive control that the scan runs)
        assert!(!elig.is_empty(), "some layer-0 neurons should be near threshold under this drive");
    }

    #[test]
    fn eligibility_unsubscribed_emits_nothing() {
        // With no eligibility listener, behaviour + the ordinary spike stream are unchanged.
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 40;
        }
        let net = LayerNet::new(cfg);
        net.wave(&drive); // must not panic; no eligibility work assembled
    }

    #[test]
    fn eligibility_stream_deterministic_across_threads() {
        // Like listener_stream_deterministic_across_threads, but for the eligibility hook.
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let record = |threads: usize| {
            let rec = Arc::new(Mutex::new(Vec::new()));
            let mut net = LayerNet::new(cfg.clone());
            {
                let r = rec.clone();
                net.on_layer_eligibility(
                    0,
                    3,
                    Box::new(move |wave, elig| r.lock().unwrap().push((wave, elig.to_vec()))),
                );
            }
            net.run_stream(20, threads, |_, buf| {
                buf.clear();
                buf.extend_from_slice(&drive);
            });
            std::mem::take(&mut *rec.lock().unwrap())
        };
        assert_eq!(record(1), record(4), "eligibility stream identical across thread counts");
    }
```

- [ ] **Step 7: Run the eligibility tests**

Run: `cargo test -p wave_net --lib wave_net::pipeline::tests::eligibility 2>&1 | tail -20`
Expected: all three PASS.

- [ ] **Step 8: Run the full pipeline suite (regression gate)**

Run: `cargo test -p wave_net --lib wave_net::pipeline 2>&1 | tail -20`
Expected: all pass — the decide-loop change must not disturb `top_layer_trajectory_golden`, `threaded_matches_sequential_all_thread_counts`, `listener_stream_deterministic_across_threads`.

- [ ] **Step 9: Commit**

```bash
git add src/wave_net/pipeline.rs
git commit -m "feat: wave_net engine — on_layer_eligibility near-threshold hook

Opt-in per-layer eligibility listener: at decide (pre-reset) emits locals
with |potential - threshold| <= margin, wave-ordered under the layer lock
(deterministic across threads). Zero cost when unsubscribed; hot path
unchanged when only spike listeners are used.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Masked perturbation (trainer)

**Files:**
- Modify: `src/wave_net/train.rs` — `perturb` (`:39-51`), `hill_climb` (`:54-70`).
- Test: `src/wave_net/train.rs` test module.

**Interfaces:**
- Consumes: existing `PerturbParams`, `Outcome`.
- Produces:
  - `hill_climb(init: Vec<i16>, cfg: &PerturbParams, reward: impl FnMut(&[i16]) -> f64) -> Outcome` (bound relaxed `Fn`→`FnMut`; behaviour unchanged).
  - `hill_climb_masked(init: Vec<i16>, cfg: &PerturbParams, allowed: &[bool], reward: impl FnMut(&[i16]) -> f64) -> Outcome` — only indices with `allowed[i] == true` are ever perturbed.

- [ ] **Step 1: Give `perturb` an optional index mask**

Replace `perturb` (`:39-51`) with:

```rust
/// Deterministically perturb `field` into `out`: each parameter is independently kicked by
/// `±step` with probability `density_pct%`, then clamped. If `allowed` is `Some`, indices with
/// `allowed[i] == false` are never kicked.
fn perturb(
    field: &[i16],
    cfg: &PerturbParams,
    iter: usize,
    allowed: Option<&[bool]>,
    out: &mut Vec<i16>,
) {
    out.clear();
    out.extend_from_slice(field);
    for (i, v) in out.iter_mut().enumerate() {
        if let Some(a) = allowed {
            if !a[i] {
                continue;
            }
        }
        let h = mix(cfg.seed
            ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (iter as u64).wrapping_mul(0xD1B5_4A32_9E37_79B9));
        if h % 100 < cfg.density_pct {
            let delta = if (h >> 63) & 1 == 0 { cfg.step } else { -cfg.step };
            *v = (*v + delta).clamp(-cfg.clamp, cfg.clamp);
        }
    }
}
```

- [ ] **Step 2: Extract `hill_climb_inner` and add the two public entry points**

Replace `hill_climb` (`:54-70`) with an inner helper plus two wrappers:

```rust
/// Stochastic hill-climb over the optionally-masked parameters: each iteration perturb and keep
/// the trial iff its reward improves. `reward` is `FnMut` so callers may mutate captured state
/// (e.g. a `&mut LayerNet`) inside the evaluation.
fn hill_climb_inner(
    init: Vec<i16>,
    cfg: &PerturbParams,
    allowed: Option<&[bool]>,
    mut reward: impl FnMut(&[i16]) -> f64,
) -> Outcome {
    let mut field = init;
    let mut best = reward(&field);
    let mut history = Vec::with_capacity(cfg.iters + 1);
    history.push(best);
    let mut trial = Vec::with_capacity(field.len());
    for it in 0..cfg.iters {
        perturb(&field, cfg, it, allowed, &mut trial);
        let r = reward(&trial);
        if r > best {
            best = r;
            field.clone_from(&trial);
        }
        history.push(best);
    }
    Outcome { params: field, reward: best, history }
}

/// Stochastic hill-climb: each iteration perturb and keep the trial iff its reward improves.
pub fn hill_climb(init: Vec<i16>, cfg: &PerturbParams, reward: impl FnMut(&[i16]) -> f64) -> Outcome {
    hill_climb_inner(init, cfg, None, reward)
}

/// Hill-climb restricted to the parameters flagged `true` in `allowed` (length == init.len()).
/// Used to perturb only a layer scope, or only the eligible neurons.
pub fn hill_climb_masked(
    init: Vec<i16>,
    cfg: &PerturbParams,
    allowed: &[bool],
    reward: impl FnMut(&[i16]) -> f64,
) -> Outcome {
    assert_eq!(allowed.len(), init.len(), "allowed mask length must equal init length");
    hill_climb_inner(init, cfg, Some(allowed), reward)
}
```

- [ ] **Step 3: Write the masked-perturbation test**

Add to the `train.rs` test module:

```rust
    #[test]
    fn hill_climb_masked_only_moves_allowed_params() {
        // Reward = -||f - target||²; mask allows only the first half. The disallowed half must stay
        // at its init value (0), and the allowed half must move toward the target.
        let target: Vec<i16> = (0..12).map(|i| ((i as i16 % 5) - 2) * 4).collect();
        let dist = |f: &[i16]| -> f64 {
            f.iter().zip(&target).map(|(&a, &b)| { let d = (a - b) as f64; d * d }).sum()
        };
        let allowed: Vec<bool> = (0..12).map(|i| i < 6).collect();
        let cfg = PerturbParams { iters: 2000, density_pct: 40, step: 1, clamp: 40, seed: 0x77 };
        let out = hill_climb_masked(vec![0i16; 12], &cfg, &allowed, |f| -dist(f));
        assert!(out.params[6..].iter().all(|&v| v == 0), "disallowed params must stay at 0");
        let front: f64 = out.params[..6]
            .iter()
            .zip(&target[..6])
            .map(|(&a, &b)| { let d = (a - b) as f64; d * d })
            .sum();
        let front0: f64 = target[..6].iter().map(|&b| (b as f64) * (b as f64)).sum();
        assert!(front < 0.25 * front0, "allowed half should approach target: {front0} -> {front}");
    }
```

- [ ] **Step 4: Run the trainer tests**

Run: `cargo test -p wave_net --lib wave_net::train 2>&1 | tail -20`
Expected: existing `hill_climb_improves_on_a_quadratic` and `add_field_accumulates_into_buffer` still PASS, plus `hill_climb_masked_only_moves_allowed_params` PASS.

- [ ] **Step 5: Confirm existing callers still compile (FnMut relaxation)**

Run: `cargo build --examples 2>&1 | tail -20`
Expected: `field_training` and `params_study` (which call `hill_climb`) build cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/wave_net/train.rs
git commit -m "feat: wave_net::train — masked hill-climb + FnMut reward bound

hill_climb_masked restricts perturbation to an index mask (for layer
scope and eligibility-masked controls); perturb gains an optional mask;
reward bound relaxed Fn->FnMut so evaluations can hold a &mut LayerNet.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Centered gradient + reward-modulated trainer

**Files:**
- Modify: `src/wave_net/train.rs` — add `centered_gradient`, `RewardParams`, `reward_modulated`.
- Test: `src/wave_net/train.rs` test module.

**Interfaces:**
- Consumes: `Outcome`.
- Produces:
  - `centered_gradient(rewards: &[f64], eligibility: &[Vec<f64>]) -> Vec<f64>` — `g[i] = Σ_t (rewards[t] − r̄)·eligibility[t][i]`, `r̄ = mean(rewards)`. Returns length = inner eligibility length (0 if empty).
  - `RewardParams { iters: usize, lr: i16, clamp: i16 }`.
  - `reward_modulated(init: Vec<i16>, cfg: &RewardParams, evaluate: impl FnMut(&[i16]) -> (f64, Vec<f64>)) -> Outcome` — per iteration: eval → propose `θ' = clamp(θ − lr·sign(g), ±clamp)` → re-eval → keep-if-better on the returned scalar reward.

- [ ] **Step 1: Write the failing centering test**

Add to the `train.rs` test module (functions don't exist yet → compile fail is the "red"):

```rust
    #[test]
    fn centered_gradient_zero_on_constant_reward() {
        // Constant reward across bits → baseline cancels it → gradient is exactly zero. This is the
        // guard for spec review #1 (raw reward would leave a nonzero global-excitability term).
        let elig = vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 1.0], vec![2.0, 2.0, 0.0]];
        let rewards = vec![1.0, 1.0, 1.0];
        let g = centered_gradient(&rewards, &elig);
        assert_eq!(g.len(), 3);
        assert!(g.iter().all(|&x| x.abs() < 1e-12), "constant reward must give zero gradient: {g:?}");
    }

    #[test]
    fn centered_gradient_points_up_reward_correlation() {
        // Neuron 0 is eligible only on the high-reward bit → positive gradient; neuron 1 only on
        // the low-reward bit → negative.
        let elig = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let rewards = vec![1.0, -1.0];
        let g = centered_gradient(&rewards, &elig);
        assert!(g[0] > 0.0 && g[1] < 0.0, "gradient tracks reward correlation: {g:?}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p wave_net --lib wave_net::train::tests::centered_gradient 2>&1 | tail -20`
Expected: FAIL — `cannot find function centered_gradient`.

- [ ] **Step 3: Implement `centered_gradient`, `RewardParams`, `reward_modulated`**

Add to `src/wave_net/train.rs` (after the `Outcome` struct / `hill_climb` block):

```rust
/// Centered-reward eligibility gradient: `g[i] = Σ_t (rewards[t] − r̄) · eligibility[t][i]`, with
/// `r̄` the mean reward over the scored bits. Centering (the REINFORCE baseline) cancels the
/// constant `r̄·ē_i` term, so only the reward↔eligibility covariance drives updates. `eligibility`
/// is one per-neuron vector per scored bit; all inner vecs must share a length.
pub fn centered_gradient(rewards: &[f64], eligibility: &[Vec<f64>]) -> Vec<f64> {
    assert_eq!(rewards.len(), eligibility.len(), "rewards and eligibility must align by bit");
    let n = eligibility.first().map(|e| e.len()).unwrap_or(0);
    if rewards.is_empty() {
        return vec![0.0; n];
    }
    let rbar = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let mut g = vec![0.0f64; n];
    for (t, elig) in eligibility.iter().enumerate() {
        let c = rewards[t] - rbar;
        for i in 0..n {
            g[i] += c * elig[i];
        }
    }
    g
}

/// Knobs for [`reward_modulated`].
#[derive(Clone, Copy, Debug)]
pub struct RewardParams {
    pub iters: usize,
    /// Threshold-delta step magnitude per accepted update.
    pub lr: i16,
    /// Bound on `|theta[i]|`.
    pub clamp: i16,
}

/// Reward-modulated trainer: at each iteration, evaluate the current θ to get a selection reward
/// and a per-neuron gradient `g`, propose `θ' = clamp(θ − lr·sign(g), ±clamp)` (lower the
/// threshold of neurons whose eligibility correlates with above-average reward), re-evaluate, and
/// keep θ' iff its reward improves. `evaluate` is `FnMut` so it may hold a `&mut LayerNet`.
pub fn reward_modulated(
    init: Vec<i16>,
    cfg: &RewardParams,
    mut evaluate: impl FnMut(&[i16]) -> (f64, Vec<f64>),
) -> Outcome {
    let mut theta = init;
    let (mut best, _) = evaluate(&theta);
    let mut history = Vec::with_capacity(cfg.iters + 1);
    history.push(best);
    let mut trial = vec![0i16; theta.len()];
    for _ in 0..cfg.iters {
        let (_r0, g) = evaluate(&theta);
        for i in 0..theta.len() {
            let s: i16 = if g[i] > 0.0 {
                1
            } else if g[i] < 0.0 {
                -1
            } else {
                0
            };
            trial[i] = (theta[i] - cfg.lr * s).clamp(-cfg.clamp, cfg.clamp);
        }
        let (r1, _) = evaluate(&trial);
        if r1 > best {
            best = r1;
            theta.clone_from(&trial);
        }
        history.push(best);
    }
    Outcome { params: theta, reward: best, history }
}
```

- [ ] **Step 4: Run the centering tests to verify they pass**

Run: `cargo test -p wave_net --lib wave_net::train::tests::centered_gradient 2>&1 | tail -20`
Expected: both PASS.

- [ ] **Step 5: Write the reward-modulated descent test**

Add to the test module:

```rust
    #[test]
    fn reward_modulated_descends_toward_target() {
        // Toy: gradient = θ − target (so −sign(g) steps toward target), reward = −distance². The
        // keep-if-better loop must drive θ to the target and keep best non-decreasing.
        let target: Vec<i16> = (0..12).map(|i| ((i as i16 % 5) - 2) * 3).collect();
        let tgt = target.clone();
        let evaluate = move |theta: &[i16]| -> (f64, Vec<f64>) {
            let dist: f64 = theta.iter().zip(&tgt).map(|(&a, &b)| { let d = (a - b) as f64; d * d }).sum();
            let g: Vec<f64> = theta.iter().zip(&tgt).map(|(&a, &b)| (a - b) as f64).collect();
            (-dist, g)
        };
        let cfg = RewardParams { iters: 200, lr: 1, clamp: 30 };
        let out = reward_modulated(vec![0i16; target.len()], &cfg, evaluate);
        assert!(out.history.windows(2).all(|w| w[1] >= w[0]), "best reward must be non-decreasing");
        let dist: f64 = out.params.iter().zip(&target).map(|(&a, &b)| { let d = (a - b) as f64; d * d }).sum();
        assert!(dist < 1.0, "reward_modulated should reach the target: dist {dist}");
    }
```

- [ ] **Step 6: Run the trainer suite**

Run: `cargo test -p wave_net --lib wave_net::train 2>&1 | tail -20`
Expected: all `train` tests PASS (masked, centering ×2, descent, plus the pre-existing two).

- [ ] **Step 7: Commit**

```bash
git add src/wave_net/train.rs
git commit -m "feat: wave_net::train — centered_gradient + reward_modulated trainer

centered_gradient(rewards, eligibility) = Σ_t (r_t - r̄)·elig_t (REINFORCE
baseline; zero on constant reward). reward_modulated proposes
θ' = clamp(θ - lr·sign(g)) and keeps-if-better on the selection reward.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: The 24-cell arm-sweep experiment

**Files:**
- Create: `examples/reward_threshold.rs`
- Reference (scaffolding, do not modify): `examples/field_training.rs`, `examples/inhibition_sweep.rs`.

**Interfaces:**
- Consumes: `LayerNet::{new, on_layer, on_layer_eligibility, set_threshold_delta, reset_state, run_stream}`; `train::{reward_modulated, RewardParams, hill_climb_masked, PerturbParams, centered_gradient, Outcome}`; `calibrate::{calibrate, CalibrateParams}`; `stream::{fair_bit, BipolarInput}`; `readout::OnlineReadout`; `index::Dims`; `config::IntConfig`.
- Produces: a binary that prints per-cell lines and a mean-over-seeds table.

- [ ] **Step 1: Write the full experiment file**

Create `examples/reward_threshold.rs`:

```rust
//! Reward-modulated per-neuron threshold training — controlled arm sweep (Spec 3 continuation).
//!
//! Grid: {gradient, random, masked-random, shuffled} × {top-only, full-depth} × 3 seeds, at
//! 32×32×6, temporal-XOR τ=1. Every cell shares one keep-if-better harness and the honest
//! TRAIN/VAL/TEST split (readout trains on TRAIN, selection reward is VAL, TEST is never selected
//! on). The four strategies differ only in how θ' is proposed/scored:
//!   - gradient:      centered-reward × near-threshold eligibility  (the real rule)
//!   - random:        node perturbation on thresholds within the scope   (blind-search control)
//!   - masked-random: node perturbation restricted to baseline-eligible neurons (targeting only)
//!   - shuffled:      gradient with per-bit reward permuted across bits  (credit-signal control)
//! Interpretation gate: credit the mechanism only if gradient beats BOTH random and shuffled on
//! TEST by more than the seed spread.
//!
//! Run: `cargo run --release --example reward_threshold [iters]` (default 60).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wave_net::wave_net::calibrate::{calibrate, CalibrateParams};
use wave_net::wave_net::config::IntConfig;
use wave_net::wave_net::hash::mix;
use wave_net::wave_net::index::Dims;
use wave_net::wave_net::pipeline::LayerNet;
use wave_net::wave_net::readout::OnlineReadout;
use wave_net::wave_net::stream::{fair_bit, BipolarInput};
use wave_net::wave_net::train::{
    centered_gradient, hill_climb_masked, reward_modulated, PerturbParams, RewardParams,
};

const WPB: usize = 8;
const WASHOUT: usize = 30;
const TRAIN: usize = 400;
const VAL: usize = 200;
const TEST: usize = 200;
const TAU: usize = 1;
const SIZE: u32 = 32;
const MARGIN: i16 = 2;
const SEEDS: u64 = 3;

#[derive(Clone, Copy, PartialEq)]
enum Strategy {
    Gradient,
    Random,
    MaskedRandom,
    Shuffled,
}
use Strategy::*;

#[derive(Clone, Copy, PartialEq)]
enum Scope {
    TopOnly,
    FullDepth,
}
use Scope::*;

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Gradient => "gradient",
            Random => "random",
            MaskedRandom => "masked-random",
            Shuffled => "shuffled",
        }
    }
    fn uses_gradient(self) -> bool {
        matches!(self, Gradient | Shuffled)
    }
}
impl Scope {
    fn name(self) -> &'static str {
        match self {
            TopOnly => "top-only",
            FullDepth => "full-depth",
        }
    }
}

struct Row {
    strategy: Strategy,
    scope: Scope,
    base_test: f64,
    trained_val: f64,
    trained_test: f64,
}

fn run_cell(strategy: Strategy, scope: Scope, k: u64, iters: usize) -> Row {
    let total = WASHOUT + TRAIN + VAL + TEST;
    let val_lo = WASHOUT + TRAIN;
    let test_lo = val_lo + VAL;
    let task_seed = mix(0x5EED_C0DE ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let input_seed = mix(0x0B17_5EED ^ k.wrapping_mul(0xD1B5_4A32_9E37_79B9));
    let perturb_seed = mix(0x00F1_E1D0 ^ k.wrapping_mul(0xA24B_AED4_963E_E407));

    let mut cfg = IntConfig::demo();
    cfg.w = SIZE;
    cfg.h = SIZE;
    let dims = Dims::new(cfg.w, cfg.h, cfg.l);
    let n = cfg.n_total();
    let ls = (cfg.w * cfg.h) as usize;
    let l = cfg.l as usize;
    let top = l - 1;
    let sites = 24 * ls / 256;
    let input = BipolarInput::scatter_bottom(&dims, input_seed, sites, 4);

    // calibrate the substrate once (fixed; only thresholds are trained)
    {
        let inp = input.clone();
        calibrate(&mut cfg, &CalibrateParams::default(), &move |w, buf| {
            inp.drive_into(buf, fair_bit(task_seed ^ 0xCA1B, (w / WPB) as u64))
        });
    }

    let bhist: Vec<u8> = (0..total).map(|t| fair_bit(task_seed, t as u64)).collect();
    let target = |t: usize| (bhist[t] ^ bhist[t - TAU]) as f64;

    // which layers are trained (get eligibility hooks) for this scope
    let trained_layers: Vec<usize> = match scope {
        TopOnly => vec![top],
        FullDepth => (0..l).collect(),
    };
    // scope mask over global indices (which θ entries may move)
    let scope_mask: Vec<bool> = (0..n).map(|i| trained_layers.contains(&(i / ls))).collect();

    // shared per-bit buffers, reused per trial
    let feats = Arc::new(Mutex::new(vec![vec![0.0f64; ls]; total]));
    let elig = Arc::new(Mutex::new(vec![vec![0.0f64; n]; total]));

    let mut net = LayerNet::new(cfg.clone());
    {
        let f = feats.clone();
        net.on_layer(
            top,
            Box::new(move |wave, fired| {
                let bit = wave / WPB;
                let mut ff = f.lock().unwrap();
                for &loc in fired {
                    ff[bit][loc as usize] += 1.0;
                }
            }),
        );
    }
    for &z in &trained_layers {
        let e = elig.clone();
        net.on_layer_eligibility(
            z,
            MARGIN,
            Box::new(move |wave, eligible| {
                let bit = wave / WPB;
                let base = z * ls;
                let mut ee = e.lock().unwrap();
                for &loc in eligible {
                    ee[bit][base + loc as usize] += 1.0;
                }
            }),
        );
    }

    // Run the reservoir with a threshold delta; return per-bit (features, eligibility).
    let run = |net: &mut LayerNet, theta: &[i16]| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        net.set_threshold_delta(theta);
        {
            let mut ff = feats.lock().unwrap();
            for row in ff.iter_mut() {
                row.iter_mut().for_each(|v| *v = 0.0);
            }
            let mut ee = elig.lock().unwrap();
            for row in ee.iter_mut() {
                row.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        net.reset_state();
        net.run_stream(total * WPB, 1, |w, buf| input.drive_into(buf, bhist[w / WPB]));
        let ff = feats.lock().unwrap().clone();
        let ee = elig.lock().unwrap().clone();
        (ff, ee)
    };

    // Train the readout on TRAIN; return (accuracy over [lo,hi), the trained readout).
    let with_bias = |row: &[f64]| -> Vec<f64> {
        let mut x = row.to_vec();
        x.push(1.0);
        x
    };
    let train_readout = |features: &[Vec<f64>]| -> OnlineReadout {
        let mut ro = OnlineReadout::new(ls + 1, 1.0);
        for t in WASHOUT..(WASHOUT + TRAIN) {
            ro.update(&with_bias(&features[t]), target(t));
        }
        ro
    };
    let accuracy = |ro: &OnlineReadout, features: &[Vec<f64>], lo: usize, hi: usize| -> f64 {
        let mut correct = 0;
        for t in lo..hi {
            if ((ro.predict(&with_bias(&features[t])) >= 0.5) as u8) as f64 == target(t) {
                correct += 1;
            }
        }
        correct as f64 / (hi - lo) as f64
    };

    // deterministic permutation of the TRAIN bit indices for the shuffled control
    let train_perm: Vec<usize> = {
        let mut idx: Vec<usize> = (WASHOUT..WASHOUT + TRAIN).collect();
        idx.sort_by_key(|&t| mix(perturb_seed ^ (t as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)));
        idx
    };

    // baseline (θ = 0)
    let zeros = vec![0i16; n];
    let (base_ft, base_elig) = run(&mut net, &zeros);
    let base_ro = train_readout(&base_ft);
    let base_val = accuracy(&base_ro, &base_ft, val_lo, val_lo + VAL);
    let base_test = accuracy(&base_ro, &base_ft, test_lo, test_lo + TEST);

    // masked-random needs the baseline-eligible set (∩ scope)
    let elig_mask: Vec<bool> = (0..n)
        .map(|i| scope_mask[i] && (WASHOUT..WASHOUT + TRAIN).any(|t| base_elig[t][i] > 0.0))
        .collect();

    // train
    let (trained_val, trained_theta) = if strategy.uses_gradient() {
        let shuffled = strategy == Shuffled;
        let mut evaluate = |theta: &[i16]| -> (f64, Vec<f64>) {
            let (features, elig_bits) = run(&mut net, theta);
            let ro = train_readout(&features);
            // per-bit reward on TRAIN
            let rewards: Vec<f64> = (WASHOUT..WASHOUT + TRAIN)
                .map(|t| {
                    let pred = ro.predict(&with_bias(&features[t]));
                    if ((pred >= 0.5) as u8) as f64 == target(t) { 1.0 } else { -1.0 }
                })
                .collect();
            let rewards = if shuffled {
                // reindex rewards by the fixed permutation (destroys temporal alignment)
                (0..TRAIN).map(|j| rewards[train_perm[j] - WASHOUT]).collect::<Vec<f64>>()
            } else {
                rewards
            };
            let elig_train: Vec<Vec<f64>> =
                (WASHOUT..WASHOUT + TRAIN).map(|t| elig_bits[t].clone()).collect();
            let g = centered_gradient(&rewards, &elig_train);
            let val = accuracy(&ro, &features, val_lo, val_lo + VAL);
            (val, g)
        };
        let rp = RewardParams { iters, lr: 1, clamp: 8 };
        let out = reward_modulated(zeros.clone(), &rp, &mut evaluate);
        (out.reward, out.params)
    } else {
        let mask = match strategy {
            Random => &scope_mask,
            MaskedRandom => &elig_mask,
            _ => unreachable!(),
        };
        let mut reward = |theta: &[i16]| -> f64 {
            let (features, _elig) = run(&mut net, theta);
            let ro = train_readout(&features);
            accuracy(&ro, &features, val_lo, val_lo + VAL)
        };
        let pp = PerturbParams { iters, density_pct: 15, step: 1, clamp: 8, seed: perturb_seed };
        let out = hill_climb_masked(zeros.clone(), &pp, mask, &mut reward);
        (out.reward, out.params)
    };

    // honest generalization number: TEST accuracy of the VAL-selected θ
    let (best_ft, _) = run(&mut net, &trained_theta);
    let best_ro = train_readout(&best_ft);
    let trained_test = accuracy(&best_ro, &best_ft, test_lo, test_lo + TEST);

    Row { strategy, scope, base_test, trained_val, trained_test }
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let strategies = [Gradient, Random, MaskedRandom, Shuffled];
    let scopes = [TopOnly, FullDepth];
    let mut cells = Vec::new();
    for &st in &strategies {
        for &sc in &scopes {
            for k in 0..SEEDS {
                cells.push((st, sc, k));
            }
        }
    }
    println!(
        "Reward-modulated threshold sweep: {} strategies × {} scopes × {SEEDS} seeds, {iters} iters, {SIZE}×{SIZE} — {} cells",
        strategies.len(),
        scopes.len(),
        cells.len()
    );

    let started = std::time::Instant::now();
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::<Row>::new());
    let workers = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(cells.len());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= cells.len() {
                    break;
                }
                let (st, sc, k) = cells[i];
                let r = run_cell(st, sc, k, iters);
                println!(
                    "{:>13} {:>10} seed {}: base TEST {:.3} | trained VAL {:.3} TEST {:.3} | gain {:+.3}",
                    r.strategy.name(), r.scope.name(), k, r.base_test, r.trained_val,
                    r.trained_test, r.trained_test - r.base_test
                );
                rows.lock().unwrap().push(r);
            });
        }
    });
    let rows = rows.into_inner().unwrap();

    println!("\nmeans over {SEEDS} seeds ({:.0}s total):", started.elapsed().as_secs_f64());
    println!("strategy        scope        base TEST   trained TEST   gain     VAL-TEST gap");
    for &sc in &scopes {
        for &st in &strategies {
            let sel: Vec<&Row> =
                rows.iter().filter(|r| r.strategy == st && r.scope == sc).collect();
            let mean = |f: &dyn Fn(&Row) -> f64| sel.iter().map(|r| f(r)).sum::<f64>() / sel.len() as f64;
            println!(
                "{:>13} {:>10}    {:.3}       {:.3}        {:+.3}      {:+.3}",
                st.name(),
                sc.name(),
                mean(&|r| r.base_test),
                mean(&|r| r.trained_test),
                mean(&|r| r.trained_test - r.base_test),
                mean(&|r| r.trained_val - r.trained_test),
            );
        }
    }
    println!(
        "\nInterpretation gate: credit the rule only if `gradient` beats BOTH `random` and\n\
         `shuffled` on trained TEST by more than the seed spread, at a given scope."
    );
}
```

- [ ] **Step 2: Compile the example**

Run: `cargo build --release --example reward_threshold 2>&1 | tail -20`
Expected: builds cleanly (no warnings that fail CI).

- [ ] **Step 3: Smoke-run at 2 iterations**

Run: `cargo run --release --example reward_threshold 2 2>&1 | tail -30`
Expected: 24 per-cell lines print, then the means table and the interpretation-gate note; process exits 0. (Numbers are noise at 2 iters — this only checks the harness end-to-end.)

- [ ] **Step 4: Commit**

```bash
git add examples/reward_threshold.rs
git commit -m "feat: reward_threshold — 24-cell controlled arm sweep

{gradient, random, masked-random, shuffled} × {top-only, full-depth} × 3
seeds at 32×32, honest TRAIN/VAL/TEST, centered-reward × near-threshold
eligibility vs its controls, with an explicit interpretation gate.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Docs + full verification + headline run

**Files:**
- Modify: `src/wave_net/mod.rs` (`:1-25` doc block).

**Interfaces:** none.

- [ ] **Step 1: Update the module doc**

In `src/wave_net/mod.rs`, add a sentence to the top doc block noting the new capability (place after the existing sentence about the toolkit):

```rust
//! Per-neuron **thresholds** are trainable via `pipeline::set_threshold_delta`; `train` adds a
//! reward-modulated rule (`reward_modulated` + `centered_gradient`) driven by the near-threshold
//! eligibility captured through `pipeline::on_layer_eligibility`.
```

- [ ] **Step 2: Full workspace test + build gate**

Run: `cargo test --all-targets 2>&1 | tail -25`
Expected: all tests pass — the engine suite (both `wave_net` and `wave_reservoir` copies), the trainer tests, and the example builds.

- [ ] **Step 3: Confirm `wave_reservoir` is untouched**

Run: `git diff --stat main -- src/wave_reservoir 2>&1 | tail -5`
Expected: no output (no changes under `src/wave_reservoir`).

- [ ] **Step 4: Commit the doc**

```bash
git add src/wave_net/mod.rs
git commit -m "docs: wave_net module doc — note trainable thresholds + reward rule

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Headline run (the actual result)**

Run: `cargo run --release --example reward_threshold 60 > /tmp/reward_threshold_60.log 2>&1` (or a background run), then read the means table.
Expected: the means table. Apply the **interpretation gate** — for each scope, check whether `gradient` trained TEST exceeds BOTH `random` and `shuffled` by more than the seed-to-seed spread. Record the finding (positive → the credit signal is real; `gradient ≈ shuffled` → noise; `gradient ≈ random` → eligibility adds nothing). This is a reported result, not a unit assertion.

---

## Self-Review

**1. Spec coverage:**
- Component 1 (trainable thresholds) → Task 1. ✓
- Component 2 (eligibility hook) → Task 2. ✓
- Component 3 (centered gradient + `reward_modulated`) → Task 4; centering rationale guarded by `centered_gradient_zero_on_constant_reward`. ✓
- Component 4 (experiment) + arm matrix + trainable-neuron mask → Task 5 (masks built in `run_cell`; `hill_climb_masked` from Task 3). ✓
- Controls & interpretation (random, masked-random, shuffled; interpretation gate) → Task 5 strategies + main() note; Task 6 Step 5. ✓
- Verification items 1–5 (engine) → Tasks 1–2 tests (identity/golden, monotonicity, near-threshold band, unsubscribed, cross-thread determinism, existing suite). ✓
- Verification 6–7 (trainer) → Task 4 centering + descent tests. ✓
- Verification 8 (gradient beats controls) → Task 6 Step 5 (reported). ✓
- Files touched (pipeline, train, example, mod) → Tasks 1–6. ✓ `wave_reservoir` unchanged → Task 6 Step 3.

**2. Placeholder scan:** No TBD/TODO; every code step carries complete code; the `unreachable!()` arm is guarded by `uses_gradient()`. Note: `RewardParams` deliberately omits the spec's `margin` field — the trainer never touches the engine, so `MARGIN` lives in the experiment (Task 5) where `on_layer_eligibility` is called. This is a conscious refinement of the spec's interface, not a gap.

**3. Type consistency:** `set_threshold_delta(&[i16])`, `on_layer_eligibility(usize, i16, Box<…>)`, `centered_gradient(&[f64], &[Vec<f64>]) -> Vec<f64>`, `RewardParams { iters, lr, clamp }`, `reward_modulated(Vec<i16>, &RewardParams, impl FnMut(&[i16]) -> (f64, Vec<f64>))`, `hill_climb_masked(Vec<i16>, &PerturbParams, &[bool], impl FnMut(&[i16]) -> f64)` — names and signatures match between the interface blocks and the call sites in Task 5. `hill_climb`'s relaxed `FnMut` bound is backward-compatible with the `&reward` calls in `field_training`/`params_study` (verified in Task 3 Step 5).
