# wave_resonate Phase 2b — trainable ω / b′ — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Checkbox steps.

**Goal:** Make each BRF neuron's intrinsic frequency `ω_j` and dampening offset `b′_j` **trainable** via
their own per-neuron HYPR forward eligibilities, so the network learns its temporal receptive fields (not
just the ternary weights). Gated by `EligParams.train_omega_b`; `false` reduces exactly to Phase 2a.

**Architecture:** Per compute neuron keep 2-state parameter eligibilities `(gˣ,gʸ)` for each of `ω` and
`b′`, recursed through the SAME BRF Jacobian as the forward pass with the parameter's own source terms
(`∂b/∂θ`, `∂ω/∂θ`), accrued during the forward compute loop (per-neuron, O(neurons) — no synapse scan).
Accumulate `grad_θ += ψ_j·gˣ_θ`; apply `θ_j += −lr_θ·signal[z][j]·grad_θ` with `δ·ω ≤ 1` / `b′ ≥ 0`
clamps. Everything is per-neuron and dense, so online == a re-run by construction (no active-set oracle
needed; validated by determinism + regression + effect).

**Tech Stack:** Rust edition 2024, std only, inline `#[cfg(test)]`.

## Global Constraints

- Std only; warning-free; no `unsafe`.
- Determinism — pure fn of `(seed, config, input)`.
- **`train_omega_b=false` is bit-identical to Phase 2a** (grads stay 0, updates are no-ops) — a regression gate.
- **Param eligibility recursion (from the spec), per compute neuron `j`, param `θ ∈ {ω_j, b′_j}`:**
  ```
  gˣ_θ(t) = gˣ_θ(t−1) + δ·[ (∂b/∂θ)·xʲ(t−1) + bʲ(t)·gˣ_θ(t−1) − (∂ω/∂θ)·yʲ(t−1) − ωʲ·gʸ_θ(t−1) ]
  gʸ_θ(t) = gʸ_θ(t−1) + δ·[ (∂ω/∂θ)·xʲ(t−1) + ωʲ·gˣ_θ(t−1) + (∂b/∂θ)·yʲ(t−1) + bʲ(t)·gʸ_θ(t−1) ]
  grad_θ += ψʲ(t) · gˣ_θ(t)
  ```
  with `θ=b′: ∂b/∂b′=−1, ∂ω/∂b′=0`; `θ=ω: ∂ω/∂ω=1, ∂b/∂ω=p′(ω)=−δω/√(1−(δω)²)`. `xʲ(t−1),yʲ(t−1)` are
  the PRE-update membrane (as in the forward loop); `bʲ(t)` the forward `b`; `ψʲ(t)` the same surrogate
  used for the weight elig. Update: `θ_j += −lr_θ·signal[z][j]·grad_θ`, clamp `ω∈[0.5, 0.99/δ]`, `b′≥0`.
- One commit per task; conventional commits; **no `Co-Authored-By` trailer**.

---

## File Structure

- Modify `src/wave_resonate/neurons.rs` — extend `TrainState` (`g_om_x/g_om_y/g_bo_x/g_bo_y`, `om_grad/bo_grad`);
  add `Layer.train_omega_b: bool`.
- Modify `src/wave_resonate/wave.rs` — accrue the param eligibility in the compute loop when
  `train && train_omega_b`.
- Modify `src/wave_resonate/network.rs` — propagate `train_omega_b` to layers; `omega_b_update`; extend
  `reset_eligibility` to clear param state.
- Modify `src/bench/wave_resonate_bench.rs` — `train_omega_b` knob + FF smoke with ω/b′ on + frozen-vs-trained diagnostic.

---

### Task 1: `TrainState` param fields + `Layer.train_omega_b`

**Files:** Modify `src/wave_resonate/neurons.rs`.

**Interfaces:** adds to `TrainState`: `g_om_x, g_om_y, g_bo_x, g_bo_y: Vec<f32>` (ls), `om_grad, bo_grad: Vec<f32>` (ls). Adds `pub train_omega_b: bool` to `Layer` (default false). `enable_training` allocates the new vectors (zeroed).

- [ ] **Step 1: Write the failing test** (append to neurons tests)

```rust
    #[test]
    fn enable_training_allocates_param_eligibility() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let mut l = Layer::new(&cfg, 0.05, 0.9, 1.0, 3, 0, size);
        assert!(!l.train_omega_b, "default off");
        l.enable_training();
        let t = l.train.as_ref().unwrap();
        let ls = l.x.len();
        assert_eq!(t.g_om_x.len(), ls);
        assert_eq!(t.g_bo_y.len(), ls);
        assert_eq!(t.om_grad.len(), ls);
        assert!(t.g_om_x.iter().all(|&v| v == 0.0) && t.om_grad.iter().all(|&v| v == 0.0));
    }
```

- [ ] **Step 2: Run to verify failure** — `cargo test wave_resonate::neurons` → FAIL.

- [ ] **Step 3: Implement.** Add the six fields to `TrainState`; add `pub train_omega_b: bool,` to `Layer`
  (init `false` in `new`); in `enable_training` add the six `vec![0f32; ls]` allocations.

- [ ] **Step 4: Run to verify pass.** Warning check.

- [ ] **Step 5: Commit** `feat(wave_resonate): TrainState param eligibility fields + Layer.train_omega_b`.

---

### Task 2: Forward accrual of the param eligibility

**Files:** Modify `src/wave_resonate/wave.rs`.

**Interfaces:** in `process_layer`'s compute loop, when `training && layer.train_omega_b`, advance the
`(gˣ,gʸ)` recursions for `ω` and `b′` (pre-update `x,y`; forward `b`; `ω`), store them, and accumulate
`om_grad[i] += ψ·gˣ_ω`, `bo_grad[i] += ψ·gˣ_b′` using the same `ψ` captured for `b_eff/psi`.

- [ ] **Step 1: Write the failing test** (append to wave tests)

```rust
    #[test]
    fn param_eligibility_accrues_only_when_enabled() {
        let dt = 0.05f32;
        let mut off = compute_layer(1, 10.0, 0.3, dt, 0.9, 1.0);
        off.enable_training();
        let mut on = compute_layer(1, 10.0, 0.3, dt, 0.9, 1.0);
        on.enable_training();
        on.train_omega_b = true;
        let mut deliv = vec![vec![0i32; 1]; 1];
        let mut fired = Vec::new();
        for _ in 0..20 {
            off.pending[0] = 30;
            on.pending[0] = 30;
            process_layer(&mut off, 0, 1, &[], &mut deliv, &mut fired);
            process_layer(&mut on, 0, 1, &[], &mut deliv, &mut fired);
        }
        assert!(off.train.as_ref().unwrap().om_grad[0] == 0.0, "off: no param grad");
        assert!(on.train.as_ref().unwrap().om_grad[0] != 0.0 || on.train.as_ref().unwrap().bo_grad[0] != 0.0, "on: param grad accrues");
    }
```

- [ ] **Step 2: Run to verify failure** — FAIL (on grads stay 0).

- [ ] **Step 3: Implement.** In the compute loop's `if training` block (after computing `nx,ny,b`, with
  pre-update `x,y,q,omega`), add, gated by `layer.train_omega_b`:

```rust
            if layer.train_omega_b {
                let t = layer.train.as_mut().unwrap();
                let pwp = -dt * omega / (1.0 - (dt * omega) * (dt * omega)).sqrt(); // p'(ω)
                // b': ∂b/∂b'=-1, ∂ω/∂b'=0
                let (gbx, gby) = (t.g_bo_x[i], t.g_bo_y[i]);
                let ngbx = gbx + dt * (-x + b * gbx - omega * gby);
                let ngby = gby + dt * (omega * gbx - y + b * gby);
                t.g_bo_x[i] = ngbx;
                t.g_bo_y[i] = ngby;
                // ω: ∂ω/∂ω=1, ∂b/∂ω=p'(ω)
                let (gwx, gwy) = (t.g_om_x[i], t.g_om_y[i]);
                let ngwx = gwx + dt * (pwp * x + b * gwx - y - omega * gwy);
                let ngwy = gwy + dt * (x + omega * gwx + pwp * y + b * gwy);
                t.g_om_x[i] = ngwx;
                t.g_om_y[i] = ngwy;
                let psi = t.psi[i]; // set just above in the training capture
                t.om_grad[i] += psi * ngwx;
                t.bo_grad[i] += psi * ngbx;
            }
```

  Ensure the existing `t.psi[i] = surrogate(...)` capture happens BEFORE this block (reuse it). Because
  `t` is borrowed mutably twice (capture block then this block), fold both into ONE `let t = ...` scope, or
  compute `psi` into a local first. Simplest: compute `let psi_val = surrogate(nx - theta_c - q);` once,
  set `t.psi[i]=psi_val; t.b_eff[i]=b;` and use `psi_val` here.

- [ ] **Step 4: Run to verify pass.** Warning check.

- [ ] **Step 5: Commit** `feat(wave_resonate): forward accrual of ω/b′ param eligibility (gated)`.

---

### Task 3: `omega_b_update` + clamps + `reset_eligibility`

**Files:** Modify `src/wave_resonate/network.rs`.

**Interfaces:**
```rust
pub fn omega_b_update(&mut self, signal: &[Vec<f32>], lr: f32);
```
For each layer `z` that is a compute layer (`train_omega_b`, not transducer, not readout), for each neuron
`j`: `ω[j] += −lr·signal[z][j]·om_grad[j]` clamped to `[0.5, 0.99/dt]`; `b_off[j] += −lr·signal[z][j]·bo_grad[j]`
clamped to `≥ 0`. `reset_eligibility` additionally zeroes the six param vectors densely. `enable_training`
and `set_elig_params` set each layer's `train_omega_b` from `elig_params.train_omega_b`.

- [ ] **Step 1: Write the failing tests** (in `training_tests`)

```rust
    #[test]
    fn omega_b_frozen_when_disabled() {
        let mut net = Network::new(two_layer(8));
        net.enable_training(); // train_omega_b default false
        let before = net.with_layer(1, |l| (l.omega.clone(), l.b_off.clone()));
        for _ in 0..30 { net.wave(&[0,1,2,8,9,10]); }
        let ls = 64usize;
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
        net.omega_b_update(&signal, 0.5);
        let after = net.with_layer(1, |l| (l.omega.clone(), l.b_off.clone()));
        assert_eq!(before, after, "ω/b′ frozen when train_omega_b=false");
    }

    #[test]
    fn omega_b_train_moves_params_within_clamp() {
        let mut net = Network::new(two_layer(8));
        net.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: true });
        net.enable_training();
        let before = net.with_layer(1, |l| l.omega.clone());
        for _ in 0..30 { net.wave(&[0,1,2,8,9,10]); }
        let ls = 64usize;
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
        net.omega_b_update(&signal, 5.0);
        net.with_layer(1, |l| {
            assert!(l.omega != before, "ω moves when trained");
            assert!(l.omega.iter().all(|&w| w >= 0.5 && w <= 0.99 / 0.05), "ω stays within δω≤1 clamp");
            assert!(l.b_off.iter().all(|&b| b >= 0.0), "b′ stays ≥ 0");
        });
    }

    #[test]
    fn reset_clears_param_eligibility() {
        let mut net = Network::new(two_layer(8));
        net.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: true });
        net.enable_training();
        for _ in 0..20 { net.wave(&[0,1,2,8,9,10]); }
        net.reset_state();
        net.with_layer(1, |l| {
            let t = l.train.as_ref().unwrap();
            assert!(t.om_grad.iter().all(|&v| v == 0.0) && t.g_om_x.iter().all(|&v| v == 0.0));
        });
    }
```

- [ ] **Step 2: Run to verify failure** — FAIL (`omega_b_update` missing).

- [ ] **Step 3: Implement.** Add `omega_b_update` (dense per-neuron; guard `layers[z].train_omega_b &&
  !transducer && !readout`; clamp via `layer.dt`). Extend `reset_eligibility` to densely zero
  `g_om_x/g_om_y/g_bo_x/g_bo_y/om_grad/bo_grad`. In `enable_training`, after enabling each layer, set
  `layers[z].train_omega_b = self.elig_params.train_omega_b` (but skip L0/readout? — leave flag set; the
  forward branches for transducer/readout never accrue, and `omega_b_update` guards them). In
  `set_elig_params`, also propagate to each layer's `train_omega_b`.

- [ ] **Step 4: Run to verify pass.** Warning check + `cargo test wave_resonate` all green (2a oracle still
  bit-exact, since default `train_omega_b=false`).

- [ ] **Step 5: Commit** `feat(wave_resonate): omega_b_update (clamped) + reset of param eligibility`.

---

### Task 4: Bench — FF trains with ω/b′ on + frozen-vs-trained diagnostic

**Files:** Modify `src/bench/wave_resonate_bench.rs`.

- [ ] **Step 1:** Add `train_omega_b: bool` + `omega_b_lr: f32` to `TaskCfg`; thread through: in
  `train_and_eval_best`, after `dfa_update`, if `cfg.train_omega_b` call `net.omega_b_update(&signal, cfg.omega_b_lr)`;
  set the net's `train_omega_b` in `make_ff` from a new param. `ff_cfg` defaults `train_omega_b=false,
  omega_b_lr=0.0` (Phase 2a behavior unchanged).

- [ ] **Step 2:** Add an `#[ignore]` smoke that trains FF with `train_omega_b=true` and asserts it still
  clears chance (`best > 600`); and a diagnostic printing frozen-vs-trained best side by side.

```rust
    #[test]
    #[ignore] // smoke: FF trains with ω/b′ trainable (--release --nocapture)
    fn wave_resonate_ff_trains_with_omega_b() {
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff_ob(seed, 16, 4, 24, 3, 0.1, (0.0, 0.2), true);
        let mut cfg = ff_cfg();
        cfg.train_omega_b = true;
        cfg.omega_b_lr = 2.0; // tune in Step 3
        let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 4, 1500);
        eprintln!("wave_resonate FF (ω/b′ trainable): best {best}@{at}");
        assert!(best > 600, "BRF+HYPR FF with trainable ω/b′ should clear chance: {best}");
    }
```
  (`make_ff_ob` = `make_ff` + a `train_omega_b` arg that sets the net's flag before/after
  `set_elig_params`+`enable_training`.)

- [ ] **Step 3: Run** `cargo test --release wave_resonate_ff_trains_with_omega_b -- --ignored --nocapture`.
  Tune `omega_b_lr` (start 2.0; the g traces are δ-scaled but injected every wave, so grads are larger than
  the weight elig — try 0.2, 1, 5) until it clears 600 and, ideally, holds near the frozen ceiling. If
  params diverge/collapse (accuracy drops vs frozen), lower `omega_b_lr`.

- [ ] **Step 4: Commit** `test(wave_resonate): FF trains with trainable ω/b′ + frozen-vs-trained diagnostic`.

---

### Task 5: Full suite + docs

- [ ] **Step 1:** `cargo test` all green; `cargo build` warning-free.
- [ ] **Step 2:** AGENTS.md: mark Phase 2b landed (trainable ω/b′), note `omega_b_update` + the param
  eligibility in `neurons.rs`/`wave.rs`; move "Next" to Phase 3 (experiments).
- [ ] **Step 3: Commit** `docs(wave_resonate): register Phase 2b (trainable ω/b′)`.

---

## Self-Review

**Spec coverage:** trainable ω/b′ via per-neuron eligibility (Tasks 1–3); `train_omega_b` gate + regression
(Task 3, default false == Phase 2a); clamps `δω≤1`/`b′≥0` (Task 3); end-to-end still trains (Task 4). ✓
**No oracle for params:** justified — the param eligibility is computed in the dense forward with no
active-set approximation, so online == re-run by construction; validated by determinism + regression +
effect + clamp tests instead. ✓
**Placeholder scan:** `omega_b_lr` in Task 4 is tuned in Step 3 (a real bring-up step with a start value),
not a shipped placeholder. ✓
**Type consistency:** `omega_b_update(signal, lr)`, `TrainState` param fields, `Layer.train_omega_b`,
`p′(ω)=−δω/√(1−(δω)²)`, clamp `[0.5, 0.99/dt]` used consistently. The g-recursion in Task 2 matches the
Global-Constraints formula (source terms `∂b/∂θ·x`, `∂ω/∂θ·y`). ✓
