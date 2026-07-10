# Complete the ALIF adaptation eligibility in e-prop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: this repo executes plans **inline and autonomously**
> (`AGENTS.md`) — use superpowers:executing-plans, never the subagent-driven option. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Add the Bellec-2020 ALIF **adaptation eligibility** (`εᵃ`, the `−β·εᵃ` term) + a normalized bump
pseudo-derivative `ψ` + scaled tunable `β` to the three temporal e-prop trainers, guarded so existing
results stay byte-identical, then re-run the ALIF recurrence benchmarks.

**Architecture:** Bench-side post-hoc from recorded per-wave traces, in `src/bench/rsnn.rs`. One shared
eligibility helper `elig_adapt_sum`. One **read-only** engine accessor `layer_effective_threshold` so `ψ` is
centered at the true adaptive firing threshold. No wave-dynamics change; determinism preserved.

**Tech Stack:** Rust 2024, std-only, inline `#[cfg(test)]` tests. Spec:
`docs/superpowers/specs/2026-07-10-eprop-alif-adaptation-eligibility-design.md`.

## Global Constraints

- **Standard library only** in `src/`; **no `unsafe`**; **warning-free build** (`cargo build`).
- **Determinism** — every result a pure function of `(seed, config, input)`; single-threaded.
- **Keep `wave_state_machine` frozen** — this plan touches only `wave_net` (one accessor) and `bench`.
- Inline `#[cfg(test)]` tests, test-first. `cargo test` must stay green and fast (experiments are `#[ignore]`d).
- **One commit per task**, conventional-commit messages. **NEVER a `Co-Authored-By` trailer.** **NEVER push.**
- Verified eligibility (do not deviate):
  ```
  εᵛ_i(t)  = pre-trace (existing `pretr`, decay 1−1/rec_tau)
  εᵃ_ij(t) = ψ_j(t−1)·εᵛ_i(t−1) + (ρ − β·ψ_j(t−1))·εᵃ_ij(t−1)
  e_ij(t)  = ψ_j(t)·( εᵛ_i(t) − β·εᵃ_ij(t) )
  ρ = 1 − 2^(−adapt_decay)   β = elig_beta (0 if adapt_bump==0)   ψ = γ·max(0, 1−|v−eff|/θ), γ=0.3
  ```

---

### Task 1: Config knobs + `layer_effective_threshold` accessor

**Files:**
- Modify: `src/bench/rsnn.rs` (`RsnnConfig` struct ~14-48; `demo()` ~51-88)
- Modify: `src/wave_net/network.rs` (import ~8; new method after `layer_decide_potential` ~161)
- Test: inline in `src/wave_net/network.rs` `#[cfg(test)]`

**Interfaces:**
- Produces: `RsnnConfig { …, elig_beta: f32, elig_bump_psi: bool }`;
  `Network::layer_effective_threshold(&self, z: usize) -> Vec<i32>`.

- [ ] **Step 1: Add the read-only accessor + failing test.** In `src/wave_net/network.rs`, change the import
  on line 8 to `use crate::wave_net::neurons::{Layer, ADAPT_SHIFT};` and add after `layer_decide_potential`
  (~line 161):

```rust
    /// Per-neuron effective ALIF firing threshold `baseline + (adapt >> ADAPT_SHIFT)` — the value the
    /// decide step compares potential against. Read-only snapshot; no dynamics change.
    pub fn layer_effective_threshold(&self, z: usize) -> Vec<i32> {
        let l = self.layers[z].lock().unwrap();
        l.threshold.iter().zip(l.adapt.iter())
            .map(|(&t, &a)| t as i32 + (a >> ADAPT_SHIFT))
            .collect()
    }
```

  Add a test in the `network.rs` `#[cfg(test)]` module (find it with `grep -n "mod tests" src/wave_net/network.rs`):

```rust
    #[test]
    fn effective_threshold_is_baseline_plus_adapt_shifted() {
        use crate::wave_net::neurons::ADAPT_SHIFT;
        let mut net = Network::new(Config::demo());
        // baseline with zero adaptation == the threshold itself
        let base = net.layer_thresholds(1);
        let eff0 = net.layer_effective_threshold(1);
        assert_eq!(eff0, base.iter().map(|&t| t as i32).collect::<Vec<_>>());
        // inject adaptation on neuron 0 of layer 1 and check the shift
        net.with_layer_mut(1, |l| l.adapt[0] = 5 << ADAPT_SHIFT);
        let eff1 = net.layer_effective_threshold(1);
        assert_eq!(eff1[0], base[0] as i32 + 5);
    }
```

- [ ] **Step 2: Run the accessor test, expect FAIL then PASS.** `cargo test -p wave_net effective_threshold`
  (or `cargo test effective_threshold`). It fails to compile first (method absent) — add the method (Step 1
  code) — then passes. Note: `Config::demo()` exists (`config.rs`); if `with_layer_mut` is `pub(crate)` the
  test is in-crate so it is reachable.

- [ ] **Step 3: Add the two config knobs.** In `src/bench/rsnn.rs`, add to `RsnnConfig` (after `rec_stab`,
  ~line 47):

```rust
    pub elig_beta: f32,      // ALIF adaptation-eligibility coupling β (0.0 = off → membrane-only, byte-identical). Active only when adapt_bump > 0.
    pub elig_bump_psi: bool, // use normalized bump pseudo-derivative ψ instead of spike/ramp post-factor (ablation: bump-ψ without the εᵃ term)
```

  And to `demo()` (after `rec_stab: 0.0,`, ~line 86):

```rust
            elig_beta: 0.0,
            elig_bump_psi: false,
```

- [ ] **Step 4: Build green.** `cargo build && cargo test --no-run` — compiles with no warnings; every
  existing `RsnnConfig::demo()`-based test still constructs (only `demo()` builds the struct literal).

- [ ] **Step 5: Commit.**

```bash
git add src/wave_net/network.rs src/bench/rsnn.rs
git commit -m "feat: layer_effective_threshold accessor + elig_beta/elig_bump_psi config knobs"
```

---

### Task 2: Shared eligibility helper + faithful `ψ`/`εᵃ` in `train_multilayer` (+ eff recording)

This is the primary path (`fair_recurrence_test` → `train_hidden_rec` → `train_multilayer`).

**Files:**
- Modify: `src/bench/rsnn.rs`: add `PSI_GAMMA` const + `elig_adapt_sum` fn (near `target_of`, ~381);
  `xor_trial_layers` (328-371) to record+return per-wave effective threshold; `train_multilayer` (885-1015);
  all 6 `xor_trial_layers` call sites (770, 870, 900, 1008, 1170, 1349).
- Test: inline `#[cfg(test)]` in `src/bench/rsnn.rs`.

**Interfaces:**
- Produces: `fn elig_adapt_sum(ttot, beta, rho, psi: impl Fn(usize)->f32, ev: impl Fn(usize)->f32) -> f32`;
  `xor_trial_layers(...) -> (Vec<f32>, Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<i16>>>, Vec<Vec<Vec<i32>>>)`
  (added 4th element `effs[z][tt][j]` = per-wave effective threshold).

- [ ] **Step 1: Add the helper + a unit test proving the recursion + sign.** Add near line 381:

```rust
/// Dampening (γ) for the normalized bump pseudo-derivative ψ = γ·max(0, 1−|v−eff|/θ). LSNN uses 0.3.
const PSI_GAMMA: f32 = 0.3;

/// Σ_t of the ALIF eligibility trace e_ij(t) = ψ_j(t)·(εᵛ_i(t) − β·εᵃ_ij(t)), with the adaptation
/// eligibility εᵃ recursed at the slow rate ρ: εᵃ(t+1) = ψ(t)·εᵛ(t) + (ρ − β·ψ(t))·εᵃ(t). β=0 reduces
/// to the membrane trace Σ_t ψ_j·εᵛ_i. (Bellec et al. 2020, Eq. 24–25.)
fn elig_adapt_sum(ttot: usize, beta: f32, rho: f32, psi: impl Fn(usize) -> f32, ev: impl Fn(usize) -> f32) -> f32 {
    let mut eps_a = 0.0f32;
    let mut e = 0.0f32;
    for tt in 0..ttot {
        let p = psi(tt);
        let v = ev(tt);
        e += p * (v - beta * eps_a);
        eps_a = p * v + (rho - beta * p) * eps_a;
    }
    e
}
```

  Unit test (add to the `rsnn.rs` tests module):

```rust
    #[test]
    fn elig_adapt_sum_matches_closed_form_and_sign() {
        // β=0 ⇒ pure membrane trace Σ ψ·εᵛ.
        let psi = [0.5f32, 0.0, 0.3];
        let ev = [1.0f32, 2.0, 4.0];
        let membrane: f32 = psi.iter().zip(ev).map(|(p, v)| p * v).sum();
        assert!((elig_adapt_sum(3, 0.0, 0.9, |t| psi[t], |t| ev[t]) - membrane).abs() < 1e-6);
        // β>0 ⇒ the −β·εᵃ term makes the total SMALLER than the membrane-only trace (adaptation is
        // suppressive: firing now raises future threshold), and εᵃ carries a slow trace forward.
        let full = elig_adapt_sum(3, 0.2, 0.9, |t| psi[t], |t| ev[t]);
        assert!(full < membrane, "adaptation term subtracts: {full} !< {membrane}");
        // hand-rolled reference for the same recursion
        let (mut eps_a, mut e) = (0.0f32, 0.0f32);
        for t in 0..3 { e += psi[t]*(ev[t]-0.2*eps_a); eps_a = psi[t]*ev[t] + (0.9-0.2*psi[t])*eps_a; }
        assert!((full - e).abs() < 1e-6);
    }
```

- [ ] **Step 2: Run it, expect FAIL→PASS.** `cargo test elig_adapt_sum_matches_closed_form` — fails
  (fn absent) → add fn → passes.

- [ ] **Step 3: Record per-wave effective threshold in `xor_trial_layers`.** In `xor_trial_layers`
  (328-371): change the return type to add `Vec<Vec<Vec<i32>>>`; add an `effs` accumulator recorded in the
  SAME `record` closure as `pots`. Edit:
  - Signature (line 328): `... -> (Vec<f32>, Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<i16>>>, Vec<Vec<Vec<i32>>>) {`
  - After `let mut pots: Vec<Vec<Vec<i16>>> = vec![Vec::new(); l];` (line 337) add:
    `let mut effs: Vec<Vec<Vec<i32>>> = vec![Vec::new(); l];`
  - In the `record` closure (338-342) change it to capture effs too. Replace the closure with:

```rust
    let record = |net: &Network, pots: &mut Vec<Vec<Vec<i16>>>, effs: &mut Vec<Vec<Vec<i32>>>| {
        for z in 1..l {
            pots[z].push(net.layer_decide_potential(z));
            effs[z].push(net.layer_effective_threshold(z));
        }
    };
```

  - Update the two `record(net, &mut pots);` calls (lines ~347, ~354) to
    `record(net, &mut pots, &mut effs);`.
  - Final return (line 370): `(act, spikes, pots, effs)`.

- [ ] **Step 4: Update the 6 `xor_trial_layers` call sites.** Add the 4th binding:
  - `train_recurrent` line 770: `let (act, spikes, pots, effs) = xor_trial_layers(...);`
  - `train_recurrent` holdout line 870: `let (act, _, _, _) = xor_trial_layers(...);`
  - `train_multilayer` line 900: `let (act, spikes, pots, effs) = xor_trial_layers(...);`
  - `train_multilayer` holdout line 1008: `let (act, _, _, _) = xor_trial_layers(...);`
  - `_gap_activity_probe` line 1170: `let (_, spikes, pots, _effs) = xor_trial_layers(...);`
  - `adapt_vs_propagation` line 1349: `let (_, spikes, _, _) = xor_trial_layers(...);`
  Then `cargo test --no-run` compiles (unused `effs` in train_recurrent until Task 4 — prefix `_effs` there
  for now to stay warning-free, rename in Task 4).

- [ ] **Step 5: Wire faithful ψ/εᵃ into `train_multilayer`.** Inside `train_multilayer`, after the trial call
  (line 900, now returning `effs`), add the mode flags near the top of the per-trial body (after `err`):

```rust
        let beta = if cfg.adapt_bump > 0 { cfg.elig_beta } else { 0.0 };
        let use_adapt = beta != 0.0;
        let use_bump = cfg.elig_bump_psi || use_adapt;
        let rho = 1.0 - (2.0f32).powi(-(cfg.adapt_decay as i32));
```

  Replace the `post[z][tt][j] = …` assignment (928-939) with the bump branch first:

```rust
                    post[z][tt][j] = if use_bump {
                        (PSI_GAMMA * (1.0 - (pots[z][tt][j] as f32 - effs[z][tt][j] as f32).abs() / theta[z][j])).max(0.0)
                    } else if cfg.subthreshold_psi {
                        (pots[z][tt][j] as f32 / theta[z][j]).clamp(0.0, 1.0)
                    } else {
                        fired[z][tt][j]
                    };
```

  Replace the eligibility accumulation (the `let mut e = 0f32; for tt … e += pretr[z][tt][i]*post[tz][tt][j];`
  block, ~973-977) with:

```rust
                        let e = if use_adapt {
                            elig_adapt_sum(ttot, beta, rho, |tt| post[tz][tt][j], |tt| pretr[z][tt][i])
                        } else {
                            let mut s = 0f32;
                            for tt in 0..ttot { s += pretr[z][tt][i] * post[tz][tt][j]; }
                            s
                        };
```

  (Keep the surrounding `if e != 0.0 { updates.push(...) }` guard.)

- [ ] **Step 6: Characterization guard test (byte-identical when off) + feature-active + determinism.**
  First capture the CURRENT default output for a tiny config, by running this throwaway once and reading the
  number:

```bash
cat > /tmp/cap.rs <<'EOF'
// (temporary) — run via a scratch test to print the baseline
EOF
```

  Instead of a scratch file, add the guard test with a placeholder, run it to observe the actual value, then
  freeze it:

```rust
    #[test]
    fn train_multilayer_elig_off_is_unchanged_and_on_differs() {
        let mut base = RsnnConfig::demo();
        base.size = 8; base.delay = 10; base.rec_count = 8; base.rate_reg = 5.0;
        base.rec_stab = 5.0; base.trials = 100;
        let off = train_hidden_rec(&base);          // elig defaults off
        assert_eq!(off, train_hidden_rec(&base), "default path deterministic");
        // FREEZE: replace 0 with the value printed on first run (characterization = no regression)
        assert_eq!(off, /*BASELINE*/ 0, "elig-off byte-identical to pre-change baseline");
        let mut on = base.clone(); on.elig_beta = 0.2;
        let a = train_hidden_rec(&on);
        assert_eq!(a, train_hidden_rec(&on), "elig-on deterministic");
        assert_ne!(a, off, "completed eligibility changes the trained result");
    }
```

  Run `cargo test train_multilayer_elig_off_is_unchanged_and_on_differs -- --nocapture`; it will fail the
  `/*BASELINE*/ 0` assertion and print the real `off` value in the panic. Replace `0` with that value and
  re-run to green. (This proves the default path did not drift AND that `elig_beta>0` is active.)

- [ ] **Step 7: Full fast suite green.** `cargo test` — all non-ignored tests pass, no warnings.

- [ ] **Step 8: Commit.**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: ALIF adaptation eligibility (εᵃ) + bump ψ in train_multilayer"
```

---

### Task 3: Faithful `ψ`/`εᵃ` in `recurrent_update` (record top-layer pots+eff)

Covers `parity_recurrence_sweep`/`distractor`/`flipflop` (→ `train_sequence`) and `alif_recurrence_vs_ff`
(→ `train_xor`), which route through `recurrent_update`. The recurrent layer is the **top** (read) layer.

**Files:**
- Modify: `src/bench/rsnn.rs`: `xor_trial` (523-555) and `sequence_trial` (561-595) to record+return top-layer
  per-wave decide_potential + eff; `recurrent_update` (600-645); call sites `train_xor_inner` (671, 698),
  `train_sequence` (721, 738), and tests `lateral_gap_survival` (1229), `sequence_trial_matches_xor` (1738-9).

**Interfaces:**
- Produces:
  `xor_trial(...) -> (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<i16>>, Vec<Vec<i32>>)` and identically
  `sequence_trial(...) -> (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<i16>>, Vec<Vec<i32>>)`
  (added: `pots_top[tt]`, `eff_top[tt]` for the read/recurrent layer);
  `recurrent_update(net, cfg, w, err, waves, pots_top: &[Vec<i16>], eff_top: &[Vec<i32>], rec_layer)`.

- [ ] **Step 1: Record top-layer pots+eff in `sequence_trial`.** In `sequence_trial` (561-595): add
  `let mut pots_top: Vec<Vec<i16>> = Vec::new(); let mut eff_top: Vec<Vec<i32>> = Vec::new();` after the
  `rec` setup. After EACH `net.wave(...)` call (the delay loop, the present loop, and the read loop), push:
  `pots_top.push(net.layer_decide_potential(top)); eff_top.push(net.layer_effective_threshold(top));`
  (`top` is already bound at line 564). Change the return (line 594) to
  `(act, waves, pots_top, eff_top)` and the signature to
  `-> (Vec<f32>, Vec<Vec<u32>>, Vec<Vec<i16>>, Vec<Vec<i32>>)`. The recorded length equals `waves.len()`
  (one push per wave), so it aligns with the fired-set index used by `recurrent_update`.

- [ ] **Step 2: Record top-layer pots+eff in `xor_trial` the same way.** In `xor_trial` (523-555):
  `top` is not bound — add `let top = net.layer_count() - 1;` (the listener is on layer 1, but the read layer
  for this 2-layer XOR net IS layer 1 = top; keep `top` = `layer_count()-1` so it matches `recurrent_update`'s
  `rec_layer`). Add the two accumulators; push after each `net.wave(...)` inside the `present` closure and the
  delay/read loops. Return `(act, waves, pots_top, eff_top)`; update signature to match Step 1.

- [ ] **Step 3: Update call sites to the new 4-tuple.**
  - `train_xor_inner` line 671: `let (act, waves, pots_top, eff_top) = xor_trial(&mut net, cfg, a, b, t);`
  - `train_xor_inner` holdout line 698: `let (act, _, _, _) = xor_trial(...);`
  - `train_sequence` line 721: `let (act, waves, pots_top, eff_top) = sequence_trial(&mut net, cfg, &classes, t);`
  - `train_sequence` holdout line 738: `let (act, _, _, _) = sequence_trial(...);`
  - test `lateral_gap_survival` line 1229: `let (_, waves, _, _) = xor_trial(&mut net, &c, 1, 0, 0);`
  - test `sequence_trial_matches_xor_on_two_cues` lines 1738-1739:
    `let (a1, w1, _, _) = xor_trial(&mut n1, &cfg, 1, 0, 3);`
    `let (a2, w2, _, _) = sequence_trial(&mut n2, &cfg, &[1, 0], 3);` (the `(act, waves)` equivalence it
    asserts still holds).

- [ ] **Step 4: Add ψ/εᵃ to `recurrent_update`.** Change its signature (line 600) to accept the new slices:
  `fn recurrent_update(net: &mut Network, cfg: &RsnnConfig, w: &[Vec<f32>], err: &[f32], waves: &[Vec<u32>], pots_top: &[Vec<i16>], eff_top: &[Vec<i32>], rec_layer: usize) {`
  After `let decay = …` / trace build, add:

```rust
    let beta = if cfg.adapt_bump > 0 { cfg.elig_beta } else { 0.0 };
    let use_adapt = beta != 0.0;
    let use_bump = cfg.elig_bump_psi || use_adapt;
    let rho = 1.0 - (2.0f32).powi(-(cfg.adapt_decay as i32));
    let theta: Vec<f32> = net.layer_thresholds(rec_layer).iter().map(|&t| (t as f32).max(1.0)).collect();
    let psi = |t: usize, j: usize| -> f32 {
        if use_bump {
            (PSI_GAMMA * (1.0 - (pots_top[t][j] as f32 - eff_top[t][j] as f32).abs() / theta[j])).max(0.0)
        } else {
            fired[t][j]
        }
    };
```

  Replace the inner eligibility accumulation (`let mut e = 0f32; for t in 0..ttot { e += tr[t][i]*fired[t][j]; }`,
  ~635-637) with:

```rust
                let e = if use_adapt {
                    elig_adapt_sum(ttot, beta, rho, |t| psi(t, j), |t| tr[t][i])
                } else {
                    let mut s = 0f32;
                    for t in 0..ttot { s += tr[t][i] * fired[t][j]; }
                    s
                };
```

  Note the closure `psi` borrows `theta`/`pots_top`/`eff_top` immutably and is called inside the
  `net.with_layer_mut(rec_layer, …)` block — capture them by reference before the `with_layer_mut` call (they
  are all locals/params, no conflict with the `&mut net` borrow inside).

- [ ] **Step 5: Update the two `recurrent_update(...)` call sites** to pass the new slices:
  - `train_xor_inner` line ~681:
    `recurrent_update(&mut net, cfg, &w, &err, &waves, &pots_top, &eff_top, rl);`
  - `train_sequence` line ~731:
    `recurrent_update(&mut net, cfg, &w, &err, &waves, &pots_top, &eff_top, rl);`

- [ ] **Step 6: Determinism + feature-active test.**

```rust
    #[test]
    fn recurrent_update_elig_on_is_deterministic_and_differs() {
        let mut base = RsnnConfig::demo();
        base.delay = 8; base.rec_count = 8; base.rec_init = 0; base.trials = 120;
        let run = |c: &RsnnConfig| train_sequence(c, |seed, t| task_parity(seed, t, 3));
        let off = run(&base);
        assert_eq!(off, run(&base), "default recurrent path deterministic");
        let mut on = base.clone(); on.elig_beta = 0.2;
        let a = run(&on);
        assert_eq!(a, run(&on), "elig-on deterministic");
        assert_ne!(a, off, "completed eligibility changes the recurrent-top result");
    }
```

- [ ] **Step 7: Run + full suite.** `cargo test recurrent_update_elig_on` then `cargo test` — all green, no
  warnings. (If `assert_ne!` unexpectedly ties, β=0.2 may be too small for this tiny 120-trial config — bump
  to 0.5 in the test; the point is only to prove the path is wired and active.)

- [ ] **Step 8: Commit.**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: ALIF adaptation eligibility + bump ψ in recurrent_update (train_xor/train_sequence)"
```

---

### Task 4: `εᵃ` in `train_recurrent`

`train_recurrent` (uniform +1/−1/−2) uses `xor_trial_layers`, which already returns `effs` from Task 2.

**Files:**
- Modify: `src/bench/rsnn.rs` `train_recurrent` (750-878).

- [ ] **Step 1: Rename the placeholder binding.** At line 770 change `_effs` → `effs` (it is now used).

- [ ] **Step 2: Add the mode flags** after `err` is computed (~773), identical to Task 2 Step 5:

```rust
        let beta = if cfg.adapt_bump > 0 { cfg.elig_beta } else { 0.0 };
        let use_adapt = beta != 0.0;
        let use_bump = cfg.elig_bump_psi || use_adapt;
        let rho = 1.0 - (2.0f32).powi(-(cfg.adapt_decay as i32));
```

- [ ] **Step 3: Bump ψ.** Replace the `post[z][tt][j] = …` assignment (798-810) with the same three-branch
  form as Task 2 Step 5 (bump → subthreshold_psi → fired), using `pots[z][tt][j]`, `effs[z][tt][j]`,
  `theta[z][j]`.

- [ ] **Step 4: εᵃ eligibility.** Replace the accumulation
  `let mut e = 0f32; for tt in 0..ttot { e += pretr[z][tt][i] * post[tz][tt][j]; }` (~845-848) with the
  `use_adapt ? elig_adapt_sum(...) : Σ` form from Task 2 Step 5.

- [ ] **Step 5: Determinism test.**

```rust
    #[test]
    fn train_recurrent_elig_on_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 16; cfg.layers = 4; cfg.back_count = 8; cfg.delay = 20;
        cfg.trials = 150; cfg.elig_beta = 0.2;
        assert_eq!(train_recurrent(&cfg), train_recurrent(&cfg));
    }
```

- [ ] **Step 6: Run + suite.** `cargo test train_recurrent_elig_on` then `cargo test` — green, no warnings.

- [ ] **Step 7: Commit.**

```bash
git add src/bench/rsnn.rs
git commit -m "feat: ALIF adaptation eligibility + bump ψ in train_recurrent"
```

---

### Task 5: New benchmark experiments (off vs on) + run them

Add ignored experiments that compare `+recurrence` with the OLD (elig off) vs the COMPLETED (elig on)
eligibility, so the scientific verdict is one reproducible test. Do NOT modify the existing experiments
(they stay as the byte-identical baseline).

**Files:**
- Modify: `src/bench/rsnn.rs` tests module — add `fair_recurrence_eprop_complete` and
  `parity_recurrence_eprop_complete`.

- [ ] **Step 1: Add the fair-test comparison (the headline: does completing the credit rule stop recurrence
  destroying the 986 baseline?).**

```rust
    #[test]
    #[ignore] // expensive; run manually in --release
    fn fair_recurrence_eprop_complete() {
        // Same fair recurrence test (deep-FF 986 vs +hidden-rec 498), now comparing the OLD eligibility
        // (elig_beta 0) against the COMPLETED ALIF eligibility (elig_beta > 0) on the recurrent path.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF, 0xA5A5_1111, 0x0F0F_2222];
        for &beta in &[0.0f32, 0.1, 0.2, 0.4] {
            let (mut ffa, mut ra) = (Vec::new(), Vec::new());
            for &s in &seeds {
                let mut ff = RsnnConfig::demo();
                ff.seed = s; ff.task_seed = s; ff.size = 16; ff.up_count = 16; ff.delay = 20;
                ff.trials = 1500; ff.rate_reg = 5.0; ff.rate_target_permille = 100; ff.rec_count = 0;
                let mut rec = ff.clone();
                rec.rec_count = 24; rec.rec_radius = 4; rec.rec_tau = 20.0; rec.rec_init = 0;
                rec.rec_stab = 5.0; rec.elig_beta = beta;
                let f = train_hidden_rec(&ff);
                let r = train_hidden_rec(&rec);
                eprintln!("β {beta}  seed {s:#x}  deep-FF {f}  +rec {r}");
                ffa.push(f); ra.push(r);
            }
            eprintln!("β {beta}: FF worst {} mean {}   +rec worst {} mean {}",
                ffa.iter().min().unwrap(), ffa.iter().sum::<u64>()/5,
                ra.iter().min().unwrap(), ra.iter().sum::<u64>()/5);
        }
    }
```

- [ ] **Step 2: Add the parity comparison.**

```rust
    #[test]
    #[ignore] // expensive; run manually in --release
    fn parity_recurrence_eprop_complete() {
        // Parity (recurrent computation), FF vs +rec with the completed ALIF eligibility, β sweep.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        for n in [3usize, 4] {
            for &beta in &[0.0f32, 0.2, 0.4] {
                let (mut wf, mut wr) = (1000u64, 1000u64);
                for &s in &seeds {
                    let mut ff = RsnnConfig::demo();
                    ff.seed = s; ff.task_seed = s; ff.delay = 8; ff.trials = 1500;
                    ff.rate_reg = 5.0; ff.rate_target_permille = 100;
                    let mut rec = ff.clone();
                    rec.rec_count = 24; rec.rec_radius = 2; rec.rec_tau = 20.0; rec.rec_init = 0;
                    rec.elig_beta = beta;
                    let fa = train_sequence(&ff, |seed, t| task_parity(seed, t, n));
                    let ra = train_sequence(&rec, |seed, t| task_parity(seed, t, n));
                    eprintln!("parity N={n} β {beta} seed {s:#x}  FF {fa}  +rec {ra}");
                    wf = wf.min(fa); wr = wr.min(ra);
                }
                eprintln!("parity N={n} β {beta}: WORST FF {wf}  +rec {wr}");
            }
        }
    }
```

- [ ] **Step 3: Determinism smoke (fast, non-ignored) for the new config on a tiny run** — already covered by
  Task 3 Step 6 (`train_sequence` + `elig_beta`). No new fast test needed; `cargo test` stays green.

- [ ] **Step 4: Run the experiments and capture output** (record raw numbers to paste into the docs):

```bash
cargo test --release fair_recurrence_eprop_complete -- --ignored --nocapture 2>&1 | tee /tmp/wave-fair.txt
cargo test --release parity_recurrence_eprop_complete -- --ignored --nocapture 2>&1 | tee /tmp/wave-parity.txt
```

  Expected runtime: minutes each (5 seeds × 4 β × 2 trainings × 1500 trials). Read the `WORST/mean` summary
  lines. **Interpretation:** β=0 must reproduce the known null (+rec ≈ 498 fair / ≤ FF parity); any β where
  `+rec` climbs toward FF (or FF+) is the completed-eligibility effect.

- [ ] **Step 5: Commit the experiments (results go in Task 6).**

```bash
git add src/bench/rsnn.rs
git commit -m "test: recurrence benchmarks comparing old vs completed ALIF eligibility"
```

---

### Task 6: Update the docs with the outcome

**Files:**
- Modify: `docs/experiments_results.md` (recurrence sections), `AGENTS.md` (recurrence bullet + remove the
  stray appended RF-neuron text at the file end).

- [ ] **Step 1: Write the result into `experiments_results.md`.** Add a new section
  `## Completing the ALIF adaptation eligibility — <verdict>` reporting the β sweep numbers from Task 5
  (`/tmp/wave-fair.txt`, `/tmp/wave-parity.txt`). State plainly which outcome occurred:
  - if `+rec` now ≥ FF at some β: recurrence earns its keep once the credit rule is complete — the prior null
    was an artifact of the incomplete eligibility;
  - if still ≤ FF at all β: the null holds *with a faithful rule* — now genuinely credible.
  **Correct** the earlier "every substrate/stabilizer/topology/neuron-model confound is now ruled out …
  surrogate-gradient BPTT is the sole remaining lever" wording: note the credit rule was itself incomplete
  when that was written, and this is the experiment that closes (or earns) that claim.

- [ ] **Step 2: Update `AGENTS.md`.** In the recurrence bullet (Learning section), replace "the sole
  remaining lever is surrogate-gradient BPTT" framing with the completed-eligibility result. **Delete the
  stray appended text** below the Architecture map (the pasted RF-neuron discussion, ~lines 253-283 —
  everything after the architecture-map code fence that is not part of the structured doc). Verify with
  `git diff AGENTS.md` that only the intended prose changed.

- [ ] **Step 3: Verify the build/tests are green and docs consistent.** `cargo test && cargo build` (green,
  no warnings). Re-read both docs for internal consistency with the new numbers.

- [ ] **Step 4: Commit.**

```bash
git add docs/experiments_results.md AGENTS.md
git commit -m "docs: <verdict> — completed ALIF eligibility re-run of the recurrence benchmarks"
```

---

## Self-Review

- **Spec coverage:** εᵃ term (Tasks 2/3/4), bump ψ (Tasks 2/3/4), scaled β knob + guard (Task 1 + β-gate in
  each), read-only accessor (Task 1), all three trainers (train_multilayer T2, recurrent_update T3,
  train_recurrent T4), guarded/byte-identical (T2 Step 6 characterization + β-off branches), benchmark
  re-runs incl. ablation β-sweep (Task 5), docs incl. correcting the premature conclusion + removing RF paste
  (Task 6). LIF unaffected — enforced by `beta = if adapt_bump>0 {elig_beta} else {0.0}`. All covered.
- **Placeholder scan:** the single intentional placeholder is `/*BASELINE*/ 0` in T2 Step 6, resolved by
  running the test once to read the real value — the step says exactly how. No other TBD/TODO.
- **Type consistency:** `elig_adapt_sum(ttot, beta, rho, psi, ev)` used identically in T2/T3/T4;
  `layer_effective_threshold -> Vec<i32>` produced in T1, consumed in T2/T3; `xor_trial_layers` 4-tuple
  (T2) and `xor_trial`/`sequence_trial` 4-tuple (T3) consumed at every listed call site; `recurrent_update`
  new signature (T3) matched at both call sites. `elig_beta: f32`, `elig_bump_psi: bool` consistent.
