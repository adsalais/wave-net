# Backward recurrence (level −1/−2) + width — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Test whether backward recurrence (`level −1/−2`) + wider layers, trained by temporal eligibility + multi-layer DFA credit, beats feed-forward on temporal XOR — and if not, cleanly implicate `ψ` as the credit blocker.

**Architecture:** New `train_recurrent` in `bench::rsnn` on a multi-layer net with a uniform `[+1,−1,−2]` topology (the router drops off-stack levels — no engine change). All synapses (forward + backward) trained by one temporal eligibility from recorded per-wave spikes, with symmetric-top / DFA-deeper credit.

**Tech Stack:** Rust edition 2024, std only. `f32` in the bench. Inline `#[cfg(test)]` tests (headline runs in `--release`).

## Global Constraints

- Std only in the engine; **no `unsafe`**; **warning-free**.
- Determinism: pure function of `(seed, task_seed, config)`; single-threaded.
- **One commit per task**, conventional commits, **no `Co-Authored-By`**, **never push**.
- On branch `feat/rsnn-backward-recurrence`. Verify each task with `cargo test` + warning-free `cargo build`.
- Existing suite (incl. the level-0 recurrence tests and `wave_state_machine`) stays green — `train_recurrent` is additive; `train_xor`/`recurrent_update` are untouched.

## File structure

| File | Change |
|---|---|
| `src/bench/rsnn.rs` | `RsnnConfig.{back_count, back_radius}`; `engine_config_recurrent`; `topo_entries`; `xor_trial_layers`; `train_recurrent`; tests |

---

### Task 1: Backward config + all-layer XOR trial recorder

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Produces `RsnnConfig.{back_count, back_radius}`, `engine_config_recurrent`, `topo_entries(cfg) -> Vec<(i32,usize,u32)>`, `xor_trial_layers(net,cfg,a,b,trial) -> (Vec<f32>, Vec<Vec<Vec<u32>>>)`.

- [ ] **Step 1: Config fields + recurrent engine config + topology list**

Add to `RsnnConfig` (after `multi_layer`): `pub back_count: u32,` and `pub back_radius: u32,`; set `back_count: 0, back_radius: 2,` in `demo()`.
Add methods:
```rust
    /// Multi-layer net with a uniform [+1, −1, −2] topology (backward levels only when back_count>0).
    /// Off-stack targets (top's +1, L0's −1/−2) are dropped by the router — harmless.
    fn engine_config_recurrent(&self) -> Config {
        let mut topo = vec![TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count }];
        if self.back_count > 0 {
            topo.push(TopologyLevel { level: -1, radius: self.back_radius, count: self.back_count });
            topo.push(TopologyLevel { level: -2, radius: self.back_radius, count: self.back_count });
        }
        let layer = LayerConfig {
            topology: topo,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: self.adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        Config { seed: self.seed, size: self.size, layers: vec![layer; self.layers] }
    }
```
Add a free fn (non-test), near `target_of`:
```rust
/// The topology entries (level, count, radius) in the same order as `engine_config_recurrent`, so the
/// training loop can walk out_weights slots and know each slot's level/target.
fn topo_entries(cfg: &RsnnConfig) -> Vec<(i32, usize, u32)> {
    let mut e = vec![(1i32, cfg.up_count as usize, cfg.up_radius)];
    if cfg.back_count > 0 {
        e.push((-1, cfg.back_count as usize, cfg.back_radius));
        e.push((-2, cfg.back_count as usize, cfg.back_radius));
    }
    e
}
```

- [ ] **Step 2: All-layer XOR trial recorder**

Add (non-test):
```rust
/// Temporal-XOR trial (reset → cue(a) → delay → cue(b) → read) on a multi-layer net. Records every
/// computational layer's per-wave fired-set and returns (top-layer read-window spike counts, spikes[z][t]).
fn xor_trial_layers(net: &mut Network, cfg: &RsnnConfig, a: usize, b: usize, trial: usize) -> (Vec<f32>, Vec<Vec<Vec<u32>>>) {
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 1..l {
        let r = rec.clone();
        net.on_layer(z, Box::new(move |_w, fired: &[u32]| r.lock().unwrap()[z].push(fired.to_vec())));
    }
    net.reset_state();
    let present = |net: &mut Network, class: usize, phase: usize| {
        for w in 0..cfg.present_waves {
            let sites = cue_realization(cfg.task_seed, cfg.size, class, trial * 2 + phase, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
            net.wave(&sites);
        }
    };
    present(net, a, 0);
    for _ in 0..cfg.delay {
        net.wave(&[]);
    }
    present(net, b, 1);
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
    }
    net.clear_listeners();
    let spikes = rec.lock().unwrap().clone();
    let ttot = spikes[l - 1].len();
    let mut act = vec![0f32; ls];
    for wv in spikes[l - 1].iter().skip(ttot - cfg.read_waves) {
        for &loc in wv {
            act[loc as usize] += 1.0;
        }
    }
    (act, spikes)
}
```

- [ ] **Step 3: Tests**

Add to `rsnn.rs` `mod tests`:
```rust
    #[test]
    fn backward_recurrence_config_builds() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 16;
        cfg.layers = 4;
        cfg.back_count = 8;
        let net = Network::new(cfg.engine_config_recurrent());
        assert_eq!(net.layer_count(), 4);
        let e = topo_entries(&cfg);
        assert_eq!(e.iter().map(|(_, c, _)| c).sum::<usize>(), cfg.up_count as usize + 2 * 8);
        assert_eq!(e.iter().map(|(lv, _, _)| *lv).collect::<Vec<_>>(), vec![1, -1, -2]);
    }
```

- [ ] **Step 4: Run + commit**

Run: `cargo test bench::rsnn::tests::backward_recurrence_config_builds` and `cargo build`.
Expected: passes; warning-free (`xor_trial_layers`/`topo_entries` may warn as unused until Task 2 — if so,
commit Task 1+2 together; note it here and proceed to Task 2 before committing).

---

### Task 2: `train_recurrent` + the FF-vs-backward headline

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Produces `train_recurrent(cfg) -> u64` (held-out test ‰) and the headline test.

- [ ] **Step 1: Write the failing headline test**

```rust
    #[test]
    fn backward_recurrence_vs_ff_on_temporal_xor() {
        // Backward recurrence (level −1/−2) + width vs feed-forward on temporal XOR (LIF, delay 20).
        // If +backward beats FF, recurrence earns its keep; if it nulls too, topology+capacity are
        // controlled out and ψ (spike-time-only) is the implicated blocker.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mut best_ff = 0u64;
        let mut best_bw = 0u64;
        for &s in &seeds {
            let mut ff = RsnnConfig::demo();
            ff.seed = s;
            ff.task_seed = s;
            ff.size = 16;
            ff.layers = 4;
            ff.adapt_bump = 0; // LIF
            ff.delay = 20;
            ff.trials = 1500;
            let mut bw = ff.clone();
            bw.back_count = 8; // turn on backward recurrence
            let ff_acc = train_recurrent(&ff);
            let bw_acc = train_recurrent(&bw);
            eprintln!("seed {s:#x}  FF {ff_acc}  +backward {bw_acc}");
            best_ff = best_ff.max(ff_acc);
            best_bw = best_bw.max(bw_acc);
        }
        eprintln!("best FF {best_ff}  best +backward {best_bw}");
        // Honest bar: either backward clears chance+margin and beats FF, or (the diagnostic outcome) it
        // does not — recorded, not forced. This assertion just guards determinism/sanity; the real read is
        // the printed comparison, interpreted in Step 3.
        assert!(best_bw >= 485, "sanity: accuracy in range");
    }
```
(The headline is the *printed comparison* + Step 3's honest write-up; the assertion is deliberately loose so
the test encodes "we ran the diagnostic," and Step 3 sets the verdict — a null is a valid result.)

- [ ] **Step 2: Implement `train_recurrent`**

```rust
/// Train (readout + all synapses via one temporal eligibility over every topology level) on temporal XOR,
/// multi-layer net. `back_count = 0` is the feed-forward baseline. Returns held-out test accuracy permille.
pub fn train_recurrent(cfg: &RsnnConfig) -> u64 {
    let mut net = Network::new(cfg.engine_config_recurrent());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let l = net.layer_count();
    let ls = (cfg.size * cfg.size) as usize;
    let top = l - 1;
    let entries = topo_entries(cfg);
    let total_slots: usize = entries.iter().map(|(_, c, _)| c).sum();
    let mut w = vec![vec![0f32; ls]; 2];
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    let decay = 1.0 - 1.0 / cfg.rec_tau.max(1.0);
    for t in 0..cfg.trials {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let label = a ^ b;
        let (act, spikes) = xor_trial_layers(&mut net, cfg, a, b, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.hidden_lr == 0.0 {
            continue;
        }
        // per-layer fired[z][t][i] and decaying pre-trace pretr[z][t][i]
        let ttot = spikes[top].len();
        let mut fired = vec![vec![vec![0f32; ls]; ttot]; l];
        let mut pretr = vec![vec![vec![0f32; ls]; ttot]; l];
        for z in 1..l {
            for (tt, wv) in spikes[z].iter().enumerate() {
                for &loc in wv {
                    fired[z][tt][loc as usize] = 1.0;
                }
            }
            for i in 0..ls {
                let mut tr = 0.0;
                for tt in 0..ttot {
                    tr = tr * decay + fired[z][tt][i];
                    pretr[z][tt][i] = tr;
                }
            }
        }
        // learning signal per (target layer, neuron): symmetric readout at top, DFA elsewhere
        let l_sig = |tz: usize, j: usize| -> f32 {
            (0..2)
                .map(|c| {
                    let b = if tz == top { w[c][j] } else { dfa_weight(cfg.seed, (tz * ls + j) as u32, c) };
                    b * err[c]
                })
                .sum()
        };
        // walk each source layer's out_weights slots by topology entry
        for z in 0..l {
            let mut updates: Vec<(usize, f32)> = Vec::new(); // (slot index, delta) for layer z
            let mut slot = 0usize;
            for &(level, count, radius) in &entries {
                let tz_i = z as i32 + level;
                if tz_i < 1 || tz_i >= l as i32 {
                    slot += count; // off-stack (or into L0) target — untrainable, skip its slots
                    continue;
                }
                let tz = tz_i as usize;
                for i in 0..ls {
                    let sg = (z * ls + i) as u32;
                    for k in 0..count {
                        let j = target_of(cfg.seed, sg, i as u32, level, k as u32, radius, cfg.size) as usize;
                        let mut e = 0f32;
                        for tt in 0..ttot {
                            e += pretr[z][tt][i] * fired[tz][tt][j];
                        }
                        if e != 0.0 {
                            updates.push((i * total_slots + slot + k, -cfg.hidden_lr * l_sig(tz, j) * e));
                        }
                    }
                }
                slot += count;
            }
            net.with_layer_mut(z, |lz| {
                for (idx, d) in &updates {
                    lz.out_shadow[*idx] += *d;
                }
                for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                    *wq = s.round().clamp(-127.0, 127.0) as i8;
                }
            });
        }
    }
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let (act, _) = xor_trial_layers(&mut net, cfg, a, b, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == (a ^ b) {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}
```
(Note: target layer must be `≥ 1` — training weights whose target is the L0 transducer is pointless, and
`tz` filtered to `1..l`. Slot bookkeeping walks entries in order, matching `engine_config_recurrent`.)

- [ ] **Step 3: Run (release) + honest verdict**

Run: `cargo test bench::rsnn::tests::backward_recurrence_vs_ff_on_temporal_xor --release -- --nocapture`
Read `FF` vs `+backward` per seed. Tune only `back_count`, `back_radius`, `rec_tau`, `hidden_lr`, `size`,
`layers` (never the rule). **Two honest outcomes, both fine:**
- **+backward beats FF** (worst clears chance+margin, above FF): recurrence works — tighten the assertion to
  `best_bw > best_ff + 100` and record the win.
- **+backward ≈ FF (both ~chance):** the diagnostic result — topology+capacity controlled out ⇒ `ψ` is the
  blocker. Keep the loose sanity assertion, and document the null + the `ψ` implication.

Do not fudge; multi-seed; the printed table is the evidence.

- [ ] **Step 4: Full suite + commit**

Run: `cargo test` and `cargo build` (all green, warning-free).
`git commit -m "feat: backward recurrence (level -1/-2) + width on temporal XOR (train_recurrent)"`

- [ ] **Step 5: Document the finding**

Append to `docs/experiments_results.md`: the FF-vs-backward table and the verdict — recurrence win, or the
null that implicates `ψ` (topology + capacity controlled out). Commit the doc.

---

## Self-review

**Spec coverage:** backward `−1/−2` + width via uniform `[+1,−1,−2]` config (Task 1); one temporal
eligibility over all levels from recorded per-wave spikes, symmetric-top/DFA-deeper credit (Task 2); temporal
XOR LIF delay 20, FF-vs-backward multi-seed held-out (Task 2); honesty gate as the diagnostic (Task 2 Step
3). No engine change (router drops off-stack levels; per-wave spikes via listeners).

**Placeholder scan:** none — full code and commands. Task 1's unused-until-Task-2 items handled by
committing together if they warn.

**Type consistency:** `topo_entries -> Vec<(i32,usize,u32)>` walked with a running `slot` matching
`engine_config_recurrent`'s topology order; `xor_trial_layers -> (Vec<f32>, Vec<Vec<Vec<u32>>>)`;
`train_recurrent -> u64`; `target_of(.., level, ..)`, `dfa_weight`, `softmax` reused; neuron-global id
`tz*ls + j`; slot stride `total_slots`. Target layer filtered to `1..l` (skip L0/off-stack).
