# Sub-threshold ψ — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give the temporal eligibility a sub-threshold pseudo-derivative (`ψ = clamp(v/θ,0,1)` from the decide-time potential) so credit flows during silent gaps, and A/B it against spike-`ψ` on the backward-recurrence temporal-XOR config that just nulled.

**Architecture:** Tiny engine addition — snapshot the decide-time potential per neuron per wave. `train_recurrent` reads it and builds `ψ` from it (behind a `subthreshold_psi` flag). Isolated A/B: only `ψ` changes.

**Tech Stack:** Rust edition 2024, std only. `f32` in the bench. Inline `#[cfg(test)]` tests (headline `#[ignore]`, release).

## Global Constraints

- Std only in the engine; **no `unsafe`**; **warning-free**.
- Determinism preserved; single-threaded; forward firing dynamics unchanged.
- **One commit per task**, conventional commits, **no `Co-Authored-By`**, **never push**.
- On branch `feat/rsnn-subthreshold-psi`. Verify each task with `cargo test` + warning-free `cargo build`.
- Whole existing suite (incl. `wave_state_machine`) stays green; `subthreshold_psi=false` is unchanged behavior.

## File structure

| File | Change |
|---|---|
| `src/wave_net/neurons.rs` | `Layer.decide_potential: Vec<i16>` + init |
| `src/wave_net/wave.rs` | snapshot `decide_potential[i]` in the decide loop; unit test |
| `src/wave_net/network.rs` | zero it in `reset_state`; `layer_decide_potential(z)` accessor |
| `src/bench/rsnn.rs` | `subthreshold_psi` flag; record decide-potentials in `xor_trial_layers`; `ψ` in `train_recurrent`; headline |

---

### Task 1: Engine — decide-time potential snapshot

**Files:** Modify `src/wave_net/{neurons,wave,network}.rs`.

**Interfaces:** Produces `Layer.decide_potential` (public), `Network::layer_decide_potential(z) -> Vec<i16>`.

- [ ] **Step 1: Field + init**

`neurons.rs`, add to `Layer` (after `elig_post`):
```rust
    pub decide_potential: Vec<i16>, // potential at the decide step (pre fire-reset/leak); per-wave snapshot
```
Init in `Layer::new` returned struct: `decide_potential: vec![0; ls],`.

- [ ] **Step 2: Snapshot in the decide loop**

`wave.rs`, in the decide loop, add the snapshot as the first line of the `for i in 0..ls` body (before the
`elig_post`/fire check, so it captures the pre-reset value for every neuron):
```rust
    for i in 0..ls {
        layer.decide_potential[i] = layer.potential[i];
        let eff = layer.threshold[i] as i32 + (layer.adapt[i] >> ADAPT_SHIFT);
        // …unchanged: elig_post box, fire check…
    }
```

- [ ] **Step 3: reset_state + accessor**

`network.rs` `reset_state`, after the `elig_post` line:
```rust
            g.decide_potential.iter_mut().for_each(|p| *p = 0);
```
Add an accessor near `layer_thresholds`:
```rust
    /// Per-neuron membrane potential captured at the last decide step (pre fire-reset/leak).
    pub fn layer_decide_potential(&self, z: usize) -> Vec<i16> {
        self.layers[z].lock().unwrap().decide_potential.clone()
    }
```

- [ ] **Step 4: Unit test**

`wave.rs` tests:
```rust
    #[test]
    fn decide_potential_snapshots_pre_reset() {
        let mut l = low_layer(4, 5, 2, vec![]); // threshold 5, no outgoing topology
        for c in l.cooldown.iter_mut() {
            *c = 0;
        }
        l.potential[0] = 8; // fires (>= 5)
        l.potential[1] = 3; // sub-threshold (< 5)
        let mut acc = vec![0i32; 16];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
        assert_eq!(l.decide_potential[0], 8, "fired neuron's decide_potential is the pre-reset value");
        assert_eq!(l.potential[0], 0, "fired neuron's potential reset post-wave");
        assert_eq!(l.decide_potential[1], 3, "sub-threshold neuron's decide_potential is its charge");
    }
```

- [ ] **Step 5: Run + commit**

Run: `cargo test` and `cargo build` (all green — existing behavior unchanged; warning-free).
`git commit -m "feat: snapshot decide-time membrane potential per wave (Layer.decide_potential)"`

---

### Task 2: Bench — sub-threshold ψ + the A/B headline

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Produces `RsnnConfig.subthreshold_psi: bool`; `xor_trial_layers` returns decide-potentials; `train_recurrent` uses sub-threshold `ψ` when the flag is set.

- [ ] **Step 1: Config flag**

Add to `RsnnConfig` (after `back_radius`): `pub subthreshold_psi: bool,`; set `subthreshold_psi: false,` in
`demo()`.

- [ ] **Step 2: Record decide-potentials in `xor_trial_layers`**

Change its return to `(Vec<f32>, Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<i16>>>)` and record decide-potentials per
layer after each wave. Replace the body's drive+record so every `net.wave(&x)` is followed by a
decide-potential read; keep the listener for spikes. Concretely, add `let mut pots: Vec<Vec<Vec<i16>>> =
vec![Vec::new(); l];` after `reset_state`, and after **each** `net.wave(...)` call (in all four phases) add
`for z in 1..l { pots[z].push(net.layer_decide_potential(z)); }`. Return `(act, spikes, pots)`.
(Simplest: replace the `present` closure with explicit `for w in 0..cfg.present_waves { let sites = …;
net.wave(&sites); for z in 1..l { pots[z].push(net.layer_decide_potential(z)); } }`, and likewise add the
`pots` push to the delay and read loops.)

- [ ] **Step 3: Build ψ and use it in `train_recurrent`**

In `train_recurrent`: update both call sites to `let (act, spikes, pots) = xor_trial_layers(…)` (held-out
call: `let (act, _, _) = …`). Read thresholds once before the loop:
`let theta: Vec<Vec<f32>> = (0..l).map(|z| net.layer_thresholds(z.max(1)).iter().map(|&t| (t as f32).max(1.0)).collect()).collect();`
(index 0 unused). After building `fired[z][t][i]`, build the postsynaptic factor:
```rust
        // postsynaptic factor: spike-time ψ, or sub-threshold ψ = clamp(decide_potential / θ, 0, 1)
        let mut post = vec![vec![vec![0f32; ls]; ttot]; l];
        for z in 1..l {
            for tt in 0..ttot {
                for j in 0..ls {
                    post[z][tt][j] = if cfg.subthreshold_psi {
                        (pots[z][tt][j] as f32 / theta[z][j]).clamp(0.0, 1.0)
                    } else {
                        fired[z][tt][j]
                    };
                }
            }
        }
```
Then in the eligibility inner loop, replace `fired[tz][tt][j]` with `post[tz][tt][j]`:
```rust
                        for tt in 0..ttot {
                            e += pretr[z][tt][i] * post[tz][tt][j];
                        }
```
(`pretr` still built from `fired` — the *pre*synaptic trace stays spike-based; only the *post* factor gains
sub-threshold sensitivity.)

- [ ] **Step 4: Determinism test**

```rust
    #[test]
    fn subthreshold_psi_is_deterministic() {
        let mut cfg = RsnnConfig::demo();
        cfg.size = 16;
        cfg.layers = 4;
        cfg.back_count = 8;
        cfg.subthreshold_psi = true;
        cfg.adapt_bump = 0;
        cfg.delay = 20;
        cfg.trials = 300;
        assert_eq!(train_recurrent(&cfg), train_recurrent(&cfg));
    }
```

- [ ] **Step 5: The three-way headline**

```rust
    #[test]
    #[ignore] // expensive; run manually in --release
    fn subthreshold_psi_vs_spike_psi_on_temporal_xor() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mk = |s: u64| {
            let mut c = RsnnConfig::demo();
            c.seed = s;
            c.task_seed = s;
            c.size = 16;
            c.layers = 4;
            c.adapt_bump = 0;
            c.delay = 20;
            c.trials = 1500;
            c
        };
        let (mut best_ff, mut best_spk, mut best_sub) = (0u64, 0u64, 0u64);
        for &s in &seeds {
            let ff = mk(s);
            let mut spk = ff.clone();
            spk.back_count = 8;
            let mut sub = spk.clone();
            sub.subthreshold_psi = true;
            let (fa, sa, ua) = (train_recurrent(&ff), train_recurrent(&spk), train_recurrent(&sub));
            eprintln!("seed {s:#x}  FF {fa}  backward+spikeψ {sa}  backward+subψ {ua}");
            best_ff = best_ff.max(fa);
            best_spk = best_spk.max(sa);
            best_sub = best_sub.max(ua);
        }
        eprintln!("best  FF {best_ff}  spikeψ {best_spk}  subψ {best_sub}");
        assert!(best_sub >= 485, "sanity; verdict is the printed comparison (Step 6)");
    }
```

- [ ] **Step 6: Run (release) + honest verdict**

Run: `cargo test bench::rsnn::tests::subthreshold_psi_vs_spike_psi_on_temporal_xor --release -- --nocapture 2>&1 | grep -E "seed 0x|best"`
Read FF / spike-ψ / sub-ψ. Tune only `subthreshold_psi`'s companions (`rec_tau`, `hidden_lr`, `back_count`).
**Two honest outcomes:**
- **sub-ψ beats FF and spike-ψ** (clears chance+margin): the credit-during-gaps fix worked — recurrence
  learns. Tighten the assertion to `best_sub > best_ff + 100 && best_sub > best_spk + 80` and record the win.
- **sub-ψ still nulls:** confound removed ⇒ the blocker is the *sustaining dynamics* (leak kills the trace)
  or needs BPTT, not the credit rule. Keep the loose assertion; document the narrowed conclusion.

Multi-seed; no fudging; the printed table is the evidence.

- [ ] **Step 7: Full suite + commit + document**

Run: `cargo test` and `cargo build` (all green, warning-free).
`git commit -m "feat: sub-threshold psi for the temporal eligibility (A/B vs spike-psi on temporal XOR)"`
Append the FF / spike-ψ / sub-ψ table + verdict to `docs/experiments_results.md`; commit the doc.

---

## Self-review

**Spec coverage:** decide-time potential snapshot + accessor (Task 1); `ψ = clamp(v/θ,0,1)` from it, behind
`subthreshold_psi`, in the temporal eligibility (Task 2); isolated A/B FF vs spike-ψ vs sub-ψ on the exact
backward config, held-out + multi-seed + honesty gate (Task 2). No firing-dynamics change; `false` path
unchanged. Deferred items untouched.

**Placeholder scan:** none — full code and commands. Step 2's edit is described precisely (push
`layer_decide_potential` after each `net.wave` in all four phases).

**Type consistency:** `Layer.decide_potential: Vec<i16>`; `layer_decide_potential(z) -> Vec<i16>`;
`xor_trial_layers -> (Vec<f32>, Vec<Vec<Vec<u32>>>, Vec<Vec<Vec<i16>>>)` (both call sites updated); `theta:
Vec<Vec<f32>>` (max(1.0) guards div-by-zero); `post: Vec<Vec<Vec<f32>>>` replaces `fired` only in the
postsynaptic slot of the eligibility. `subthreshold_psi` set in `demo()`.
