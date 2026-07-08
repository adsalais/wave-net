# Recurrence via trained level-0 weights + temporal e-prop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Show `wave_net` recurrence earns its keep: on **temporal XOR** (which a feed-forward net can't do), trained `level 0` lateral weights + a temporal (per-wave) e-prop eligibility lift accuracy above the FF baseline — held-out, multi-seed.

**Architecture:** All in `bench::rsnn` (no engine change — `level 0` topology already supported, per-wave spikes via listeners). Task 1 builds temporal XOR and *verifies FF is ~chance*. Task 2 adds level-0 recurrence + a temporal eligibility computed from recorded spikes.

**Tech Stack:** Rust edition 2024, std only. `f32` shadow/eligibility in the bench. Inline `#[cfg(test)]` tests.

## Global Constraints

- Std only in the engine; **no `unsafe`**; **warning-free**.
- Determinism: pure function of `(seed, task_seed, config)`; single-threaded.
- **One commit per task**, conventional commits, **no `Co-Authored-By`**, **never push**.
- On branch `feat/rsnn-recurrence`. Verify each task with `cargo test` + warning-free `cargo build`.
- Whole existing suite (incl. `wave_state_machine`) stays green.

## File structure

| File | Change |
|---|---|
| `src/bench/rsnn.rs` | `target_of` gains a `level` arg; XOR config fields + `engine_config_xor`; `pick_ab`; `xor_trial`; `train_xor`; tests |

---

### Task 1: Temporal XOR task + verify the FF baseline is ~chance

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Produces `RsnnConfig.{rec_count, rec_radius, rec_tau}`, `engine_config_xor`, `pick_ab`, `xor_trial`, `train_xor` (recurrence off when `rec_count == 0`).

- [ ] **Step 1: Add XOR config fields + a per-layer XOR engine config**

Add to `RsnnConfig` (after `hidden_lr`):
```rust
    pub rec_count: u32,  // level-0 lateral synapses per neuron (0 = feed-forward, no recurrence)
    pub rec_radius: u32, // level-0 recurrence radius
    pub rec_tau: f32,    // presynaptic-trace decay time constant (waves) for the temporal eligibility
```
Set in `demo()`: `rec_count: 0, rec_radius: 2, rec_tau: 4.0,`.
Add a method (the XOR net is L0 input-transducer + one recurrent hidden layer L1; readout reads L1):
```rust
    fn engine_config_xor(&self) -> Config {
        use crate::wave_net::synapse::TopologyLevel;
        // L0: feed the input up to L1 (level+1). L1: lateral recurrence within itself (level 0), or empty
        // when rec_count == 0 (the feed-forward baseline).
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: self.up_radius, count: self.up_count }],
            leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32,
            baseline_init: 6, adapt_bump: 20, adapt_decay: 6,
        };
        let l1_topo = if self.rec_count > 0 {
            vec![TopologyLevel { level: 0, radius: self.rec_radius, count: self.rec_count }]
        } else {
            vec![]
        };
        let l1 = LayerConfig { topology: l1_topo, ..l0.clone() };
        Config { seed: self.seed, size: self.size, layers: vec![l0, l1] }
    }
```

- [ ] **Step 2: `pick_ab`, `xor_trial`, and generalize `target_of`**

Generalize `target_of` to take the topology level, and update the existing call in `train_eprop` from
`target_of(cfg.seed, sg, i as u32, kk as u32, cfg.up_radius, cfg.size)` to pass level `1`:
```rust
fn target_of(seed: u64, source_global: u32, src_local: u32, level: i32, k: u32, radius: u32, size: u32) -> u32 {
    let (sx, sy) = xy_of(src_local, size);
    let h = mix(key(seed, source_global, level, k, P_TARGET));
    let span = 2 * radius + 1;
    let dx = map_range24((h >> 40) as u32, span) as i32 - radius as i32;
    let dy = map_range24(((h >> 16) as u32) & 0x00FF_FFFF, span) as i32 - radius as i32;
    local_of(wrap(sx, dx, size), wrap(sy, dy, size), size)
}
```
(Update the `train_eprop` call site to `target_of(cfg.seed, sg, i as u32, 1, kk as u32, cfg.up_radius, cfg.size)`.)

Add the XOR task. `pick_ab` draws two independent bits; `xor_trial` runs the four phases and records L1's
per-wave fired-sets, returning the **read-window** L1 activity (readout input) and the per-wave record
(eligibility input):
```rust
/// Two independent input bits for trial `t` (deterministic).
fn pick_ab(seed: u64, t: usize) -> (usize, usize) {
    let a = (mix(key(seed, t as u32, 0, 0, 51)) & 1) as usize;
    let b = (mix(key(seed, t as u32, 0, 0, 53)) & 1) as usize;
    (a, b)
}

/// reset → present cue(a) → delay → present cue(b) → read (silent). Records L1 per-wave fired-sets and
/// returns (read-window L1 spike counts, per-wave L1 fired-sets over the whole trial).
fn xor_trial(net: &mut Network, cfg: &RsnnConfig, a: usize, b: usize, trial: usize) -> (Vec<f32>, Vec<Vec<u32>>) {
    let ls = (cfg.size * cfg.size) as usize;
    let rec: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new())); // [wave] = fired locals in L1
    {
        let r = rec.clone();
        net.on_layer(1, Box::new(move |_w, fired: &[u32]| r.lock().unwrap().push(fired.to_vec())));
    }
    net.reset_state();
    let mut present = |net: &mut Network, class: usize, phase: usize| {
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
    let read_start = rec.lock().unwrap().len();
    for _ in 0..cfg.read_waves {
        net.wave(&[]);
    }
    net.clear_listeners();
    let waves = rec.lock().unwrap().clone();
    // read-window L1 spike counts (last read_waves waves)
    let mut act = vec![0f32; ls];
    for wave in waves.iter().skip(read_start) {
        for &loc in wave {
            act[loc as usize] += 1.0;
        }
    }
    (act, waves)
}
```

- [ ] **Step 3: `train_xor` (readout on L1; recurrence off in this task)**

```rust
/// Train a readout on L1's read-window activity for temporal XOR; with rec_count>0 also trains the L1
/// level-0 recurrent weights (Task 2). Returns held-out test accuracy permille.
pub fn train_xor(cfg: &RsnnConfig) -> u64 {
    let mut net = Network::new(cfg.engine_config_xor());
    net.calibrate(&cfg.calib, &random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16));
    let ls = (cfg.size * cfg.size) as usize;
    let mut w = vec![vec![0f32; ls]; 2]; // 2-class readout on L1
    let score = |w: &[Vec<f32>], a: &[f32]| -> Vec<f32> {
        (0..2).map(|c| w[c].iter().zip(a).map(|(wi, ai)| wi * ai).sum()).collect()
    };
    for t in 0..cfg.trials {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let label = a ^ b;
        let (act, waves) = xor_trial(&mut net, cfg, a, b, t);
        let p = softmax(&score(&w, &act));
        let err: Vec<f32> = (0..2).map(|c| p[c] - if c == label { 1.0 } else { 0.0 }).collect();
        for c in 0..2 {
            for j in 0..ls {
                w[c][j] -= cfg.readout_lr * err[c] * act[j];
            }
        }
        if cfg.rec_count > 0 {
            recurrent_update(&mut net, cfg, &w, &err, &waves); // Task 2
        }
    }
    let mut correct = 0usize;
    let holdout = 400usize;
    for t in cfg.trials..cfg.trials + holdout {
        let (a, b) = pick_ab(cfg.task_seed, t);
        let (act, _) = xor_trial(&mut net, cfg, a, b, t);
        let s = score(&w, &act);
        let pred = if s[1] > s[0] { 1 } else { 0 };
        if pred == (a ^ b) {
            correct += 1;
        }
    }
    (correct as u64 * 1000) / holdout as u64
}
```
For Task 1, add a no-op stub so it compiles: `fn recurrent_update(_: &mut Network, _: &RsnnConfig, _: &[Vec<f32>], _: &[f32], _: &[Vec<u32>]) {}` (replaced in Task 2).

- [ ] **Step 4: Test — FF baseline is ~chance**

```rust
    #[test]
    fn temporal_xor_ff_is_near_chance() {
        // Feed-forward (rec_count = 0) cannot hold A across the gap and cannot do XOR -> ~chance.
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mut best = 0u64;
        for &s in &seeds {
            let mut cfg = RsnnConfig::demo();
            cfg.seed = s;
            cfg.task_seed = s;
            cfg.trials = 1500;
            let acc = train_xor(&cfg);
            eprintln!("FF temporal-XOR seed {s:#x}  held-out {acc}");
            best = best.max(acc);
        }
        assert!(best < 640, "feed-forward should NOT solve temporal XOR (best {best})");
    }
```

- [ ] **Step 5: Run + honesty check**

Run: `cargo test bench::rsnn::tests::temporal_xor_ff_is_near_chance -- --nocapture`
Expected: all seeds near chance (best < 640). **If FF already solves it** (adaptation holds A), stop and
report — make the task harder (longer `delay`, lower `adapt_bump`) so it genuinely requires recurrence,
before Task 2. Full suite green, warning-free.

- [ ] **Step 6: Commit** — `git commit -m "feat: temporal XOR benchmark; feed-forward baseline is ~chance (needs recurrence)"`

---

### Task 2: level-0 recurrence + temporal eligibility lifts XOR

**Files:** Modify `src/bench/rsnn.rs`.

**Interfaces:** Replaces the `recurrent_update` stub; produces the headline `recurrence_lifts_temporal_xor` test.

- [ ] **Step 1: Write the failing headline test**

```rust
    #[test]
    fn recurrence_lifts_temporal_xor() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        let mut worst_rec = 1000u64;
        for &s in &seeds {
            let mut ff = RsnnConfig::demo();
            ff.seed = s; ff.task_seed = s; ff.trials = 2000;
            let mut rc = ff.clone();
            rc.rec_count = 12; // turn on level-0 recurrence
            let ff_acc = train_xor(&ff);
            let rc_acc = train_xor(&rc);
            eprintln!("seed {s:#x}  FF {ff_acc}  +recurrence {rc_acc}");
            worst_rec = worst_rec.min(rc_acc);
        }
        assert!(worst_rec > 640, "recurrence lifts temporal XOR above chance across seeds (worst {worst_rec})");
    }
```

- [ ] **Step 2: Implement `recurrent_update` (temporal eligibility)**

Replace the stub. For the L1 level-0 recurrent synapses: build a decaying presynaptic trace per neuron over
the recorded waves, correlate with postsynaptic spikes (`e_ij = Σ_t pre_trace_i(t)·fired_j(t)`), and update
the stored weights via the symmetric-feedback learning signal.
```rust
fn recurrent_update(net: &mut Network, cfg: &RsnnConfig, w: &[Vec<f32>], err: &[f32], waves: &[Vec<u32>]) {
    let ls = (cfg.size * cfg.size) as usize;
    let up = cfg.rec_count as usize;
    let ttot = waves.len();
    // per-wave fired matrix F[t][j] and decaying pre-trace TR[t][i]
    let mut fired = vec![vec![0f32; ls]; ttot];
    for (t, wv) in waves.iter().enumerate() {
        for &loc in wv {
            fired[t][loc as usize] = 1.0;
        }
    }
    let decay = 1.0 - 1.0 / cfg.rec_tau.max(1.0);
    let mut tr = vec![vec![0f32; ls]; ttot];
    for i in 0..ls {
        let mut trace = 0f32;
        for t in 0..ttot {
            trace = trace * decay + fired[t][i];
            tr[t][i] = trace;
        }
    }
    // learning signal per L1 neuron j (symmetric feedback from the readout)
    let l_sig: Vec<f32> = (0..ls).map(|j| (0..2).map(|c| w[c][j] * err[c]).sum()).collect();
    net.with_layer_mut(1, |l1| {
        for i in 0..ls {
            let sg = (ls + i) as u32; // L1 global id = layer 1 * ls + i
            for kk in 0..up {
                let j = target_of(cfg.seed, sg, i as u32, 0, kk as u32, cfg.rec_radius, cfg.size) as usize;
                // e_ij = Σ_t pre_trace_i(t) · fired_j(t)
                let mut e = 0f32;
                for t in 0..ttot {
                    e += tr[t][i] * fired[t][j];
                }
                l1.out_shadow[i * up + kk] += -cfg.hidden_lr * l_sig[j] * e;
            }
        }
        for (wq, s) in l1.out_weights.iter_mut().zip(&l1.out_shadow) {
            *wq = s.round().clamp(-127.0, 127.0) as i8;
        }
    });
}
```

- [ ] **Step 3: Run + tune (do not fudge)**

Run: `cargo test bench::rsnn::tests::recurrence_lifts_temporal_xor -- --nocapture`
Read the FF vs +recurrence numbers per seed. Tune **only** `rec_count`, `rec_radius`, `rec_tau`,
`hidden_lr`, `trials`, `delay`. Expected: recurrence-on clears chance (worst > 640) while FF stays ~chance.

**Honesty gate:** temporal XOR is hard; the spike-timing eligibility (`fired_j` as `ψ`) is crude. If
recurrence can't beat the FF baseline after reasonable tuning, **stop and report** — that localizes the gap
to the eligibility/`ψ` approximation or the need for `level −1`/BPTT. A documented null is the result; never
a single seed or a fudged threshold.

- [ ] **Step 4: Full suite + commit**

Run: `cargo test` and `cargo build` (all green, warning-free).
`git commit -m "feat: level-0 recurrence + temporal e-prop eligibility lifts temporal XOR over FF"`

---

## Self-review

**Spec coverage:** temporal XOR task + FF-fails verification (Task 1); `level 0` trainable recurrence, no
engine change (Task 2); temporal eligibility `Σ_t pre_trace_i·fired_j` from recorded spikes, symmetric
feedback, quantized shadow (Task 2); held-out + multi-seed + honesty gate (both). `level 0` only; memory-opt
deferred (per-synapse `e` computed transiently from `O(T·ls)` recorded activity — the recompute path).

**Placeholder scan:** Task 1 Step 3's `recurrent_update` is an explicit no-op stub (labelled), replaced in
Task 2 Step 2 with full code. No other placeholders.

**Type consistency:** `target_of(.., level: i32, ..)` updated at its one existing call site; `xor_trial ->
(Vec<f32>, Vec<Vec<u32>>)`; `train_xor -> u64`; `recurrent_update(&mut Network, &RsnnConfig, &[Vec<f32>],
&[f32], &[Vec<u32>])` matches stub and impl. `pick_ab` uses fresh hash purposes (51/53), distinct from
`pick_class` (41). Readout `w: Vec<Vec<f32>>` (2×ls); L1 global id `ls + i`; level-0 slot stride `rec_count`.

**Note:** eligibility uses `fired_j` as the pseudo-derivative (spike-time surrogate) — the crude-but-simple
choice; the honesty gate in Task 2 Step 3 covers its adequacy.
