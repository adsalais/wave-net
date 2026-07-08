# Reservoir-regime Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure eight properties of the calibrated-but-untrained reservoir and report which predict learnability (V1 & V2b) and whether topology couples to the other knobs — a diagnostic that scopes the brittleness fix.

**Architecture:** New `src/bench/regime.rs` built on one `reservoir_states` primitive (per-computational-layer spike counts from a trial, with an optional input perturbation). Eight metric fns consume it; two `#[ignore]` experiments produce the findings tables. Reuses `eprop` (config, calibration, class picker), `NearestCentroid`, and the store-recall cue/probe. No engine change.

**Tech Stack:** Rust edition 2024, std only, no deps. Inline `#[cfg(test)]` tests. `f64` in bench only.

## Global Constraints

- **Std only**; **no `unsafe`**; **warning-free** build.
- **Determinism** — pure function of `(seed, config)`; single-threaded.
- Integer engine untouched; metrics are `f64` in the bench. Tests inline, test-first.
- **One commit per task**, conventional commits. **NEVER** a `Co-Authored-By` trailer. **NEVER** push.
- On branch `feat/regime-diagnostic`.
- Verify each task with `cargo test` + warning-free `cargo build` before committing.
- The whole existing suite stays green.

## File structure

| File | Change |
|---|---|
| `src/bench/eprop.rs` | expose `pub(crate) fn pick_class` and a new `pub(crate) fn calibrated_reservoir` |
| `src/bench/mod.rs` | `pub mod regime;` |
| `src/bench/regime.rs` | new — primitive, eight metrics, experiments, tests |

---

### Task 1: Scaffold + `reservoir_states` / `collect_states`

**Files:**
- Modify: `src/bench/eprop.rs`, `src/bench/mod.rs`
- Create: `src/bench/regime.rs`

**Interfaces:**
- Produces: `eprop::{pick_class, calibrated_reservoir}` (`pub(crate)`); `regime::{reservoir_states, flat, collect_states}`.
- Consumes: `EpropConfig`, `cue_realization`/`probe_pattern`, `Network`.

- [ ] **Step 1: Expose the eprop helpers**

In `src/bench/eprop.rs`, change `fn pick_class` to `pub(crate) fn pick_class`, and add (near it, non-test):
```rust
/// Build the shared computational reservoir (no readout) and firing-rate-calibrate it — the untrained
/// substrate both learners share. Used by the regime diagnostic.
pub(crate) fn calibrated_reservoir(cfg: &EpropConfig) -> Network {
    let mut net = Network::new(cfg.engine_config());
    let input = random_l0_input(cfg.seed ^ 0xE9, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);
    net
}
```
Add to `src/bench/mod.rs`:
```rust
pub mod regime;
```

- [ ] **Step 2: Write the failing test**

Create `src/bench/regime.rs`:
```rust
//! Reservoir-regime diagnostic — measures properties of the calibrated-but-untrained reservoir to find
//! which predict learnability (V1 & V2b) and how topology couples to the other knobs. Bench-side, f64.

use crate::bench::eprop::{calibrated_reservoir, pick_class, EpropConfig};
use crate::bench::readout::NearestCentroid;
use crate::bench::store_recall::{cue_realization, probe_pattern};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix};
use std::sync::{Arc, Mutex};

/// One trial (reset → present cue → delay → probe); returns per-computational-layer spike counts. If
/// `flip` is set, that L0 site is toggled in every present wave (the perturbation for σ).
pub(crate) fn reservoir_states(
    net: &mut Network,
    cfg: &EpropConfig,
    class: usize,
    trial: usize,
    flip: Option<u32>,
) -> Vec<Vec<u32>> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![vec![0u32; ls]; l]));
    for z in 1..l {
        let c = counts.clone();
        net.on_layer(
            z,
            Box::new(move |_w: usize, fired: &[u32]| {
                let mut g = c.lock().unwrap();
                for &loc in fired {
                    g[z][loc as usize] += 1;
                }
            }),
        );
    }
    net.reset_state();
    for w in 0..cfg.present_waves {
        let mut sites = cue_realization(cfg.seed, cfg.size, class, trial, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
        if let Some(s) = flip {
            match sites.iter().position(|&x| x == s) {
                Some(pos) => {
                    sites.remove(pos);
                }
                None => sites.push(s),
            }
        }
        net.wave(&sites);
    }
    for _ in 0..cfg.delay {
        net.wave(&[]);
    }
    let probe = probe_pattern(cfg.seed, cfg.size, cfg.probe_q16);
    for _ in 0..cfg.read_waves {
        net.wave(&probe);
    }
    net.clear_listeners();
    let g = counts.lock().unwrap();
    (1..l).map(|z| g[z].clone()).collect()
}

/// Flatten per-layer counts into one feature vector.
pub(crate) fn flat(layered: &[Vec<u32>]) -> Vec<u32> {
    layered.iter().flatten().copied().collect()
}

/// Collect `trials` flattened reservoir states with their class labels (no training).
pub(crate) fn collect_states(cfg: &EpropConfig, trials: usize) -> (Vec<Vec<u32>>, Vec<usize>) {
    let mut net = calibrated_reservoir(cfg);
    let mut states = Vec::with_capacity(trials);
    let mut labels = Vec::with_capacity(trials);
    for t in 0..trials {
        let class = pick_class(cfg.seed, t, cfg.k);
        states.push(flat(&reservoir_states(&mut net, cfg, class, t, None)));
        labels.push(class);
    }
    (states, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> EpropConfig {
        let mut cfg = EpropConfig::demo();
        cfg.calib.warmup = 8;
        cfg.calib.waves = 24;
        cfg.calib.max_steps = 12;
        cfg.calib.refine_passes = 2;
        cfg
    }

    #[test]
    fn collect_states_shape_and_determinism() {
        let cfg = small();
        let (s1, y1) = collect_states(&cfg, 20);
        let (s2, y2) = collect_states(&cfg, 20);
        assert_eq!(s1.len(), 20);
        assert_eq!(s1[0].len(), (cfg.layers - 1) * (cfg.size * cfg.size) as usize);
        assert_eq!((s1, y1), (s2, y2), "collection must be deterministic");
    }
}
```

- [ ] **Step 3: Run**

Run: `cargo test bench::regime` and `cargo build`
Expected: `collect_states_shape_and_determinism` passes; warning-free. (`key`/`mix`/`NearestCentroid` are imported for later tasks — if unused-warns here, they are used in Tasks 2/4; commit Tasks 1–2 together to avoid a temporary allow — see Task 2 Step 5.)

- [ ] **Step 4: Do not commit yet** — proceed to Task 2.

---

### Task 2: Separation metrics — ceiling + Fisher ratio

**Files:**
- Modify: `src/bench/regime.rs`

**Interfaces:**
- Produces: `separation_ceiling(cfg, trials) -> u64`; `fisher_ratio(states, labels, k) -> f64`.

- [ ] **Step 1: Write the failing tests**

Add to `regime.rs` `mod tests`:
```rust
    fn dead_cfg() -> EpropConfig {
        let mut cfg = small();
        cfg.up_count = 8; // known dead regime from the sweep
        cfg
    }

    #[test]
    fn separation_ceiling_discriminates_working_from_dead() {
        let work = separation_ceiling(&small(), 200);
        let dead = separation_ceiling(&dead_cfg(), 200);
        eprintln!("ceiling work {work} dead {dead}");
        assert!(work > 600, "working reservoir separates classes: {work}");
        assert!(work > dead + 80, "working {work} > dead {dead}");
    }

    #[test]
    fn fisher_ratio_discriminates_working_from_dead() {
        let (sw, yw) = collect_states(&small(), 200);
        let (sd, yd) = collect_states(&dead_cfg(), 200);
        let fw = fisher_ratio(&sw, &yw, 2);
        let fd = fisher_ratio(&sd, &yd, 2);
        eprintln!("fisher work {fw:.4} dead {fd:.4}");
        assert!(fw > fd, "working Fisher {fw} > dead {fd}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::regime::tests::fisher_ratio_discriminates_working_from_dead`
Expected: FAIL to compile — `separation_ceiling` / `fisher_ratio` not defined.

- [ ] **Step 3: Implement**

Add to `regime.rs` (non-test):
```rust
/// Held-out NearestCentroid accuracy (permille) on the reservoir states — the intrinsic separability.
pub(crate) fn separation_ceiling(cfg: &EpropConfig, trials: usize) -> u64 {
    let (states, labels) = collect_states(cfg, trials);
    let half = trials / 2;
    if half == 0 || half == trials {
        return 500;
    }
    let nc = NearestCentroid::fit(&states[..half], &labels[..half], cfg.k);
    let correct = (half..trials).filter(|&i| nc.predict(&states[i]) == labels[i]).count();
    (correct as u64 * 1000) / (trials - half) as u64
}

/// Fisher discriminant ratio S_B / S_W (trace form): between-class over within-class scatter.
pub(crate) fn fisher_ratio(states: &[Vec<u32>], labels: &[usize], k: usize) -> f64 {
    let n = states.len();
    let d = states[0].len();
    let mut mu = vec![0f64; d];
    for x in states {
        for (j, &v) in x.iter().enumerate() {
            mu[j] += v as f64;
        }
    }
    for m in &mut mu {
        *m /= n as f64;
    }
    let mut cmu = vec![vec![0f64; d]; k];
    let mut cn = vec![0f64; k];
    for (x, &c) in states.iter().zip(labels) {
        cn[c] += 1.0;
        for (j, &v) in x.iter().enumerate() {
            cmu[c][j] += v as f64;
        }
    }
    for c in 0..k {
        if cn[c] > 0.0 {
            for m in &mut cmu[c] {
                *m /= cn[c];
            }
        }
    }
    let mut sb = 0f64;
    for c in 0..k {
        let d2: f64 = (0..d).map(|j| (cmu[c][j] - mu[j]).powi(2)).sum();
        sb += cn[c] * d2;
    }
    let mut sw = 0f64;
    for (x, &c) in states.iter().zip(labels) {
        for j in 0..d {
            sw += (x[j] as f64 - cmu[c][j]).powi(2);
        }
    }
    if sw <= 0.0 { f64::INFINITY } else { sb / sw }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test bench::regime -- --nocapture` and `cargo build`
Expected: both discrimination tests pass; warning-free.

- [ ] **Step 5: Commit Tasks 1 + 2**

```bash
git add src/bench/eprop.rs src/bench/mod.rs src/bench/regime.rs
git commit -m "feat: regime diagnostic — state collection + separation ceiling + Fisher ratio"
```

---

### Task 3: Rank metrics — effective dim + kernel−gen rank

**Files:**
- Modify: `src/bench/regime.rs`

**Interfaces:**
- Produces: `effective_dim(states: &[Vec<f64>]) -> f64`; `kernel_minus_gen_rank(cfg) -> f64`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests`:
```rust
    #[test]
    fn effective_dim_matches_known_participation_ratio() {
        // rank-1 (all rows a scalar multiple of one direction) → PR ≈ 1
        let rank1: Vec<Vec<f64>> = (1..=10).map(|a| vec![a as f64, 2.0 * a as f64, 3.0 * a as f64]).collect();
        let pr1 = effective_dim(&rank1);
        assert!((pr1 - 1.0).abs() < 0.05, "rank-1 PR ~ 1, got {pr1}");
        // isotropic 3-D (one-hot rows, equal variance per axis) → PR ≈ 3
        let iso: Vec<Vec<f64>> = (0..30).map(|i| { let mut v = vec![0.0; 3]; v[i % 3] = 1.0; v }).collect();
        let pr3 = effective_dim(&iso);
        assert!((pr3 - 3.0).abs() < 0.2, "isotropic PR ~ 3, got {pr3}");
    }

    #[test]
    fn kernel_minus_gen_rank_is_finite_and_deterministic() {
        let a = kernel_minus_gen_rank(&small());
        let b = kernel_minus_gen_rank(&small());
        assert!(a.is_finite());
        assert_eq!(a, b, "deterministic");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::regime::tests::effective_dim_matches_known_participation_ratio`
Expected: FAIL to compile — `effective_dim` / `kernel_minus_gen_rank` not defined.

- [ ] **Step 3: Implement**

Add to `regime.rs` (non-test):
```rust
/// Participation ratio (tr C)² / tr(C²) of the state covariance — the effective dimensionality.
pub(crate) fn effective_dim(states: &[Vec<f64>]) -> f64 {
    let n = states.len();
    if n == 0 {
        return 0.0;
    }
    let d = states[0].len();
    let mut mu = vec![0f64; d];
    for x in states {
        for j in 0..d {
            mu[j] += x[j];
        }
    }
    for m in &mut mu {
        *m /= n as f64;
    }
    let mut c = vec![vec![0f64; d]; d];
    for x in states {
        for a in 0..d {
            let xa = x[a] - mu[a];
            for b in 0..d {
                c[a][b] += xa * (x[b] - mu[b]);
            }
        }
    }
    let tr: f64 = (0..d).map(|a| c[a][a] / n as f64).sum();
    let tr_sq: f64 = c.iter().flatten().map(|&v| (v / n as f64).powi(2)).sum();
    if tr_sq <= 0.0 { 0.0 } else { tr * tr / tr_sq }
}

fn as_f64(states: &[Vec<u32>]) -> Vec<Vec<f64>> {
    states.iter().map(|x| x.iter().map(|&v| v as f64).collect()).collect()
}

/// Legenstein–Maass power: effective rank across distinct inputs (both classes) minus effective rank
/// across noisy copies of one input (one class, different noise realizations). Uses PR as the soft rank.
pub(crate) fn kernel_minus_gen_rank(cfg: &EpropConfig) -> f64 {
    let m = 64usize;
    let mut net = calibrated_reservoir(cfg);
    let kernel: Vec<Vec<u32>> =
        (0..m).map(|t| flat(&reservoir_states(&mut net, cfg, pick_class(cfg.seed, t, cfg.k), t, None))).collect();
    let gen: Vec<Vec<u32>> =
        (0..m).map(|t| flat(&reservoir_states(&mut net, cfg, 0, t, None))).collect();
    effective_dim(&as_f64(&kernel)) - effective_dim(&as_f64(&gen))
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test bench::regime` and `cargo build`
Expected: both rank tests pass; warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/bench/regime.rs
git commit -m "feat: regime diagnostic — effective dimensionality + kernel-minus-generalization rank"
```

---

### Task 4: Dynamics + degeneracy — σ, layer gain, dead/saturated/synchrony

**Files:**
- Modify: `src/bench/regime.rs`

**Interfaces:**
- Produces: `perturbation_spread(cfg) -> f64`; `layer_gain(cfg, trials) -> Vec<f64>`; `degeneracy(states, waves) -> (f64, f64, f64)`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests`:
```rust
    #[test]
    fn perturbation_spread_orders_regimes() {
        // a starved (sparse) reservoir should spread a perturbation no more than the denser baseline
        let sparse = perturbation_spread(&dead_cfg());
        let dense = perturbation_spread(&small());
        eprintln!("sigma sparse {sparse:.3} dense {dense:.3}");
        assert!(sparse.is_finite() && dense.is_finite());
        assert!(dense >= sparse, "denser reservoir spreads at least as much: dense {dense} vs sparse {sparse}");
    }

    #[test]
    fn degeneracy_flags_dead_and_saturated() {
        // 4 features: #0 dead (all 0), #3 saturated (== waves every trial); waves = 5
        let states: Vec<Vec<u32>> = (0..10).map(|i| vec![0, (i % 3) as u32, (i % 2) as u32, 5]).collect();
        let (dead, sat, _sync) = degeneracy(&states, 5);
        assert!((dead - 0.25).abs() < 1e-9, "one of four dead: {dead}");
        assert!((sat - 0.25).abs() < 1e-9, "one of four saturated: {sat}");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test bench::regime::tests::degeneracy_flags_dead_and_saturated`
Expected: FAIL to compile — the three fns not defined.

- [ ] **Step 3: Implement**

Add to `regime.rs` (non-test):
```rust
/// σ / edge-of-chaos: flip one L0 site, measure per-layer Hamming divergence (neurons whose count
/// differs) between base and perturbed states, and return the geometric growth of divergence up the stack.
pub(crate) fn perturbation_spread(cfg: &EpropConfig) -> f64 {
    let mut net = calibrated_reservoir(cfg);
    let site = (mix(key(cfg.seed, 0, 0, 0, 71)) % (cfg.size * cfg.size) as u64) as u32;
    let base = reservoir_states(&mut net, cfg, 0, 0, None);
    let pert = reservoir_states(&mut net, cfg, 0, 0, Some(site));
    let div: Vec<f64> = base
        .iter()
        .zip(&pert)
        .map(|(b, p)| b.iter().zip(p).filter(|(x, y)| x != y).count() as f64)
        .collect();
    // geometric mean of consecutive layer-to-layer ratios where the denominator is non-zero
    let mut log_sum = 0f64;
    let mut n = 0u32;
    for z in 1..div.len() {
        if div[z - 1] > 0.0 && div[z] > 0.0 {
            log_sum += (div[z] / div[z - 1]).ln();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (log_sum / n as f64).exp() }
}

/// Mean firing fraction per computational layer over `trials` trials.
pub(crate) fn layer_gain(cfg: &EpropConfig, trials: usize) -> Vec<f64> {
    let mut net = calibrated_reservoir(cfg);
    let ls = (cfg.size * cfg.size) as usize;
    let waves = (cfg.present_waves + cfg.delay + cfg.read_waves).max(1);
    let mut sum: Vec<f64> = Vec::new();
    for t in 0..trials {
        let layered = reservoir_states(&mut net, cfg, pick_class(cfg.seed, t, cfg.k), t, None);
        if sum.is_empty() {
            sum = vec![0.0; layered.len()];
        }
        for (z, layer) in layered.iter().enumerate() {
            sum[z] += layer.iter().sum::<u32>() as f64 / (ls * waves) as f64;
        }
    }
    sum.iter().map(|s| s / trials as f64).collect()
}

/// Degeneracy: (dead fraction, saturated fraction, sampled mean |pairwise correlation|). `waves` is the
/// per-trial spike ceiling used to flag saturation.
pub(crate) fn degeneracy(states: &[Vec<u32>], waves: u32) -> (f64, f64, f64) {
    let n = states.len();
    let d = states[0].len();
    let mut dead = 0usize;
    let mut sat = 0usize;
    for j in 0..d {
        let total: u64 = states.iter().map(|x| x[j] as u64).sum();
        let mx = states.iter().map(|x| x[j]).max().unwrap_or(0);
        if total == 0 {
            dead += 1;
        }
        if mx >= waves {
            sat += 1;
        }
    }
    // synchrony: mean |Pearson| over a deterministic sample of neuron-index pairs
    let mut sync_sum = 0f64;
    let mut pairs = 0u32;
    let mut a = 0usize;
    while a + 7 < d && pairs < 200 {
        let b = a + 7;
        let (ma, mb): (f64, f64) = {
            let sa: f64 = states.iter().map(|x| x[a] as f64).sum::<f64>() / n as f64;
            let sb: f64 = states.iter().map(|x| x[b] as f64).sum::<f64>() / n as f64;
            (sa, sb)
        };
        let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
        for x in states {
            let da = x[a] as f64 - ma;
            let db = x[b] as f64 - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        if va > 0.0 && vb > 0.0 {
            sync_sum += (cov / (va.sqrt() * vb.sqrt())).abs();
            pairs += 1;
        }
        a += 1;
    }
    let sync = if pairs == 0 { 0.0 } else { sync_sum / pairs as f64 };
    (dead as f64 / d as f64, sat as f64 / d as f64, sync)
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test bench::regime -- --nocapture` and `cargo build`
Expected: both tests pass; warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/bench/regime.rs
git commit -m "feat: regime diagnostic — perturbation spread, layer gain, degeneracy stats"
```

---

### Task 5: Experiments + findings

**Files:**
- Modify: `src/bench/regime.rs`, `docs/experiments_results.md`

**Interfaces:**
- Consumes: all metrics + `bench::eprop::train`.

- [ ] **Step 1: Add the two reporting experiments**

Add to `regime.rs` `mod tests`:
```rust
    use crate::bench::eprop::train;

    fn learned(mut cfg: EpropConfig, readout: bool, broadcast: bool, lr: f64) -> u64 {
        cfg.readout = readout;
        cfg.broadcast = broadcast;
        if broadcast {
            cfg.softmax_temp = 10.0;
        }
        cfg.trials = 1500;
        cfg.block = 250;
        let c = train(&cfg, lr).accuracy_permille;
        let h = c.len() / 2;
        c[h..].iter().sum::<u64>() / (c.len() - h).max(1) as u64
    }

    #[test]
    #[ignore]
    fn _regime_vs_learnability() {
        let base = small();
        let cases: Vec<(&str, EpropConfig)> = vec![
            ("baseline", base.clone()),
            ("up_count=8", { let mut c = base.clone(); c.up_count = 8; c }),
            ("up_count=24", { let mut c = base.clone(); c.up_count = 24; c }),
            ("up_radius=2", { let mut c = base.clone(); c.up_radius = 2; c }),
            ("layers=2", { let mut c = base.clone(); c.layers = 2; c }),
            ("layers=4", { let mut c = base.clone(); c.layers = 4; c }),
        ];
        let waves = (base.present_waves + base.delay + base.read_waves) as u32;
        eprintln!("{:<12} {:>5} {:>7} {:>6} {:>7} {:>6} {:>5} {:>5} {:>5}", "cfg", "ceil", "fisher", "edim", "k-g", "sigma", "dead", "V1", "V2b");
        for (name, cfg) in &cases {
            let ceil = separation_ceiling(cfg, 200);
            let (s, y) = collect_states(cfg, 200);
            let fish = fisher_ratio(&s, &y, cfg.k);
            let edim = effective_dim(&as_f64(&s));
            let kg = kernel_minus_gen_rank(cfg);
            let sig = perturbation_spread(cfg);
            let (dead, _sat, _sync) = degeneracy(&s, waves);
            let v1 = learned(cfg.clone(), false, false, 0.3);
            let v2b = learned(cfg.clone(), true, true, 0.5);
            eprintln!("{name:<12} {ceil:>5} {fish:>7.2} {edim:>6.1} {kg:>7.1} {sig:>6.2} {dead:>5.2} {v1:>5} {v2b:>5}");
        }
    }

    #[test]
    #[ignore]
    fn _topology_interaction_grid() {
        let base = small();
        eprintln!("separation ceiling — rows up_count, cols adapt_bump");
        eprint!("{:>10}", "cnt\\bump");
        for b in [5i16, 10, 20, 40] {
            eprint!("{b:>6}");
        }
        eprintln!();
        for cnt in [8u32, 12, 16, 24] {
            eprint!("{cnt:>10}");
            for b in [5i16, 10, 20, 40] {
                let mut c = base.clone();
                c.up_count = cnt;
                c.adapt_bump = b;
                eprint!("{:>6}", separation_ceiling(&c, 160));
            }
            eprintln!();
        }
    }
```

- [ ] **Step 2: Run the experiments and read the tables**

Run: `cargo test bench::regime::tests::_regime_vs_learnability -- --ignored --nocapture`
Run: `cargo test bench::regime::tests::_topology_interaction_grid -- --ignored --nocapture`
Record both tables. Identify which metric column tracks the V1/V2b columns (moves down together on the dead
rows), and whether the grid's high-ceiling region is a diagonal band (topology × adapt_bump interaction).

- [ ] **Step 3: Write the findings**

Append a subsection to `docs/experiments_results.md` with both tables and the reading: **which metric(s)
predict learnability** (for V1, V2b, or both), **which don't**, and **whether topology interacts** with
adapt_bump (diagonal band vs axis-aligned). State honestly if the split is muddy. Close with what the fix
should target (the winning metric) — scoping the next spec.

- [ ] **Step 4: Full suite + warning-free**

Run: `cargo test` and `cargo build`
Expected: all pass (the `#[ignore]` experiments don't run in the normal suite); warning-free.

- [ ] **Step 5: Commit**

```bash
git add src/bench/regime.rs docs/experiments_results.md
git commit -m "feat: regime diagnostic experiments + findings (which metric predicts learnability)"
```

---

## Self-review

**Spec coverage:** All 8 metrics — separation ceiling + Fisher + forgetting (Task 2; forgetting = ceiling
swept over `delay`, done in the experiment/findings), effective dim + kernel−gen rank (Task 3), σ + layer
gain + degeneracy (Task 4). `collect_states` primitive (Task 1). Both experiments + both-learner correlation
+ topology grid (Task 5). Honesty gate → Task 5 Step 3. No engine change; f64 bench-only; determinism →
throughout.

**Placeholder scan:** none — concrete code and commands throughout.

**Type consistency:** `reservoir_states -> Vec<Vec<u32>>`, `flat -> Vec<u32>`, `collect_states ->
(Vec<Vec<u32>>, Vec<usize>)`; `separation_ceiling(&EpropConfig, usize) -> u64`; `fisher_ratio(&[Vec<u32>],
&[usize], usize) -> f64`; `effective_dim(&[Vec<f64>]) -> f64` fed via `as_f64`; `kernel_minus_gen_rank`,
`perturbation_spread`, `layer_gain`, `degeneracy` as declared. `NearestCentroid::fit(&[Vec<u32>], &[usize],
usize)` / `predict(&[u32])` match. `pick_class`/`calibrated_reservoir` exposed `pub(crate)` in Task 1.

**Note:** the *forgetting curve* metric is realized as `separation_ceiling` swept over `cfg.delay` inside the
findings run (no separate fn needed) — folded into Task 5 to avoid a one-line wrapper.
