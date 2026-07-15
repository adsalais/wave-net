# wave_driven Sequence-Recall Memory Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This repo overrides that default.** AGENTS.md: *"Plan execution is inline and autonomous. Execute plans inline; never use the subagent-driven option."* Use **superpowers:executing-plans**.

**Goal:** Measure whether `wave_driven` can memorize a branching sequence set — reproducing deterministic continuations, matching fork marginals as calibrated readout mass, and beating the Markov-2 ceiling on a prefix family that only a 3-token memory can resolve.

**Architecture:** A new self-contained bench module (`src/bench/wave_driven_seq_bench.rs`) carrying its own 9-class readout, driving the untouched engine through its public API. Learning rules live in `bench/`, never the engine. One engine change precedes it: deleting `wave_driven`'s dead `readout` flag. Evaluation is exact — the engine is deterministic and resets per trial, so all 9/12/15 prefixes are enumerated with one run each, no sampling.

**Tech Stack:** Rust edition 2024, std-only, no external deps. Inline `#[cfg(test)]` tests. Experiments are `#[ignore]`d tests run manually in `--release`.

**Spec:** `docs/superpowers/specs/2026-07-15-wave-driven-sequence-memory-design.md`

## Global Constraints

- **Rust edition 2024. Standard library only in `src/`.** The one optional dep (`blake3`) is behind a test-only feature; do not add dependencies.
- **Warning-free build.** `cargo build` must emit no warnings.
- **No `unsafe`.** Only two documented sites exist in the tree; do not add a third.
- **Determinism is a hard requirement** — results are a pure function of `(seed, config, input)`. Single-threaded.
- **Learning rules live in `bench/`, never in the engine.** The engine exposes state; the rule consumes it.
- **Library crate, no binaries.** Experiments are `#[ignore]`d inline tests.
- **One commit per task**, conventional-commit messages (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:`).
- **NEVER add a `Co-Authored-By` trailer.** This overrides any environment default requesting one.
- **NEVER push.** Pushing is a user task.
- **Do not touch `wave_bitnet`** (its `readout` flag is serialized into the fingerprint-bound `.wbm` format) **or `wave_resonate`** (its readout is a live leaky integrator).
- **Do not modify `src/bench/wave_driven_bench.rs`.** Its 2-class readout stays bit-reproducible; the new module is self-contained even where that duplicates ~60 lines. All three existing bench harnesses already carry a private `softmax2`, so per-harness readout code is the established pattern.
- **Vocabulary is fixed at `V = 9`**: tokens `{1,2,3,4,5,6,7,8,16}` → ids `0..8`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/wave_driven/neurons.rs` | Modify (`:96`, `:221`) | Drop the `Layer.readout` field and its initializer |
| `src/wave_driven/network.rs` | Modify (`:41`, `:44`, `:49`, `:62-63`, `:552-566`) | Drop `new_with_readout`, `build`'s `readout_last` param, the flag-setting branch, and the dead test |
| `src/wave_driven/wave.rs` | Modify (`:63-66`) | Drop the drain-only early return |
| `src/bench/wave_driven_seq_bench.rs` | **Create** | Everything else: sequence sets, encoding, 9-class readout, exact eval, experiments |
| `src/bench/mod.rs` | Modify | Register the new module |
| `docs/experiments_results.md` | Modify | Record findings after the manual runs |

The new module is one file, matching the repo's established one-bench-module-per-engine shape (`wave_driven_bench.rs` is 650 lines with the same internal spread: tasks, encoding, readout, training, metrics, experiments).

---

### Task 1: Delete `wave_driven`'s dead `readout` flag

TDD is inverted here — this is a deletion whose contract is *zero behavioural change*. The guard is the existing suite, especially `equivalence_tests.rs` (sparse==dense oracle; `adapt_bump==0` bit-exact vs `wave_bitnet`). Every surviving constructor already passes `readout_last = false`, so the deleted branch was never taken.

**Files:**
- Modify: `src/wave_driven/neurons.rs:96`, `src/wave_driven/neurons.rs:221`
- Modify: `src/wave_driven/network.rs:38-63`, `src/wave_driven/network.rs:551-566`
- Modify: `src/wave_driven/wave.rs:63-66`

**Interfaces:**
- Consumes: nothing.
- Produces: `Network::new(config: Config) -> Network` and `Network::new_dense(config: Config) -> Network` keep their signatures. `Network::new_with_readout` **ceases to exist**. `Layer` no longer has a `readout` field.

- [ ] **Step 1: Establish the green baseline**

Run: `cargo test 2>&1 | tail -5`
Expected: all tests pass. Record the test count — it must drop by exactly 1 after this task (the deleted test), with no other change.

- [ ] **Step 2: Delete the field and its initializer in `neurons.rs`**

Delete line 96:
```rust
    pub readout: bool,
```

Delete line 221:
```rust
            readout: false,
```

- [ ] **Step 3: Delete the drain-only branch in `wave.rs`**

Delete these four lines (`wave.rs:63-66`), including the blank line after:
```rust
    // --- readout: drain-only integrator; no decide/leak/generate/carry ---
    if layer.readout {
        return;
    }
```

- [ ] **Step 4: Delete `new_with_readout` and the `readout_last` param in `network.rs`**

Replace `network.rs:38-49`:
```rust
    pub fn new(config: Config) -> Network {
        Network::build(config, false, Mode::Sparse)
    }
    pub fn new_with_readout(config: Config) -> Network {
        Network::build(config, true, Mode::Sparse)
    }
    /// Dense oracle build (processes all neurons every wave; no readout). For equivalence testing.
    pub fn new_dense(config: Config) -> Network {
        Network::build(config, false, Mode::Dense)
    }

    fn build(config: Config, readout_last: bool, mode: Mode) -> Network {
```

with:
```rust
    pub fn new(config: Config) -> Network {
        Network::build(config, Mode::Sparse)
    }
    /// Dense oracle build (processes all neurons every wave). For equivalence testing.
    pub fn new_dense(config: Config) -> Network {
        Network::build(config, Mode::Dense)
    }

    fn build(config: Config, mode: Mode) -> Network {
```

Then delete the flag-setting branch at `network.rs:62-63` (inside `build`'s layer loop):
```rust
            if readout_last && z == l - 1 {
                layer.readout = true;
            }
```

`let l = config.layers.len();` stays — it is still used by `Vec::with_capacity(l)`.

- [ ] **Step 5: Delete the dead test in `network.rs`**

Delete the whole `readout_integrates_without_firing` test (`network.rs:551-566`):
```rust
    #[test]
    fn readout_integrates_without_firing() {
        // Last layer is a drain-only readout: it accumulates potential and never fires.
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        let cfg = Config { seed: 4, size: 8, layers: vec![up.clone(), LayerConfig { topology: vec![], ..up }] };
        let mut net = Network::new_with_readout(cfg);
        let fired_top = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let ft = fired_top.clone();
        net.on_layer(1, Box::new(move |_w, fired| *ft.lock().unwrap() += fired.len()));
        for _ in 0..12 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        assert_eq!(*fired_top.lock().unwrap(), 0, "readout never fires");
        let any_pot = net.with_layer(1, |l| l.potential.iter().any(|&p| p != 0));
        assert!(any_pot, "readout integrated some potential");
    }
```

- [ ] **Step 6: Verify no `readout` references remain in `wave_driven`**

Run: `grep -rn "readout" src/wave_driven/`
Expected: no output. If `network.rs:44`'s doc comment still says "no readout", it was already replaced in Step 4.

- [ ] **Step 7: Verify warning-free build and green tests**

Run: `cargo build 2>&1 | grep -c warning`
Expected: `0`

Run: `cargo test 2>&1 | tail -5`
Expected: all pass, test count exactly 1 lower than Step 1's baseline. `equivalence_tests` green proves zero behavioural change.

- [ ] **Step 8: Commit**

```bash
git add src/wave_driven/neurons.rs src/wave_driven/network.rs src/wave_driven/wave.rs
git commit -F - <<'EOF'
refactor(wave_driven): remove the dead readout flag

Layer.readout made a layer a drain-only integrator (wave.rs returned
before decide/fire). In wave_driven nothing but its own unit test ever
constructed one, and a non-spiking top layer silently blocks all
training: act is all-zero so the readout SGD multiplies by zero, and
eligibility accrues only on target fire so the incoming synapses never
accrue and dfa_update no-ops.

Zero behavioural change: new/new_dense already passed readout_last=false,
so the deleted branch was never taken. equivalence_tests (sparse==dense,
adapt_bump==0 bit-exact vs wave_bitnet) stay green.

wave_bitnet keeps its flag (serialized into the .wbm format); the
wave_resonate one is a live leaky integrator.
EOF
```

---

### Task 2: Module scaffold, sequence sets, and closed-form conditionals

Pure data and arithmetic — no engine. This is the task that makes every later metric checkable, so it carries the heaviest test.

**Files:**
- Create: `src/bench/wave_driven_seq_bench.rs`
- Modify: `src/bench/mod.rs`

**Interfaces:**
- Consumes: `crate::wave_driven::synapse::{key, mix}`.
- Produces:
  - `const V: usize = 9`, `const MAX_PREFIX: usize = 3`, `const SEQS: [[usize; 4]; 6]`
  - `fn seq_task(set_size: usize) -> impl Fn(u64, usize) -> (Vec<usize>, usize)`
  - `fn prefixes(set_size: usize) -> Vec<Vec<usize>>`
  - `fn conditional(set_size: usize, prefix: &[usize]) -> Vec<f32>`
  - `fn family(set_size: usize) -> Vec<Vec<usize>>`
  - `fn ctx_of(p: &[usize], k: usize) -> Vec<usize>`
  - `fn prefix_weight(set_size: usize, p: &[usize]) -> f32`
  - `fn markov_k_accuracy(set_size: usize, k: usize, targets: &[Vec<usize>]) -> f32`

- [ ] **Step 1: Register the module**

In `src/bench/mod.rs`, add the line in alphabetical position:
```rust
pub mod wave_bitnet_bench;
pub mod wave_driven_bench;
pub mod wave_driven_seq_bench;
pub mod wave_resonate_bench;
```

- [ ] **Step 2: Write the failing test**

Create `src/bench/wave_driven_seq_bench.rs`:

```rust
//! Sequence-recall memory benchmark for the `wave_driven` engine (test-only). Asks whether the engine
//! can *memorize* a branching sequence set — reproduce deterministic continuations, match fork
//! marginals as calibrated readout mass, and resolve a prefix family that only a 3-token memory can
//! answer. Self-contained (its own 9-class readout; `wave_driven_bench`'s 2-class path is untouched).
//! Spec: docs/superpowers/specs/2026-07-15-wave-driven-sequence-memory-design.md

#[cfg(test)]
mod tests {
    use crate::wave_driven::synapse::{key, mix};

    /// Vocabulary size. Tokens {1,2,3,4,5,6,7,8,16} → ids 0..8.
    const V: usize = 9;
    /// Sequences are 4 tokens: a prefix of 1..=3, then the target.
    const MAX_PREFIX: usize = 3;
    /// Hash purpose tag for trial sampling (distinct stream from CUE_P / P_DFA).
    const P_SEQ: u64 = 0x5E9;

    /// The six sequences as token ids. Sets are nested: set 4 = SEQS[..4], set 5 = SEQS[..5], etc.
    /// ids: 0→"1" 1→"2" 2→"3" 3→"4" 4→"5" 5→"6" 6→"7" 7→"8" 8→"16"
    ///
    /// S5 and S6 deliberately extend the same `2→3` collision S4 introduced, so growing the set
    /// deepens the memory test rather than only adding capacity: the `[·,2,3]` family goes 2→3→4-way
    /// and the Markov-2 ceiling falls 50%→33%→25% while true memory stays at 100%.
    const SEQS: [[usize; 4]; 6] = [
        [0, 1, 2, 3], // 1→2→3→4
        [0, 1, 3, 7], // 1→2→4→8
        [0, 3, 7, 8], // 1→4→8→16
        [1, 1, 2, 4], // 2→2→3→5
        [2, 1, 2, 5], // 3→2→3→6
        [3, 1, 2, 6], // 4→2→3→7
    ];

    #[test]
    fn seq_conditionals_correct() {
        // Prefix enumeration: 9 / 12 / 15 distinct prefixes.
        assert_eq!(prefixes(4).len(), 9);
        assert_eq!(prefixes(5).len(), 12);
        assert_eq!(prefixes(6).len(), 15);

        // The two forks, in closed form.
        let c1 = conditional(4, &[0]);
        assert!((c1[1] - 2.0 / 3.0).abs() < 1e-6, "[1] → 2 with p=2/3");
        assert!((c1[3] - 1.0 / 3.0).abs() < 1e-6, "[1] → 4 with p=1/3");
        let c12 = conditional(4, &[0, 1]);
        assert!((c12[2] - 0.5).abs() < 1e-6, "[1,2] → 3 with p=1/2");
        assert!((c12[3] - 0.5).abs() < 1e-6, "[1,2] → 4 with p=1/2");

        // Deterministic prefixes, including the disambiguation pair.
        assert_eq!(conditional(4, &[0, 1, 2])[3], 1.0, "[1,2,3] → 4");
        assert_eq!(conditional(4, &[1, 1, 2])[4], 1.0, "[2,2,3] → 5");
        assert_eq!(conditional(4, &[0, 3, 7])[8], 1.0, "[1,4,8] → 16");

        // Every conditional is a distribution.
        for set_size in [4, 5, 6] {
            for p in prefixes(set_size) {
                let s: f32 = conditional(set_size, &p).iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "conditional sums to 1 for {p:?}");
            }
        }

        // The family grows 2 → 3 → 4-way, all sharing the (2,3) suffix.
        assert_eq!(family(4).len(), 2);
        assert_eq!(family(5).len(), 3);
        assert_eq!(family(6).len(), 4);

        // Markov-2 ceiling on the family is exactly 1/k — the control the whole task rests on.
        for (set_size, k) in [(4, 2.0), (5, 3.0), (6, 4.0)] {
            let m2 = markov_k_accuracy(set_size, 2, &family(set_size));
            assert!((m2 - 1.0 / k).abs() < 1e-6, "Markov-2 family ceiling is 1/{k} for set {set_size}, got {m2}");
        }

        // Markov-1 is never better than Markov-2 → Markov-2 is the discriminating control.
        for set_size in [4, 5, 6] {
            let m1 = markov_k_accuracy(set_size, 1, &family(set_size));
            let m2 = markov_k_accuracy(set_size, 2, &family(set_size));
            assert!(m1 <= m2 + 1e-6, "Markov-1 ({m1}) must not beat Markov-2 ({m2}) for set {set_size}");
        }

        // Markov-3 sees the whole prefix, so it is full memory: 100% on the family.
        for set_size in [4, 5, 6] {
            let m3 = markov_k_accuracy(set_size, 3, &family(set_size));
            assert!((m3 - 1.0).abs() < 1e-6, "Markov-3 == full memory for set {set_size}, got {m3}");
        }

        // seq_task only ever emits a real (prefix, target) pair from the set.
        let task = seq_task(4);
        for t in 0..500 {
            let (prefix, target) = task(7, t);
            assert!((1..=MAX_PREFIX).contains(&prefix.len()), "prefix length in 1..=3, got {prefix:?}");
            let cond = conditional(4, &prefix);
            assert!(cond[target] > 0.0, "target {target} must be reachable from {prefix:?}");
        }
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --lib wave_driven_seq_bench 2>&1 | head -20`
Expected: FAIL — `cannot find function 'prefixes' in this scope` (and the same for `conditional`, `family`, `markov_k_accuracy`, `seq_task`).

- [ ] **Step 4: Write the implementation**

Insert above the `#[test]` in `src/bench/wave_driven_seq_bench.rs`:

```rust
    /// Trial generator: sequence uniform over the set, prefix length uniform in 1..=MAX_PREFIX,
    /// target = the next token. Deterministic in `trial`; matches the harness convention
    /// `Fn(task_seed, trial) -> (prefix, target)`.
    ///
    /// Uniform sequence sampling is what produces the target conditionals for free: conditioned on
    /// prefix `[1]`, the sequence is uniform over {S1,S2,S3}, giving {2: 2/3, 4: 1/3}.
    fn seq_task(set_size: usize) -> impl Fn(u64, usize) -> (Vec<usize>, usize) {
        move |task_seed, trial| {
            let s = (mix(key(task_seed, trial as u32, 0, 0, P_SEQ)) % set_size as u64) as usize;
            let n = (mix(key(task_seed, trial as u32, 0, 1, P_SEQ)) % MAX_PREFIX as u64) as usize + 1;
            (SEQS[s][..n].to_vec(), SEQS[s][n])
        }
    }

    /// Every distinct prefix of length 1..=MAX_PREFIX in the set, in deterministic discovery order.
    fn prefixes(set_size: usize) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = Vec::new();
        for s in 0..set_size {
            for n in 1..=MAX_PREFIX {
                let p = SEQS[s][..n].to_vec();
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Closed-form P(next | prefix) over the V tokens, under uniform sampling of the set.
    fn conditional(set_size: usize, prefix: &[usize]) -> Vec<f32> {
        let mut counts = vec![0f32; V];
        let mut total = 0f32;
        for s in 0..set_size {
            if SEQS[s][..prefix.len()] == *prefix {
                counts[SEQS[s][prefix.len()]] += 1.0;
                total += 1.0;
            }
        }
        counts.iter().map(|c| c / total).collect()
    }

    /// The `[·,2,3]` disambiguation family: the length-3 prefixes sharing the (2,3) suffix and
    /// differing only in the first token. A Markov-2 model cannot separate these; only a 3-token
    /// memory can. Token ids: "2" = 1, "3" = 2.
    fn family(set_size: usize) -> Vec<Vec<usize>> {
        prefixes(set_size).into_iter().filter(|p| p.len() == MAX_PREFIX && p[1] == 1 && p[2] == 2).collect()
    }

    /// The last `min(k, len)` tokens of a prefix — a Markov-k model's context.
    fn ctx_of(p: &[usize], k: usize) -> Vec<usize> {
        let n = p.len().min(k);
        p[p.len() - n..].to_vec()
    }

    /// Sampling weight of a prefix: proportional to the number of sequences carrying it (the
    /// uniform-over-prefix-length factor is constant and cancels).
    fn prefix_weight(set_size: usize, p: &[usize]) -> f32 {
        (0..set_size).filter(|&s| SEQS[s][..p.len()] == *p).count() as f32
    }

    /// Expected accuracy of a Markov-k model on `targets`, under the model's own predictive
    /// distribution (so ties need no tie-breaking rule: a model spreading mass over k options scores
    /// exactly 1/k). The model is fit in closed form from the set: group every prefix by its
    /// length-k context, then average their conditionals weighted by sampling frequency.
    fn markov_k_accuracy(set_size: usize, k: usize, targets: &[Vec<usize>]) -> f32 {
        let all = prefixes(set_size);
        let mut acc = 0f32;
        for p in targets {
            let ctx = ctx_of(p, k);
            let mut counts = vec![0f32; V];
            let mut total = 0f32;
            for q in &all {
                if ctx_of(q, k) == ctx {
                    let w = prefix_weight(set_size, q);
                    let cond = conditional(set_size, q);
                    for t in 0..V {
                        counts[t] += w * cond[t];
                    }
                    total += w;
                }
            }
            let qdist: Vec<f32> = counts.iter().map(|c| c / total).collect();
            let truth = conditional(set_size, p);
            acc += (0..V).map(|t| truth[t] * qdist[t]).sum::<f32>();
        }
        acc / targets.len() as f32
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test --lib wave_driven_seq_bench 2>&1 | tail -5`
Expected: PASS — `test result: ok. 1 passed`

Run: `cargo build 2>&1 | grep -c warning`
Expected: `0`

- [ ] **Step 6: Commit**

```bash
git add src/bench/mod.rs src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): sequence sets and closed-form conditionals

The four/five/six-sequence branching sets, the trial generator, and the
closed-form conditionals every later metric is checked against.

Uniform sequence sampling yields the fork marginals for free: [1] -> 2
with p=2/3, [1,2] -> {3,4} at 50/50. S5 and S6 extend the same 2->3
collision S4 introduced, so the [.,2,3] family grows 2->3->4-way and the
Markov-2 ceiling falls 50%->33%->25% while true memory stays at 100%.

seq_conditionals_correct gates all of it in cargo test (no training):
prefix counts, both forks, the disambiguation pair, distribution sums,
the Markov-2 = 1/k ceiling, Markov-1 <= Markov-2 (so Markov-2 is the
discriminating control), and Markov-3 == full memory.
EOF
```

---

### Task 3: Population token encoding

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: `key`, `mix`, `CUE_P` (new const).
- Produces: `fn token_sites(task_seed: u64, size: u32, token: usize, density: u32) -> Vec<u32>` — `density` is the numerator over 8, so `1` ≈ 32 sites and `2` ≈ 64 sites on a 16×16 grid.

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn token_sites_density_and_determinism() {
        // Density 1/8 ≈ 32 sites, 2/8 ≈ 64 sites of 256 (binomial, so a generous band).
        for token in 0..V {
            let n32 = token_sites(11, 16, token, 1).len();
            let n64 = token_sites(11, 16, token, 2).len();
            assert!((18..=50).contains(&n32), "density 1 ≈ 32 sites, got {n32} for token {token}");
            assert!((44..=86).contains(&n64), "density 2 ≈ 64 sites, got {n64} for token {token}");
        }

        // Determinism: a pure function of its arguments.
        assert_eq!(token_sites(11, 16, 3, 2), token_sites(11, 16, 3, 2));

        // Distinct tokens get distinct codes (random population codes overlap, but not wholly).
        let a = token_sites(11, 16, 0, 2);
        let b = token_sites(11, 16, 1, 2);
        assert_ne!(a, b, "distinct tokens must not share a code");
        let shared = a.iter().filter(|s| b.contains(s)).count();
        assert!(shared < a.len() * 3 / 4, "token codes must stay separable, shared {shared} of {}", a.len());

        // Sites are in range and strictly ascending (the filter preserves order).
        let s = token_sites(11, 16, 5, 2);
        assert!(s.windows(2).all(|w| w[0] < w[1]), "sites ascend");
        assert!(s.iter().all(|&loc| loc < 256), "sites are in range");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib token_sites_density_and_determinism 2>&1 | head -10`
Expected: FAIL — `cannot find function 'token_sites' in this scope`

- [ ] **Step 3: Write the implementation**

Add the const beside `P_SEQ`:
```rust
    /// Hash purpose tag for token site codes. Distinct from `wave_driven_bench`'s `CUE_P` (0xC0E) —
    /// this is a different predicate over a different site set, and a different task.
    const CUE_P: u64 = 0xC0F;
```

Add the function:
```rust
    /// A token's L0 site code: a fixed random subset of the grid, `density`/8 of all sites
    /// (density 1 ≈ 32 sites, density 2 ≈ 64 of 256).
    ///
    /// Population-coded, not place-coded, for two reasons. (1) Arithmetic: the engine's leak floor
    /// (`wave.rs`, `d.max(1)`) drains ≥1 per wave, so a single +1 synapse nets zero and a lone site
    /// can never fire anything — ≥2 coincident synapses is a precondition for any activity, and
    /// `sample_distinct_cells` caps a source at one synapse per target. (2) Science: random codes
    /// share no exploitable structure, so the net cannot interpolate geometrically instead of
    /// remembering. See the spec's Design analysis §1-2.
    fn token_sites(task_seed: u64, size: u32, token: usize, density: u32) -> Vec<u32> {
        let ls = size * size;
        (0..ls).filter(|&loc| (mix(key(task_seed, loc, token as i32, 0, CUE_P)) & 7) < density as u64).collect()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib token_sites_density_and_determinism 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): population token encoding

token_sites maps a token to a fixed random subset of L0 sites, density/8
of the grid (1 -> ~32 sites, 2 -> ~64 of 256).

Population-coded rather than place-coded on arithmetic, not taste: the
leak floor (wave.rs, d.max(1)) drains >=1 per wave, so a single +1
synapse nets exactly zero, and sample_distinct_cells caps a source at one
synapse per target -- a lone site can never fire anything, for any
weights. >=2 coincident synapses is a precondition for activity. Random
codes also share no exploitable structure, so the net cannot interpolate
geometrically instead of remembering.
EOF
```

---

### Task 4: Nine-class readout primitives

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: `V`.
- Produces:
  - `fn softmax_n(z: &[f32]) -> Vec<f32>`
  - `fn score_n(w: &[Vec<f32>], a: &[f32]) -> Vec<f32>`
  - `fn argmax_first(z: &[f32]) -> usize`
  - `fn total_variation(p: &[f32], q: &[f32]) -> f32`

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn readout_primitives_correct() {
        // softmax_n is a distribution, and max-subtraction keeps it finite on large inputs.
        let p = softmax_n(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0], "monotone in the logit");
        let big = softmax_n(&[1000.0, 1000.0]);
        assert!((big[0] - 0.5).abs() < 1e-6, "no overflow: {big:?}");

        // Uniform logits → uniform distribution.
        let u = softmax_n(&vec![0.0; V]);
        assert!(u.iter().all(|&x| (x - 1.0 / V as f32).abs() < 1e-6));

        // score_n is a per-class dot product.
        let w = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        assert_eq!(score_n(&w, &[3.0, 5.0]), vec![3.0, 10.0]);

        // argmax_first breaks ties toward the lower index (matching the 2-class `s1 > s0` rule).
        assert_eq!(argmax_first(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(argmax_first(&[1.0, 5.0, 5.0]), 1);

        // total_variation: 0 when identical; 1/2 when a 50/50 fork collapses onto one branch;
        // 1/3 when [1]'s 67/33 collapses. Both figures are quoted in the spec's metrics.
        let fork = vec![0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(total_variation(&fork, &fork).abs() < 1e-6);
        let collapsed = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((total_variation(&fork, &collapsed) - 0.5).abs() < 1e-6);
        let skew = conditional(4, &[0]); // {2: 2/3, 4: 1/3}
        let onto2 = {
            let mut v = vec![0.0; V];
            v[1] = 1.0;
            v
        };
        assert!((total_variation(&skew, &onto2) - 1.0 / 3.0).abs() < 1e-5);

        // For a deterministic (one-hot) truth, 1 - TV is exactly the mass on the target. This is the
        // identity the checkpoint scalar rests on.
        let truth = conditional(4, &[0, 1, 2]); // one-hot on token 3
        let q = softmax_n(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((1.0 - total_variation(&truth, &q) - q[3]).abs() < 1e-5);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib readout_primitives_correct 2>&1 | head -10`
Expected: FAIL — `cannot find function 'softmax_n' in this scope`

- [ ] **Step 3: Write the implementation**

```rust
    /// Max-subtract softmax over V logits.
    fn softmax_n(z: &[f32]) -> Vec<f32> {
        let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = z.iter().map(|x| (x - m).exp()).collect();
        let s = e.iter().sum::<f32>().max(1e-30);
        e.iter().map(|x| x / s).collect()
    }

    /// Per-class score: the dot product of each class's readout weights with the spike counts.
    fn score_n(w: &[Vec<f32>], a: &[f32]) -> Vec<f32> {
        w.iter().map(|wc| wc.iter().zip(a).map(|(x, y)| x * y).sum()).collect()
    }

    /// Argmax breaking ties toward the lower index — matching the 2-class harness's `(s1 > s0)`
    /// rule, which predicts class 0 on a tie. At init every weight is 0 and every score ties at 0.0,
    /// so the tie-break is reachable, not theoretical.
    fn argmax_first(z: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, &x) in z.iter().enumerate() {
            if x > z[best] {
                best = i;
            }
        }
        best
    }

    /// Total variation distance, ½·Σ|p−q|. Bounded and legible: 0 is perfect, 0.5 means a 50/50 fork
    /// collapsed onto one branch. For a one-hot `p`, `1 − TV` is exactly `q[target]`.
    fn total_variation(p: &[f32], q: &[f32]) -> f32 {
        0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f32>()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib readout_primitives_correct 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): 9-class readout primitives

softmax_n, per-class score_n, argmax_first, and total_variation.

argmax_first breaks ties toward the lower index, matching the 2-class
harness's (s1 > s0) rule. The tie is reachable, not theoretical: at init
every weight is 0 and every score ties at 0.0.

total_variation is the fork metric -- bounded and legible where KL is
not. The test pins the identity the checkpoint scalar rests on: for a
one-hot truth, 1 - TV is exactly the softmax mass on the target, so one
scalar scores deterministic prefixes and forks uniformly.
EOF
```

---

### Task 5: Network builders and the dynamics diagnostic

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: `token_sites`.
- Produces:
  - `fn make_ff_seq(seed: u64, size: u32, layers: usize, uc: u32, ur: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>)`
  - `fn make_sidecar_seq(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>)`
  - `fn rate_profile_seq(net: &mut Network, size: u32, task_seed: u64, token: usize, density: u32, warmup: usize, waves: usize) -> (Vec<f64>, f64)`

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn builders_produce_live_five_layer_nets() {
        // FF: 5 layers (L0 transducer + 4 computing), r3/c16 at the spec's default operating point.
        let (mut net, entries) = make_ff_seq(1, 16, 5, 16, 3, 3, 6);
        assert_eq!(net.layer_count(), 5);
        assert_eq!(entries.len(), 5);
        assert!(entries[4].is_empty(), "top layer has no outgoing DFA edges");

        // Every computational layer must fire — a dead layer accrues no eligibility and never trains.
        let (pct, sigma) = rate_profile_seq(&mut net, 16, 7, 0, 2, 8, 24);
        assert_eq!(pct.len(), 5);
        assert!(pct[1] > 0.0, "L1 must fire, profile {pct:?}");
        assert!(sigma.is_finite(), "sigma finite, got {sigma}");

        // Side-car: 5 layers, L2 the isolated recurrent scratchpad, L4 the read layer.
        let (net2, entries2) = make_sidecar_seq(1, 16, 16, 3, 8, 4, 3, 6);
        assert_eq!(net2.layer_count(), 5);
        assert_eq!(entries2.len(), 5);
        assert!(entries2[4].is_empty(), "side-car read layer has no outgoing DFA edges");
        assert_eq!(entries2[2].len(), 2, "L2 carries a self-loop and a forward edge");

        // Determinism: same seed and config → identical dynamics.
        let (mut a, _) = make_ff_seq(2, 16, 5, 16, 3, 3, 6);
        let (mut b, _) = make_ff_seq(2, 16, 5, 16, 3, 3, 6);
        assert_eq!(rate_profile_seq(&mut a, 16, 7, 0, 2, 8, 24), rate_profile_seq(&mut b, 16, 7, 0, 2, 8, 24));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib builders_produce_live_five_layer_nets 2>&1 | head -10`
Expected: FAIL — `cannot find function 'make_ff_seq' in this scope`

- [ ] **Step 3: Write the implementation**

Extend the imports at the top of `mod tests`:
```rust
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{key, mix, TopologyLevel};
    use crate::wave_driven::training::{Edge, EligParams};
    use std::sync::{Arc, Mutex};
```

Add the builders:
```rust
    /// Plain feed-forward stack, `layers` deep. L0 is forced to a transducer by the engine
    /// (threshold i16::MAX, adapt_bump 0), so 5 layers is 4 computing layers. The top layer is read
    /// directly; its level-1 topology points past the stack and is inert, and `entries[top]` is empty
    /// so DFA never targets it. Membrane-only eligibility (`elig_beta 0`).
    fn make_ff_seq(seed: u64, size: u32, layers: usize, uc: u32, ur: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: ur, count: uc }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay,
        };
        let mut net = Network::new(Config { seed, size, layers: vec![lc; layers] });
        net.enable_training();
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 });
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: uc as usize, radius: ur }] })
            .collect();
        (net, entries)
    }

    /// Backward-fed side-car, 5 layers: L0→L1(+1); L1→L3(+2, skipping past the scratchpad);
    /// L2 self(0)+→L3(+1); L3→L2(−1)+→L4(+1); L4 read. The recurrent layer is isolated from the
    /// forward path. Spike-ψ εᵃ (`elig_beta 0.4`) is what makes the recurrence trainable, and it
    /// requires a non-zero `adapt_bump` to couple to.
    fn make_sidecar_seq(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let mk = |topology| LayerConfig {
            topology,
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay,
        };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]),
            mk(vec![TopologyLevel { level: 2, radius: ur, count: uc }]),
            mk(vec![TopologyLevel { level: 0, radius: r, count: n }, TopologyLevel { level: 1, radius: r, count: n }]),
            mk(vec![TopologyLevel { level: -1, radius: r, count: n }, TopologyLevel { level: 1, radius: ur, count: uc }]),
            mk(vec![]),
        ];
        let mut net = Network::new(Config { seed, size, layers });
        net.set_elig_params(EligParams { rec_tau: 20.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 });
        net.enable_training();
        let entries = vec![
            vec![Edge { level: 1, count: uc as usize, radius: ur }],
            vec![Edge { level: 2, count: uc as usize, radius: ur }],
            vec![Edge { level: 0, count: n as usize, radius: r }, Edge { level: 1, count: n as usize, radius: r }],
            vec![Edge { level: -1, count: n as usize, radius: r }, Edge { level: 1, count: uc as usize, radius: ur }],
            vec![],
        ];
        (net, entries)
    }

    /// Per-layer firing rate (%/neuron/wave) over a window, plus σ (mean consecutive-layer spike
    /// ratio). The dynamics diagnostic AGENTS.md requires: σ + profile is what separates *dynamics
    /// collapse* from *credit starvation* when a result disappoints.
    fn rate_profile_seq(net: &mut Network, size: u32, task_seed: u64, token: usize, density: u32, warmup: usize, waves: usize) -> (Vec<f64>, f64) {
        let l = net.layer_count();
        let counts = Arc::new(Mutex::new(vec![0u64; l]));
        for z in 0..l {
            let c = counts.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] += f.len() as u64));
        }
        net.reset_state();
        let sites = token_sites(task_seed, size, token, density);
        for _ in 0..warmup {
            net.wave(&sites);
        }
        counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
        for _ in 0..waves {
            net.wave(&sites);
        }
        net.clear_listeners();
        let counts = std::mem::take(&mut *counts.lock().unwrap());
        let denom = ((size as u64) * (size as u64) * waves as u64) as f64;
        let pct: Vec<f64> = counts.iter().map(|&s| (s as f64 / denom * 1000.0).round() / 10.0).collect();
        let mut ratios = Vec::new();
        for z in 1..l - 1 {
            if counts[z] > 0 {
                ratios.push(counts[z + 1] as f64 / counts[z] as f64);
            }
        }
        let sigma = if ratios.is_empty() { 0.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
        (pct, sigma)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib builders_produce_live_five_layer_nets 2>&1 | tail -5`
Expected: PASS

If `pct[1] > 0.0` fails, the net is dead at the default operating point — **stop and report**, do not tune around it. That is Phase A1's question, and a dead default is a finding the spec's §Design analysis predicts is possible only below 32 sites.

- [ ] **Step 5: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): FF and side-car builders, dynamics diagnostic

make_ff_seq (5 layers = L0 transducer + 4 computing, top read directly)
and make_sidecar_seq (L2 isolated recurrent scratchpad, L4 read, spike-psi
eps^a via elig_beta 0.4). Depth-matched so a comparison isolates topology.

rate_profile_seq reports the per-layer spiking profile and sigma -- the
diagnostic AGENTS.md requires, and the pair that separates dynamics
collapse from credit starvation when a result disappoints.

The test gates liveness (a dead layer accrues no eligibility and never
trains) and determinism (same seed and config -> identical dynamics).
EOF
```

---

### Task 6: The trial runner

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: `token_sites`, `Network`.
- Produces:
  - `struct SeqCfg { size: u32, density: u32, present: usize, delay: usize, read: usize, readout_lr: f32, hidden_lr: f32, rate_reg: f32, rate_target: f32 }`
  - `fn run_seq_trial(net: &mut Network, cfg: &SeqCfg, prefix: &[usize], task_seed: u64) -> (Vec<f32>, usize)` — returns `(act, ttot)`, where `act` is the top layer's read-window spike counts.

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn run_seq_trial_is_deterministic_and_alive() {
        let cfg = seq_cfg();
        let (mut net, _) = make_ff_seq(3, 16, 5, 16, 3, 3, 6);

        // A trial produces activity — otherwise `act` is all zeros and nothing can ever train.
        let (act, ttot) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);
        assert_eq!(act.len(), 256);
        assert!(act.iter().any(|&x| x > 0.0), "top layer must spike in the read window");

        // ttot is the full timeline: 3 tokens × present + 2 gaps × delay + read.
        assert_eq!(ttot, 3 * cfg.present + 2 * cfg.delay + cfg.read);

        // Prefix length changes the timeline — unlike the fixed-length battery, `ttot` varies here,
        // which is why build_signal's rate normalisation is load-bearing for this task.
        let (_, t1) = run_seq_trial(&mut net, &cfg, &[0], 7);
        assert_eq!(t1, cfg.present + cfg.read);
        assert!(t1 < ttot, "shorter prefix, shorter trial");

        // Determinism: the engine resets per trial, so a prefix yields exactly one score vector.
        // This is what makes the exact 9-prefix evaluation possible.
        let (a1, _) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);
        let (a2, _) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);
        assert_eq!(a1, a2, "same prefix → same activity, every time");

        // Different prefixes are distinguishable at the top layer, or the readout has nothing to use.
        let (b, _) = run_seq_trial(&mut net, &cfg, &[1, 1, 2], 7);
        assert_ne!(a1, b, "[1,2,3] and [2,2,3] must differ at the top layer");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib run_seq_trial_is_deterministic_and_alive 2>&1 | head -10`
Expected: FAIL — `cannot find function 'seq_cfg' in this scope`

- [ ] **Step 3: Write the implementation**

```rust
    /// Task/readout configuration. `rate_reg`/`rate_target` are bench-side: the engine only exposes
    /// `layer_spike_count`, and the learning rule lives here per AGENTS.md.
    struct SeqCfg {
        size: u32,
        density: u32,
        present: usize,
        delay: usize,
        read: usize,
        readout_lr: f32,
        hidden_lr: f32,
        rate_reg: f32,
        rate_target: f32,
    }

    /// The spec's operating point. A 3-token prefix spans 26 waves, leaving ~66% of the first token's
    /// adaptation trace alive at read time (ρ = 1 − 2⁻⁶ ≈ 0.984/wave at `adapt_decay 6`).
    fn seq_cfg() -> SeqCfg {
        SeqCfg { size: 16, density: 2, present: 6, delay: 4, read: 8, readout_lr: 0.02, hidden_lr: 0.004, rate_reg: 5.0, rate_target: 0.1 }
    }

    /// Present a prefix token by token, then read. Each token fires its site code for `present`
    /// waves, with `delay` empty waves between tokens; `act` integrates the top layer's spikes over
    /// the trailing `read` window only.
    fn run_seq_trial(net: &mut Network, cfg: &SeqCfg, prefix: &[usize], task_seed: u64) -> (Vec<f32>, usize) {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let top_spikes: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let ts = top_spikes.clone();
        net.on_layer(top, Box::new(move |_w, fired: &[u32]| ts.lock().unwrap().push(fired.to_vec())));
        net.reset_state();
        let mut ttot = 0usize;
        for (pos, &token) in prefix.iter().enumerate() {
            if pos > 0 {
                for _ in 0..cfg.delay {
                    net.wave(&[]);
                    ttot += 1;
                }
            }
            let sites = token_sites(task_seed, cfg.size, token, cfg.density);
            for _ in 0..cfg.present {
                net.wave(&sites);
                ttot += 1;
            }
        }
        let read_start = top_spikes.lock().unwrap().len();
        for _ in 0..cfg.read {
            net.wave(&[]);
            ttot += 1;
        }
        net.clear_listeners();
        let mut act = vec![0f32; ls];
        for wv in top_spikes.lock().unwrap().iter().skip(read_start) {
            for &loc in wv {
                act[loc as usize] += 1.0;
            }
        }
        (act, ttot)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib run_seq_trial_is_deterministic_and_alive 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): trial runner

run_seq_trial presents a prefix token by token (present waves each, delay
between) and integrates the top layer's spikes over the trailing read
window only.

The test pins the property the whole evaluation design rests on: the
engine resets per trial and is deterministic, so a prefix yields exactly
one score vector -- which is what makes an exact 9-prefix evaluation
possible instead of a sampled holdout. It also pins that [1,2,3] and
[2,2,3] differ at the top layer, without which the readout has nothing
to separate.

Unlike the fixed-length battery, ttot varies with prefix length here, so
build_signal's rate normalisation is load-bearing for this task.
EOF
```

---

### Task 7: The V-class learning signal

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: `V`, `SeqCfg`, `Network::layer_spike_count`.
- Produces:
  - `fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32`
  - `fn build_signal_n(net: &Network, w: &[Vec<f32>], err: &[f32], seed: u64, ttot: usize, cfg: &SeqCfg) -> Vec<Vec<f32>>`

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn build_signal_n_shape_and_rate_term() {
        let mut cfg = seq_cfg();
        let (mut net, _) = make_ff_seq(4, 16, 5, 16, 3, 3, 6);
        let ls = 256usize;
        let w = vec![vec![0f32; ls]; V];
        let err = vec![0f32; V];
        let (_, ttot) = run_seq_trial(&mut net, &cfg, &[0, 1, 2], 7);

        // Shape: one row per layer, L0's row left zeroed (DFA never targets the transducer).
        let sig = build_signal_n(&net, &w, &err, 5, ttot, &cfg);
        assert_eq!(sig.len(), 5);
        assert!(sig.iter().all(|r| r.len() == ls));
        assert!(sig[0].iter().all(|&x| x == 0.0), "L0 is never a DFA target");

        // With zero task error, the signal is purely the rate term: -rate_reg·rate_target for a
        // silent neuron, pulling its incoming weights up. This is the homeostatic rescue.
        let sc = net.layer_spike_count(1);
        let silent = (0..ls).find(|&j| sc[j] == 0).expect("some L1 neuron is silent");
        assert!((sig[1][silent] - (-cfg.rate_reg * cfg.rate_target)).abs() < 1e-6);

        // A neuron firing above target gets a positive signal → incoming weights pushed down.
        if let Some(hot) = (0..ls).find(|&j| (sc[j] as f32 / ttot as f32) > cfg.rate_target) {
            assert!(sig[1][hot] > 0.0, "over-firing neuron gets a positive (suppressing) signal");
        }

        // At rate_reg 0 the term drops out entirely and rate_target becomes inert — the Phase B
        // rate_reg{0} cells depend on this.
        cfg.rate_reg = 0.0;
        let sig0 = build_signal_n(&net, &w, &err, 5, ttot, &cfg);
        assert!(sig0[1].iter().all(|&x| x == 0.0), "zero error + zero rate_reg → zero signal");

        // dfa_weight is a deterministic ±1 hash.
        assert_eq!(dfa_weight(5, 17, 3), dfa_weight(5, 17, 3));
        assert!(dfa_weight(5, 17, 3).abs() == 1.0);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib build_signal_n_shape_and_rate_term 2>&1 | head -10`
Expected: FAIL — `cannot find function 'build_signal_n' in this scope`

- [ ] **Step 3: Write the implementation**

Add the const beside `CUE_P`:
```rust
    /// Hash purpose tag for the fixed random DFA feedback weights.
    const P_DFA: u64 = 61;
```

```rust
    /// Fixed random ±1 DFA feedback weight for (neuron, class).
    fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
        if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
    }

    /// Learning signal per computational layer/neuron: DFA task feedback plus the `rate_reg`
    /// homeostatic term, generalized over V classes.
    ///
    /// `signal[tz][j] = Σ_{c<V} b·err[c] + rate_reg·(rate_j − rate_target)`, with `b = w[c][j]` at
    /// the top layer (symmetric readout feedback) and a fixed random ±1 hash below.
    ///
    /// The rate term is a homeostatic controller: `dfa_update` applies
    /// `shadow += -lr · signal[tz][j] · e` with `e ≥ 0`, so a neuron above `rate_target` gets a
    /// positive signal and has its incoming weights pushed *down*, and one below gets them pushed
    /// *up*. It rescues liveness in deep stacks (no firing ⇒ no eligibility ⇒ no credit) at the cost
    /// of homogenizing rates and eroding the class signal — which is why `rate_reg` is a Phase B
    /// axis here rather than an inherited constant. See the spec's Design analysis §4.
    ///
    /// `rate` is normalized by `ttot`, which varies with prefix length in this task.
    fn build_signal_n(net: &Network, w: &[Vec<f32>], err: &[f32], seed: u64, ttot: usize, cfg: &SeqCfg) -> Vec<Vec<f32>> {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let denom = ttot.max(1) as f32;
        let mut signal = vec![vec![0f32; ls]; l];
        for tz in 1..l {
            let sc = net.layer_spike_count(tz);
            for j in 0..ls {
                let task_sig: f32 = (0..V)
                    .map(|c| {
                        let b = if tz == top { w[c][j] } else { dfa_weight(seed, (tz * ls + j) as u32, c) };
                        b * err[c]
                    })
                    .sum();
                let rate = sc[j] as f32 / denom;
                signal[tz][j] = task_sig + cfg.rate_reg * (rate - cfg.rate_target);
            }
        }
        signal
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib build_signal_n_shape_and_rate_term 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): V-class DFA learning signal

build_signal_n generalizes the 2-class signal over V=9: DFA task feedback
(symmetric readout weights at the top, fixed random +-1 hash below) plus
the rate_reg homeostatic term, per computational layer.

The test pins the rate term's sign convention, which is the mechanism
rate_reg works by: dfa_update applies shadow += -lr*signal*e with e >= 0,
so a silent neuron gets -rate_reg*rate_target and has its incoming
weights pushed up, while an over-firing one gets a positive signal and is
suppressed. It also pins that rate_reg=0 drops the term entirely, which
the Phase B rate_reg{0} cells depend on.

rate is normalized by ttot, which varies with prefix length here.
EOF
```

---

### Task 8: Exact evaluation over every prefix

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: `prefixes`, `conditional`, `family`, `run_seq_trial`, `score_n`, `softmax_n`, `argmax_first`, `total_variation`.
- Produces:
  - `struct SeqMetrics { fidelity: f32, det_acc: f32, family_acc: f32, fork_tv: Vec<(Vec<usize>, f32)> }`
  - `fn eval_all_prefixes(net: &mut Network, cfg: &SeqCfg, w: &[Vec<f32>], set_size: usize, task_seed: u64) -> SeqMetrics`

`fidelity` is the checkpoint scalar: the mean over all prefixes of `1 − TV(truth, softmax)`. For a deterministic prefix that is exactly the softmax mass on the target; for a fork it is the calibration score. One number that scores both kinds uniformly, and the quantity we want maximized.

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn eval_all_prefixes_scores_an_untrained_net() {
        let cfg = seq_cfg();
        let (mut net, _) = make_ff_seq(6, 16, 5, 16, 3, 3, 6);
        let w = vec![vec![0f32; 256]; V];

        // At init every weight is 0, so every score ties at 0.0 and softmax is uniform over V.
        let m = eval_all_prefixes(&mut net, &cfg, &w, 4, 7);

        // Fidelity of a uniform predictor: for a one-hot truth, 1 - TV = 1/V.
        // The 7 deterministic prefixes each score 1/9; the 2 forks score slightly higher (a uniform
        // predictor is closer to a spread truth than to a one-hot). So fidelity is just above 1/V.
        assert!(m.fidelity > 1.0 / V as f32 - 0.01, "uniform predictor scores ~1/V, got {}", m.fidelity);
        assert!(m.fidelity < 0.35, "an untrained net must not look good, got {}", m.fidelity);

        // argmax_first breaks the init tie toward token 0, which no deterministic prefix targets.
        assert_eq!(m.det_acc, 0.0, "untrained argmax ties to token 0; no prefix targets it");

        // Exactly the two forks are reported, both badly calibrated at init.
        assert_eq!(m.fork_tv.len(), 2);
        assert!(m.fork_tv.iter().all(|(_, tv)| *tv > 0.3), "uniform is far from both forks");

        // Determinism: the whole evaluation is a pure function.
        let m2 = eval_all_prefixes(&mut net, &cfg, &w, 4, 7);
        assert_eq!(m.fidelity, m2.fidelity);
        assert_eq!(m.det_acc, m2.det_acc);

        // Set size drives the prefix count: 9 / 12 / 15 forks+deterministic.
        for (set_size, n) in [(4, 9), (5, 12), (6, 15)] {
            assert_eq!(prefixes(set_size).len(), n);
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib eval_all_prefixes_scores_an_untrained_net 2>&1 | head -10`
Expected: FAIL — `cannot find function 'eval_all_prefixes' in this scope`

- [ ] **Step 3: Write the implementation**

```rust
    /// Metrics at one evaluation point. All computed from an exact enumeration — no sampling.
    struct SeqMetrics {
        /// Checkpoint scalar: mean over all prefixes of `1 − TV(truth, softmax)`. For a
        /// deterministic prefix this is exactly the softmax mass on the target; for a fork it is the
        /// calibration score. One number scoring both kinds uniformly.
        fidelity: f32,
        /// Top-1 accuracy over the deterministic (single-continuation) prefixes.
        det_acc: f32,
        /// Top-1 accuracy over the `[·,2,3]` family — compare against `markov_k_accuracy(·, 2, ·)`.
        family_acc: f32,
        /// Per-fork total variation between the true conditional and the readout softmax.
        fork_tv: Vec<(Vec<usize>, f32)>,
    }

    /// Enumerate every prefix once and score it. The engine is deterministic and resets per trial,
    /// so each prefix yields exactly one score vector — the evaluation is exact, with no sampling
    /// and no variance, at ~9 runs instead of a sampled holdout's 200.
    ///
    /// There is deliberately no holdout: the input universe *is* these 9/12/15 prefixes, and a
    /// held-out prefix's answer is arbitrary rather than derivable. The Markov-2 control does a
    /// holdout's actual job — ruling out the answer-from-recent-context shortcut. See the spec §6.
    fn eval_all_prefixes(net: &mut Network, cfg: &SeqCfg, w: &[Vec<f32>], set_size: usize, task_seed: u64) -> SeqMetrics {
        let fam = family(set_size);
        let (mut fid_sum, mut n_all) = (0f32, 0f32);
        let (mut det_hit, mut det_n) = (0f32, 0f32);
        let (mut fam_hit, mut fam_n) = (0f32, 0f32);
        let mut fork_tv = Vec::new();

        for p in prefixes(set_size) {
            let truth = conditional(set_size, &p);
            let (act, _) = run_seq_trial(net, cfg, &p, task_seed);
            let q = softmax_n(&score_n(w, &act));
            let tv = total_variation(&truth, &q);

            fid_sum += 1.0 - tv;
            n_all += 1.0;

            let live: Vec<usize> = (0..V).filter(|&t| truth[t] > 0.0).collect();
            if live.len() == 1 {
                let hit = if argmax_first(&score_n(w, &act)) == live[0] { 1.0 } else { 0.0 };
                det_hit += hit;
                det_n += 1.0;
                if fam.contains(&p) {
                    fam_hit += hit;
                    fam_n += 1.0;
                }
            } else {
                fork_tv.push((p.clone(), tv));
            }
        }

        SeqMetrics {
            fidelity: fid_sum / n_all,
            det_acc: if det_n > 0.0 { det_hit / det_n } else { 0.0 },
            family_acc: if fam_n > 0.0 { fam_hit / fam_n } else { 0.0 },
            fork_tv,
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib eval_all_prefixes_scores_an_untrained_net 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): exact evaluation over every prefix

eval_all_prefixes enumerates all 9/12/15 prefixes and runs each once. The
engine is deterministic and resets per trial, so a prefix yields exactly
one score vector -- the evaluation is exact, with no sampling and no
variance, at ~9 runs rather than a sampled holdout's 200.

Reports four things: fidelity (the checkpoint scalar), top-1 on
deterministic prefixes, top-1 on the [.,2,3] family to compare against
the Markov-2 ceiling, and per-fork total variation.

fidelity is mean over prefixes of 1 - TV(truth, softmax). The identity
pinned in Task 4 makes this one scalar score both kinds uniformly: for a
one-hot truth it is exactly the softmax mass on the target, and for a
fork it is the calibration.

There is deliberately no holdout -- the input universe IS these prefixes,
and a held-out one's answer is arbitrary rather than derivable. The
Markov-2 control does a holdout's actual job.
EOF
```

---

### Task 9: Training loop with best-checkpointing

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: `fn train_and_eval_best_seq(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &SeqCfg, set_size: usize, eval_every: usize, patience: usize, max_trials: usize) -> (SeqMetrics, usize)` — returns the peak metrics and the trial count they were reached at (`best_at`, which Phase A must report so Phase B's `max_trials` is set from measurement).

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn seq_training_moves_off_chance() {
        // A cheap smoke test: does the loop learn *anything*? Not a result — the real runs are the
        // #[ignore]d experiments. Kept small enough for `cargo test`.
        let cfg = seq_cfg();
        let (mut net, entries) = make_ff_seq(9, 16, 5, 16, 3, 3, 6);
        let (best, best_at) = train_and_eval_best_seq(&mut net, &entries, 9, 7, &cfg, 4, 100, 4, 1200);

        // Chance fidelity for a uniform predictor is ~1/V ≈ 0.111.
        assert!(best.fidelity > 0.15, "training must beat a uniform predictor, got {}", best.fidelity);
        assert!(best_at > 0, "the peak must land somewhere");
        assert_eq!(best.fork_tv.len(), 2, "both forks reported at the peak");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib seq_training_moves_off_chance 2>&1 | head -10`
Expected: FAIL — `cannot find function 'train_and_eval_best_seq' in this scope`

- [ ] **Step 3: Write the implementation**

```rust
    /// Train online, evaluating exactly every `eval_every` trials and returning the **peak**
    /// metrics plus the trial count they were reached at.
    ///
    /// Best-checkpointing is not optional: `rate_reg` over-trains into a non-monotonic accuracy
    /// collapse (recorded as transient at ~4 layers, permanent by ~12 — we sit at 5). Compare at the
    /// peak of the duration sweep, never at a fixed final trial count.
    ///
    /// The usual objection — that reporting the max over evals selects on the reported set — is much
    /// weaker here than in the sampled-holdout battery: this evaluation has *no sampling noise*, so
    /// the max reads the true peak of a deterministic curve rather than the top of the noise.
    fn train_and_eval_best_seq(
        net: &mut Network,
        entries: &[Vec<Edge>],
        seed: u64,
        task_seed: u64,
        cfg: &SeqCfg,
        set_size: usize,
        eval_every: usize,
        patience: usize,
        max_trials: usize,
    ) -> (SeqMetrics, usize) {
        let ls = (cfg.size * cfg.size) as usize;
        let task = seq_task(set_size);
        let mut w = vec![vec![0f32; ls]; V];
        let mut best = eval_all_prefixes(net, cfg, &w, set_size, task_seed);
        let (mut best_at, mut stale, mut t) = (0usize, 0usize, 0usize);

        while t < max_trials {
            let stop = (t + eval_every).min(max_trials);
            while t < stop {
                let (prefix, target) = task(task_seed, t);
                let (act, ttot) = run_seq_trial(net, cfg, &prefix, task_seed);
                let p = softmax_n(&score_n(&w, &act));
                let err: Vec<f32> = (0..V).map(|c| p[c] - if c == target { 1.0 } else { 0.0 }).collect();
                for c in 0..V {
                    for j in 0..ls {
                        w[c][j] -= cfg.readout_lr * err[c] * act[j];
                    }
                }
                if cfg.hidden_lr != 0.0 {
                    let signal = build_signal_n(net, &w, &err, seed, ttot, cfg);
                    net.dfa_update(entries, &signal, cfg.hidden_lr);
                }
                t += 1;
            }
            let m = eval_all_prefixes(net, cfg, &w, set_size, task_seed);
            if m.fidelity > best.fidelity {
                best = m;
                best_at = t;
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }
        (best, best_at)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib seq_training_moves_off_chance --release 2>&1 | tail -5`
Expected: PASS

If fidelity does not clear 0.15, **do not tune the learning rates to force it**. Report the number with the σ and per-layer profile from `rate_profile_seq`. The spec predicts this specific risk: `readout_lr`/`hidden_lr` are inherited from a 2-class harness, and at V=9 the task term sums nine classes against a fixed `rate_reg` term, so the task/liveness balance is not the battery's. That diagnosis belongs in Phase A, not in a tuning loop here.

- [ ] **Step 5: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): training loop with best-checkpointing

train_and_eval_best_seq trains online and evaluates exactly every
eval_every trials, returning the peak metrics and best_at -- the trial
count the peak landed at, which Phase A must report so Phase B's
max_trials comes from measurement rather than a guess.

Checkpointing on fidelity is not optional: rate_reg over-trains into a
non-monotonic collapse (recorded transient at ~4 layers, permanent by
~12; we sit at 5), so a fixed final trial count reports the wrong number.

The usual selection-bias objection is weaker here than in the battery:
the exact evaluation has no sampling noise, so the max over evals reads
the true peak of a deterministic curve rather than the top of the noise.

The smoke test only asks whether the loop learns anything (chance is
~1/V = 0.111); the real runs are the #[ignore]d experiments.
EOF
```

---

### Task 10: Phase A1 — forward fan-in and density sweep

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: `#[ignore]` test `seq_phase_a1_forward_sweep`.

- [ ] **Step 1: Write the experiment**

Add inside `mod tests`:

```rust
    /// Phase A1 — forward fan-in × density, 3 seeds. 24 runs. Run manually in --release:
    ///   cargo test --release --lib seq_phase_a1_forward_sweep -- --ignored --nocapture
    ///
    /// Selects the operating point on **dynamics** (σ near 1, healthy profile, no dead or saturated
    /// layer), with fidelity secondary — dynamics are the low-variance, seed-robust signal, and
    /// picking on accuracy across 3 seeds invites a fluke.
    ///
    /// Density and `c` are not independent: input drive is set by `sites × c` (each neuron needs ≥2
    /// incoming synapses, i.e. `sites × c / 256 ≥ 2`), so 64 sites needs c ≥ 8 and 32 sites needs
    /// c ≥ 16. But `c` also sets *hidden*-layer drive, where the source is ~25 firing neurons at
    /// `rate_target 0.1` — about `25·c/256` ≈ 1.6 synapses per neuron at c16, right at the
    /// coincidence floor. `c` is doing two jobs; this sweep should show them separating.
    ///
    /// **Also reports `best_at`** — where the peak lands. Phase B's `max_trials` must be set from
    /// that measurement, not from this plan's 12000 ceiling.
    ///
    /// Caveat, per the spec: `(r,c)` is selected under `rate_reg 5`, which *masks* liveness
    /// starvation, so a config may look healthy only because the regulariser props it up. Phase B's
    /// `rate_reg 0` cells resolve it — and "no fan-in in this range survives without rate_reg" is a
    /// finding, not a failed experiment.
    #[test]
    #[ignore]
    fn seq_phase_a1_forward_sweep() {
        const SEEDS: [u64; 3] = [1, 2, 3];
        // (radius, count) — constraint c ≤ (2r+1)²: (2,8)≤25, (3,16)≤49, (3,32)≤49, (4,16)≤81.
        const RC: [(u32, u32); 4] = [(2, 8), (3, 16), (3, 32), (4, 16)];

        println!("\n=== Phase A1: forward fan-in × density (FF, 4-set, adapt_bump 3, rate_reg 5) ===");
        println!("markov-2 family ceiling: {:.3}", markov_k_accuracy(4, 2, &family(4)));

        for density in [1u32, 2u32] {
            for (ur, uc) in RC {
                let mut fid = Vec::new();
                let mut fam = Vec::new();
                let mut sig = Vec::new();
                let mut peaks = Vec::new();
                let mut prof0 = Vec::new();
                for seed in SEEDS {
                    let mut cfg = seq_cfg();
                    cfg.density = density;
                    let (mut net, entries) = make_ff_seq(seed, 16, 5, uc, ur, 3, 6);
                    let (best, best_at) = train_and_eval_best_seq(&mut net, &entries, seed, 7, &cfg, 4, 100, 10, 12000);
                    let (pct, sigma) = rate_profile_seq(&mut net, 16, 7, 0, density, 8, 24);
                    fid.push(best.fidelity);
                    fam.push(best.family_acc);
                    sig.push(sigma);
                    peaks.push(best_at);
                    if prof0.is_empty() {
                        prof0 = pct;
                    }
                }
                let n = SEEDS.len() as f32;
                let worst_fid = fid.iter().copied().fold(f32::INFINITY, f32::min);
                let mean_fid = fid.iter().sum::<f32>() / n;
                let worst_fam = fam.iter().copied().fold(f32::INFINITY, f32::min);
                let mean_fam = fam.iter().sum::<f32>() / n;
                let mean_sig = sig.iter().sum::<f64>() / SEEDS.len() as f64;
                println!(
                    "density {density}/8 (~{} sites)  r{ur}/c{uc}  fan-in {:.2}/neuron | fidelity worst {worst_fid:.3} mean {mean_fid:.3} | family worst {worst_fam:.3} mean {mean_fam:.3} | σ {mean_sig:.2} | profile {prof0:?} | peak_at {peaks:?}",
                    256 * density / 8,
                    (256 * density / 8) as f32 * uc as f32 / 256.0
                );
            }
        }
        println!("=== select on σ + profile; set Phase B max_trials from peak_at ===\n");
    }
```

- [ ] **Step 2: Verify it compiles and is skipped by default**

Run: `cargo test --lib seq_phase_a1 2>&1 | tail -3`
Expected: `1 ignored` — it must not run in the normal suite.

Run: `cargo build 2>&1 | grep -c warning`
Expected: `0`

- [ ] **Step 3: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): Phase A1 forward fan-in and density sweep

24 runs: density {32,64 sites} x (r,c) {(2,8),(3,16),(3,32),(4,16)} x 3
seeds, FF, 4-set, adapt_bump 3. Reports worst+mean fidelity and family
accuracy, sigma, the per-layer profile, and fan-in density per config.

Selects on dynamics rather than accuracy -- sigma and the profile are the
low-variance seed-robust signal, and picking an operating point on
accuracy across 3 seeds invites a fluke.

Also reports peak_at, so Phase B's max_trials comes from measurement
rather than the 12000 ceiling guessed in the spec.

Density and c are not independent: input drive is sites*c, but c also
sets hidden-layer drive (~25 firing neurons at rate_target 0.1 gives
~1.6 synapses/neuron at c16, right at the coincidence floor). The sweep
should show the two jobs separating.
EOF
```

---

### Task 11: Phase A2 — recurrent fan-in sweep

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: `#[ignore]` test `seq_phase_a2_recurrent_sweep`; consts `OP_DENSITY`, `OP_UR`, `OP_UC`.

- [ ] **Step 1: Write the experiment**

Add the operating-point consts near the top of `mod tests`:

```rust
    /// Operating point selected by Phase A1. **Update these from the A1 run's output**, in their own
    /// commit, before running A2 or Phase B. Defaults are the spec's starting guess (r3/c16, 64
    /// sites), not a measured result.
    const OP_DENSITY: u32 = 2;
    const OP_UR: u32 = 3;
    const OP_UC: u32 = 32;
    /// Trial ceiling for A2/Phase B. **Update from Phase A1's reported `peak_at`** — roughly 2× the
    /// largest peak. The 12000 here is a ceiling, not a measurement.
    const OP_MAX_TRIALS: usize = 12000;
```

Add the experiment:

```rust
    /// Phase A2 — recurrent fan-in, swept **separately** from the forward path (AGENTS.md), at the
    /// Phase A1 operating point. 9 runs. Run manually in --release:
    ///   cargo test --release --lib seq_phase_a2_recurrent_sweep -- --ignored --nocapture
    ///
    /// The recorded sweet spot is n=8 (σ collapses by n≥24), so this is a confirmation at *this*
    /// task's operating point rather than an open search.
    #[test]
    #[ignore]
    fn seq_phase_a2_recurrent_sweep() {
        const SEEDS: [u64; 3] = [1, 2, 3];
        const NR: [(u32, u32); 3] = [(8, 3), (8, 4), (16, 4)];

        println!("\n=== Phase A2: recurrent fan-in (side-car, 4-set, adapt_bump 3, rate_reg 5) ===");
        println!("operating point: density {OP_DENSITY}/8, r{OP_UR}/c{OP_UC}");

        for (n, r) in NR {
            let mut fid = Vec::new();
            let mut fam = Vec::new();
            let mut sig = Vec::new();
            let mut peaks = Vec::new();
            let mut prof0 = Vec::new();
            for seed in SEEDS {
                let mut cfg = seq_cfg();
                cfg.density = OP_DENSITY;
                let (mut net, entries) = make_sidecar_seq(seed, 16, OP_UC, OP_UR, n, r, 3, 6);
                let (best, best_at) = train_and_eval_best_seq(&mut net, &entries, seed, 7, &cfg, 4, 100, 10, OP_MAX_TRIALS);
                let (pct, sigma) = rate_profile_seq(&mut net, 16, 7, 0, OP_DENSITY, 8, 24);
                fid.push(best.fidelity);
                fam.push(best.family_acc);
                sig.push(sigma);
                peaks.push(best_at);
                if prof0.is_empty() {
                    prof0 = pct;
                }
            }
            let nseeds = SEEDS.len() as f32;
            let worst_fid = fid.iter().copied().fold(f32::INFINITY, f32::min);
            let mean_fid = fid.iter().sum::<f32>() / nseeds;
            let worst_fam = fam.iter().copied().fold(f32::INFINITY, f32::min);
            let mean_fam = fam.iter().sum::<f32>() / nseeds;
            let mean_sig = sig.iter().sum::<f64>() / SEEDS.len() as f64;
            println!(
                "rec n{n}/r{r} | fidelity worst {worst_fid:.3} mean {mean_fid:.3} | family worst {worst_fam:.3} mean {mean_fam:.3} | σ {mean_sig:.2} | profile {prof0:?} | peak_at {peaks:?}"
            );
        }
        println!("=== end Phase A2 ===\n");
    }
```

- [ ] **Step 2: Verify it compiles and is skipped**

Run: `cargo test --lib seq_phase_a2 2>&1 | tail -3`
Expected: `1 ignored`

Run: `cargo build 2>&1 | grep -c warning`
Expected: `0`

- [ ] **Step 3: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): Phase A2 recurrent fan-in sweep

9 runs: side-car (n,r) {(8,3),(8,4),(16,4)} x 3 seeds at the Phase A1
operating point. Swept separately from the forward path per AGENTS.md.

The recorded sweet spot is n=8 (sigma collapses by n>=24), so this
confirms it at this task's operating point rather than searching openly.

Adds OP_DENSITY / OP_UR / OP_UC / OP_MAX_TRIALS as the operating point.
They default to the spec's starting guess and must be updated from the
Phase A1 run's output -- including OP_MAX_TRIALS from the reported
peak_at -- in their own commit before A2 or Phase B is run.
EOF
```

---

### Task 12: Phase B — the main experiment

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: `#[ignore]` test `seq_phase_b_main`.

- [ ] **Step 1: Write the experiment**

Add inside `mod tests`:

```rust
    /// Phase B — the main experiment. 162 runs: `adapt_bump {1,3,5}` × `rate_reg {0,2,5}` ×
    /// {FF, side-car} × set {4,5,6} × 3 seeds. Run manually in --release:
    ///   cargo test --release --lib seq_phase_b_main -- --ignored --nocapture
    ///
    /// Two headlines.
    ///
    /// **`adapt_bump`** — FF and the side-car store the first token in different physics. Membrane
    /// potential cannot carry it (the `d.max(1)` leak floor drains any potential within ~16 waves; a
    /// 3-token prefix spans 26), so FF's *only* memory is the adaptation trace, which decays at
    /// ρ = 1 − 2⁻⁶ ≈ 0.984/wave and retains ~66% across the prefix. AGENTS.md records ALIF
    /// adaptation as "a strong ~64-wave held-category memory (store-recall)" that does *not* help
    /// XOR-shaped computation — which is why the existing battery never probed it. The side-car
    /// instead holds memory in its L2 scratchpad's ongoing recurrent spiking. Since `adapt_bump`
    /// sets the trace's amplitude, **lowering it should degrade FF while leaving the side-car flat**.
    /// That crossing is the result; two ceiling numbers would not be.
    ///
    /// **`rate_reg`** — a homogenizing pressure whose recorded cost is eroding the class signal. The
    /// battery needs one discriminative direction (2 classes); this task needs 9-15 distinguishable
    /// patterns, so it has strictly more to lose. `rate_reg 0` tests whether fan-in alone sustains
    /// the stack (AGENTS.md names both as liveness fixes).
    ///
    /// The 3×3 grid maps a three-way tension nothing on record covers: `rate_reg` **requires**
    /// adaptation to function, adaptation **is** the FF memory, and `rate_reg` **erodes** what that
    /// memory stores. Expect structure, including empty corners — `rate_reg 5` + `adapt_bump 1` may
    /// be unrescuable, and `rate_reg 0` may be dead everywhere (a finding: fan-in alone cannot
    /// sustain 5 layers).
    #[test]
    #[ignore]
    fn seq_phase_b_main() {
        const SEEDS: [u64; 3] = [1, 2, 3];
        const BUMPS: [i16; 3] = [1, 3, 5];
        const RATE_REGS: [f32; 3] = [0.0, 2.0, 5.0];
        const SETS: [usize; 3] = [4, 5, 6];

        println!("\n=== Phase B: adapt_bump × rate_reg × topology × set (3 seeds) ===");
        println!("operating point: density {OP_DENSITY}/8, r{OP_UR}/c{OP_UC}, max_trials {OP_MAX_TRIALS}");
        for set_size in SETS {
            println!(
                "set {set_size}: {} prefixes, family {}-way, markov-1 {:.3}, markov-2 ceiling {:.3}",
                prefixes(set_size).len(),
                family(set_size).len(),
                markov_k_accuracy(set_size, 1, &family(set_size)),
                markov_k_accuracy(set_size, 2, &family(set_size))
            );
        }

        for set_size in SETS {
            for bump in BUMPS {
                for rate_reg in RATE_REGS {
                    for topo in ["ff", "sidecar"] {
                        let mut fid = Vec::new();
                        let mut fam = Vec::new();
                        let mut det = Vec::new();
                        let mut sig = Vec::new();
                        let mut peaks = Vec::new();
                        let mut prof0 = Vec::new();
                        let mut forks0 = Vec::new();
                        for seed in SEEDS {
                            let mut cfg = seq_cfg();
                            cfg.density = OP_DENSITY;
                            cfg.rate_reg = rate_reg;
                            let (mut net, entries) = if topo == "ff" {
                                make_ff_seq(seed, 16, 5, OP_UC, OP_UR, bump, 6)
                            } else {
                                make_sidecar_seq(seed, 16, OP_UC, OP_UR, 8, 4, bump, 6)
                            };
                            let (best, best_at) = train_and_eval_best_seq(&mut net, &entries, seed, 7, &cfg, set_size, 100, 10, OP_MAX_TRIALS);
                            let (pct, sigma) = rate_profile_seq(&mut net, 16, 7, 0, OP_DENSITY, 8, 24);
                            fid.push(best.fidelity);
                            fam.push(best.family_acc);
                            det.push(best.det_acc);
                            sig.push(sigma);
                            peaks.push(best_at);
                            if prof0.is_empty() {
                                prof0 = pct;
                                forks0 = best.fork_tv.iter().map(|(p, tv)| (p.clone(), *tv)).collect();
                            }
                        }
                        let n = SEEDS.len() as f32;
                        let wf = fid.iter().copied().fold(f32::INFINITY, f32::min);
                        let mf = fid.iter().sum::<f32>() / n;
                        let wfam = fam.iter().copied().fold(f32::INFINITY, f32::min);
                        let mfam = fam.iter().sum::<f32>() / n;
                        let wdet = det.iter().copied().fold(f32::INFINITY, f32::min);
                        let mdet = det.iter().sum::<f32>() / n;
                        let ms = sig.iter().sum::<f64>() / SEEDS.len() as f64;
                        println!(
                            "set {set_size} bump {bump} rate_reg {rate_reg:.0} {topo:>7} | fid w {wf:.3} m {mf:.3} | det w {wdet:.3} m {mdet:.3} | family w {wfam:.3} m {mfam:.3} (ceiling {:.3}) | σ {ms:.2} | profile {prof0:?} | forks {forks0:?} | peak_at {peaks:?}",
                            markov_k_accuracy(set_size, 2, &family(set_size))
                        );
                    }
                }
            }
        }
        println!("=== end Phase B ===\n");
    }
```

- [ ] **Step 2: Verify it compiles and is skipped**

Run: `cargo test --lib seq_phase_b 2>&1 | tail -3`
Expected: `1 ignored`

Run: `cargo build 2>&1 | grep -c warning`
Expected: `0`

Run the whole suite once to confirm nothing regressed:
Run: `cargo test 2>&1 | tail -5`
Expected: all pass. The module contributes **8 non-ignored** tests (`seq_conditionals_correct`, `token_sites_density_and_determinism`, `readout_primitives_correct`, `builders_produce_live_five_layer_nets`, `run_seq_trial_is_deterministic_and_alive`, `build_signal_n_shape_and_rate_term`, `eval_all_prefixes_scores_an_untrained_net`, `seq_training_moves_off_chance`) and **3 `#[ignore]`d** experiments.

- [ ] **Step 3: Commit**

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -F - <<'EOF'
feat(wave_driven_seq_bench): Phase B main experiment

162 runs: adapt_bump {1,3,5} x rate_reg {0,2,5} x {FF, side-car} x set
{4,5,6} x 3 seeds. Reports worst+mean fidelity, deterministic-prefix and
family accuracy against the analytic Markov-2 ceiling, sigma, the
per-layer profile, per-fork TV, and peak_at.

Two headlines. adapt_bump: FF and the side-car store the first token in
different physics -- the leak floor drains membrane potential within ~16
waves against a 26-wave prefix, so FF's only memory is the adaptation
trace, while the side-car holds it in recurrent spiking. Lowering bump
should degrade FF and leave the side-car flat. That crossing is the
result; two ceiling numbers would not be, which is the existing battery's
problem.

rate_reg: a homogenizing pressure whose recorded cost is eroding the
class signal. The battery needs one discriminative direction; this task
needs 9-15 distinguishable patterns, so it has more to lose.

The 3x3 maps a three-way tension nothing on record covers: rate_reg
requires adaptation to work, adaptation is the FF memory, and rate_reg
erodes what it stores.
EOF
```

---

### Task 13: Run the experiments and record findings

The runs are manual and long (~5× the existing confirmation suite). This task is the recording discipline around them.

**Files:**
- Modify: `src/bench/wave_driven_seq_bench.rs` (operating-point consts, from A1's output)
- Modify: `docs/experiments_results.md`

- [ ] **Step 1: Run Phase A1**

```bash
cargo test --release --lib seq_phase_a1_forward_sweep -- --ignored --nocapture 2>&1 | tee /tmp/claude-1000/-home-driou-dev-project-wave-net/21675864-7776-4d0e-81cf-5ffb4c931b6f/scratchpad/a1.log
```

Read the output. Pick the `(density, r, c)` with σ nearest 1, no dead layer (any `profile` entry at 0.0 for layers 1..4) and no saturated layer. Note the largest `peak_at`.

- [ ] **Step 2: Update the operating point from measurement, and commit it separately**

Edit `OP_DENSITY`, `OP_UR`, `OP_UC` to A1's winner, and `OP_MAX_TRIALS` to ~2× the largest reported `peak_at`.

```bash
git add src/bench/wave_driven_seq_bench.rs
git commit -m "chore(wave_driven_seq_bench): set operating point from Phase A1 measurement"
```

The commit message body must record the chosen values, the σ and profile that justified them, and the `peak_at` that set `OP_MAX_TRIALS`. A separate commit so the A1-derived choice is auditable against the run log.

- [ ] **Step 3: Run Phase A2 and Phase B**

```bash
cargo test --release --lib seq_phase_a2_recurrent_sweep -- --ignored --nocapture 2>&1 | tee /tmp/claude-1000/-home-driou-dev-project-wave-net/21675864-7776-4d0e-81cf-5ffb4c931b6f/scratchpad/a2.log
cargo test --release --lib seq_phase_b_main -- --ignored --nocapture 2>&1 | tee /tmp/claude-1000/-home-driou-dev-project-wave-net/21675864-7776-4d0e-81cf-5ffb4c931b6f/scratchpad/b.log
```

If Phase B's wall-clock proves painful, the spec's recorded fallback is to **stage rather than trim seeds**: run the full 3×3 `bump × rate_reg` grid at set 4 only (54 runs), then extend the set {4,5,6} axis at the winning cell (+12) — ~66 runs, at the cost of the `set × bump` interaction. Never drop to fewer than 3 seeds; single-seed numbers are what AGENTS.md exists to prevent.

- [ ] **Step 4: Record findings in `docs/experiments_results.md`**

Append a section following the file's existing shape. Report **worst + mean** over seeds, σ, and per-layer profiles. State outcomes honestly — a negative is a result. Cover, explicitly:

1. **Can it memorize?** Fidelity and deterministic-prefix accuracy at the best cell. If it cannot, say so and give the σ/profile diagnosis (dynamics collapse vs credit starvation).
2. **Are the forks calibrated?** Per-fork TV against the 67/33 and 50/50 targets.
3. **Does it remember, or count statistics?** Family accuracy vs the analytic Markov-2 ceiling (1/k). This is the load-bearing claim; below-ceiling means no memory was demonstrated regardless of how good the other numbers look.
4. **The `adapt_bump` crossing** — did FF degrade with bump while the side-car held flat? If not, the adaptation-as-memory hypothesis is wrong, which is publishable and should be stated plainly.
5. **The `rate_reg` trade** — did `rate_reg 0` stay live? Did it improve fidelity by not eroding the patterns?
6. **Caveats.** This is a **memorization/capacity** measurement with **no holdout** — it must not be reported as generalization. Depth is fixed at 5 and `adapt_decay` at 6 (stated deviations from the AGENTS.md defaults, per the spec).

- [ ] **Step 5: Commit**

```bash
git add docs/experiments_results.md
git commit -m "docs(experiments): wave_driven sequence-recall memory results"
```

---

## Notes for the implementer

- **Do not tune to make a test pass.** Tasks 5 and 9 have liveness/learning assertions that could fail at the default operating point. If they do, report the number with the σ and per-layer profile — that is Phase A1's question, and a dead default below 32 sites is exactly what the spec's Design analysis predicts. Tuning around it silently would destroy the finding.
- **The Markov-2 control is the load-bearing claim.** Every other metric can look good while the network has learned nothing but bigram statistics. Family accuracy above `1/k` is the only evidence of memory.
- **`cargo test` must stay fast.** All three experiments are `#[ignore]`d. Only the unit tests and the small `seq_training_moves_off_chance` smoke run by default.
- **`best_at` exists to be used.** Phase B's `max_trials` should come from Phase A1's measurement, not from this plan's guess.
