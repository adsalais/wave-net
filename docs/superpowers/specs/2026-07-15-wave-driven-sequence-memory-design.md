# wave_driven — sequence recall: adaptation-memory vs recurrent-memory

- **Date:** 2026-07-15
- **Status:** design approved; ready for an implementation plan
- **Scope:** A new **memorization** benchmark for `wave_driven`: recall the next token of a memorized
  branching sequence set, where forks make the correct answer a **distribution** and one prefix family
  can only be resolved by remembering the token from three steps back. Bench-only — **the engine is not
  touched**. Research/validation work, not a performance suite.

## Motivation

Every task in the current `wave_driven` battery (temporal XOR, parity-4, distractor-XOR, flip-flop) is a
**2-class function** of the cue sequence: the net computes an answer. None of them ask it to **store an
arbitrary token and reproduce it**. Recall and computation are different capabilities, and we have only
ever measured the second.

There is also a methodological problem the recorded findings already flag: the recurrence-confirmation
suite found spike-ψ `εᵃ` side-car beats FF **4/4 tasks, 3-seed, matched baseline — but with all
conditions at ceiling**. A battery where both topologies saturate cannot rank them; the delta is real but
unquantified. We need a task with **headroom**, and ideally one where FF and side-car are predicted to
diverge for a mechanistic reason rather than by luck.

This task supplies both. Its analysis (below) shows FF and the side-car must store the first token in
**physically different places** — FF in the adaptation trace, the side-car in recurrent spiking — which
turns `adapt_bump` into an axis along which they should come apart.

## Goal (success criterion)

Answer two questions, and record the outcome honestly (per the project rule: conclusive results, don't
overclaim):

1. **Can `wave_driven` memorize a branching sequence set?** Reproduce deterministic continuations, and
   match the fork marginals (`[1]`→67/33, `[1,2]`→50/50) as *calibrated readout mass*.
2. **Does it remember, or is it counting statistics?** Beat the analytic Markov-2 ceiling on the
   `[·,2,3]` disambiguation family — the one place where only a 3-token memory suffices.

There is **no hard pass/fail assertion** on either; the experiment prints the numbers. A cheap unit test
does gate the task generator and the closed-form conditionals.

## Design analysis (why the obvious encoding is impossible)

Three findings from reading the engine drove the design. They are recorded here because they are
non-obvious and a future reader will otherwise re-propose the rejected options.

### 1. The engine is a coincidence detector by construction

`wave.rs:92`:

```rust
let d = (pot >> la) + (pot >> lb);
layer.potential[i] = pot - if pot > 0 { d.max(1) } else { d };
```

The **`d.max(1)`** floor drains ≥1 potential per wave from any positive membrane. A single +1 synapse
delivers exactly +1 per wave, so **net drift is zero** — the potential oscillates between 0 and 1 and
never climbs. With **2** coincident synapses the drift is +1/wave, climbing to a steady state of
**p\* = 16** (where `pot>>3` reaches 2 and balances the input, under the bench leak `(3,5)`).

One synapse → 0. Two → 16. This is a cliff, not a gradient, and **training cannot cross it**: weights are
ternary, so a synapse's maximum delivery is +1, and the leak floor eats it. Thresholds are not trainable
either (`neurons.rs:167-180`; only weights are). **≥2 coincident synapses is an arithmetic precondition
for any activity at all.**

### 2. Therefore one-hot ("each grid position is a number") cannot work

`sample_distinct_cells` (`synapse.rs:123`) guarantees a source sends **at most one** synapse to any given
target. So a single firing L0 site puts every target on the wrong side of the cliff, and **no neuron
anywhere can ever fire** — provably, for any weights, any training, any threshold > 1.

The failure is structural, not a tuning problem. Even granting a retuned threshold that lets one synapse
fire (`baseline_init: 5, threshold_jitter: 8` yields σ ≈ 0.93), the resulting neuron fires on *one* input
— an OR gate, a relay. The network degenerates into a ~1-neuron-per-layer chain that faithfully relays
*which* token fired and can never form the conjunction "1 **and then** 2". A conjunction needs two
sites' synapses to land on a shared target, which under one-hot place coding only happens for
grid-adjacent tokens — making `1→2` learnable and `1→16` geometrically impossible, for reasons with
nothing to do with the task.

**Sparsity is cheap precisely because it is not computing.** Density is where the AND lives.

### 3. Adaptation is FF's only memory

Membrane potential cannot carry the first token: the `.max(1)` floor drains any potential to zero within
~16 waves, and a 3-token prefix spans 26. But the adaptation trace decays at ρ = 1 − 2⁻⁶ = **0.984/wave**
(`adapt_decay: 6`), retaining **~66%** across those 26 waves.

So the two topologies store the first token in different physics — FF in the **adaptation trace**, the
side-car in its L2 scratchpad's ongoing **recurrent spiking**. Since `adapt_bump` sets the trace's
amplitude, **lowering it should degrade FF while leaving the side-car flat.** That predicted crossing is
the experiment's headline result, and it is why `adapt_bump` is the main axis rather than a setting.

## Components

### 1. New module `src/bench/wave_driven_seq_bench.rs` (registered in `src/bench/mod.rs`)

Self-contained, `#[cfg(test)] mod tests`, following the existing bench harnesses. The N-class readout
lives here rather than generalizing `wave_driven_bench.rs` in place: all three harnesses
(`wave_driven`, `wave_bitnet`, `wave_resonate`) already carry a private `softmax2`, so per-harness
readout code is the established pattern, and the confirmed battery's recorded numbers stay
bit-reproducible. (An N-class softmax is bit-exact with `softmax2` at V=2 *except* at init, where all
scores tie at 0.0 and `(s1 > s0)` breaks toward class 0 while a naive `max_by` argmax breaks toward the
last class — a needless reproducibility risk for zero benefit.)

### 2. The sequence sets

Vocabulary of **9 tokens** `{1,2,3,4,5,6,7,8,16}` → ids 0..8. Sets are nested, so the 4-set's
conditionals survive into the 5- and 6-sets (S5/S6 start on fresh tokens):

| set | sequences | prefixes | `[·,2,3]` family | Markov-2 ceiling |
|---|---|---|---|---|
| 4 | `1→2→3→4`, `1→2→4→8`, `1→4→8→16`, `2→2→3→5` | 9 | 2-way | 50% |
| 5 | + `3→2→3→6` | 12 | 3-way | 33% |
| 6 | + `4→2→3→7` | 15 | 4-way | 25% |

S5 and S6 deliberately extend the **same** `2→3` collision, so growing the set deepens the memory test
rather than merely adding capacity: the Markov-2 ceiling falls while true memory stays at 100%.

**Trial sampling:** sequence uniform over the set; prefix length uniform in 1..3; target = the next
token. Uniform sequence sampling yields the target conditionals for free — conditioned on prefix `[1]`,
the sequence is uniform over {S1,S2,S3}, giving `{2: 2/3, 4: 1/3}`.

**Closed-form conditionals (4-set), all 9 prefixes:**

| prefix | true conditional | kind |
|---|---|---|
| `[1]` | `{2: 2/3, 4: 1/3}` | **fork** |
| `[1,2]` | `{3: 1/2, 4: 1/2}` | **fork** |
| `[2]` | `{2: 1}` | det |
| `[1,4]` | `{8: 1}` | det |
| `[2,2]` | `{3: 1}` | det |
| `[1,2,4]` | `{8: 1}` | det |
| `[1,4,8]` | `{16: 1}` | det |
| `[1,2,3]` | `{4: 1}` | det, **family** |
| `[2,2,3]` | `{5: 1}` | det, **family** |

**Generator:** `seq_task(set_size) -> impl Fn(u64, usize) -> (Vec<usize> /* token ids */, usize /*
target id */)` — the set size is captured, and the returned closure matches the existing task convention
`Fn(task_seed, trial)`, deterministic in `trial`.

**Unit test `seq_conditionals_correct`** (runs in `cargo test`, no training): asserts the prefix
enumeration, the closed-form conditionals, and the Markov-1/2 ceilings for all three set sizes.

### 3. Encoding

`token_sites(task_seed, size, token, density)` — a new generator alongside `cue_sites`, selecting sites
by `mix(key(task_seed, loc, token, 0, CUE_P)) & 7 < density`, so `density ∈ {1, 2}` → **32 or 64** of the
256 sites. (A distinct predicate from `cue_sites`' `& 3 == 0`, hence a distinct site set — irrelevant, as
this is a new task.) Token codes are random population codes: overlapping, unstructured, and therefore
free of any geometric shortcut the net could exploit instead of remembering.

**Windows:** `present 6 / delay 4 / read 8`. A 3-token prefix spans 26 waves, leaving ~66% of token 1's
adaptation trace alive at read time.

**Known stressor:** `2→2` (in S4) presents the same site pattern twice consecutively. L0 injection forces
a fire regardless of cooldown (`wave.rs:58-61`, `potential = i16::MAX; cooldown = 0`) and L0 has
`adapt_bump = 0` forced (`network.rs:57-61`), so L0 repeats cleanly — but L1 is adapted by the first
presentation and will answer the second more weakly. This is a genuine interaction with the `adapt_bump`
axis and is expected to show up in the results, not a defect.

### 4. Topologies (5 layers each — matched)

L0 is a forced injection transducer that does not compute, so 5 layers is 4 computing layers.
Per AGENTS.md the **top spiking layer is read directly**; no dedicated readout layer.

**Do not call `Network::new_with_readout` — it would silently block all training.** The engine's
`readout: bool` flag makes a layer a drain-only integrator: `wave.rs:63-66` returns *before*
decide/fire, so the layer never spikes. Two consequences, both silent:

1. `act[j]` (top-layer read-window spike counts) is all zeros, so the readout SGD
   `w[c][j] -= readout_lr * err[c] * act[j]` multiplies by zero and never learns.
2. Eligibility accrues only when the **target** fires (`e_ij += pretr_i`), so the incoming L3→L4
   synapses never accrue, and `dfa_update` is a no-op on them.

Use plain `Network::new` (which passes `readout_last = false`, `network.rs:38-40`), as both `make_ff`
and `make_sidecar` already do. The top layer is then an ordinary spiking layer whose only peculiarity is
having no outgoing work — `make_sidecar`'s L4 has genuinely empty topology, while `make_ff`'s top layer
carries an inert level-1 topology aimed at a nonexistent layer 5 (`entries[top] = vec![]` keeps DFA off
it). It receives from L3, integrates, and fires normally, with `rate_reg` driving it toward
`rate_target`.

- **FF:** `make_ff(seed, size 16, layers 5, up_count 16, up_radius 3, adapt_bump, adapt_decay 6)`,
  membrane-only (`elig_beta 0`).
- **Side-car:** canonical `make_sidecar(seed, size 16, uc 16, ur 3, n 8, r 4, adapt_bump, adapt_decay 6)`
  — L0→L1(+1); L1→L3(+2); L2 self(0)+→L3(+1); L3→L2(−1)+→L4(+1); L4 read. Spike-ψ `εᵃ`
  (`elig_beta 0.4`, `rec_tau 20`), rec_count 8 (the recorded sweet spot).

`adapt_bump` must stay **> 0**: the side-car's `elig_beta 0.4` needs an adaptation trace to couple to,
and `rate_reg` requires ALIF.

### 5. Readout and training (bench-side, N-class)

- `w: Vec<Vec<f32>>` shaped **V × size²** (V = 9); `score` = per-class dot product over the read-window
  spike counts; `softmax_n` (max-subtract) → cross-entropy `err[c] = p[c] − onehot[c]`; readout SGD
  `w[c][j] -= readout_lr * err[c] * act[j]`.
- `build_signal` generalized over V: `signal[tz][j] = Σ_{c<V} b·err[c] + rate_reg·(spike_count[j]/ttot −
  rate_target)`, with `b = w[c][j]` at the top layer and `dfa_weight(seed, tz*ls+j, c)` below.
- `TaskCfg`-equivalent: `size 16, present 6, delay 4, read 8, readout_lr 0.02, hidden_lr 0.004,
  rate_reg 5.0, rate_target 0.1`.
- **Duration:** `train_and_eval_best`-equivalent, `eval_every 200, patience 10, max_trials 12000`.
  Per AGENTS.md, compare at the **peak** of the duration sweep, never at a fixed trial count —
  `rate_reg` over-trains and collapses accuracy non-monotonically after convergence.

### 6. Evaluation — exact, not sampled

**This task has no holdout, by design.** The 4-set has only 9 distinct prefixes (12 / 15 for the 5- and
6-sets); train and test are the same items, which is correct because remembering *is* the task. It is a
memorization and capacity measurement and must not be reported as generalization.

The engine is deterministic and resets per trial (`reset_state()`), so **each prefix yields exactly one
score vector**. Evaluation enumerates all 9/12/15 prefixes, one run each — exact, no sampling, no
variance, ~20× cheaper than the battery's `holdout: 200`. This is what affords `eval_every 200`.

Metrics per (topology, bump, set, seed):

- **Deterministic prefixes:** top-1 accuracy in permille (repo convention).
- **Forks:** total variation `TV = ½ Σ|p_true − p_softmax|` against the closed-form conditional. Bounded
  and legible: TV=0 is perfect; TV=0.5 means a 50/50 fork collapsed onto one branch; TV=⅓ means `[1]`'s
  67/33 collapsed. Reported per fork prefix, not averaged away.
- **Disambiguation:** accuracy on the `[·,2,3]` family against its analytic Markov-2 ceiling (1/k).
- **Controls:** Markov-1 and Markov-2 ceilings computed in closed form from the sequence set — free, no
  training. Markov-3 equals full memory (prefixes are ≤3 tokens), so **Markov-2 is the discriminating
  control**; beating it is the evidence of recall.
- **Required by AGENTS.md:** σ branching ratio, per-layer spiking profile, fan-in density; **worst +
  mean** over seeds, never single-seed.

## Run matrix

**Phase A — liveness pilot (6 runs).** density {32, 64} × `adapt_bump` {1,3,5}, FF only, 4-set, 1 seed.
Purpose is to pick density on σ and the spiking profile. The analysis says 32 is the floor (≈2 synapses
per neuron: `sites × 16 / 256 ≥ 2`); this measures it rather than trusting the arithmetic. A 1-seed pilot
for operating-point selection is not the reported result.

**Phase B — main experiment (54 runs).** `adapt_bump` {1,3,5} × {FF, side-car} × set {4,5,6} × 3 seeds,
at the density Phase A picks.

60 runs at size 16 / 5 layers ≈ 13 size-32-equivalents — below the existing confirmation suite's 24.
Both phases are `#[ignore]`d inline tests, run manually in `--release`.

## Non-goals and deviations

- **BPTT** — permanently out of scope, project-wide. Not proposed, not benchmarked.
- **Generalization** — this task cannot measure it (§6). No held-out claim will be made.
- **Engine changes** — none. Learning rules live in `bench/`; the engine is untouched.
- **A dedicated readout layer** — rejected per AGENTS.md; top spiking layer read directly.
- **Sparse / one-hot input drive** — rejected on the arithmetic in §Design analysis 1–2, not on taste.
  If revisited, it is a separate spike answering "can `wave_driven` compute under ~1-neuron-per-layer
  drive at all?", with its own σ sweep — never confounded with this question.
- **Deviation from AGENTS.md:** r3/c16 is **fixed**, not swept, and depth is fixed at 5. AGENTS.md
  requires sweeping radius/count and depth. Mitigation: an r/c sanity check at the winning `adapt_bump`
  once the headline result is in. Flagged explicitly so the result is not over-read.
- **`adapt_decay`** — fixed at 6 (τ ≈ 64 waves vs a 26-wave prefix), matching the battery. Sweeping it
  alongside `adapt_bump` would map FF's memory horizon in two dimensions; deferred to a follow-up once
  the crossing is confirmed.

## Risks

- **Both topologies ceiling again.** The failure mode this task exists to escape. Mitigated by the
  `adapt_bump` axis (a crossing curve survives even if the endpoints saturate) and by the 4/5/6 capacity
  axis, which raises difficulty monotonically.
- **The adaptation-memory prediction is wrong** and FF holds the first token some other way. That is a
  publishable negative and the reason `adapt_bump` is swept rather than assumed.
- **9 tokens × 64 sites on 256 sites** means each site belongs to ~2.25 token codes. Two tokens share
  ~16 of their ~64 sites, leaving a large Hamming distance — expected fine, but Phase A's spiking profile
  will show if token codes have smeared together.
- **Nine prefixes may be too few** to prevent an unanticipated shortcut. The Markov-2 control is the
  guard: a shortcut that beats Markov-2 on the family is, by construction, memory.
