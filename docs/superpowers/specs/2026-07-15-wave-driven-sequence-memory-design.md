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

This is corroborated by a recorded finding rather than resting on the arithmetic above — AGENTS.md
already states: *"ALIF adaptation is both a working memory **and** load-bearing for liveness. It is a
strong **~64-wave held-category memory (store-recall)**; it does **not** help linear echo (MC) or
nonlinear temporal computation (XOR) feed-forward."* The timescale matches exactly (`adapt_decay: 6` →
τ ≈ 64 waves), and "store-recall" is precisely this task. It also explains the gap this benchmark fills:
the existing battery is all XOR-shaped computation, which is the regime where adaptation is recorded as
*not* helping — so adaptation-as-memory has never actually been probed. **This task is adaptation's home
turf.**

So the two topologies store the first token in different physics — FF in the **adaptation trace**, the
side-car in its L2 scratchpad's ongoing **recurrent spiking**. Since `adapt_bump` sets the trace's
amplitude, **lowering it should degrade FF while leaving the side-car flat.** That predicted crossing is
the experiment's headline result, and it is why `adapt_bump` is the main axis rather than a setting.

### 4. `rate_reg` homogenizes, and this task has the most to lose

`rate_reg` is a **homeostatic rate controller**, active during **training only** (`build_signal` is called
at `wave_driven_bench.rs:128`, inside the train loop; the eval path at `:135` never touches it). The
chain: `signal[tz][j] = task_sig + rate_reg·(rate_j − rate_target)`, then `dfa_update` applies
`shadow[widx] += -lr · signal[tz][j] · e` (`network.rs:422`), where `e` is the eligibility of a synapse
*into* `j` and is non-negative. So a neuron firing above `rate_target` gets its incoming weights pushed
down, one firing below gets them pushed up — negative feedback, per neuron, per trial, toward 10%.

Its purpose is liveness: deep FF stacks starve (activity decays layer by layer; eligibility accrues only
on target fire, so no firing ⇒ no credit ⇒ no learning). AGENTS.md records it as a *"conclusive liveness
rescue… chance → ~980 on temporal XOR, 5-seed robust"*, requiring ALIF.

**But it is a homogenizing pressure, and its recorded cost is exactly what this task is made of.** The
caveat states it *"homogenizes firing rates and erodes the class signal"*. The battery needs **one**
discriminative direction in the 256-dim spike-count space — two classes. This task needs **9–15
distinguishable patterns**, one per prefix. There is far more structure here for a homogenizing term to
erode, so **this task is more exposed to the `rate_reg` pathology than anything in the existing
battery**, and `rate_reg: 5.0` must not be inherited as a constant.

The opening: AGENTS.md names **two** fixes for liveness starvation — `rate_reg`, and *"the fix is more
fan-in"*. Phase A1 sweeps fan-in. If an `(r,c)` is intrinsically live, `rate_reg` may be unnecessary,
which would remove the over-training collapse, make best-checkpointing belt-and-braces rather than
load-bearing, and lift the pressure eroding the prefix patterns. Hence `rate_reg` is a **Phase B axis**.

Three-way tension worth naming, since the 3×3 `adapt_bump × rate_reg` grid maps it: `rate_reg` **requires**
adaptation to function; adaptation **is** the FF memory (§3); and `rate_reg` **erodes** what that memory
stores. Nothing in the recorded findings maps this interaction.

## Components

### 0. Remove `wave_driven`'s dead `readout` flag (engine, prerequisite)

The flag makes a layer drain-only (`wave.rs:63-66` returns before decide/fire) and is a silent training
killer: zero `act` ⇒ the readout SGD multiplies by zero; eligibility accrues only on **target** fire ⇒
the synapses feeding it never accrue and `dfa_update` no-ops. It is also **dead** — in `wave_driven`,
`new_with_readout` is called by nothing but its own unit test.

Delete, from `wave_driven` only: the `Layer.readout` field (`neurons.rs:96`), `Network::new_with_readout`
(`network.rs:41`), `build`'s `readout_last` parameter (`network.rs:49,62-63`), the `wave.rs:64-66`
branch, and the `readout_integrates_without_firing` test (`network.rs:552`).

**Zero behavioural change**, which is the point: every surviving constructor (`new`, `new_dense`) already
passes `readout_last = false`, so the deleted branch was never taken and the recorded battery reproduces
bit-exactly. The `equivalence_tests.rs` oracles (sparse==dense; `adapt_bump==0` bit-exact vs
`wave_bitnet`) are unaffected — neither uses the flag.

**Not touched:** `wave_bitnet` (the flag is serialized into the fingerprint-bound `.wbm` format —
`persist.rs:82,108,117`, asserted at `:357`) and `wave_resonate` (its readout is a live leaky integrator:
`tau_out`, and `dfa_update` skips ω/b′ training for readout layers at `network.rs:373`). AGENTS.md's
*Readout layers* paragraph is updated to scope the feature to those two engines and record the footgun.

This resolves a standing self-contradiction in AGENTS.md, which described readout layers as "the output
symmetry" in the architecture section while the benchmarking conventions said "no dedicated readout
layer" — plausibly why the flag existed, unused, in all three engines.

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
The **top spiking layer is read directly**; no dedicated readout layer (see Phase 0). The top layer is an
ordinary spiking layer whose only peculiarity is having no outgoing work — `make_sidecar`'s L4 has
genuinely empty topology, while `make_ff`'s top layer carries an inert level-1 topology aimed at a
nonexistent layer 5 (`entries[top] = vec![]` keeps DFA off it). It receives from L3, integrates, and
fires normally, with `rate_reg` driving it toward `rate_target`.

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
  rate_target)`, with `b = w[c][j]` at the top layer and `dfa_weight(seed, tz*ls+j, c)` below. `rate_reg`
  is **bench-side**, as in the existing harness: the engine only exposes `layer_spike_count(z)`
  (`network.rs:183`, "for rate_reg"), and the rule lives here per AGENTS.md.
- `TaskCfg`-equivalent: `size 16, present 6, delay 4, read 8, readout_lr 0.02, hidden_lr 0.004,
  rate_reg 5.0, rate_target 0.1`.
- **`ttot` varies per trial here, unlike the battery.** `build_signal` normalizes rate by
  `denom = ttot.max(1)`, and our prefixes differ in length (`[1]` spans 14 waves, `[1,2,3]` spans 34).
  `rate_target` targets a firing *frequency*, not a count, so this is already correct — but it is
  load-bearing in a way it is not for the fixed-length battery, and worth watching in Phase A.
- **Duration:** `train_and_eval_best`-equivalent, **`eval_every 100`**, `patience 10`,
  `max_trials 12000`. Compare at the **peak**, never at a fixed trial count — `rate_reg` over-trains into
  a non-monotonic collapse (recorded: transient at ~4 layers, permanent by ~12; we are at 5).
  `eval_every 100` rather than the battery's 300 because the exact 9-prefix eval costs ~9% overhead
  against `max_trials` where a sampled `holdout: 200` would cost ~200% — cheap exactness buys finer peak
  resolution, which is the whole point of best-checkpointing under `rate_reg`.

### 6. Evaluation — exact, not sampled

**This task has no holdout, and cannot have one.** Not a compromise — a consequence of what is being
measured.

A holdout answers *"does it work on inputs it has never seen?"*. The battery can ask that because XOR and
parity are **functions** over a huge input space: `task(seed, t)` trains and `task(seed, EVAL_OFFSET+i)`
evaluates on genuinely fresh cue bits, and only a network that learned the *rule* can answer them.

Here the entire universe of inputs is **9 prefixes** (12 / 15 for the 5- and 6-sets). There is no 10th.
Holding out `[1,2,3]` would present a prefix whose answer (`4`) is **arbitrary** — a memorized fact, not
a rule instance, derivable from nothing the network could have seen. It would guarantee failure and
measure nothing. Train and test being the same items is what memorization *means*; the question is "can
it store and reproduce 9 arbitrary associations?", and there is deliberately no rule to infer.

**The Markov-2 control does the job a holdout normally does.** The degenerate solution here is not
memorizing (that is the task) but answering from recent context alone — and a bigram/trigram lookup
provably scores 1/k on the `[·,2,3]` family while genuine 3-token memory scores 100%. That is the
validity guard, and it is why the family exists.

**Consequence:** this is a memorization/capacity measurement and must never be reported as
generalization. The 4/5/6 set axis is the capacity probe.

**Exactness.** The engine is deterministic and resets per trial (`reset_state()`), so **each prefix
yields exactly one score vector**. Evaluation enumerates all 9/12/15 prefixes, one run each — exact, no
sampling, no variance, ~20× cheaper than the battery's `holdout: 200`.

This inverts the usual objection to best-checkpointing. Reporting the max over evals normally selects on
the reported set, an optimistic bias — and the battery has it, since its sampled `holdout: 200` carries
variance the max cherry-picks upward. **Our eval has no sampling noise**, so the max over evals reads the
true peak of a deterministic curve rather than the top of the noise. Having no holdout makes
best-checkpointing *cleaner* here, not dirtier. Residual caveat, stated plainly: the peak is still an
upward-biased estimate of "accuracy at a fixed sensible stopping point", since a jagged trajectory is
read at its top — but the jaggedness is real learning dynamics, and both topologies are measured
identically, so the FF/side-car delta is unaffected.

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

The r/c sweep runs **first**, as a reported result rather than a hedge — it is the axis AGENTS.md most
wants swept, and its outcome fixes the operating point everything after it uses.

**Phase A1 — forward fan-in + density (24 runs).** density {32, 64} × forward `(r,c)` ∈ {(2,8), (3,16),
(3,32), (4,16)} × 3 seeds. FF, 4-set, `adapt_bump 3` (the recorded good point; the interaction with bump
is Phase B's business). Constraint: `c ≤ (2r+1)²`, so r2 caps at 25 and r3 at 49.

Select on **dynamics** — σ near 1, a healthy per-layer profile, no dead or saturated layer — with
accuracy secondary. Dynamics are the low-variance, seed-robust signal; picking an operating point on
accuracy across 3 seeds invites a fluke.

**These two axes are not independent, and that is itself a result.** Input drive is set by the product
`sites × c` (§Design analysis 1: each neuron needs ≥2 incoming synapses, i.e. `sites × c / 256 ≥ 2`), so
density and `c` trade off directly at L0 — 64 sites needs `c ≥ 8`, 32 sites needs `c ≥ 16`. But `c` also
sets **hidden**-layer drive, where the source is ~25 firing neurons (`rate_target 0.1`), giving
`25 × c / 256` ≈ 1.6 synapses per neuron at c16 — right at the coincidence floor. So `c` is doing two
different jobs and the sweep should show them separating.

**Phase A2 — recurrent fan-in, swept separately (9 runs).** Side-car `(n, r)` ∈ {(8,3), (8,4), (16,4)} ×
3 seeds, at A1's forward winner, `adapt_bump 3`, 4-set. Separate from the forward sweep per AGENTS.md;
the recorded sweet spot is n=8 (σ collapses by n≥24), so this is a confirmation at this task's operating
point rather than an open search.

**Phase B — main experiment (162 runs).** `adapt_bump` {1,3,5} × **`rate_reg` {0,2,5}** × {FF, side-car}
× set {4,5,6} × 3 seeds, at the Phase A operating point. Two headlines: the predicted FF/side-car
crossing in `adapt_bump`, and the liveness-vs-erosion trade in `rate_reg` (§Design analysis 4). At
`rate_reg 0`, `rate_target` is inert — the term drops out of `build_signal` entirely.

**A1's `(r,c)` is selected under `rate_reg 5`, which masks liveness starvation.** A config may look
healthy only because the regulariser props it up. Accepted rather than fixed: running A1 at
`rate_reg 0` risks 24 null runs if nothing in the `(r,c)` range is intrinsically live at 5 layers.
Phase B's `rate_reg 0` cells resolve it directly — and "no fan-in in this range survives without
`rate_reg`" is itself a legitimate finding, not a failed experiment.

**Cost, stated honestly:** ≈195 runs at size 16 / 5 layers. Per-run cost is ~0.6× a battery run (5× the
trials, ¼ the neurons at size 16, ½ the fan-in at c16), so this is **≈5× the existing confirmation
suite** — not the "under 24 equivalents" an earlier draft of this spec claimed by ignoring `max_trials`.
Mitigation: `max_trials: 12000` is a **ceiling**, not the expected cost — `patience 10` exits ~1000
trials past the peak, and a 9-prefix task may converge well short of the cap. **Phase A must report where
the peak lands**, and Phase B's `max_trials` should be set from that measurement rather than from this
estimate. If wall-clock proves painful, the fallback is to stage rather than trim seeds: run the full 3×3
`bump × rate_reg` grid at set 4 only (54 runs), then extend the set {4,5,6} axis at the winning cell
(+12), reaching ~66 runs at the cost of the `set × bump` interaction.

All phases are `#[ignore]`d inline tests, run manually in `--release`.

## Non-goals and deviations

- **BPTT** — permanently out of scope, project-wide. Not proposed, not benchmarked.
- **Generalization** — this task cannot measure it (§6). No held-out claim will be made.
- **Engine changes** — none. Learning rules live in `bench/`; the engine is untouched.
- **A dedicated readout layer** — rejected per AGENTS.md; top spiking layer read directly.
- **Sparse / one-hot input drive** — rejected on the arithmetic in §Design analysis 1–2, not on taste.
  If revisited, it is a separate spike answering "can `wave_driven` compute under ~1-neuron-per-layer
  drive at all?", with its own σ sweep — never confounded with this question.
- **Defaults dropped, and why** (AGENTS.md *Defaults* asks that these be stated, not that they never
  happen):
  - **Depth is fixed at 5**, not swept. Depth is matched between FF and the side-car so the comparison
    isolates topology; sweeping it would confound the headline axis. Depth-reach is also already
    characterised for `wave_driven` (the skip-topology work reaching 24).
  - **`adapt_decay` fixed at 6** (τ ≈ 64 waves vs a 26-wave prefix), matching the battery. It sets the
    adaptation trace's *horizon* where `adapt_bump` sets its *amplitude*; sweeping both would map FF's
    memory capacity in two dimensions and is the natural follow-up once the crossing is confirmed.
  - **`adapt_bump 3` is pinned during Phase A**, so the r/c winner is selected under one bump. Accepted:
    the alternative is a 3× larger pilot to tune a variable Phase B then sweeps anyway.

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
- **`rate_reg 0` may be dead everywhere**, collapsing a third of the Phase B grid to chance. That is a
  finding (fan-in alone cannot sustain 5 layers; `rate_reg` is mandatory), not a wasted third — but it
  would make `rate_reg 2` the only informative non-default cell.
- **`rate_reg 5` + `adapt_bump 1` may be an empty corner.** `rate_reg` requires ALIF, and bump 1 is the
  weakest non-zero adaptation; the regulariser may be unable to rescue anything there. Expected
  structure in the 3×3 grid rather than a defect, but worth predicting before seeing it.
- **`readout_lr`/`hidden_lr` are inherited from a 2-class harness.** At V=9 the task term
  `Σ_{c<9} b·err[c]` sums nine terms rather than two, so its magnitude relative to the fixed `rate_reg`
  term shifts — the effective task/liveness balance is not the battery's, even at identical `rate_reg`.
  Phase A should sanity-check that training moves at all before Phase B commits 162 runs.
