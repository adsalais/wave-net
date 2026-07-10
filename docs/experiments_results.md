# Experiment results — the `wave_net` RSNN

**What this is:** the standing record of results on `wave_net`, the stored-weight RSNN engine
(`src/bench/` drives the experiments). Every result here is **held-out and multi-seed** — the bar the
pre-pivot approach failed to meet. Design rationale lives in the per-spec docs under
`docs/superpowers/specs/`; literature framing lives in `docs/related-work.md`.

> **History.** This file previously logged the pre-pivot investigation — an ALIF-vs-LIF
> characterization of the fixed random `±1` reservoir and a long arc on training per-neuron *thresholds*
> over it (the approach now frozen in `wave_state_machine`). That blow-by-blow was **removed as
> superseded** once it established, held-out and multi-seed, that threshold-only training does not
> reliably learn. It lives in git history; the load-bearing conclusion is summarised below.

## Setup

- **Substrate:** the integer `wave_net` engine — synapse **addresses** stay procedural (hash-regenerated
  at fire time, free), while **weights** are **stored** (`i8` + `f32` shadow) and trainable. ALIF =
  per-neuron adaptive threshold (`adapt_bump > 0`, Q12 fixed point, τ ≈ `2^adapt_decay` waves); LIF =
  the same network with `adapt_bump = 0`.
- **Readouts:** bench-side `f64` (ridge regression factored once; integer nearest-centroid); the engine
  stays integer.
- **Methodology:** every number is a pure function of `(seed, config, params)` and is reported
  **held-out** (frozen params, unseen realizations) and **multi-seed**. Single-seed / prequential
  numbers proved unreliable in the pre-pivot arc and are no longer trusted.

## The pre-pivot dead end (why we train weights, not thresholds)

Before `wave_net` stored weights, the plan was to keep the reservoir a **fixed random `±1` procedural
projection** and learn by adapting only **per-neuron thresholds** (plus firing-rate calibration) — the
approach now frozen in `wave_state_machine`. A long arc explored it: a gradient-free three-factor
threshold rule (a spiking-population reward variant, and a broadcast-error / feedback-alignment variant
on a non-spiking readout), a reservoir-regime diagnostic (separation ceiling as a learnability proxy),
and a multi-seed held-out verification. The verdict was negative, and it is why the project pivoted:

- **Threshold-only training does not reliably learn.** On a **held-out, multi-seed** test it succeeds
  only when the (reservoir × task) seed pair happens to align — roughly a 1-in-3-to-4 coin flip.
  Single-seed "successes" (an apparent ~770‰ store-recall learner; a broadcast-readout variant that
  looked like it learned) were artifacts of the wrong metric (prequential, not held-out) and of a
  **hash defect**: the default `key`/`mix` correlates the reservoir and task streams that share a seed,
  so `net == task` (our whole dev setup) was a rigged draw. Swapping in BLAKE3
  (`--features strong_hash`) reshuffles the lottery but does not remove it.
- **Three independent rescue attempts all failed** to make it reliable: scaling **width** (to 4096
  neurons/layer), a cryptographic **hash** (BLAKE3), and **richer static weights** (procedural
  `sign × 1..16`). The ceiling is structural — thresholds can only *gate* a fixed random projection,
  never *shape* it to the task.

This is exactly what GeNN (Knight & Nowotny 2021) predicts: procedural connectivity is a
**static**-connectivity technique; *plastic synapses that change their weights must be stored*. Hence
the pivot — **keep the procedural addresses, store and train the weights** — for everything below.

Two substrate facts from that arc carry forward and are used later: **the class signal separates less
well the deeper a fixed feed-forward stack runs** (motivates multi-layer credit), and **ALIF adaptation
is a strong ~64-wave working memory** (the bar recurrence must beat). The **floored-leak** engine fix
(next) also came out of it.

## Engine finding — the floored leak

An early experiment surfaced a real substrate bug. The potential leak `p -= (p>>a)+(p>>b)` is `0` for
`0 < p < 2^a`, so small sub-threshold potentials **froze forever** — the network had *infinite*
sub-threshold memory, and plain LIF never forgot a cue (the ALIF-vs-LIF distinction vanished). Fixed by
flooring positive decay at 1 (`p -= max((p>>a)+(p>>b), 1)`), giving a finite membrane time constant.
Cost: the 1/wave floor starves sparse cascades, so configs need denser drive (the fix and its density
cost are in `docs/related-work.md`, with the fixed-point-`potential` upgrade as the follow-up). This
floored leak is now load-bearing for the recurrence results below — it caps how long a trace can
persist.

## Resolution — the RSNN pivot: stored weights + e-prop learns reliably

The verification arc's conclusion (train the *weights*, not just the thresholds) was implemented in
`wave_net` (the `wave_state_machine` fork stays the pure-procedural reference) and it **works**:

- **Substrate:** synapse **addresses stay procedural** (regenerated from the hash — free); only the plastic
  **weights are stored (int8 + f32 shadow)**. This is the GeNN split — procedural static structure, stored
  plastic weights — at a scale where storing weights is trivially cheap.
- **Learning:** **e-prop** — a factored per-neuron eligibility `e_ij = pre_i · psi_j` (pre-trace × a box
  pseudo-derivative near threshold, O(neurons) engine state) × a symmetric-feedback learning signal from a
  trained readout, updating the stored weights through the shadow.

**Result (held-out, multi-seed — the bar threshold-only failed):**

| | seed s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| fixed-reservoir top-layer readout | 727 | 500 | 655 | 507 |
| **+ e-prop hidden-weight training** | **1000** | **965** | **890** | **987** |

A top-layer readout on the *fixed* reservoir is seed-fragile (two seeds at chance) — the class doesn't
reliably propagate upward. **e-prop on the hidden `L1→L2` weights shapes the reservoir so the top layer is
reliably separable on *every* seed (890–1000).** Training weights *shapes* the projection (real feature
learning); training thresholds could only *gate* a fixed one — which is exactly why weights generalize
across seeds where thresholds did not. `hidden_lr = 0.004` (0.008 destabilizes one seed; ≤0.002 is too slow
for the fragile seeds).

Also banked, as a reliable baseline: a trained readout on the **full** reservoir hits **1000‰ held-out on
all seeds** (the class is linearly present across the layers) — the classic LSM, first reliable learning on
`wave_net`.

**Deferred (next):** recurrence (`level 0/−1` trained → full RSNN); multi-layer credit assignment (train all
feed-forward layers, not just the last); int8 readout + weight sharing (conv kernels / hashing trick) for
sub-linear memory; a task that *needs* reservoir computation (so hidden training must beat a full-reservoir
readout, which here already saturates).

## Multi-layer credit (DFA) — training every layer makes depth usable

Extended feed-forward e-prop to train **all** layers, not just the last: the top layer keeps symmetric
readout feedback, each deeper layer gets **Direct Feedback Alignment** (fixed random hash-derived feedback
broadcasting the output error), and every layer's weights update by the factored eligibility × its signal.

**Result (deep net, layers 4, held-out, multi-seed):**

| seed | single-layer (train last only) | multi-layer (train all, DFA) |
|---|---|---|
| s0 | 535 | **1000** |
| s1 | 500 | **1000** |
| s2 (deadbeef) | 485 | **1000** |

Single-layer e-prop is at **chance on every seed** at depth 4 — training only the last projection can't
recover the class signal after the fixed intermediate layers have eroded it (the class separates less
well the deeper a fixed feed-forward stack runs — see the pre-pivot summary above). **Multi-layer DFA
credit succeeds perfectly (1000) on all seeds** — training every layer (including the input projection
`L0→L1`) lets each layer preserve/transform the signal, so depth becomes usable. This is the capability
that turns the earlier "depth erodes separation" finding from a wall into something trainable.

(At depth 5 the globally-degenerate `deadbeef` reservoir is beyond even multi-layer — both fail — while the
other seeds still hit 1000; depth 4 is reliable for all. DFA's random feedback is noisy but here it was
clean across seeds.)

**Depth sweep — how far does it hold?** Worst-seed held-out over 3 seeds:

| depth | single-layer | multi-layer |
|---|---|---|
| 3 | 890 | 1000 |
| 4 | 485 | 1000 |
| 5 | 485 | 485 |
| 6 | 485 | 902 |
| 7 | 485 | 485 |

Single-layer is at chance for **every** depth ≥ 4 — the advantage of training all layers is robust and only
grows with depth. But multi-layer's *worst-seed* reliability breaks down past depth ~4: at ≥5 the hardest
seed can still fail (the non-monotonic 485/902/485 is just which seed is unluckiest varying). So multi-layer
makes depth **usable** (vs single-layer's total failure) but **not arbitrarily reliable** — depth 4 is the
clean sweet spot; deeper, the fixed random reservoir's seed lottery + noisy DFA feedback cap it. Pushing
deeper needs a better reservoir (recurrence/criticality) and/or less-noisy credit (symmetric feedback / BPTT).

**Does width fix the depth degradation? Yes — for multi-layer.** Worst-seed held-out, size × depth:

| size (N/layer) | depth 4 single / multi | depth 5 | depth 6 |
|---|---|---|---|
| 8 (64) | 485 / 1000 | 485 / **485** | 485 / 902 |
| 16 (256) | 485 / 1000 | 485 / **1000** | 485 / **1000** |
| 32 (1024) | 485 / 1000 | 485 / 1000 | — |

**Width fixes multi-layer's deep reliability** (size 16 → clean 1000 through depth 6, hard seed included) —
the "brutal" size-8 degradation was a **capacity shortage**: 64 neurons/layer is too few to reliably train
the *hard* seed's deep layers; 256 is enough. **But width does nothing for single-layer** (chance at every
size/depth ≥ 4): its problem is *untrained* intermediate layers eroding the signal — no width fixes that;
only *training* them does. So the levers **compound**: depth erodes → multi-layer training preserves →
but that training needs enough **width** to succeed on hard seeds. **Wide + multi-layer = reliable deep
learning**; either alone fails.

**How deep is it stable (size 16, multi-layer)?** Reliable to **~depth 16** — with two distinct limits:

| depth | fixed 16-wave trial | trial length scaled to depth |
|---|---|---|
| 10 | 1000 | 1000 |
| 12 | **485** (chance) | **1000** |
| 16 | 485 | **1000** |
| 20 | 465 | **485** (still fails) |

- **The depth-12 cliff (fixed trial) is a *timing* artifact, not a learning limit.** The cue is injected
  only during the present-waves, so its wavefront needs ~`depth` waves to reach the top; a fixed 16-wave
  trial closes the read window before it arrives. Scaling `present`/`read` to `depth` **fully recovers
  depths 12 and 16 to 1000** — the net trains fine that deep given time for the signal to propagate.
- **A genuine ceiling appears ~depth 18–20** — depth 20 fails even with scaled waves. This is the known
  **Direct Feedback Alignment** limitation: random per-layer feedback becomes too noisy to credit very deep
  layers (DFA degrades where backprop wouldn't). Pushing past ~16 layers is the "less-noisy credit
  (symmetric feedback / BPTT)" lever, not a substrate problem.

So: **size 16 + multi-layer + trial length matched to depth ⇒ reliable to ~16 layers**; the wall beyond is
the *credit rule* (DFA), not capacity or the reservoir.

## Recurrence — trainable in the deep + wide + sub-critical regime (completed ALIF eligibility)

Trained recurrence earns its keep on this substrate **only** in a specific regime, and reaching it required
completing e-prop's ALIF credit rule. The eligibility used in every earlier recurrence test was the **crude
spike-timing** form — only the fast membrane term `e = Σ_t ψ_j·εᵛ_i`. Textbook e-prop for ALIF neurons
(Bellec 2020, Eq. 24–25) adds a second, **slow** component — the adaptation eligibility `εᵃ`, recursed at the
adaptation rate `ρ` — giving `e = ψ·(εᵛ − β·εᵃ)`. That term carries credit over the ~64-wave adaptation
horizon (the substrate's delay memory) and was never implemented. It is now built (`bench::rsnn`;
`RsnnConfig.elig_beta` / `elig_bump_psi`; a decide-time effective-threshold snapshot `Layer.decide_eff`; a
fixed-width bump ψ of half-width `PSI_WIDTH`), verified against the paper and the reference implementation
(`IGITUGraz/eligibility_propagation`). Two ψ bugs were found and fixed on the way: `eff` must be sampled at
the decide step (before the fire-bump), and the bump must use a fixed absolute band (not the θ≈1 baseline,
which collapses ψ to ~0 since integer ±1 drive overshoots `eff` by O(2–26)).

**Three levers make it work — all necessary** (deep 4-layer hidden-recurrent stack `train_hidden_rec_task`:
L0→L1→L2→L3, recurrence on L2, *every* forward layer trained via multi-layer DFA + `rate_reg` forward
liveness; temporal XOR, delay 20, worst-seed / 3 seeds unless noted):

- **Depth + trained forward layers.** A shallow recurrent-top net, or a deep net with *untrained* forward
  layers, does not get there — the forward projection must be trained (multi-layer DFA).
- **Width.** rec_count 8, FF vs trained recurrence: **990 / 725** (size 16) → **1000 / 982** (size 32) —
  width nearly erases the gap.
- **Sub-critical recurrence density.** A sharp cliff at **rec_count ≈ 12** (size 16): below it, trainable;
  at/above it, super-critical ±1 recurrence collapses the read layer to chance regardless of the credit rule.

  | rec_count (size 16) | crude | completed |
  |---|---|---|
  | 4 | 707 | **855** |
  | 8 | 672 | **725** |
  | ≥ 12 | 475 | 475 (collapse) |

**The completed eligibility consistently beats the crude rule** wherever recurrence is trainable, and its edge
is **largest where credit assignment is hardest** (light density / smaller width): rc4 855 vs 707, rc8 725 vs
672, size 32 982 vs 975.

**Recurrence out-performs feed-forward on a headroom task.** Temporal XOR is FF-saturated (ALIF-FF ≈ 1000), so
recurrence can only *tie*. On **parity N=4** — non-monotone, FF does not saturate (~620) — trained recurrence
with the completed eligibility (size 32, 4 layers, rec_count 8) **beats FF on 2 of 3 seeds** (687 vs 680, 642
vs 637; worst-seed 560 vs 620, held back by the hardest seed) and beats the crude rule on all three. This is
the first place trained recurrence beats feed-forward on this substrate.

**Verdict.** Trained recurrence via the completed ALIF e-prop eligibility **works in the deep + wide +
sub-critical-density regime** and beats FF where the task has headroom — a reversal of the earlier "airtight
null," which was measured with an incomplete credit rule in shallow / over-dense / FF-saturated regimes. The
knobs are depth, width, recurrence density (keep it sub-critical), and β·`PSI_WIDTH`.

**Open (next):** robustness on the hardest seed; whether more depth (a 5th layer / a second recurrent layer),
more width + rec_count, the backward side-car topology, or β·W tuning pushes it to a *robust* win; and a BPTT
comparison. Tools in place: `train_hidden_rec_task` / `sequence_trial_layers` (task-parameterized deep
trainer), `elig_beta` / `elig_bump_psi`, and the decide-time `eff` snapshot.
