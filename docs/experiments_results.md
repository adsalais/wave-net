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

## Recurrence attempt — an honest null (and a finding about ALIF vs the leak)

Enabled trainable `level 0` lateral weights + a temporal e-prop eligibility
(`e_ij = Σ_t pre_trace_i(t)·fired_j(t)`, computed from recorded per-wave spikes) and tested on **temporal
XOR** (present `A` → delay → present `B`, label `A XOR B`).

**Finding 1 — ALIF adaptation solves temporal XOR feed-forward.** With adaptation on, a *feed-forward* net
solves it at every delay tested (997 / 965 / 882 at delay 4 / 20 / 40). Adaptation is a powerful per-neuron
memory; it holds `A` and supplies the nonlinearity XOR needs. So the task only tests recurrence if we strip
adaptation.

**Finding 2 — the trained recurrence's memory horizon ≈ the membrane-leak horizon, so it doesn't beat FF.**
With **LIF** (`adapt_bump = 0`):

| delay | FF (rec off) | + level-0 recurrence |
|---|---|---|
| 12 | **835** (leak already holds ~12 waves) | 732 (no help — worse) |
| 20 | 455–552 (**~chance**) | 475–595 (**~chance**) |

There is **no delay where FF fails and recurrence succeeds**: at short delays the LIF membrane leak already
gives the feed-forward net ~12 waves of memory (so FF wins), and at delay 20 — where FF is at chance — the
level-0 recurrence **can't sustain `A` across the silent gap** either (activity dies; a stronger recurrent
init `∈{3,6,10}` doesn't rescue it — it saturates). Tuned across `delay`, `rec_count`, `rec_radius`,
`rec_tau`, `hidden_lr`, `rec_init`.

**Reading.** The **floored leak** — the fix that gave the engine a *finite* membrane memory (killing the
infinite-memory bug) — now also **caps how long recurrence can hold a memory**: trained level-0 recurrence
holds ~as long as the leak, not longer. So on this substrate, level-0 lateral recurrence with a crude
spike-timing pseudo-derivative adds no net memory over the leak. Honest null.

**What would move it (deferred):** a proper pseudo-derivative `ψ` (sub-threshold, not just spikes) so credit
flows during silent gaps; `level −1` backward recurrence; a slower/finer leak on the recurrent layer (the
fixed-point-potential upgrade) so recurrent memory can outlast the membrane; or surrogate-gradient BPTT for
true temporal credit. The reliable results stand: feed-forward e-prop learns robustly, and **ALIF
adaptation** is itself a strong, already-working temporal memory.

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

## Backward recurrence (level −1/−2) + sub-threshold ψ — the blocker is sustaining dynamics

> **Correction.** An earlier version of this section read the backward-recurrence null as "ψ (spike-time)
> is the blocker." That conclusion was **wrong** — it rested on a hidden bug (below). The corrected finding
> is that the blocker is the *sustaining dynamics*, not the credit rule.

To separate "topology/capacity too weak" from "credit rule too weak," we tried a *stronger* recurrence —
cross-layer **backward** loops (`level −1` and `−2`, i.e. `Lz → Lz−1 → Lz`) — with the **width fix**
(size 16, depth 4) and multi-layer credit, on temporal XOR (LIF, delay 20). It nulled (FF, +backward both
~chance). We took that as implicating the spike-time `ψ` and built a **sub-threshold `ψ`**
(`clamp(decide_potential/θ, 0, 1)` from a new decide-time potential snapshot) so credit could flow to
neurons that are charged-but-not-firing.

**The bug it exposed — a dead readout layer.** A per-layer activity probe showed the 4-layer recurrent net's
readout layer (L3) was **completely silent** under the default drive: the LIF reservoir is **sub-critical**
(feed-forward gain < 1), so the sparse transient cue decays with depth (L1 fires ~30/256, L2/L3 = 0).
Calibration set deep-layer thresholds too high (42/21/11) for the transient to overcome; ALIF only ever
"worked" because adaptation floors the calibrated baseline threshold to 1. **So the backward and first
sub-ψ nulls were invalid — they read a dead layer, where nothing can work.** Fixed with an *alive-LIF*
config (`up_count=32, present_waves=12, base_q16=30000`) that raises gain so all four layers fire ~7%.

**Re-run on the alive net — the valid result:**

| variant | best over seeds |
|---|---|
| FF | ~500 (chance) |
| backward + spike-ψ | ~500 |
| backward + sub-ψ | ~500 |

All at chance, and sub-ψ **byte-identical** to spike-ψ. But sub-ψ was *not* a no-op — it flowed credit to
thousands of charged-silent neurons/layer (mean ψ ≈ 0.44). It didn't matter because a **per-wave probe shows
the recurrent activity dies in the gap**: after cue A ends, spikes stop within ~6 waves and the membrane
trace decays 28→2 over the 20-wave delay — A's trace is gone before B arrives. So **no `ψ` (spike or
sub-threshold) can help — there is no sustained trace to credit.** The blocker is the **sustaining
dynamics** (floored leak + weak ±1 recurrence don't self-sustain), confirming that the floored leak caps
memory (see the floored-leak finding above) with both the dead-readout *and* credit-rule confounds now
removed.

**Open issue — generic calibration.** Calibration tunes thresholds to a *sustained random* input at one
operating point; it does **not** guarantee a *transient, sparse, task-specific* cue propagates through an
arbitrary-depth stack. Keeping the reservoir near-critical (gain ≈ 1) across depths and input distributions
is unsolved here — we hand-tuned drive to revive the deep net. A principled generic calibration (per-layer
gain control / criticality target) is future work.

**What would actually move recurrence (deferred):** self-sustaining recurrent dynamics (trained strong
recurrent weights at gain ≈ 1 — a criticality/attractor regime the ±1 init never reaches); or a
slower/finer recurrent-layer leak so the trace outlasts the gap (risking the passive-memory that the floored
leak fixed); or surrogate-gradient BPTT. Note ALIF adaptation *already* provides a working ~64-wave memory —
recurrence has to beat that, which is a high bar on this substrate.

## The depth wall is the credit rule, not liveness — firing-rate regularization isolates it

The multi-layer DFA ceiling (reliable to ~16 layers; at **depth 20** the worst seed is **485 = chance**, even
with trial length scaled to depth) has two candidate blockers: (a) the deep layers go **dead** (the
sub-critical net's transient cue dies with depth), or (b) the **credit rule** (DFA feedback too noisy to
train very deep layers). This isolates them.

**Abandoned first attempt — external "criticality gain calibration."** We tried to fix (a) with a
per-(layer, level) gain `β` *outside* weight storage, tuned by a branching estimator toward gain ≈ 1. It was
discarded, with two findings that carry forward:
- The **analytic branching estimator (`Σ|w|·β / θ`) is invalid** — a *static* weight/threshold ratio, blind
  to whether spikes propagate. It read branching **16** (super-critical) for layers that *never fire* (rate
  0.00, because calibration floors their thresholds to ~1). Driving a controller with it **cut** forward gain
  16× and killed the reservoir → uniform chance across all depths/targets.
- A **separate measure-and-invert gain calibration is a non-standard bolt-on.** The field does not calibrate
  a trained RSNN to criticality as a phase; it folds a **firing-rate regularization** into the loss,
  co-trained by the same rule (LSNN, Bellec et al. 2018/2020). Memory comes from **adaptation** (ALIF) or
  learned structure (BPTT), not from sitting at σ = 1.

**The field-standard fix — rate regularization in the e-prop learning signal.** Add `c_reg·(r_j − r_target)`
to each target neuron's learning signal, carried by the *same* eligibility `e_ij = pre_i·ψ_j`
(`r_j = elig_pre_j / n_waves`, `r_target` = the calibration rate). A too-quiet neuron pulls its incoming
weights up. Bench-side only, **no engine change**; guarded so `rate_reg = 0` is byte-identical.

- **Mechanism (works).** At depth 20 (size 16, multi-layer), `rate_reg = 0` leaves layers 8–15 **dead**
  (rate 0.00); `c_reg ≈ 5` brings **every layer alive** (0.04–0.19, reaching/exceeding the 0.1 target in the
  deep layers). Liveness climbs the stack bottom-up (the eligibility gates each layer's revival on its
  feeder's spikes, and L1 is always driven by the L0 transducer). This is the propagation the invalid static
  estimator could not see. `c_reg` must be ~10s (comparable to `L_j^task ~ O(1)`); a small coefficient is
  negligible, a large one (≥ 20) over-suppresses.

- **Depth wall (null).** Held-out accuracy at depth 20 is **byte-identical** across `c_reg ∈ {0, 5, 20}`
  (per-seed 535 / 500 / 485; worst **485 = chance**). Reviving all 20 layers moves the accuracy **not at
  all**.

**Reading.** Liveness is **necessary but not sufficient**. With the dead-layer confound *definitively*
removed (every layer firing near target), the depth-20 wall stands — so it is the **credit rule**: DFA's
random feedback is too noisy to train 20 deep layers to carry the *class* signal, even when they are all
active. The regularizer adds *class-agnostic* activity, not class information. This confirms the
long-suspected DFA ceiling and points the next lever at **symmetric feedback / surrogate-gradient BPTT**, not
the dynamics. (Rate regularization stands as a working, field-standard liveness mechanism — `RsnnConfig.rate_reg`
— now available for any deep config, just not the depth-20 bottleneck.)

## Field-standard check: ALIF holds temporal XOR alone; recurrence doesn't extend it

A literature review (see `related-work.md`, 2026-07-09) corrected the recurrence setup: in the e-prop/LSNN
world **ALIF adaptation is the delay-memory mechanism**, rate regularization is a soft efficiency term (not a
memory tool, and a *uniform* target erodes input coding), and threshold calibration is a one-time *sensible
init* — not a rate that must transfer to the task (our `calibrate` measures ~10% on a *sparser random* drive
than the denser task cue, so the task actually runs at 20–40%; that's fine as an init, it just isn't a
target). Prior recurrence experiments had **stripped ALIF** ("to isolate recurrence"), removing the very
mechanism the field relies on — so those LIF nulls were expected, not informative about recurrence's value.

Re-ran the question the field's way — **ALIF-FF vs ALIF + lateral recurrence** on temporal XOR, delays
bracketing ALIF's horizon (single seed, calibration as init, rate reg off, `rec_count 24`, `rec_tau = delay`):

| delay | ALIF-FF | ALIF + lateral recurrence |
|---|---|---|
| 40 | 885 | 750 |
| 80 | 947 | 652 |
| 120 | 835 | 672 |

- **ALIF alone solves it out to 120+ waves** — no horizon reached in range (835–947, well above chance 500).
  Adaptation is a strong, *sufficient* delay memory; there is no gap for recurrence to fill.
- **Recurrence hurts at every delay** (652–750 vs 835–947) — lateral recurrence + the crude spike-timing
  e-prop adds interference to the adaptation-held representation rather than extending it. Consistent with the
  earlier "recurrence hurts XOR" sweep, now confirmed on the *field-standard ALIF* setup.

**Reading.** On the correct setup the result matches LSNN theory: **adaptation holds the memory; recurrence
does not extend it** (and degrades it here). Temporal XOR is therefore the *wrong task* to demonstrate
recurrence — ALIF dominates the whole delay range. Recurrence's potential value is *computation*, not memory;
showing it earn its keep needs a task that genuinely demands recurrent computation beyond ALIF + feed-forward
— an open problem on this substrate (the ALIF-vs-LIF sweeps have found no such task yet). Single-seed; the
direction is consistent across all three delays. The `rate_reg`-in-recurrence machinery
(`recurrent_update` / `train_recurrent`) remains available as the field's light liveness term, not a memory
mechanism.

## Recurrence benchmark suite: trained recurrence doesn't earn its keep (and hurts parity)

To avoid concluding from one task, built a suite of tasks that need recurrent *computation* — a monotone
adaptation accumulator provably can't fake them — and ran **ALIF, FF vs +lateral-recurrence, 3 seeds**:
**sequential parity** (N-bit, `label = ⊕ bits`), **delayed XOR with a distractor cue**, and a **set/reset
flip-flop**. (`bench::rsnn`: `sequence_trial` / `train_sequence` / `task_parity` / `task_distractor` /
`task_flipflop`.)

**Parity (worst-seed over 3 seeds):**

| N | FF (no recurrence) | + lateral recurrence |
|---|---|---|
| 2 | 967 | 825 |
| 3 | 817 | 700 |
| 4 | 670 | 522 |
| 5 | 590 | 515 |

- FF degrades with N (parity is non-monotone → harder) but stays **above chance even at N=5** — the ALIF
  *feed-forward* reservoir + readout captures *partial* parity on its own.
- **+rec is worse than FF at every N.** On the canonical recurrence task, trained lateral recurrence
  consistently *degrades* performance.

**Distractor-XOR:** FF wins 2/3 seeds (worst-seed FF 857 > rec 800); one seed showed a rec advantage
(972 vs 857) but not reliably across seeds. **Flip-flop:** FF ≈ rec, both high (880–982) — **ALIF solves it
as held memory; recurrence neither needed nor helpful** (flip-flop is memory, not computation).

**Reading.** Robust across tasks *and* N: **trained lateral recurrence via the crude spike-timing e-prop
never reliably helps — it hurts parity at every N, mostly loses distractor, and ties flip-flop.** Because the
ALIF feed-forward reservoir captures *partial* parity while trained recurrence *degrades* it, the bottleneck
is the **credit rule** (the temporal eligibility injects noise rather than useful recurrent computation), not
substrate capacity. This is the same conclusion the depth wall (credit rule, not liveness) and the temporal-
XOR check (adaptation holds; recurrence hurts) reached — now robust across a purpose-built task suite, not a
single measurement.

**Caveat — this is an under-powered regime, not a fair test vs the literature.** An LSNN/e-prop literature
check shows our recurrence setup is below the field's norms on *every* axis: **64 neurons** (size 8) vs
**100–400** (TIMIT 400; sMNIST ~100–200; N-MNIST 120+84); **sparse lateral** recurrence (`rec_count 24`) bolted
onto a *fixed procedural* feed-forward projection vs a **fully/densely recurrent hidden layer where recurrence
IS the computation**; **toy tasks** (parity/XOR) vs sMNIST/TIMIT; and a **crude spike-timing eligibility** vs
proper e-prop (itself only *approaching* BPTT — an open question, ICML 2025). So "trained recurrence doesn't
earn its keep" holds *in this small, sparse, thin-recurrence regime* — it does **not** establish a substrate
limit, and "BPTT is the lever" is only one hypothesis among four (**capacity, recurrence density, task
difficulty, credit rule**). A fair test needs the LSNN regime: 100s of neurons, dense/full recurrence *as the
main substrate*, harder tasks, and proper temporal credit. Until then the null is provisional.

**Fair-regime data point (size 16 = 256 neurons, dense recurrence).** Testing the capacity + density
hypotheses on parity (worst-seed over 3 seeds):

| N | FF (size 16) | + dense recurrence (`rec_count 96`, full radius) | (under-powered: size 8, `rec_count 24`) |
|---|---|---|---|
| 3 | **980** | **465 (chance)** | FF 817 / rec 700 |
| 4 | **900** | **487 (chance)** | FF 670 / rec 522 |

Two findings, and neither matches the naive hypothesis: **(1) capacity was a real gap — but it rescues the
*feed-forward* reservoir**, not recurrence (scaling to 256 neurons lifted FF parity 817/670 → 980/900; the
task is FF-solvable with enough neurons). **(2) Dense ±1 recurrence *collapses to chance*** — 96 recurrent
synapses/neuron at ±1 init is strongly **super-critical** (runaway), drowning the class signal, so densifying
made recurrence *catastrophically worse*, not better. Dense recurrence is a **gain liability**, not a free
win: it needs the recurrent gain controlled to **σ ≈ 1** to be stable — which **re-surfaces the abandoned
criticality thread, now with the valid estimator**: the *perturbation probe* (`probe_avalanche`, built but
never wired to a controller — the *analytic* `Σ|w|/θ` estimator was the invalid one). **Net: capacity was
real (fixed FF); dense recurrence is untested fairly until its gain is controlled.** The next honest test of
recurrence is probe-gain-controlled (σ≈1) dense recurrence — not raw ±1 dense recurrence, and not the
sparse under-powered version.

**Deep hidden-recurrent architecture (forward layer under a recurrent top).** Testing the "go deeper + a
real hidden recurrent layer" hypothesis: `L0 → L1(forward) → L2(recurrent top, read)`, size 16, *modest*
recurrence (`rec_count 24`, off the super-critical cliff), parity worst-seed:

| N | deep-FF | + hidden recurrence | (shallow-wide FF, 2-layer size 16) |
|---|---|---|---|
| 3 | 637 | 465 (chance) | 980 |
| 4 | 677 | 502 (chance) | 900 |

Both effects negative: **(1) depth hurt** — deep-FF (637/677) is far below shallow-wide FF (980/900) because
the extra *fixed procedural* forward layer **erodes** the parity signal (documented depth-erosion of a fixed
feed-forward stack); **(2) recurrence collapsed to chance** even at modest density. So the deeper
hidden-recurrent net is *worse* than wide-shallow FF on both counts. **But this is still not the LSNN
regime**: the forward layers here are **fixed/untrained**, whereas LSNN *trains* input→hidden — and our own
multi-layer DFA result shows *training* every layer is exactly what makes depth usable (untrained depth just
erodes). So the honest remaining gap is **trained forward layers**: `train_sequence` trains the recurrence +
readout, not the feed-forward projection. A fair deep test would combine multi-layer DFA forward training
(`train_eprop`) *with* recurrence. **Until that build: the substrate's best parity config remains wide,
shallow, feed-forward + ALIF (980/900); depth and recurrence both hurt when the forward projection is
untrained.**

## `rate_reg` is a feed-forward liveness tool — it revives deep FF but *hurts* recurrence

Turned `rate_reg = 5.0` on across **all** recurrence benchmarks (both variants) and re-ran. The result splits
cleanly by *where* the regularizer's class-agnostic term lands:

- **On recurrent weights → hurts, everywhere.** Every `+rec` variant got worse vs `rate_reg = 0`, most
  crashing toward chance: parity N=3 700→**562**; distractor-XOR (worst) 800→**475**; flip-flop (worst)
  880→**485**; ALIF+rec (delay 40/80) 750/652→**475/475**. `rate_reg` adds `c_reg·(rate−target)` to the
  signal that trains the *recurrent* weights, so at `c_reg=5` it overrides the task signal — the recurrence
  learns to hit a *rate*, not carry the *class* → class-agnostic reverberation → worse.
- **On forward weights → can dramatically revive deep FF.** The side-car's matched **deep-FF(4)** baseline
  jumped from **chance to near-perfect** on temporal XOR: 480/497/492 → **982/987/997**. Here `rate_reg` on the
  forward (DFA) weights revived the dead deep layers, the cue reached L3, and XOR got solved — the depth-wall
  mechanism finally moving *accuracy*, not just liveness. (Config-sensitive: the l2l3loop deep-FF, denser
  drive / shorter delay, stayed at chance.)

**Lesson:** `rate_reg` belongs on the **feed-forward** path (revive dead depth — sometimes a huge accuracy
win), **not** on recurrent weights (destroys the class signal). It does not rescue recurrence; it confirms
harder that recurrence + crude e-prop fails. The one keeper is the FF revival (deep-FF 480→982) — evidence
that, for feed-forward depth, liveness genuinely *can* be the blocker.

**Robustness (5 seeds) — the FF revival is real, and `rate_reg` is a *targeted liveness rescue*.** Swept the
plain 4-layer deep FF (no recurrence) on temporal XOR, `delay × up_count × rate_reg`, 5 seeds (mean):

| drive `up_count` | `rate_reg` | delay 12 | delay 20 |
|---|---|---|---|
| 16 (sparse) | 0 | 498 (chance) | 498 (chance) |
| 16 (sparse) | 5 | **957** | **986** |
| 32 (dense) | 0 | 615 | 890 |
| 32 (dense) | 5 | 504 (hurt) | 801 (hurt) |

The config-sensitivity resolves cleanly: it's whether the deep FF is **liveness-starved**. Sparse drive →
the deep stack is **dead**, and `rate_reg` **reliably revives it** (475→957/986 across all 5 seeds, both
delays — the 982 reproduced). Dense drive → the stack is **already alive** (615–890 with no reg), and there
`rate_reg` **hurts** (890→801, class-agnostic noise). So `rate_reg` rescues a *starved* deep FF and degrades
an *alive* one — a targeted tool, not a universal one. (This also explains the l2l3loop deep-FF "failure" at
507: it was the `up32 + reg` alive-and-hurt cell.)

**And liveness-vs-credit is task-dependent.** The earlier depth-wall finding (parity) was "reviving deep
layers doesn't fix accuracy — the wall is credit." On **temporal XOR** the opposite holds: reviving the deep
FF **does** fix accuracy (chance→980). So whether liveness or the credit rule is the wall depends on the
task — on XOR, deep-FF liveness was the entire blocker, and the online `rate_reg` (the e-prop-literature
mechanism) is the fix.

## The fair recurrence test (every confound removed) — recurrence *destroys* a working baseline

Assembled the fairest possible recurrence test and ran it (5 seeds, temporal XOR): a **working, live deep-FF
baseline** (L0→L1→L2→L3, sparse drive + per-neuron `rate_reg` on the *forward* path → ~986) plus **level-0
recurrence on L2**, stabilized by a **class-preserving per-layer stabilizer** (`rec_stab`, a uniform-bias
rate control — *not* the per-neuron `rate_reg` that homogenizes the class signal away). This finally removes
every confound that plagued earlier recurrence tests: dead baseline (now 986), wrong stabilizer (now
class-preserving), too few neurons (256), wrong topology (clean hidden-recurrent, signal flows *through* L2).

| | deep-FF | + hidden recurrence (stabilized) |
|---|---|---|
| worst / mean (5 seeds) | 970 / **986** | 475 / **498 (chance)** |

**Adding recurrence didn't fail to help — it destroyed a working 986 baseline down to chance, all 5 seeds.**
With every substrate/stabilizer confound eliminated, the only thing left is the **credit rule**: the crude
spike-timing temporal eligibility can only train the recurrence to *scramble* the signal, never to compute
with it. This is the airtight version of the session-long null: **on this substrate, trained recurrence via
the crude e-prop is not merely useless but actively harmful, and no amount of liveness / gain-stabilizer /
capacity / topology fixing rescues it.** The single remaining lever is proper temporal credit —
**surrogate-gradient BPTT** — everything short of that has now been ruled out.

**The session's net positives (the working regime):** wide + shallow + feed-forward + ALIF is where this
substrate learns; multi-layer DFA + width makes depth usable to ~16 layers; and `rate_reg` on the forward
path reliably rescues liveness-starved depth (chance→980 on XOR). Recurrence is a documented dead end short
of BPTT.

**Final LIF check — removing adaptation *refutes* "ALIF fights the loop"; ALIF is load-bearing.** Re-ran the
fair recurrence test with **LIF** (`adapt_bump = 0`), to test the hypothesis that ALIF's adaptation was
quenching the recurrent gain (5 seeds, temporal XOR):

| | LIF-FF | LIF+rec (`rate_reg`) | LIF+rec (`rec_stab`) |
|---|---|---|---|
| worst / mean | 472 / **489** | 472 / 486 | 472 / 486 |

All at chance — and the *reason* is the finding. **The LIF-FF baseline is dead (489)**: `rate_reg` revived
the *ALIF* deep-FF to 986 but **cannot** revive the *LIF* one (only difference: `adapt_bump`). So **ALIF's
self-regulation is *necessary* for the deep-FF liveness that `rate_reg` fine-tunes** — LIF deep stacks starve
on the floored leak with nothing to self-regulate them. Recurrence stays at chance either way, but that's
*inconclusive here* (dead baseline, no headroom). **So the "ALIF quenches the loop, LIF would free it"
hypothesis is refuted — removing adaptation didn't rescue recurrence, it killed the substrate.** ALIF is not
an obstacle; it is load-bearing.

**Combined verdict on recurrence:** it *crashes a working baseline with ALIF* (986→498, fair test) and there
is *no working baseline without ALIF* to test it on (489). Recurrence has **no regime** on this substrate +
crude e-prop where it earns its keep, ALIF is essential, and **surrogate-gradient BPTT (proper temporal
credit) is the sole remaining lever** — every substrate/stabilizer/topology/neuron-model confound is now
ruled out.

## Completing the ALIF adaptation eligibility — the credit rule *was* incomplete, and it still doesn't rescue recurrence

> **Correction to the verdict above.** "Every confound ruled out … BPTT is the sole remaining lever" was
> **premature on one axis**: the *credit rule itself was never complete*. All recurrence experiments above used
> a **crude spike-timing eligibility** — only the fast membrane term `e_ij = Σ_t εᵛ_i(t)·ψ_j(t)`, with `ψ`
> the hard spike. Textbook e-prop for **ALIF** neurons (Bellec et al. 2020, Eq. 24–25) has a **second**
> component — the **adaptation eligibility** `εᵃ`, recursed at the *slow* adaptation rate `ρ`, entering as
> `e = ψ·(εᵛ − β·εᵃ)`. That term is exactly what carries credit over the ~64-wave adaptation horizon (the
> substrate's actual delay-memory), and it **was never implemented**. This section builds it faithfully
> (`bench::rsnn`, guarded behind `RsnnConfig.elig_beta`/`elig_bump_psi`; a decide-time effective-threshold
> snapshot `Layer.decide_eff` in the engine) and re-runs the ALIF recurrence benchmarks.

**Two implementation bugs found and fixed on the way (both would have faked a result).** A first cut looked
like it *helped* recurrence — it was an artifact:

1. **ψ read the effective threshold post-wave.** `eff = baseline + adapt` was sampled *after* the wave, but a
   fire **bumps `adapt` by ~`adapt_bump` during that wave** — so the recorded `eff` was ~20 too high and the
   at-spike overshoot `v−eff` read as **−16** (impossible; a spike has `v ≥ eff`). Fixed by snapshotting the
   **decide-time** `eff` in the engine (before the fire-bump), paired with `decide_potential`.
2. **ψ normalized by θ = baseline**, which calibration floors to ~1, while integer ±1 drive makes potentials
   overshoot `eff` by **O(2–26)** at a spike. So `ψ = γ·max(0,1−|v−eff|/θ)` collapsed to **~0 everywhere**
   (>99% of entries), silently disabling the eligibility **and** `rate_reg` (which is carried by the same
   eligibility). Fixed with a **fixed-width band** `PSI_WIDTH=16` matched to the potential scale (the engine's
   own working `elig_post` uses `PSI_BAND=8`; deeper layers overshoot more) → ψ non-degenerate for 70–100% of
   spikes. With the degenerate ψ, `elig_beta>0` merely *disabled* recurrent learning (leaving the less-harmful
   ±1 init), which *looked* like an improvement (parity 562→700). The corrected ψ removes that confound.

**Result 1 — parity (the informative test: read layer alive, eligibility verified to engage), size 16, 3
seeds.** Four columns isolate *what* about recurrence helps or hurts — **FF** (no recurrence, trained
forward); **fixed-rec** (±1 recurrence, `hidden_lr 0` — untrained reservoir + trained readout, a classic
LSM); **crude-rec** (recurrence trained via the spike-ψ eligibility); **completed-rec** (trained via the
faithful ALIF `e = ψ·(εᵛ − β·εᵃ)`, `elig_beta 0.4`). Worst-seed:

| N | FF | fixed-rec (readout only) | crude-rec | completed-rec |
|---|---|---|---|---|
| 3 | **980** | **950** | 542 | 617 |
| 4 | **900** | **817** | 487 | 472 |

The result is clean and it **relocates the blocker**: the **fixed random recurrence is a strong reservoir**
(950 / 817, right behind FF) — recurrence is *not* useless. What craters performance is **training the
recurrent weights**: e-prop drops it from 950 → ~550, and **completing the ALIF eligibility does not fix
this** — `completed ≈ crude`, both cratering (N=3 617 vs 542, N=4 472 vs 487; per-seed mixed, no consistent
edge). The eligibility is verifiably *engaged*, not a no-op: an L2 magnitude probe shows `Σ|e|` differs
between the rules (forward 5188 crude vs 7145 completed; recurrent 16680 vs 6155,
`_diag_fair_elig_magnitude`). So the credit *is* flowing differently — it just doesn't help, because e-prop's
factorized credit (itself only *approaching* BPTT) pushes the recurrent weights off the good random operating
point rather than toward a better one. (Earlier size-8 numbers hinted "completed slightly ≤ crude"; the
size-16 multi-seed picture shows they are effectively tied — both destroy the reservoir.)

**Result 2 — fair hidden-rec test (the 986→498 null): byte-identical across `elig_beta ∈ {0, 0.1, 0.2, 0.4}`,
because that config is inert to *all* hidden training.** A `hidden_lr` × `elig_beta` probe is decisive:
`hidden_lr 0` (no hidden training at all) gives **475**, identical to `hidden_lr 0.02, elig_beta 2.0` — every
combination is **475** (`_diag_fair_hidden_lr_sensitivity`). So held-out here is set *entirely* by the readout
on the recurrence-collapsed reservoir, independent of any hidden weights; a differing `e` (the eligibility
*does* engage) cannot move a number that hidden training cannot move. This is the **reservoir-collapse**
blocker — the ±1 recurrence erases the class signal from the read layer — **downstream of and orthogonal to**
the credit rule. Byte-identical-across-β is therefore *expected*, not a bug.

**Corrected verdict.** Completing the credit rule was the right thing to rule out, and it is now ruled out:
the eligibility matches Bellec 2020 (Eq. 24–25, cross-checked against the official implementation) and is
verified to engage on-substrate, yet it **does not make trained recurrence earn its keep anywhere** — fixed
recurrence beats trained recurrence, FF beats both, and the fair-test collapse is untouched. The earlier
"only BPTT left, every confound ruled out" was premature *because the credit rule was incomplete*; with it
completed, the honest picture is **two separated, still-open blockers**:

- **(a) Training the recurrence is destructive.** Fixed random recurrence is a good reservoir; e-prop
  training (crude *or* complete) craters it. The lever is either **don't train the loop** (use it as a fixed
  reservoir — which already works) or **exact temporal credit (surrogate-gradient BPTT)** in place of
  e-prop's approximation.
- **(b) Reservoir collapse in the deep hidden-rec regime.** The ±1 recurrence drives the read layer to a
  class-agnostic chance state; needs recurrent-gain control (σ≈1) so the loop neither dies nor runs away —
  independent of the credit rule.

**Scope / limitations:** `elig_beta ∈ {0.1, 0.2, 0.4}` (plus a `2.0` spot-check on the inert config),
`PSI_WIDTH = 16`, `rec_tau` for εᵛ rather than the true membrane α, and `z(t)` rather than `z(t−1)` in the
pre-trace — the eligibility *structure* is faithful (εᵃ recursion, `−β·εᵃ` sign, ρ from the substrate's real
adaptation decay), but W, β, and the εᵛ time-constant are not exhaustively swept, and the `elig_bump_psi`
ablation (bump-ψ without εᵃ) was not run. The completed eligibility, the decide-time `eff` snapshot
(`Layer.decide_eff`), and the `elig_beta` / `elig_bump_psi` knobs are now in place for a future BPTT
comparison and for a "fixed vs trained recurrence" study — the latter being the more promising near-term lead
(fixed recurrence *works*).
