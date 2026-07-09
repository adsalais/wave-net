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
