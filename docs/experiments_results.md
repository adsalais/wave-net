# Experiment results — ALIF vs LIF on the wave-net bench

**Date:** 2026-07-08
**What this is:** a consolidated record of the ALIF-vs-LIF experiments run on the integer bench
(`src/bench/`), across store-recall (Spec 1), Memory Capacity (Spec 2), temporal XOR (Spec 2b), and an
architecture robustness sweep over all three. Design rationale lives in the per-spec docs under
`docs/superpowers/specs/`; the literature framing and forward-looking notes live in
`docs/related-work.md`. This file is the *results*.

## Setup

- **Substrate:** the integer wave-net engine. **ALIF** = per-neuron adaptive threshold (`adapt_bump > 0`,
  fixed-point Q12 adaptation, τ ≈ `2^adapt_decay` waves). **LIF** = the *same* network with
  `adapt_bump = 0`. Every comparison is that single knob, each variant calibrated to the same firing rate.
- **Isolation:** most experiments use a **feed-forward** topology so recurrence can't carry memory on its
  own (recurrence confounds the comparison — see the store-recall note below). L0 is a non-adapting input
  transducer; the computational layers `1..L` carry the dynamics.
- **Readouts (bench, `f64`; engine stays integer):** integer nearest-centroid classifier (store-recall);
  ridge regression, factored once (MC and XOR).
- **Determinism:** every number below is a pure function of `(seed, config, params)`. Configs are small
  demo networks (size 8–16, 3–6 layers); the findings are qualitative and were stress-tested by the sweep.

## Experiment 1 — Store-recall (delayed match): *held-category* memory

Present one of `K=4` cues, wait a silent delay `N`, then inject a fixed probe and decode which cue it was
from the spike response. Only ALIF's residual adaptation footprint should survive the delay.

**Memory-horizon (feed-forward), decode accuracy ‰ (chance = 250):**

| delay N | 0 | 8 | 24 |
|---|---|---|---|
| **ALIF** | 900 | 550 | 550 |
| **LIF** | 1000 | 250 | 250 |

**Result: ALIF wins.** LIF collapses to chance by delay 8 (it forgets); ALIF holds the cue at ~550‰ out to
delay 24. The cue lives in the slow adaptation state, and the probe converts it into a readable spike
pattern.

## Experiment 2 — Memory Capacity (MC): *delayed linear echo*

Stream i.i.d. bits `u(t)` in bins; fit a linear readout to reconstruct `u(t−k)` from the state; `MC = Σ_k
r²_k`. Measures how well the reservoir *linearly echoes* specific past bits.

**MC total (feed-forward and recurrent):**

| | feed-forward | recurrent |
|---|---|---|
| **LIF** | 1.57 | 1.58 |
| **ALIF** | 0.39 | 0.38 |

LIF reconstructs the most-recent bit near-perfectly (`r²₁ ≈ 1.0`, via the one-hop delay).

**Result: LIF wins, by ~4×.** MC measures linear echo; adaptation is a slow **low-pass integrator** that
can't pinpoint one past bit — so it *lowers* MC. `adapt_bump = 0` is the max-MC point. Exposing the raw
adaptation state to the readout did not help (it's low-pass, not echo).

## Experiment 3 — Temporal XOR: *nonlinear temporal* computation

Target `y(t) = u(t) ⊕ u(t−τ)`, swept over `τ`; thresholded ridge classifier. XOR is not linearly
separable in the inputs, so the reservoir must provide the nonlinear features.

**Feed-forward accuracy ‰ (chance = 500), inhibitory reservoir:**

| τ | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| **LIF** | 861 | 661 | 561 | 552 |
| **ALIF** | 595 | 480 | 476 | 576 |

**Result: LIF wins; ALIF near chance.** The hypothesis that adaptation's nonlinearity would *buy* nonlinear
temporal computation is **falsified** — ALIF does not help.

## Architecture robustness sweep

Each experiment was re-run across the same nine feed-forward architectures — varying width, depth,
refractory period, connectivity density, and inhibition — to check whether the findings are robust or
config artifacts.

**Store-recall** (decode ‰ at delay 24, chance 250):

| arch | LIF | ALIF |
|---|---|---|
| baseline (dense) | 250 | 950 |
| wider (size 16) | 250 | 1000 |
| deeper (6 layers) | 250 | 950 |
| refractory = 1 | 250 | 950 |
| refractory = 4 | 250 | 950 |
| inhibition 0.15 | 250 | 850 |
| **sparse (count 6)** | 250 | **350** |
| wide + inhibition | 250 | 1000 |

**MC total** (higher = more linear memory):

| arch | LIF | ALIF |
|---|---|---|
| baseline | 1.42 | 0.40 |
| wider (size 16) | 1.71 | 0.33 |
| deeper (6 layers) | 1.44 | 0.34 |
| refractory = 1 | 1.41 | 0.40 |
| refractory = 4 | 1.41 | 0.38 |
| inhibition 0.15 | 1.64 | 0.39 |
| sparse (count 6) | 1.65 | 0.23 |
| **wide + inhibition** | **2.04** | 0.41 |
| recurrent | 1.51 | 0.38 |

**XOR** (accuracy ‰ at τ = 1, chance 500):

| arch | LIF | ALIF |
|---|---|---|
| baseline (dense) | 727 | 544 |
| wider (size 16) | 761 | 544 |
| deeper (6 layers) | 694 | 527 |
| refractory = 1 | 683 | 544 |
| refractory = 4 | 738 | 544 |
| inhibition 0.15 | 800 | 550 |
| **sparse (count 6)** | **927** | 544 |
| wide + inhibition | 888 | 638 |
| recurrent | 555 | 533 |

**Robustness verdict:**
- **Store-recall (ALIF wins): robust** across width/depth/refractory/inhibition (850–1000‰), **except
  sparse connectivity**, where ALIF collapses to 350‰.
- **MC (LIF wins): robust** — richer/inhibitory reservoirs *raise* LIF's MC (up to 2.04) while ALIF stays
  pinned at ~0.3–0.4, widening the gap.
- **XOR (LIF wins): robust** — sparsity/inhibition *raise* LIF's accuracy; recurrence hurts; ALIF never
  helps in any architecture.

## Synthesis

**ALIF's adaptation buys exactly one thing: categorical memory held across a delay and read by a probe
(store-recall). It does not help linear echo (MC) or nonlinear temporal computation (XOR) — in any
architecture.** LIF and ALIF occupy different points on a fast-echo ↔ slow-held memory spectrum:

| axis | who wins | why |
|---|---|---|
| held-category across a delay | **ALIF** | adaptation footprint persists in a silent, sub-threshold state |
| linear echo of a specific past bit | **LIF** | fading spike echo; adaptation is low-pass, can't pinpoint |
| nonlinear temporal computation | **LIF** | adaptation adds history-dependent gain that scrambles bit-level features |

### The connectivity-density tradeoff

The sweep exposed opposite density preferences for the two winning behaviors:

| | dense (count 16) | sparse (count 6) |
|---|---|---|
| **Store-recall** (ALIF held memory) | strong (950‰) | weak (350‰) |
| **XOR** (LIF nonlinear) | weak (727‰) | strong (927‰) |

**Dense fan-out spreads the cue's adaptation footprint across many neurons → good for ALIF held memory;
sparse fan-out gives more distinct, less-redundant features → good for LIF nonlinear separation.** So a
heterogeneous network should mix *densities per role*, not just neuron types: dense-ALIF layers to hold
context, sparse-LIF layers to compute (`adapt_bump` is already per-layer).

## Experiment 4 — e-prop-like threshold learning (Spec 3, v1): *it learns*

A gradient-free three-factor rule: per-neuron **eligibility** (trial spike count) × a **global
reward-prediction-error** `(R − R̄)` nudges each baseline threshold, `Δθ = −lr·(R−R̄)·e`, accumulated in an
`f64` shadow written back to the integer engine. Task: the `K=2` held-category store-recall (cue → delay →
probe); output = the top layer split into `K` **population groups**; reward = the correct group's spike
margin. No engine change; no trained readout — the thresholds *are* the learned parameters.

**Result: the rule learns.** Late-training accuracy vs a frozen-threshold control (`lr = 0`, same seed):

| | late-half accuracy ‰ (chance 500) |
|---|---|
| **learning** (`lr = 0.3`) | ~770 |
| **frozen control** | ~271 |

The learning curve climbs from ~chance to ~77% and clearly beats the frozen control; it's noisy
block-to-block (crude credit + threshold random-walk), so the metric is the late-half mean.

**Two findings from getting it working:**
- **Population coding was required.** Reading two *single* output neurons failed — they're almost always
  silent, so `R = 0` every trial at any `lr`. Worse, a silent neuron has **zero eligibility**, so the rule
  can never wake it (chicken-and-egg). The spike-count eligibility can *reshape active* neurons but not
  *recruit silent* ones — which is exactly why the deferred **non-spiking potential readout** (potential is
  non-zero even when silent) is the principled next step.
- **The `f64` shadow is essential.** `lr` an order of magnitude below 0.3 never moved the integer
  thresholds at all — tiny updates round to 0 without the fractional accumulator.

**V2a — non-spiking potential readout: engine works, global-reward learning is a null.** A dedicated
non-spiking *readout layer* (drain-only integrator, `Layer.readout`; the L0/L_top symmetry) was added to
the engine and works. But learning with a **global scalar reward** fails: readout accuracy sits at chance
(~490‰ vs frozen ~510‰) at *every* `lr`, while V1's spiking-trainable-output path learns (~770‰). Reason: a
non-spiking readout has no trainable output, so learning is entirely *internal* (feedback-alignment); the
fixed ±1 readout projection doesn't separate classes, so `(R − R̄) → 0` — a scalar reward is too weak to
shape the reservoir to a fixed projection. **This pins the boundary of the crude rule: global reward learns
only when the output itself is trainable.** The readout must pair with **per-output broadcast-error credit**
(V2b), for which the readout layer is now infrastructure.

**V2b — broadcast-error alignment: the readout learns.** Replacing the global scalar with a **per-output
broadcast error** (softmax error × fixed random hash-derived feedback weights, `Δθⱼ = −lr·Σᵢ B(j,i)·errᵢ·eⱼ`)
makes the non-spiking readout learn — **~687‰ (peaks ~740)** vs frozen ~500, on par with V1's ~770. The
learning arc, end to end:

| variant | output | credit | late accuracy ‰ |
|---|---|---|---|
| V1 | spiking populations (trainable) | global reward | ~770 |
| V2a | non-spiking potential readout | global reward | ~490 (null) |
| **V2b** | non-spiking potential readout | **broadcast error** | **~687** |

So all-internal (feedback-alignment) learning works **once the credit is per-output** — exactly the boundary
V2a exposed. (`softmax_temp = 10`: the potential-sum scores are large, so a low temperature is needed to keep
the error from washing out to uniform.)

**Parameter sensitivity — V1 vs V2b (the surprising part: V1 is the brittle one).** One-at-a-time sweep of
both learners around their shared baseline (late-half accuracy ‰, 2000 trials, chance 500; `485` is the
dead/no-learning value — argmax defaults to the majority class). **Bold** = collapse-to-chance or notable
peak.

| knob (values) | V1 late ‰ (base 805) | V2b late ‰ (base 652) |
|---|---|---|
| size 8 / 16 | 805 / **485** | 652 / 641 |
| layers 2 / 3 / 4 | 663 / 805 / **485** | **485** / 652 / **485** |
| cooldown 1 / 2 / 4 | 805 / 805 / **485** | 652 / 652 / 611 |
| up_count 8 / 16 / 24 | **485** / 805 / 554 | **485** / 652 / 501 |
| up_radius 2 / 3 / 4 | **485** / 805 / 527 | **485** / 652 / 544 |
| adapt_bump 0/10/20/40 | 609 / 548 / **805** / **485** | 587 / **901** / 652 / **504** |
| adapt_decay 4 / 6 / 8 | **531** / 805 / **485** | 705 / 652 / 626 |
| present 3 / 6 / 12 | 483 / **805** / 594 | 504 / 652 / **696** |
| delay 0/4/8/16 | 653 / 805 / 726 / 597 | 573 / 652 / 625 / 513 |
| read 3 / 6 / 12 | **485** / 805 / **485** | 573 / 652 / **382** |

**Three findings:**

1. **V1 wins on peak, V2b wins on robustness.** V1 reaches higher (~805 vs ~652) but lives in a *much*
   tighter pocket: it collapses to chance on `size=16`, `cooldown=4`, `adapt_decay∈{4,8}`, `read∈{3,12}`,
   `present=3` — all of which V2b tolerates (graceful degradation, not collapse). The broadcast learner's
   *distributed per-neuron* credit is more forgiving of off-nominal dynamics than V1's *global* reward,
   which needs the spiking output populations precisely in regime. A genuine robustness/performance tradeoff
   between the two rules.
2. **Shared brittleness = the reservoir's dynamical regime, not the rule.** *Both* collapse on depth
   (only 3 layers), `up_count` (only 16), and `up_radius` (only 3) — the [density
   tradeoff](#the-connectivity-density-tradeoff) again: neither credit signal can learn unless the
   connectivity puts the reservoir in the right regime first.
3. **Adaptation strength is resonant for both.** `adapt_bump` is non-monotonic: V1 peaks at 20, V2b spikes
   at 10 (a *fragile* spike — neighbours 5→377, 15→498 are null, so **not** adopted). Timing: a delay of ~4
   is the working-memory sweet spot for both; V1 wants a tight `present=6`/`read=6` while V2b likes longer
   presentation but still needs a short read window (its drain-only readout over-integrates — `read=12`
   drops *below* chance).

### Regime diagnostic — what predicts learnability (and topology × adaptation coupling)

To explain the brittleness we measured eight properties of the *calibrated-but-untrained* reservoir (top
computational layer — what the readout accesses) and correlated them with learned accuracy for both rules:

| cfg | sep.ceiling | Fisher | eff.dim | k−g rank | σ | V1 | V2b |
|---|---|---|---|---|---|---|---|
| baseline | 750 | 0.48 | 1.2 | 0.0 | 0.33 | 772 | 621 |
| up_count=8 | 490 | 0.00 | 0.0 | 0.0 | 0.00 | 505 | 505 |
| up_count=24 | 600 | 0.12 | 4.1 | −1.0 | 0.20 | 557 | 558 |
| up_radius=2 | 490 | 0.26 | 1.0 | 0.0 | 0.00 | 505 | 505 |
| layers=2 | 1000 | 1.52 | 2.6 | −6.7 | 0.00 | 788 | 505 |
| layers=4 | 490 | 0.00 | 0.0 | 0.0 | 0.33 | 505 | 505 |

**Findings:**

1. **Separation ceiling predicts V1 cleanly.** `ceiling ≥ 600 → V1 learns` (baseline, up_count=24, layers=2);
   `ceiling ≈ 490 (chance) → V1 dead` (up_count=8, up_radius=2, layers=4). A perfect split. **This is the
   metric a fix should target** — the reservoir must make the classes linearly separable *at the top layer*.
2. **V2b needs separation *and depth*.** `layers=2` separates perfectly (ceiling 1000) and V1 reads it off
   (788), but **V2b stays at chance (505)** — with only one computational layer the broadcast credit has no
   internal layer to shape. So separation is *necessary but not sufficient* for feedback-alignment; it also
   needs ≥2 trainable layers. (Explains the depth knife-edge differing by rule.)
3. **The dynamical / rank metrics do *not* predict here.** σ (edge-of-chaos), kernel−gen rank, effective
   dim, and degeneracy all fail to split learns-from-collapses — σ and k−g are confounded by depth
   (`layers=2` learns at σ=0; `layers=4` is dead at σ=0.33). **The working pocket is about top-layer
   *separation*, not criticality.** An honest negative for the fancier metrics.
4. **Topology × adaptation is a coupled ridge (hypothesis confirmed).** Separation ceiling over `up_count ×
   adapt_bump`:

   | cnt\bump | 5 | 10 | 20 | 40 |
   |---|---|---|---|---|
   | 8 | 787 | 637 | 525 | 525 |
   | 12 | 650 | **925** | 525 | 525 |
   | 16 | 675 | 575 | **750** | 525 |
   | 24 | 662 | **800** | 512 | 525 |

   The best `up_count` **shifts with `adapt_bump`**: weak adaptation (bump=5) separates at any density; the
   optimum moves to count=12 at bump=10, count=16 at bump=20, and nothing survives bump=40. More adaptation
   → more resistive neurons → needs denser fan-out to reach threshold. The OAT sweep's "fragile spike"
   (bump=10 → 901) was simply crossing this ridge. **The knobs are not independent — they trade off.**

**Scopes the fix:** a **separation-targeting calibration** that tunes the reservoir (thresholds, and given
the coupling, likely co-tuning drive/adaptation) to maximise the top-layer separation ceiling — so learning
stops depending on landing on the ridge by hand. For V2b specifically, also enforce depth ≥ 2 computational
layers.

### Deeper regime scans (separation ceiling as a cheap learnability proxy)

> Caveat: the ceiling is a single-seed held-out estimate over ~70–100 test trials, so it carries **±~100‰
> noise** (the *same* config reads 490 at 200 trials but 628 at 140). Read the **trends**, not single cells.

**adapt_bump saturates with trial length — your intuition, confirmed.** Ceiling over `adapt_bump × read_waves`:

| bump\read | 3 | 6 | 12 | 24 |
|---|---|---|---|---|
| 20 | 628 | 728 | 757 | 700 |
| **40** | 628 | 628 | 628 | 628 |
| **80** | 628 | 628 | 628 | 628 |

`bump=40` and `bump=80` are **identical (628) at every read length** — beyond a critical bump, adaptation is
inert: one fire already maxes its within-trial effect (the effective threshold jumps past reach), so a
larger bump, or a longer trial, changes nothing. The mechanism: adaptation accumulates `≈ N·bump` over `N`
fires, and once one fire silences the neuron for the rest of the (~16-wave) trial, extra bump is wasted.
**So `adapt_bump` should be read relative to trial length, and capped ~one-fire-to-silence.**

**Depth: the class signal *decays* upward — deeper does not help.** Ceiling vs `layers` (rows) × `up_count`:

| L\cnt | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| **2** | 1000 | 1000 | 971 | 1000 |
| 3 | 628 | 728 | 414 | 614 |
| 4–6 | ~628 | ~628 | ~628 | ~400–628 |

One computational layer (`layers=2`) separates **perfectly** (the top layer *is* the cue); every added
layer *loses* separation, flattening at ~628 by depth 3 and **not recovering at 5–6 layers**. The class
signal attenuates through the feed-forward stack. (This is why V1 — reading the top layer directly — is fine
shallow, while V2b needs depth for broadcast but pays a separation cost for it.)

**Width helps; initial threshold does not.** Wider layers separate better — at depth 3, `size=64` (64²
neurons) reaches **1000** vs `size=8`'s ~728 (more neurons → higher-dimensional, more separable top-layer
code); depth still erodes it (`size=64, layers=5 → 680`). But sweeping `baseline_init ∈ {1..256}` and
`threshold_jitter ∈ {0..512}` against `up_count` left the ceiling **essentially flat** — **calibration
overrides the initial thresholds** (it tunes them to the firing-rate target regardless of where they start),
so the initial-threshold-vs-synapse-count coupling is **decoupled by calibration**, not a usable lever. The
residual density structure (`up_count=16` robustly good, `24` a persistent bad resonance) is **structural**,
independent of the initial threshold.

**Net for the fix:** target **separation** (thresholds are calibration's job, so tune *drive/adaptation/
density*, not init); **scale width** to buy separation headroom; keep **`adapt_bump` bounded relative to
trial length**; and note that **more depth costs separation**, so the fix must actively *preserve* the
class signal up the stack (e.g. skip/residual-style projection or per-layer separation targets), not just
add layers.

**Width × depth interact — width buys ~one extra layer of separation.** Ceiling over `size × layers`:

| size\L | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| 8 | 1000 | 728 | 628 | 628 | 628 |
| 16 | 1000 | 628 | 628 | 628 | 628 |
| 32 | 1000 | 680 | 680 | 680 | 680 |
| 64 | 1000 | **1000** | 680 | 680 | 680 |

`size=64` holds *full* separation through depth 3 (where `size=8` has already decayed to 728), and raises
the deep-net floor (680 vs 628). But every width plateaus by depth 4–5 — **wider carries the class signal
about one layer deeper, not arbitrarily.** So a deep network needs *proportionally wider* layers to keep
separating, and even then depth eventually wins.

**Calibration target rate — and the confound it hid (firing rate is *not* the lever).** First pass looked
tidy: ceiling collapses below 10% and is flat above, so "10% is the knee." But 10% is the value we chose, so
we measured the *achieved* rate and varied the *drive* to check for circularity — and the tidy story broke:

- **Achieved task rate is decoupled from the target.** At `up_count=16`, targeting 5% / 10% / 20% / 40% all
  produce the **same ~6.3%** task firing rate (the delay dilutes it and the drive pins it) — yet the ceiling
  is 525 at target 5% but 750 at ≥10%. **Same rate, different separation** → the rate is not what changed;
  the *threshold level* is (targeting ≥10% lands thresholds low enough, 5% leaves them too high → collapse).
- **Denser drive fires faster but separates *worse*.** Raising `up_count` *does* lift the achievable rate
  (16 → 6.3%, 32 → 13%, 48 → 19%), directly refuting "higher targets are unreachable" — but higher firing
  **hurts**: ceiling 750 (up_count=16, 6.3%) → 525 (32, 13%) → 475 (48, 19%).

| up_count | achieved rate | ceiling |
|---|---|---|
| **16** | 6.3% | **750** |
| 32 | 13.3% | 525 |
| 48 | 18.7% | 475 |

**Correction:** the earlier "10% is optimal / denser codes need more drive" was an **artifact of hand-tuning
the drive to `up_count=16`**. The firing rate is an *epiphenomenon*, not a control knob. Calibration's only
job here is to set thresholds *low enough* (any target ≥10% does that). (Credit: this correction came from
*not* trusting a suspiciously convenient result.)

### Verification pass — multi-seed + held-out (most single-seed findings do NOT survive)

Prompted by the 10%-artifact, we stress-tested the load-bearing claims with a **held-out test split** in
`train()` and **4 seeds**. The results are sobering — a lot of the single-seed story was noise.

**① V2b does not generalize.** The headline "V2b learns to 687‰" was **prequential** (predict-then-update on
training trials). With a held-out test (frozen thresholds, unseen cue realizations):

| | train-late ‰ | held-out test ‰ |
|---|---|---|
| V1 (spiking, trainable output) | 770 | **697** (generalizes) |
| **V2b (broadcast readout)** | 687 | **505 = chance** (does NOT) |

With `lr=0.5` the broadcast update *chases recent trials* (inflating the online curve); the frozen classifier
is at chance. **The V2b "success" was an artifact of the wrong metric.** Only V1 learns a stable classifier.

**② `up_count=16` optimum — refuted (seed artifact).** Separation ceiling across seeds:

| up_count | s0 | s1 | s2 | s3 | mean |
|---|---|---|---|---|---|
| 8 | 490 | 550 | 470 | 520 | 507 |
| **16** | **750** | 550 | 470 | 520 | 572 |
| 24 | 600 | 920 | 560 | 970 | **762** |
| 32 | 490 | 570 | 650 | 960 | 667 |

`up_count=16` looked optimal **only at s0** (our dev seed); elsewhere it's near chance, and `24` has the best
mean. Same failure mode as 10%. **There is no stable density optimum** — the ceiling swings 470–970 for one
config, so it's a **±~250‰ instrument**, far noisier than the ±100 estimated. Fine-grained density findings
do not survive.

**③ V1 learnability is itself seed-fragile.** Held-out V1 test for the *baseline*: s0 → 845, s1 → 587, s2
(`0xdeadbeef`) → **472 (chance)**. At s2 *every* config's ceiling is ~470 and every V1 test ~472 — the
reservoir is globally degenerate for that seed. So even "V1 reliably learns" is really "V1 learns at ~2 of 3
seeds"; some random-`±1` reservoirs simply can't be trained.

**④ What *does* survive:** the **causal** V2a-null → V2b-flip (one factor changed behavior — though now known
not to *generalize*); the **floored-leak** bug; **`layers=2` separates robustly** (~987 across seeds — but
trivial: the top layer *is* the cue); and the **`adapt_bump` saturation mechanism** (`40≡80`). The
separation ceiling retains only *coarse* value: a very low ceiling (~470) reliably means chance learning; a
high one (~750) means it learns — but the broad middle is uninformative.

**Verdict:** most specific regime "optima" were single-seed coincidences, and V2b never generalized. **Any
future claim must be multi-seed and held-out from the start** — and the deeper question is whether a fixed
random-`±1` feed-forward integer reservoir is a trainable substrate at all, given even V1 fails on some seeds.

### Does width fix the seed-fragility? No.

The one hypothesis that could redeem the substrate: seed-fragility is a *small-N* problem (64 neurons/layer is
a tiny random-feature sample; at large N the law of large numbers should make *some* neurons reliably
discriminative). Test — V1 **held-out** test accuracy over width × 4 seeds (layers 3, lr 0.3):

| size (N/layer) | s0 | s1 | s2 | s3 | mean | worst |
|---|---|---|---|---|---|---|
| 8 (64) | 845 | 587 | 472 | 512 | 604 | 472 |
| 16 (256) | 510 | 457 | 472 | 532 | 492 | 457 |
| 32 (1024) | 490 | 455 | 527 | 952 | 606 | 455 |
| 64 (4096) | 505 | 460 | 527 | 710 | 550 | 460 |

**Refuted.** The worst seed sits at chance (~460) at *every* width; the mean shows no trend; and even at 4096
neurons/layer learning is still a coin flip (~1–2 of 4 seeds learn, the rest chance). Width does **not** tame
the fragility — so the problem is **not** the reservoir's random-feature *coverage* (which width would fix).

**Where that leaves it.** The `seed` controls *both* the reservoir *and* the seeded cue/probe patterns, so the
per-instance coin flip could live in either: a "bad seed" may mean a bad random projection **or** a
degenerate task encoding (the two class patterns not separably re-triggered by the probe). Width refuting the
reservoir-coverage story shifts suspicion toward the **task encoding** — but they're currently entangled under
one seed. Disentangling them (separate network vs task seeds) is the next decisive test; "just scale up" is
dead either way.

### Split the seed (network vs task) + BLAKE3 — a hash artifact *and* a real ceiling

Split `seed` (reservoir/calibration/feedback) from `task_seed` (cue/probe/class sequence) and measured V1
**held-out** test accuracy varying one axis while fixing the other — under the default `key`/`mix` hash and
again under BLAKE3 (`--features strong_hash`, a cryptographic hash swapped in at the single `mix` choke point):

| | net/task=s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| default, fix task / vary **net** | 845 | 505 | 505 | 507 |
| default, fix net / vary **task** | 845 | 460 | 472 | 560 |
| **blake3**, fix task / vary **net** | 417 | 442 | **877** | 442 |
| **blake3**, fix net / vary **task** | 417 | 505 | **885** | 507 |

**① The "same-seed works" pattern was a hash artifact.** Under the default hash, *only* `(s0,s0)` learned and
every off-diagonal was chance — i.e. learnability required the reservoir and task to be hashed from the *same*
seed. Under BLAKE3 that vanishes: `(s0,s0)` drops to chance (417) and *different* pairs learn instead
(`net=s2`→877, `task=s2`→885). So `key`/`mix` **correlates the network and task streams that share a seed** —
a genuine engine defect. Our entire dev setup ran with `net==task` (one seed), so the original **"V1 learns
~770" was partly propped up by this artifact.**

**② A clean hash does *not* fix the fragility.** BLAKE3 only *reshuffles* the lottery — still ~1 of 4 pairs
learns, the rest chance. So the ceiling is real and hash-independent: **threshold-only training on frozen
random ±1 weights learns only when the (reservoir × task) pair happens to align.** No calibration, width, or
hash change makes it reliable.

**Conclusion of the whole verification arc.** The learning results were far weaker than they looked: V2b never
generalized, most regime "optima" were single-seed coincidences, and even V1's apparent success leaned on a
hash correlation. The mechanistic core stands (V2a→V2b flip, floored-leak, adapt saturation), but **the
substrate is not a reliable learner as built.** The trainable degrees of freedom (per-neuron *thresholds*)
can only gate a fixed random projection; they cannot *shape* it to the task — so success needs luck. The
decisive next direction is **architectural: make the weights/input projection trainable** (so the network can
align the reservoir to the task) rather than searching for a cleverer calibration of a fixed random one.

### "One last chance" — richer procedural weights (still static) don't fix it

Test: replace the ±1 synapse sign with a procedural **random magnitude** (`sign × 1..=16`, hash-drawn — a
richer but still *static* projection, as GeNN samples procedural weights from a distribution), run under
BLAKE3 to avoid the hash confound, and re-measure the V1 held-out seed-split:

| | net/task=s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| ±1, vary net | 417 | 442 | **877** | 442 |
| **rand-wt, vary net** | 570 | 442 | 312 | 410 |
| ±1, vary task | 417 | 505 | **885** | 507 |
| **rand-wt, vary task** | 570 | 687 | 472 | — |

**No improvement — arguably worse.** Peaks are *lower* (687 vs 885), there are *more* below-chance cells, and
it's still a 1-in-4 lottery. Enriching the fixed random projection does not make threshold-only training
reliable. This is exactly what Knight & Nowotny predict: procedural connectivity is a **static**-connectivity
technique — *"plastic synapses which change their weights must be simulated in the traditional way"* (i.e.
**stored**). A richer static projection is still static.

**Final verdict.** Three independent attempts to rescue the substrate — **scale (width)**, **hash quality
(BLAKE3)**, and **projection richness (random weights)** — all failed to make learning reliable. The ceiling
is structural: **you cannot reliably train a network whose only plastic degrees of freedom are per-neuron
thresholds over frozen random weights.** The GeNN-sanctioned path is a **hybrid — keep the procedural static
reservoir, add a small *stored, trained* readout/projection** (cheap at our scale). See
`related-work.md` §1 for the paper grounding.

## Engine finding along the way — the floored leak

Store-recall *found a real substrate bug*. The potential leak `p -= (p>>a)+(p>>b)` is `0` for `0 < p <
2^a`, so small sub-threshold potentials **froze forever** — the network had *infinite* sub-threshold
memory, and plain LIF never forgot a cue (the ALIF-vs-LIF distinction vanished). Fixed by flooring positive
decay at 1 (`p -= max((p>>a)+(p>>b), 1)`), giving a finite membrane time constant. Cost: the 1/wave floor
starves sparse cascades, so configs need denser drive (the fix and its density cost are in
`docs/related-work.md`, with the fixed-point-`potential` upgrade as the follow-up).

## Implications

- **For training (Spec 3, e-prop-style threshold learning) — now demonstrated (Experiment 4):** trained
  against a **held-category** task (store-recall) — the only thing adaptation buys — the three-factor rule
  learns (~770‰ vs frozen ~271‰). **Not** MC or XOR (bit-level tasks LIF already does better). e-prop's
  eligibility trace on the per-neuron threshold is the machinery that credits exactly this slow held-state.
  Ensure ALIF layers have **dense**
  fan-out.
- **Heterogeneous networks** (mixed LIF/ALIF layers, mixed densities) are the natural way to span both
  memory axes — worth a bench experiment (a mixed config on store-recall *and* XOR) before or during Spec 3.
- **Always control for passive memory** when attributing memory to adaptation: recurrence carries memory on
  its own (use feed-forward to isolate), and — before the floored-leak fix — frozen potentials gave even
  LIF infinite memory.
