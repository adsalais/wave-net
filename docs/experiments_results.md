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

**V2b parameter sensitivity — the learnable regime is a narrow pocket.** One-at-a-time sweep around the
working baseline (late-half accuracy ‰, 2000 trials, chance 500, baseline ~652):

| knob | values → late ‰ | reading |
|---|---|---|
| size (width) | 8→652, 16→641 | robust — width barely matters |
| **layers (depth)** | 2→485, **3→652**, 4→485 | **knife-edge — only depth 3 learns; ±1 collapses to chance** |
| refractory (cooldown) | 1→652, 2→652, 4→611 | robust for low/mid |
| **up_count (density)** | 8→485, **16→652**, 24→501 | **narrow band — sparse *and* dense collapse** |
| **up_radius** | 2→485, **3→652**, 4→544 | brittle — needs radius 3 |
| **adapt_bump** | 0→587, 5→377, **10→901**, 15→498, 20→652, 40→504 | **non-monotonic/knife-edge — 10 is a fragile spike, not a trend** |
| adapt_decay | 4→705, 6→652, 8→626 | robust; faster decay mildly better |
| present_waves | 3→504, 6→652, 12→696 | longer cue → better |
| delay | 0→573, 2→513, **4→652**, 8→625, 16→513 | ~4 optimal (needs *some* held delay, but memory decays if too long) |
| read_waves | 3→573, **6→652**, 12→382 | ~6 optimal; **long read over-integrates and destroys the signal** |

**Takeaway:** V2b learns only inside a narrow pocket of *connectivity/depth/adaptation* space — depth,
`up_count`, `up_radius`, and `adapt_bump` each collapse to chance under small perturbations, while *width,
refractory, and adapt_decay* are robust. This is the [density-tradeoff finding](#the-connectivity-density-tradeoff)
resurfacing under learning: broadcast credit only shapes the reservoir when its dynamics already sit in the
right regime. The `adapt_bump=10 → 901‰` result is a *fragile resonance* (neighbours 5→377, 15→498 are
null), **not** a robust improvement — so it was **not** adopted; the committed baseline (`adapt_bump=20`)
sits in a stable pocket. On timing: more presentation helps, a delay of ~4 is the working-memory sweet spot,
and the readout window must stay short (long integration washes the class signal out — a readout-specific
caveat).

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
