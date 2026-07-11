# Experiment results — the `wave_net` RSNN

**What this is:** the standing record of results on `wave_net`, the stored-weight RSNN engine (`src/bench/`
drives the experiments). Every number is held-out, multi-seed, and a pure function of `(seed, config,
params)`. Design rationale lives under `docs/superpowers/specs/`; literature framing in `docs/related-work.md`.

## Substrate

The integer `wave_net` engine: synapse **addresses** stay procedural (hash-regenerated at fire time, free);
**weights** are stored (`i8` + `f32` shadow) and trained. ALIF = per-neuron adaptive threshold (`adapt_bump >
0`, Q12 fixed point, τ ≈ `2^adapt_decay` waves); LIF = `adapt_bump = 0`. Readouts are bench-side `f64`; the
engine stays integer. One engine fix underlies everything: the potential leak is **floored at 1**
(`p -= max((p>>a)+(p>>b), 1)`) — the old leak had a dead zone (`0` for `0 < p < 2^a`) that froze sub-threshold
potentials forever (infinite passive memory); flooring gives a finite membrane time constant and caps how
long a trace persists.

## The pivot: train weights, not thresholds

Training only per-neuron **thresholds** over a *fixed random ±1* projection does **not** reliably learn
(held-out, multi-seed: works only when the reservoir×task seed pair aligns, ~1-in-3). Thresholds can only
*gate* a fixed projection, never *shape* it — scaling width, a crypto hash (BLAKE3), and richer static
weights all failed to rescue it. This is what GeNN (Knight & Nowotny 2021) predicts: procedural connectivity
is *static*-only; **plastic weights must be stored**. So `wave_net` stores + trains the weights (the
`wave_state_machine` fork stays the frozen pure-procedural reference), and **e-prop on the stored weights
learns reliably** where thresholds failed:

| | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| fixed-reservoir top-layer readout | 727 | 500 | 655 | 507 |
| **+ e-prop hidden-weight training** | **1000** | **965** | **890** | **987** |

(A trained readout on the *full* reservoir also hits 1000 on all seeds — the classic LSM.) Two facts carry
forward: the class signal separates **less well the deeper** a fixed feed-forward stack runs, and **ALIF
adaptation is a strong ~64-wave working memory** (the bar recurrence must beat).

## Depth is usable, and width sets how deep

Train **every** layer, not just the last: the top layer gets symmetric readout feedback, deeper layers get
**Direct Feedback Alignment** (fixed random hash-derived feedback of the output error). Single-layer e-prop is
at chance for every depth ≥ 4 (untrained intermediate layers erode the class signal); **multi-layer DFA is
reliable to ~16 layers — but only with enough width.** Width is the lever that sets the depth the credit can
reach (worst-seed / 3 seeds, multi-layer):

| size (N/layer) | depth 4 | depth 5 | depth 6 |
|---|---|---|---|
| 8 (64) | 1000 | **485** | 902 |
| 16 (256) | 1000 | **1000** | **1000** |

Size 8 collapses past depth 4 (too few neurons to train the hard seed's deep layers); size 16 stays clean
through depth 6, and (with trial length scaled to depth so the cue reaches the top) holds to ~16 layers.
Beyond that, DFA's random feedback is too noisy. **Wide + multi-layer = reliable deep learning; either alone
fails.**

## Recurrence — the side-car topology robustly beats feed-forward

Trained recurrence earns its keep once two things are in place: **the completed ALIF credit rule** and **a
topology that isolates the recurrent layer from the forward path**. The winning architecture is the
backward-fed **side-car** (`train_sidecar_task`, `engine_config_sidecar`): the forward signal **skips past**
the recurrent layer (L1 → L3 via a +2 skip) while a separate **recurrent scratchpad** (L2: self-loop + a
L2→L3 forward, with L3 feeding L2 back) holds state alongside — so the loop computes *without* injecting its
reverberation into the clean forward projection. *(These numbers are the original **direct-read** side-car —
the readout reads the recurrent layer **L3 directly**, so L3 is the top layer with clean symmetric feedback.
The code's `engine_config_sidecar` has since been extended with a dedicated read layer L4 for the scaling
study below; reading L3 directly scored higher — see "Scaling study".)*

**Result (size 32, worst-seed / 3 seeds) — a strict improvement over feed-forward on every benchmark:**

| task | FF | **side-car** |
|---|---|---|
| temporal XOR (delay 20) | 990 | **1000** |
| flip-flop (delay 12) | 985 | **1000** |
| distractor-XOR (delay 20) | 700 | **995** |
| parity N=4 (delay 8) | 587 | **837** |

Where FF **saturates** (XOR, flip-flop) the side-car ties at ceiling — no cost. Where FF **struggles** it wins
big: parity N=4 587 → 837, and distractor-XOR 700 → 995 (the task hides A behind an irrelevant middle cue; the
recurrent scratchpad holds A across it where a pure forward stack can't). Verified across seeds *and* tasks —
the first robust, general result where trained recurrence beats feed-forward here.

**The credit rule that trains it.** Earlier recurrence tests used the **crude spike-timing** eligibility (only
the fast membrane term `e = Σ_t ψ_j·εᵛ_i`). Textbook e-prop for ALIF neurons (Bellec 2020, Eq. 24–25) adds a
**slow** adaptation eligibility `εᵃ`, recursed at the adaptation rate `ρ`: `e = ψ·(εᵛ − β·εᵃ)` — carrying
credit over the ~64-wave adaptation horizon. It is now built and verified against the paper + reference
implementation (`RsnnConfig.elig_beta` / `elig_bump_psi` / `elig_psi_width`; a decide-time `eff` snapshot
`Layer.decide_eff`; a fixed-width bump ψ). Two ψ bugs fixed on the way: `eff` must be read at the decide step
(before the fire-bump), and the bump needs a fixed absolute band (the θ≈1 baseline collapses ψ to ~0, since
±1 drive overshoots `eff` by O(2–26)).

**What the sweeps found (hidden-recurrent stack, the route to the side-car):**

- **Width sets the ceiling** — rec_count 8, FF/rec: 990/725 (size 16) → 1000/982 (size 32).
- **Recurrence density must stay sub-critical** — sharp collapse cliff at rec_count ≈ 12 (size 16); above it
  the ±1 loop is super-critical and drowns the read layer to chance regardless of the credit rule.
- **Completed eligibility ≥ crude wherever it's trainable**, its edge largest where credit is hardest (light
  density / small width). But β·`elig_psi_width` tuning that helped the hidden-rec stack *hurt* the side-car
  (**topology >> hyperparameter tuning**) — the plain side-car (β 0.4, W 16) is best.

**Open:** generality to larger sizes; and *why* the side-car structure (forward skip + isolated recurrent
scratchpad + backward loop) is what unlocks it — working hypothesis: isolating the loop from the forward path
keeps the class projection clean while the scratchpad holds temporal state (a hypothesis to check against the
LSNN/e-prop literature). Tools: `train_sidecar_task` / `train_hidden_rec_task` / `sequence_trial_layers`,
`elig_beta` / `elig_bump_psi` / `elig_psi_width`, `hidden_rec_depth`.

## Scaling study (in progress) — forward drive, width, and read-layer topology

**Status: ongoing systematic exploration, mostly single-seed / preliminary** — the single-threaded integer
engine is too slow to run these sweeps multi-seed at the sizes needed. Recorded so the direction survives;
*re-verify multi-seed once the engine is faster.*

**Topology note (direct-read vs read-layer).** The robust headline result above (parity N=4 = 837, plus
XOR/flip-flop/distractor) is the **direct-read** side-car — readout on the recurrent layer **L3 directly** (L3
is the top layer → clean symmetric readout feedback). The code's `engine_config_sidecar` now instead has a
**dedicated read layer L4** (`… L3 → L4 (+1); L4 read`) — the *read-layer variant* used for this study.
Reading L3 directly scored **higher** (parity N=4, same params: direct-read **837** vs read-layer **~700–765**
single-seed): a separate read layer **demotes** the recurrent layer from symmetric feedback to noisy DFA
(≈ −200). So *reading the recurrent computation directly is itself load-bearing.* The read-layer variant is
kept because it cleanly separates "readout" from "computation" for the parameter sweeps.

**Forward drive × width (read-layer variant, parity N=4, rec 16/r4, single seed 0xE9…):**

| size ↓ / up_count → | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| 16 (256) | 540 | 540 | 557 | 560 |
| 32 (1024) | 540 | 540 | 712 | 710 |
| 64 (4096) | 540 | 542 | 697 | **765** |

Three relationships (preliminary):
- **Per-neuron forward-drive threshold ≈ up_count 32** (at radius 3), roughly **width-independent** — below
  it *everything* is chance. `up_count` is synapses/neuron into a fixed window, so it's a per-neuron drive
  switch, not a total-count one.
- **Width is a capacity floor** — size 16 (256) never leaves chance; ≥ size 32 (1024) is needed to have a
  working regime at all.
- **Above both bars, width × forward scale *jointly*** — size 64 + up 64 (765) beats either lever alone
  (~710). It's an AND, not a sum, and hasn't plateaued (size/up 128 may keep climbing).

**Recurrence density** wants to stay **sparse** — rec_count ≈ 8–16 is the sweet spot (16 slightly best at high
forward drive); rec_count 32 goes super-critical and collapses toward chance (radius trades against count:
wider radius lowers local density and partly rescues a high count). **Depth hurts** — every deeper (6-layer)
side-car variant underperformed the compact one (DFA credit noise + propagation lag + starved deep read
layers), across couplings, an L3 self-loop, and forward/recurrence density tuning.

**Direction:** harder / deeper recurrent computation appears to need **more width + more forward drive
together**, and the **forward topology** (skip distances, where the side-car couples) is a large untested
axis. **Immediate blocker → next step: engine performance optimization** (the single-threaded integer engine
can't run these sweeps multi-seed at size ≥ 64), then resume the systematic scaling / forward-topology study.

## Criticality init — homeostatic weight-training to σ≈1 (the calibration replacement; FF-validated)

**Motivation.** Firing-rate `calibrate` is a brittle *init*: it tunes per-layer **baselines**, but a
threshold only *gates* a fixed projection — it can't manufacture drive that never arrives, so on a
sub-critical (cue-dies-with-depth) stack, lowering a starved layer's threshold to 1 still can't revive it.
And rate is only a **loose proxy for σ** (branching ratio), the real order parameter. Replaced by a
**homeostatic weight-training init** (`bench::critical_init`) — e-prop-style, layer-wise greedy bottom-up —
that shapes the stored *weights* (the gain), not the thresholds.

**σ diagnostic.** `forward_avalanche` — per-hop single-spike **damage spreading**, `footprint[z+1]/
footprint[z]` = the forward branching ratio σ_hop (a *burst* of injected spikes cuts the ratio noise). The
whole-network `sigma_probe` **accumulates across layers** and mis-reads FF criticality (reads σ>1
everywhere from cross-layer accumulation); the per-hop measure is the right one for a feed-forward stack.

**Findings (32×32×5 FF, single-entry level+1, `adapt_bump 5`; multi-seed where noted).**

1. **Weight-training revives what calibration can't.** A rate-homeostatic e-prop init (`rate_reg_init`,
   `Δw ∝ −(r_j − r_target)·pre_i·ψ_j`, greedy) revives a sub-critical stack to σ≈1 where calibration dies
   with depth (up_count 16: forward avalanche *maintained* `0.6→0.5` vs calibration's `dies to 0`).
   Confirms the pivot mechanism — **weights set gain/σ; thresholds only gate.** ALIF is load-bearing (LIF →
   dead: no near-threshold ψ → no gradient). ψ is load-bearing *for the rate rule* (dropping it breaks
   low-density revival).

2. **σ is a density property; ALIF masks super-criticality.** σ is set by fan-out/radius; there is a
   **critical density** (~up_count 16 at radius 3 *with trained weights* — lower than the ±1 forward-drive
   threshold ~32, because trained gain reaches σ=1 at lower fan-out). Below → sub-critical; above →
   super-critical. Crucially, **ALIF holds the rate at target while σ runs >1** — a rate-stable net can be
   super-critical, and the rate proxy hides it (LIF at high density goes overtly super-critical instead).

3. **rate ≠ σ off the critical density.** Rate-targeting → *flat rate* but *super-critical* σ at high
   density; σ-targeting → *σ≈1* but a *decaying rate*. They **coincide only at the critical density**
   (there the simple rate init gives both flat rate and σ≈1 for free).

4. **Rate-free σ-init.** `sigma_eprop_init` drives σ_hop→1 with a per-synapse update `Δw ∝ −(σ_hop−1)·pre_i`
   (rate emergent, *no set-point*). A uniform gain-*scaling* controller fails at high density — int8
   weights are all-or-nothing near the quantization floor → oscillation between super-critical and dead;
   the per-synapse update's `pre_i` heterogeneity thins weights smoothly and fixes it. The f32 latent
   shadow crossing zero makes **sign-flip to inhibition** available (unused here — sparsifying excitation
   reaches σ≤1 first — but it is the lever for the planned BitNet **ternary** path, where magnitude is fixed).

5. **σ≈1 is the robust computational target (multi-seed × two tasks).** Intrinsic top-layer quality
   (held-out nearest-centroid accuracy + effective-dim), 5 seeds. On the **linear pattern** task, σ-eprop
   wins at *every* density and the flat-rate init **craters toward chance where it goes super-critical**
   (uc24: flat 334‰ vs σ 798‰; chance 250‰) — a rate-stable super-critical net is *computationally dead*
   for linear readout. On the **nonlinear spatial-XOR** task both tie well above chance (super-critical
   chaos *helps* nonlinear mixing). Textbook edge-of-chaos: **σ≈1 maximizes linear separation; super-critical
   favors nonlinear mixing; σ≈1 is the safe general target.**

**End-to-end FF training gate (2026-07-11) — σ≈1 does NOT cleanly beat calibration on *trained readout*
accuracy.** Deep (5-layer), width-32 FF, full multi-layer e-prop + trained readout, `critical_init` vs
the calibration fallback, 3 seeds (chance 500‰):

| up_count | calibration | critical_init |
|---|---|---|
| 8  | 830‰ | **1000‰** |
| 12 | **1000‰** | 820‰ |
| 16 | 1000‰ | 1000‰ |
| 32 | 1000‰ | 982‰ |

Two lessons: (1) **training compensates for a brittle init** — `rate_reg`/e-prop revives a
calibration-starved deep stack, so both hit ceiling at up_count ≥ 16 (the init barely matters once you
train). (2) **σ≈1 is not the right target for a *readout*-based objective** — its emergent *rate decays
with depth*, so the top layer can be too **sparse** to read, and `critical_init` *regresses* at up_count 12
(820 vs 1000). It wins only where calibration is genuinely un-revivable (up_count 8). **Consequence: the FF default was
NOT flipped to `critical_init`** — it stays an available init (a decisive revive for un-trainable
sub-critical stacks), calibration stays the default.

**Follow-up (2026-07-11) — the regression is NOT a sparse-top problem; density is not the lever.** The
initial diagnosis ("σ≈1's decaying rate leaves the top too sparse for the readout") was **wrong**. The
pre-training per-layer rate profiles (5-layer, width 32) show σ-init actually keeps the **densest, most
alive top** of the σ/calibration pair (uc32 σ = `[30.6, 10.0, 4.0, 3.9, 4.4]` vs calibration
`[30.6, 6.0, 2.0, 0.6, 0.4]`, whose top is *dead* pre-training and only revived by training's `rate_reg`).
To test density directly, a third init — `Network::rate_match_init` (drive each layer's rate to the layer
below, `rate[z]→rate[z-1]`, via the same e-prop weight update; converged at rounds 100 / lr 0.15) —
produces a **flat, dense, alive** profile everywhere (uc8 top **11%**, uc12 **15%**, uc32 **16%**), by far
the densest of the three. Its trained accuracy: `[819, 832, 988, 992]‰` over uc `[8,12,16,32]` — the
**worst or tied-worst almost everywhere**. Decisive point: at uc8 the *densest* init (rate-match, top 11%)
trains to **819**, while the *sparsest* (σ-init, top 1%) trains to **1000**. So (1) the init's static rate
profile does **not** predict trained accuracy, and (2) **denser is worse**, not better. The real substrate
quality is **σ≈1 = optimal information propagation at the edge of chaos**, *not* firing rate: where the
substrate is the bottleneck (uc8) σ=1 preserves the signal while an over-driven dense regime saturates it.
Net: `rate_match_init` is a **confirmed dead end** for trained-readout accuracy; calibration stays the FF
default, `critical_init` (σ≈1) stays the rescue init for un-revivable starved stacks.

**Is calibration itself redundant with training? — No (`None`-init test, 2026-07-11).** A fourth init,
`FfInit::None` (train from *raw* weights, no init pass, `rate_reg` only), added as a column to the gate:

| up_count | none | calibration | σ-init | rate-match+ |
|---|---|---|---|---|
| 8  | 498 (chance) | 830 | **1000** | 819 |
| 12 | 498 (chance) | **1000** | 820 | 832 |
| 16 | 1000 | 1000 | 1000 | 988 |
| 32 | 1000 | 1000 | 982 | 992 |

`None` sits at **chance** at uc8/uc12 and only reaches 1000 at uc16/uc32. So training's `rate_reg` *cannot*
bootstrap a raw stack at marginal fan-in — it needs calibration's threshold pre-pass to reach a trainable
operating point first. This gives a clean **three-regime picture by substrate richness**: (1) **generous
fan-in (uc16, 32)** — init is irrelevant, even `None` trains to 1000, calibration *is* redundant here; (2)
**marginal fan-in (uc12)** — calibration is *decisive* (chance→perfect), `rate_reg` alone can't bootstrap
raw weights; (3) **severe starvation (uc8)** — even calibration limps (830), σ≈1 is the unique rescue
(1000). A standing puzzle: at uc12 the *more-alive* inits (σ-init, rate-match) train *worse* than
calibration's dead-top init — "more firing at init" is not the good; calibration lands a specific operating
point ideal for `rate_reg` training that the weight-only inits miss (cf. Pennington: forward liveness ≠ a
trainable basin). Literature frame: untrained reservoirs (ESN/LSM) need edge-of-chaos init because the
hidden weights are never trained (our uc8); trained RSNNs (Bellec e-prop/LSNN) rely on firing-rate
regularization and treat init as a warm-start (our uc16/32); deep-FF signal-propagation theory
(Poole/Schoenholz, Pennington) says criticality enables depth but normalization can substitute — all three
consistent with the regimes above.

**Status.** FF-validated; **`sigma_eprop_init` (rate-free, σ≈1) is the keeper** and the intended calibration
replacement. Recurrent criticality (the side-car *loop gain* — a separate quantity `sigma_probe(Some(z))`
measures) is **not yet handled**: the greedy-FF structure doesn't map to the cyclic (L2↔L3) topology. That
extension + validation on the recurrent benchmarks is required before it replaces calibration for recurrent
configs; calibration is retained (downgraded to a bench tool) as the fallback until then.
