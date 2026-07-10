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
