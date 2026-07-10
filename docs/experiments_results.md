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

## Recurrence — trainable in the deep + wide + sub-critical regime

Trained recurrence earns its keep **only** in a specific regime, reached by completing e-prop's ALIF credit
rule. Every earlier recurrence test used the **crude spike-timing** eligibility (only the fast membrane term
`e = Σ_t ψ_j·εᵛ_i`). Textbook e-prop for ALIF neurons (Bellec 2020, Eq. 24–25) adds a **slow** adaptation
eligibility `εᵃ`, recursed at the adaptation rate `ρ`: `e = ψ·(εᵛ − β·εᵃ)`. That term carries credit over the
~64-wave adaptation horizon (the substrate's delay memory) and was never implemented. It is now built and
verified against the paper + reference implementation (`RsnnConfig.elig_beta` / `elig_bump_psi`; a decide-time
`eff` snapshot `Layer.decide_eff`; a fixed-width bump ψ, half-width `elig_psi_width`). Two ψ bugs fixed on the
way: `eff` must be read at the decide step (before the fire-bump), and the bump needs a fixed absolute band
(the θ≈1 baseline collapses ψ to ~0, since integer ±1 drive overshoots `eff` by O(2–26)).

**Three levers, all necessary** (deep 4-layer hidden-recurrent stack `train_hidden_rec_task`: recurrence on
the second-from-top layer, *every* forward layer trained via DFA + `rate_reg` liveness; temporal XOR, delay
20, worst-seed / 3 seeds):

- **Depth + trained forward layers** — a shallow recurrent-top net, or untrained forward layers, don't get there.
- **Width** — rec_count 8, FF vs trained recurrence: **990 / 725** (size 16) → **1000 / 982** (size 32).
- **Sub-critical recurrence density** — sharp collapse cliff at **rec_count ≈ 12** (size 16); above it the
  ±1 loop is super-critical and drowns the read layer to chance regardless of the credit rule:

  | rec_count (size 16) | crude ψ | completed εᵃ |
  |---|---|---|
  | 4 | 707 | **855** |
  | 8 | 672 | **725** |
  | ≥ 12 | 475 | 475 (collapse) |

The **completed eligibility consistently beats the crude rule** where recurrence is trainable, most where
credit is hardest (light density / small width): rc4 855 vs 707, size 32 982 vs 975.

**It beats feed-forward where the task has headroom.** Temporal XOR is FF-saturated (ALIF-FF ≈ 1000) so
recurrence can only tie. On **parity N=4** (non-monotone, FF ≈ 620) trained recurrence + completed eligibility
(size 32, depth 4, rec_count 8) **beats FF on 2 of 3 seeds** (687 vs 680, 642 vs 637; worst-seed 560 vs 620,
held by the hardest seed) and beats crude on all three — the first substrate result where trained recurrence
out-performs feed-forward.

**Levers to push it to a robust win (open):** more depth (a 2nd recurrent layer), more width + rec_count, the
backward side-car topology, and β·`elig_psi_width` tuning — the hardest seed is the remaining gap. Tools:
`train_hidden_rec_task` / `sequence_trial_layers` (task-parameterized deep trainer, configurable
`hidden_rec_depth`), `elig_beta` / `elig_bump_psi` / `elig_psi_width`, `train_sidecar_task`.
