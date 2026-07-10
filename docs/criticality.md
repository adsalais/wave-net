# Criticality — the order parameter behind the scaling relationships

**Purpose.** The scaling relationships the side-car study is uncovering (a forward-drive threshold, a
recurrence-density collapse cliff, a width capacity floor, joint width×forward scaling, depth limits) are not
five separate mysteries — the literature frames most of them through one measurable order parameter:
**network criticality**. This note records that framing, the references, and — importantly — how a criticality
controller would interact with the rate-equilibration we already do during training.

## The order parameter: branching ratio σ (≈ spectral radius)

Recurrent / spiking networks have an order parameter — the **branching ratio σ** (mean number of descendant
spikes per spike), equivalently the **spectral radius of the recurrent Jacobian**:

- **σ < 1 — sub-critical:** activity dies out → the readout sees nothing → **chance**.
- **σ ≈ 1 — critical (edge of chaos):** maximal dynamic range, memory, and linear separation — the
  computational sweet spot.
- **σ > 1 — super-critical:** one spike triggers many → runaway → the class signal is drowned → **chance**.

σ is *measurable* (perturbation / avalanche statistics), and it is **not** the firing rate — see the caveat
section. Firing-rate calibration is only a loose *proxy* for σ≈1 (already noted in `related-work.md`: "rate is
easy to measure, but branching ratio / σ is the quantity that actually governs memory and computation").

## Mapping to the side-car scaling findings

| our finding (`experiments_results.md` scaling study) | criticality reading |
|---|---|
| **forward-drive threshold** (below up_count ≈ 32 → chance) | sub-critical: too little drive/gain, σ < 1, activity dies |
| **recurrence-density collapse** (rec_count ≳ 12–32 → chance) | super-critical: too much recurrent gain, σ > 1, runaway |
| **the working band in between** (~700–800) | operating near σ ≈ 1 |
| **width capacity floor** (size 16 can't; ≥ 32 needed) | separation capacity — need enough dimension (Cover) |
| **joint width × forward scaling** | compute-optimal balanced-scaling frontier |
| **depth hurts** | finite signal-propagation depth off the critical line |

So the forward-drive threshold and the recurrence-density collapse are the **two sides of the same σ = 1
transition** — one approached from below (starved), one from above (runaway).

## References

- **Criticality / branching in spiking nets:** [A mean-field approach to criticality in SNNs for reservoir
  computing](https://www.nature.com/articles/s41598-025-18004-y) (Sci Rep 2025); [Control of criticality and
  computation in spiking neuromorphic networks with plasticity](https://www.nature.com/articles/s41467-020-16548-3)
  (Nat Comms 2020) — σ ≈ 1 optimizes information transmission while preventing runaway excitation.
- **Width / separation capacity:** Cover's theorem (P patterns separable in D dims w.h.p. if P < 2D);
  [Separation capacity of linear reservoirs with random connectivity](https://arxiv.org/abs/2404.17429) (2024)
  — separation scales with reservoir dimension N (entries ~ρ/√N — again the √N gain scaling of criticality).
- **Joint scaling:** neural scaling laws (Kaplan et al.; Hoffmann et al. "Chinchilla" — scale model *and* data
  together); [4+3 Phases of Compute-Optimal Neural Scaling Laws](https://arxiv.org/pdf/2405.15074) — the
  phase-plane of which resource binds.
- **Depth / signal propagation:** mean-field theory of trainability (Poole/Schoenholz/Pennington;
  [Xiao et al. 2018, training 10,000-layer nets](https://arxiv.org/abs/1806.05393)) — trainable *iff* signal
  reaches the top, which requires sitting on the critical line; [Dynamical Isometry and a Mean Field Theory of
  RNNs: Gating Enables Signal Propagation](https://arxiv.org/abs/1806.05394) — architectural structure
  (gating/isolation) is what lets signal propagate through recurrence (the propagation-theory view of our
  "isolate the loop" side-car result).

## The highest-leverage move: measure σ as a diagnostic

We currently measure only **firing rate** (`Network::measure_layer_rates`) — not σ. A branching/avalanche
probe is **not implemented** (the "probe_avalanche" mentioned in old doc history was an idea, never built; the
one criticality controller that *was* tried used an analytic `Σ|w|/θ` estimator that proved **invalid** — it
read "branching 16" for layers that never fire, blind to actual propagation, and was abandoned).

**A valid σ estimator is a perturbation probe:** flip one extra spike, measure how the population response
diverges over the next few waves (or fit the avalanche-size power law). This would let us *predict* the
drive/collapse transitions instead of grid-searching them — the biggest single win for the scaling study.

## ⚠️ Interaction with our existing rate-equilibration (does criticality conflict?)

We already equilibrate activity three ways; a σ-controller would coexist with all three:

1. **ALIF adaptation** (runtime, per-neuron): threshold rises with firing and quenches it — a local
   negative-feedback **rate** controller, every wave. Load-bearing (removing it kills the deep-FF stack).
2. **`rate_reg`** (training-time, per-neuron, forward path): `c_reg·(r_j − r_target)` folded into the e-prop
   learning signal → pulls incoming weights so each neuron approaches the target rate.
3. **`rec_stab`** (training-time, per-layer, recurrent levels): a uniform, class-preserving bias toward
   r_target (unlike per-neuron `rate_reg`, which homogenizes and *hurts* recurrence).

**The key fact: firing rate ≠ branching ratio.** Rate is a 1-point statistic (how often a neuron fires); σ is
a gain / 2-point statistic (how much one spike propagates). A network can sit at the same 10% rate with σ < 1,
σ = 1, or σ > 1. Consequences:

- **Not redundant, so not directly in conflict.** Rate-control alone does *not* put you at σ ≈ 1 (it's only a
  proxy). A σ-controller adds information the rate controllers don't have, and targets a different variable
  (recurrent *gain*), so it isn't just re-doing `rate_reg`.
- **But they couple.** All of these change network activity, so stacking feedback controllers on one system
  risks **oscillation / fighting** unless their timescales are separated (homeostasis ≪ σ-control ≪ wave).
  This is exactly the risk `related-work.md` "Design notes A" already flagged for online homeostasis.
- **ALIF is *already* a partial σ-controller.** Adaptation raises thresholds → less propagation → lower
  effective σ. So an explicit weight-based σ-controller would **overlap** with ALIF's gain role: the two can
  chase each other (σ-controller raises recurrent weights to reach σ = 1 → ALIF raises thresholds to quench
  the extra firing → σ drops → repeat). Watch this specifically.
- **`rate_reg`-on-recurrence was found to hurt** (it homogenizes the class signal). A σ-controller on the
  recurrent **gain** is plausibly the *right* recurrence lever — it targets the variable `rate_reg` was
  clumsily proxying, so it could **replace** `rate_reg`/`rec_stab` on the recurrence rather than conflict.

**Safe design.**
1. **First, measure σ as a pure diagnostic** — no controller, zero conflict. It tells us where the current
   ALIF + `rate_reg` operating point sits relative to σ = 1, and should explain the drive threshold and the
   collapse cliff directly.
2. **If we add a σ-controller:** put it on the recurrent **weights/gain**, on a **slow** timescale, and keep
   ALIF + `rate_reg` for per-neuron **rate**. Do **not** stack a σ-controller and `rate_reg`/`rec_stab` on the
   same recurrent weights, and separate its timescale from ALIF's (which already moves σ via thresholds).

**Bottom line:** no hard conflict — σ and firing rate are different quantities — but there is real coupling,
and ALIF's implicit gain control is the one to watch. Start with σ *measurement* (safe, high-value); only then
consider a slow, weight-targeted σ *controller* on the recurrence.
