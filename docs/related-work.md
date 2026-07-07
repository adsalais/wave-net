# Related work — where wave_net sits, and what to borrow

**Date:** 2026-07-08
**Purpose:** `wave_net` is a novel *combination*, but every major ingredient maps to an established
line of work. This note records the closest matches and the concrete idea to steal from each — a
starting point for designing the learning layer.

## What wave_net is, in one line

A **procedural-connectivity liquid state machine with intrinsic-plasticity threshold learning, in a
TrueNorth-style integer substrate**, laid out on a 2D grid with deferred wave propagation. Each
quadrant below is real and studied; the specific fusion (and the grid/wave deferred-propagation
topology) is the part that looks like our own contribution.

## 1. Hash-generated synapses → "procedural connectivity" (closest match)

Knight & Nowotny, *Larger GPU-accelerated brain simulations with procedural connectivity* (GeNN,
Nature Computational Science 2021) do exactly our "synapses are never stored — regenerated on the
fly": connectivity + weights are generated on demand from a **per-neuron-seeded RNG** (a PRNG is a
hash) whenever a spike must be transmitted, storing zero connectivity. It enabled 4.1M neurons /
24B synapses on one GPU. Our `synapse::key`/`mix` → `generate_into` is the same trick.

- **Steal:** the determinism discipline and the memory-vs-compute tradeoff argument — the published
  justification that our storage-free, FPGA-clean approach scales. Also worth checking how they
  keep per-neuron RNG streams independent and reproducible.

## 2. Fixed substrate + train per-neuron params → Liquid State Machine / spiking reservoir computing

Our "fixed procedurally-wired dynamical substrate, then learn on top" *is* an LSM (Maass): a random
recurrent spiking reservoir with fixed synapses + a trained readout. Large literature on
*conditioning* the reservoir.

- **Steal:** the **edge-of-chaos / criticality** result — reservoirs compute best when the
  branching ratio ≈ 1. That is the deeper target our firing-rate calibration is a proxy for: rate
  is easy to measure, but **branching ratio / spatial-temporal σ** is the quantity that actually
  governs memory and computation. Calibration direction is validated; reservoir theory says *where*
  the sweet spot is. (This is what the earlier, now-deleted criticality/homeostasis idea was
  reaching for — revisit it when conditioning matters.)

## 3. Training per-neuron thresholds → intrinsic / homeostatic plasticity + adaptive-threshold neurons

Our `Layer::calibrate_step` — "boost excitability if it rarely fires, depress if it fires too much"
— is textbook **intrinsic plasticity / homeostasis**. Modern RSNNs make the firing **threshold** an
adaptive per-neuron state variable (**ALIF** — adaptive leaky integrate-and-fire), which is exactly
our chosen trainable parameter.

- **Steal:** intrinsic-plasticity rules are a proven, gradient-free way to train the per-neuron
  threshold; ALIF shows adaptive thresholds + recurrence is what makes RSNNs expressive. The common
  "dynamic threshold" rule (raise on spike, decay at rest) is a ready-made homeostatic primitive.

## 4. Integer, ±1 weights, deterministic → neuromorphic hardware

The engine is shaped like a neuromorphic chip. **IBM TrueNorth** is the closest sibling: strictly
digital, **deterministic integer neurons, binary synaptic weights** — nearly our ±1 / i16 /
deterministic design. **Intel Loihi 2** is fixed-point (8-bit weights, 24-bit membrane) with
on-chip learning.

- **Steal:** TrueNorth confirms deterministic-integer-binary-weight SNNs are a real, effective
  design point, not a compromise. Loihi's fixed-point layout is a reference for how many bits to
  budget for thresholds vs. membrane potential if precision ever bites.

## Training the learning layer (next phase) — and the multi-wave rule

**e-prop** (Bellec et al. 2020, *A solution to the learning dilemma for recurrent networks of
spiking neurons*) is the key RSNN training rule, and two things align eerily with our design:

- It solves **temporal credit assignment** with **eligibility traces**: because a recurrent net's
  response is spread over time, learning signals must integrate over a window. That is our
  "**a read requires several waves; single-wave training is an error**" rule (AGENTS.md), formalized.
- In their words, e-prop's gradient information "flows through slow hidden variables **like firing
  thresholds**" — the trainable slow variable they lean on is exactly the one we chose.

So the multi-wave rule is not just correct — it is the principle e-prop is built on. When we get to
training, the menu is:

1. **Intrinsic plasticity / homeostasis** — unsupervised conditioning (what calibration already is).
2. **Reward-modulated plasticity / node perturbation** — gradient-free, integer-friendly task
   learning over a multi-wave window (no differentiable shadow needed).
3. **e-prop-style eligibility traces** — if we want a per-neuron-threshold learning rule with proper
   temporal credit assignment; needs a surrogate-gradient shadow since the integer engine is
   non-differentiable.

Any of these must operate on a **multi-wave window**, per the engine's deferred + recurrent dynamics.

## Sources

- [GeNN procedural connectivity (Nature Comp. Sci. 2021)](https://www.nature.com/articles/s43588-020-00022-7)
- [Reservoir Computing: Foundations, Advances, Toward Neuromorphic Intelligence (review)](https://www.mdpi.com/2673-2688/7/2/70)
- [Reinforcement Learning with Low-Complexity Liquid State Machines](https://pmc.ncbi.nlm.nih.gov/articles/PMC6718696/)
- [Biologically Inspired Dynamic Thresholds for SNNs](https://arxiv.org/pdf/2206.04426)
- [Information-Theoretic Intrinsic Plasticity for Online Unsupervised Learning in SNNs](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00031/full)
- [Implementing SNNs on Neuromorphic Architectures: A Review (TrueNorth/Loihi)](https://arxiv.org/pdf/2202.08897)
- [Integer-State Dynamics of Quantized SNNs](https://arxiv.org/pdf/2604.01042)
- [A solution to the learning dilemma for recurrent SNNs (e-prop, Bellec et al.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/)
