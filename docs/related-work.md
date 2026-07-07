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

## Online, constant-memory training over long windows (FPTT / HYPR family)

The methods below all solve the same problem the multi-wave rule creates: learning over a long
temporal window in a recurrent spiking net **without storing the window**. Two facts about wave_net
filter what's borrowable:

- **Weights are fixed ±1 procedural** → per-*synapse* rules don't apply; the only trainable state is
  **per-neuron** (threshold, maybe leak/adaptation). Point these methods' machinery at per-neuron
  parameters, not synapses.
- **Integer hard threshold → no gradients.** Gradient-based methods (FPTT, HYPR, StochEP) need a
  **differentiable shadow twin**; the gradient-free three-factor form (eligibility × reward) needs
  no shadow and fits the FPGA-clean integer substrate — that's the natural path.

Ranked by value to this engine:

1. **Adaptive threshold (ALIF) as the trainable slow state** — highest leverage. Make the per-neuron
   threshold a slow dynamic variable (rises on fire, decays at rest) instead of a static calibrated
   value. Gives long-range memory, is per-neuron (fits the constraint), and is exactly the variable
   e-prop/FPTT know how to train (e-prop's ALIF eligibility has a threshold-adaptation component).
   Cheap: one extra per-neuron slow variable + decay. Bridges static-threshold *calibration* to
   dynamic-threshold *learning*.
2. **Constant-memory online learning** — don't store the multi-wave window; carry a constant-size
   **eligibility trace** forward and update online. Borrow (a) e-prop's threshold eligibility trace
   as the credit-assignment mechanism (gradient-free when the top-down signal is a reward =
   three-factor), and (b) FPTT's **dynamic regularizer** (couple each online update to a slowly
   moving reference parameter) as the stabilizer — transplantable even into a reward-modulated rule.
3. **Locality ladder for on-substrate learning** — **ETLP** (event-based three-factor local
   plasticity) is e-prop made hardware-local; the target shape for a rule that runs inside the
   substrate rather than an offline trainer.
4. **Liquid / heterogeneous time constants** — FPTT matched BPTT paired with a liquid (adaptive)
   time-constant neuron. Making `leak` per-neuron / heterogeneous / adaptive is another cheap
   per-neuron parameter buying multi-timescale memory (adaptive-LIF / resonate-and-fire family).
5. **HYPR parallelization** — file for later: forward-mode learning parallelizes, and the deferred
   one-hop model is already layer-parallelizable (read last wave's inbox, write next wave's outbox),
   so this lands naturally at scale-up.
6. **StochEP / Equilibrium Propagation** — a stretch: EP needs the net to relax to an equilibrium
   (convergent RNN, free/nudged phases); a driven wave engine only fits if a settling regime is
   carved out. Lower priority.

**Suggested path:** per-neuron threshold as an adaptive (ALIF) state, trained by an e-prop-style
eligibility trace under a reward/three-factor signal (gradient-free, no shadow), regularized
FPTT-style toward a running reference — with a surrogate-gradient shadow as a ceiling/benchmark and
ETLP as the on-chip-local target.

## Sources

Format: *Title* (tag) — Venue Year — link(s).

- Knight & Nowotny, *Larger GPU-accelerated brain simulations with procedural connectivity* — Nature Computational Science 2021 — [nature](https://www.nature.com/articles/s43588-020-00022-7)
- *Reservoir Computing: Foundations, Advances, and Challenges Toward Neuromorphic Intelligence* (review) — MDPI AI 2026 — [mdpi](https://www.mdpi.com/2673-2688/7/2/70)
- *Reinforcement Learning with Low-Complexity Liquid State Machines* — Frontiers in Neuroscience 2019 — [pmc](https://pmc.ncbi.nlm.nih.gov/articles/PMC6718696/)
- *Biologically Inspired Dynamic Thresholds for Spiking Neural Networks* — arXiv 2022 — [arxiv](https://arxiv.org/pdf/2206.04426)
- *Information-Theoretic Intrinsic Plasticity for Online Unsupervised Learning in SNNs* — Frontiers in Neuroscience 2019 — [frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00031/full)
- *Implementing Spiking Neural Networks on Neuromorphic Architectures: A Review* (TrueNorth/Loihi) — arXiv 2022 — [arxiv](https://arxiv.org/pdf/2202.08897)
- *Integer-State Dynamics of Quantized Spiking Neural Networks* — arXiv 2026 — [arxiv](https://arxiv.org/pdf/2604.01042)
- Bellec et al., *A solution to the learning dilemma for recurrent networks of spiking neurons* (e-prop) — Nature Communications 2020 — [nature](https://www.nature.com/articles/s41467-020-17236-y) · [pmc](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/)
- Yin, Corradi & Bohté, *Accurate online training of dynamical SNNs through Forward Propagation Through Time* (FPTT) — Nature Machine Intelligence 2023 — [nature](https://www.nature.com/articles/s42256-023-00650-4) · [arxiv](https://arxiv.org/abs/2112.11231) — origin: Kag & Saligrama, ICML 2021
- Baronig et al., *A scalable hybrid training approach for recurrent SNNs* (HYPR) — Neuromorphic Computing and Engineering 2026 — [doi](https://doi.org/10.1088/2634-4386/ae46d4) · [arxiv](https://arxiv.org/abs/2506.14464)
- *ETLP: event-based three-factor local plasticity for online learning with neuromorphic hardware* — Neuromorphic Computing and Engineering 2024
- *Stochastic Equilibrium Propagation for Spiking Convergent Recurrent Neural Networks* (StochEP) — arXiv 2025 — [arxiv](https://arxiv.org/abs/2511.11320)
- Bellec et al., *Long short-term memory and learning-to-learn in networks of spiking neurons* (LSNN / adaptive-LIF) — NeurIPS 2018 — [arxiv](https://arxiv.org/abs/1803.09574)
