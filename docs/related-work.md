# Related work — where wave_bitnet sits, and what to borrow

**Date:** 2026-07-08
**Purpose:** `wave_bitnet` is a novel *combination*, but every major ingredient maps to an established
line of work. This note records the closest matches and the concrete idea to borrow from each — the
literature backing for the engine's design and its learning layer.

## What wave_bitnet is, in one line

A **ternary-native integer spiking network with ALIF adaptive-threshold neurons, trained by e-prop /
multi-layer temporal-DFA, in a TrueNorth-style integer substrate** — laid out on a 2D grid with
deferred one-hop wave propagation, its topology materialized once into per-neuron occupancy bitsets and
its plastic weights stored as 2-bit packed ±1/0 codes plus an f32 training shadow. Each quadrant below is
real and studied; the specific fusion (and the grid/wave deferred-propagation topology) is the part that
looks like our own contribution.

## 1. Materialized procedural topology → "procedural connectivity" (closest match)

Knight & Nowotny, *Larger GPU-accelerated brain simulations with procedural connectivity* (GeNN,
Nature Computational Science 2021) do the "synapses need never be stored — regenerate on the fly":
connectivity + weights are generated on demand from a **per-neuron-seeded RNG** (a PRNG is a hash)
whenever a spike must be transmitted, storing zero connectivity. It enabled 4.1M neurons / 24B synapses
on one GPU. `wave_bitnet` uses the same address trick but resolves the memory-vs-compute tradeoff the
other way: it **samples each neuron's neighborhood once at construction and materializes it into an
occupancy bitset** (one bit per `(2r+1)²` cell), then iterates the set bits at fire time — no per-wave
hashing. The address scheme stays procedural (a pure function of `(seed, config)`); only the resolved
bitset is kept.

- **Steal:** the determinism discipline and the memory-vs-compute tradeoff argument. Because the
  addresses are procedural, materializing them costs almost nothing (~0.2 B/synapse) and buys back the
  per-wave hash cost — the deliberate inverse of GeNN's never-store choice, correct at our scale. Worth
  matching how they keep per-neuron RNG streams independent and reproducible.

**Two points from a full read (2026-07-08), now the settled design rationale:**

1. **RNG quality is load-bearing; GeNN uses Philox4×32-10** (a *counter-based* cryptographic PRNG,
   Salmon et al. 2011) precisely to get independent, correlation-free streams per neuron. The
   neighborhood sampler must be counter-based / crypto-quality so the sampled per-neuron streams stay
   correlation-free — **hold to a BLAKE3-quality mixer** (the `strong_hash` control is the standard to
   match). This is a real defect the paper's design explicitly avoids.
2. **Procedural = *static* connectivity only; plastic weights must be stored.** Verbatim: *"applicable
   whenever synapses are static – plastic synapses which change their weights during a simulation will
   have to be simulated in the traditional way."* GeNN **never learns procedural weights**; plastic
   weights are **stored**. That is exactly `wave_bitnet`'s split: the synapse **addresses** are
   procedural (materialized once, free), and the **plastic weights** are stored — as 2-bit packed ±1/0
   codes plus an f32 training shadow (quantized through a prune threshold). Addresses cost storage once;
   the trainable part lives where it must. The GeNN payoff (pure procedural) is memory at *scale*
   (24×10⁹ synapses); at `wave_bitnet`'s ~10³–10⁵ synapses, storing the plastic part costs nothing —
   which is precisely why it stores the ternary codes outright and keeps only the free part procedural.
   The sanctioned architecture is this **hybrid: procedural static structure + a small *stored* plastic
   part**, and it is the one `wave_bitnet` implements.

## 2. Fixed substrate + trained readout → Liquid State Machine / spiking reservoir computing

The grid of adaptive integer neurons *is* a spiking reservoir (Maass LSM): a recurrent spiking
substrate driven by input, read (and here also trained) on top. A trained readout on the full reservoir
(classic LSM) is a reliable baseline; the engine additionally trains the reservoir weights themselves.
Large literature on *conditioning* the reservoir.

- **Steal:** the **edge-of-chaos / criticality** result — reservoirs compute best when the branching
  ratio ≈ 1. That is the deeper operating-point target. `rate_reg` keeps the substrate live, but the
  quantity that actually governs memory and computation is **branching ratio / spatial-temporal σ**,
  measured on the *task* drive — reservoir theory says *where* the sweet spot is.

## 3. Adaptive-threshold neurons → intrinsic / homeostatic plasticity + ALIF

The engine's per-neuron **adaptive threshold** — the ALIF adaptation state that rises when a neuron
fires and decays at rest, throttling its own excitability — is the neuromorphic form of textbook
**intrinsic plasticity / homeostasis** ("depress if it fires too much, boost if it rarely fires").
Modern RSNNs make the firing **threshold** an adaptive per-neuron state variable (**ALIF** — adaptive
leaky integrate-and-fire), which is exactly `wave_bitnet`'s delay-memory mechanism; the soft `rate_reg`
liveness term is its co-trained homeostatic companion.

- **Borrowed context / steal:** intrinsic-plasticity rules are the literature frame for the adaptive
  threshold and `rate_reg`; ALIF shows adaptive thresholds + recurrence are what make RSNNs expressive.
  The common "dynamic threshold" rule (raise on spike, decay at rest) is precisely the ALIF adaptation
  primitive the engine runs.

## 4. Integer, ternary weights, deterministic → neuromorphic hardware

The engine is shaped like a neuromorphic chip. **IBM TrueNorth** is the closest sibling: strictly
digital, **deterministic integer neurons, binary synaptic weights** — nearly `wave_bitnet`'s 2-bit ±1/0
codes / i16 potential / deterministic design. **Intel Loihi 2** is fixed-point (8-bit weights, 24-bit
membrane) with on-chip learning.

- **Steal:** TrueNorth confirms deterministic-integer, low-bit-weight SNNs are a real, effective design
  point, not a compromise. Loihi's fixed-point layout is a reference for how many bits to budget for
  thresholds vs. membrane potential if precision ever bites.

## The training rule — e-prop, eligibility traces, and the multi-wave rule

**e-prop** (Bellec et al. 2020, *A solution to the learning dilemma for recurrent networks of spiking
neurons*) is the key RSNN training rule, and two things align eerily with `wave_bitnet`'s design:

- It solves **temporal credit assignment** with **eligibility traces**: because a recurrent net's
  response is spread over time, learning signals must integrate over a window. That is the
  "**a read requires several waves; single-wave training is an error**" rule (AGENTS.md), formalized.
- In their words, e-prop's gradient information "flows through slow hidden variables **like firing
  thresholds**" — and the engine's ALIF adaptation is exactly that slow hidden state, carried in the
  eligibility as the `εᵃ` term (`e = ψ·(εᵛ − β·εᵃ)`, Bellec 2020). The trained parameter is the stored
  ternary weight, moved through its f32 shadow.

So the multi-wave rule is not just correct — it is the principle e-prop is built on. The engine's
realized rule sits in the gradient-based branch of the family:

1. **Intrinsic plasticity / homeostasis** — the unsupervised-conditioning role, now filled by the soft
   `rate_reg` liveness term.
2. **Reward-modulated plasticity / node perturbation** — gradient-free, integer-friendly task learning
   over a multi-wave window (no differentiable shadow needed).
3. **e-prop-style eligibility traces** — a factored per-neuron eligibility (`e = pre-trace × ψ`, both
   O(neurons) engine state) times a top-down learning signal, updating the stored weights through the
   `f32` shadow. **This is the path `wave_bitnet` took**; the `f32` shadow is the differentiable twin the
   non-differentiable integer engine needs.

Any such rule operates on a **multi-wave window**, per the engine's deferred + recurrent dynamics.

## Online, constant-memory training over long windows (FPTT / HYPR family)

The methods below all solve the same problem the multi-wave rule creates: learning over a long temporal
window in a recurrent spiking net **without storing the window**. Two facts about `wave_bitnet` filter
what's borrowable:

- **Weights are stored and trainable** (2-bit ternary codes + per-synapse `f32` shadow) → per-*synapse*
  rules apply directly; point their machinery at the stored weights. The per-neuron slow states (ALIF
  adaptation, and optionally leak) are available too.
- **Integer hard threshold → no gradients** in the forward engine. Gradient-based methods (FPTT, HYPR,
  StochEP) need a **differentiable shadow twin** — which the engine already carries as the `f32` weight
  shadow. The gradient-free three-factor form (eligibility × reward) needs no shadow and fits the
  integer substrate.

Ranked by value to this engine:

1. **Adaptive threshold (ALIF) as the slow state** — highest leverage, and built. The per-neuron
   threshold is a slow dynamic variable (rises on fire, decays at rest): long-range memory, per-neuron,
   and exactly the variable e-prop/FPTT know how to train (e-prop's ALIF eligibility has a
   threshold-adaptation component — the `εᵃ` term the engine uses). Cheap: one extra per-neuron slow
   variable + decay.
2. **Constant-memory online learning** — don't store the multi-wave window; carry a constant-size
   **eligibility trace** forward and update online. Borrow (a) e-prop's threshold eligibility trace as
   the credit-assignment mechanism (gradient-free when the top-down signal is a reward = three-factor),
   and (b) FPTT's **dynamic regularizer** (couple each online update to a slowly moving reference
   parameter) as the stabilizer — transplantable even into a reward-modulated rule.
3. **Locality ladder for on-substrate learning** — **ETLP** (event-based three-factor local plasticity)
   is e-prop made hardware-local; the target shape for a rule that runs inside the substrate rather than
   an offline trainer.
4. **Liquid / heterogeneous time constants** — FPTT matched BPTT paired with a liquid (adaptive)
   time-constant neuron. Making `leak` per-neuron / heterogeneous / adaptive is another cheap per-neuron
   parameter buying multi-timescale memory (adaptive-LIF / resonate-and-fire family).
5. **HYPR parallelization** — file for later: forward-mode learning parallelizes, and the deferred
   one-hop model is already layer-parallelizable (read last wave's inbox, write next wave's outbox), so
   this lands naturally at scale-up.
6. **StochEP / Equilibrium Propagation** — a stretch: EP needs the net to relax to an equilibrium
   (convergent RNN, free/nudged phases); a driven wave engine only fits if a settling regime is carved
   out. Lower priority.

**Realized path:** a factored per-neuron eligibility trace (`pre-trace × ψ`, with the ALIF adaptation
`εᵃ` component) under a learning signal from a trained readout / multi-layer DFA feedback, updating the
stored ternary weights through the `f32` shadow, with `rate_reg` as a light co-trained liveness term.
Borrowable next: FPTT's dynamic regularizer as a stabilizer, and ETLP as the on-substrate-local target.

## Design note from the ALIF iteration (2026-07-08) — heterogeneous leak weighed

After the ALIF-threshold design landed, an alternative direction was analysed as a critical-thinking
exercise. It is not built; captured here so the reasoning is not lost.

### Adaptive / heterogeneous leak (vs. the adaptive threshold)

- **What.** Make the **leak** (integration time constant) per-neuron — heterogeneous-fixed or dynamic.
  Memory then lives in the *existing* `potential` (an **input-integration** memory), giving a spectrum
  of temporal receptive fields (coincidence-detector → accumulator). This is a *different* memory from
  ALIF's **output/firing-rate** memory, and composes natively with the sub-threshold-integration dead
  zone the engine deliberately keeps (slow-leak → near-perfect accumulator; fast-leak → leaky).
- **Memory usage — much lighter.** `adapt` is `i32` = **4 bytes/neuron**, the single heaviest
  per-neuron field (44% of the 9-byte footprint `potential`(i16)/`cooldown`(u8)/`threshold`(i16)/`adapt`(i32)),
  inflated to i32 by the Q-fixed-point dead-zone fix. Per-neuron leak is a `u8` shift = **1 byte**
  stored, or **0 bytes** if derived on demand from the seed. The fixed-heterogeneous form adds **no new
  dynamic state at all** — the multi-timescale memory is just `potential` decaying at per-neuron rates.
  (Deriving on demand trades 1 byte of storage for one mix per neuron per wave in the leak loop; storing
  the byte is likely the better trade.)
- **Trade.** Leak buys input-timescale memory + reservoir richness but **not** ALIF's rate
  self-regulation. The two are **not mutually exclusive**.
- **Recommendation.** Add heterogeneous leak **additively** first — cheap, deterministic, on-brand for a
  store-light engine — and measure what each contributes before deciding whether leak makes `adapt`
  droppable (reclaiming the 4 bytes). Final adjudication needs a temporal task.

Meta-point (shared with the ranked list above): mechanism should be validated by a task, not tuned in a
vacuum. See ranked items 1 (ALIF threshold) and 4 (liquid/heterogeneous leak).

## Follow-up: fixed-point potential — remove the floored-leak density cost (store-recall bench)

The Tier-0 store-recall bench forced a real engine change and left one open upgrade. The potential leak
`p -= (p>>a)+(p>>b)` had a dead zone (`0` for `0 < p < 2^a`), freezing small sub-threshold potentials
*forever* — the network had **infinite** sub-threshold memory, so plain LIF never forgot a cue and the
ALIF-vs-LIF distinction vanished. Fixed by flooring the positive decay at 1
(`p -= max((p>>a)+(p>>b), 1)`), giving a finite membrane time constant. LIF then forgot to chance while
ALIF held ~55% decoding at delay 24 — the adaptive-threshold memory is real and readable.

- **Cost, now on the books:** the 1/wave floor **starves weakly-driven (sparse) cascades** — a neuron
  receiving < 1 delivery/wave loses more to the floor than it gains, so it can't integrate. Upper layers
  need much denser drive (the fixture's fan-in went `level+1 count 6 → 16`). The sub-threshold
  integration property we'd defended is now confirmed to matter.
- **The upgrade when it bites — fixed-point `potential`.** Mirror the `adapt` fix: store `potential`
  scaled up (`i32`, Q-fixed-point), with `±1` deliveries becoming `±scale`. Then the geometric leak
  `p -= p>>a` decays exponentially to ~0 (finite τ, **no** dead zone) *and* keeps fine sub-threshold
  integration (accumulation at the fine scale, so a lone delivery leaves a trace over the membrane
  window). Cost: `potential` i16→i32, redo the drain clamp/overflow, retune rates — a real engine change
  touching everything (much bigger than the floored-leak two-liner). Do it if/when the density
  requirement of floored-leak proves limiting; otherwise floored-leak suffices.

## Bench findings: ALIF buys only held-category memory (three experiments), and what training should target

The Tier-0/1 bench (Specs 1, 2, 2b) ran three ALIF-vs-LIF experiments. The result is sharper — and
narrower — than expected, which directly scopes what an e-prop-style rule (Spec 3) should target.

- **Store-recall / delayed match (Spec 1):** **ALIF wins** — holds a cue across a silent delay that plain
  LIF forgets. *Held / probed* memory: the cue lives in the slow adaptation state and a probe converts it to
  a readable spike pattern. **Robust across the architecture sweep** — ALIF decodes 850–1000‰ (LIF always at
  chance 250‰) across width/depth/refractory/inhibition — **except under sparse connectivity**, where ALIF
  collapses to ~350‰: held memory needs enough fan-out to *spread the cue's adaptation footprint* across
  many neurons.
- **Memory Capacity (Spec 2):** **LIF wins** (~1.57 vs ~0.39). MC = *delayed linear echo* (reconstruct a
  specific past bit `u(t−k)`). LIF's fading spike echo does it; adaptation is a slow low-pass integrator
  that can't pinpoint a past bit. Exposing the raw adaptation state to the readout did not help. Also
  **robust across the same architecture sweep** — richer/inhibitory reservoirs raise LIF's MC (up to ~2.0)
  while ALIF stays pinned at ~0.3–0.4, *widening* the gap.
- **Temporal XOR (Spec 2b):** **LIF wins** (~86% at `τ=1`); **ALIF near chance**. And this survived an
  architecture sweep — width, depth, refractory, density, inhibition: XOR solvability varies a lot
  (sparsity/inhibition help; the dense all-excitatory topology the floored leak favors is *poor* for
  nonlinear separation; recurrence hurts), but **ALIF never helps in any of them**. So "adaptation buys
  nonlinear computation" is **falsified**, robustly.

**Conclusion: ALIF's adaptation buys *exactly one* thing — categorical memory held across a delay and read
by a probe. Not linear echo (MC), not nonlinear temporal computation (XOR).** The two prior notes'
guess that XOR/adding would reward ALIF was wrong; correct it here.

Implications for Spec 3 (e-prop-style training):

- **Train against a held-category / working-memory task** — the store-recall / delayed-match family (hold
  *which pattern*, recall after a delay/distractor). **Not** MC, **not** XOR/adding (bit-level tasks LIF
  already does better) — those don't exercise what adaptation provides.
- **e-prop's eligibility trace, carrying the ALIF adaptation state (the `εᵃ` component)**, is precisely the
  machinery that credits this slow held-state; the slow variable and the memory it carries line up.
- **Control for passive memory** when attributing memory to adaptation: recurrence carries memory on its
  own (feed-forward isolates adaptation), and — before the floored-leak fix — frozen sub-threshold
  potentials gave even LIF infinite memory. The harness supports feed-forward isolation.
- **Architecture note:** the dense drive the floored leak requires trades against nonlinear-task
  performance (sparse/inhibitory reservoirs separate better). If nonlinear tasks matter later, revisit the
  fixed-point-`potential` upgrade (which removes the density requirement) alongside sparser topology.
- **A connectivity-density tradeoff → mix densities, not just neuron types.** The sweeps exposed opposite
  density preferences: **dense fan-out favors ALIF's held-category memory** (store-recall 950‰ dense vs
  350‰ sparse), while **sparse favors LIF's nonlinear separation** (XOR 927‰ sparse vs 727‰ dense). Since
  `adapt_bump` is already per-layer, a heterogeneous network can place *dense ALIF* layers (hold context)
  and *sparse LIF* layers (compute) — the LSNN-style mix, but tuned per-role on both `adapt_bump` **and**
  topology density. This is the natural way to span both memory axes; worth a bench experiment (a mixed
  config on store-recall *and* XOR) before or alongside Spec 3.

## What e-prop does with firing rates — and how rate_reg follows it (2026-07-09)

The e-prop / LSNN literature pins down what firing-rate machinery should and shouldn't do; `wave_bitnet`'s
`rate_reg` follows it on three points.

1. **Rate regularization is a soft *loss* term, co-trained.** e-prop/LSNN adds an L1/L2 firing-rate penalty
   to the loss, minimized *jointly* with the task, pulling rates to a modest biologically-plausible level
   (~16 Hz; tasks solved with no neuron above ~12 Hz) for **efficiency / sparsity**. `wave_bitnet`'s
   `rate_reg` is exactly this: a light per-neuron term folded into the e-prop learning signal and minimized
   jointly with the task — not a separate pre-training pass on a proxy drive.
2. **A fixed, uniform rate target is a documented liability, not a virtue.** The SNN regularization
   literature warns it "forces neurons to have similar firing rates for different inputs instead of allowing
   them to use different rates to represent different inputs" — i.e. it erodes input-specific coding. This is
   exactly the **"class-agnostic activity, not information"** failure the depth- and recurrence-`rate_reg`
   experiments hit (reviving / keeping-alive added activity but not class content), and why `rate_reg` is
   held to a *light* co-trained term rather than a hard set-point: strict per-neuron rate homeostasis
   homogenizes rates → hurts reservoir richness.
3. **LSNN's delay-task memory is ALIF adaptation, not a self-sustaining recurrent loop.** LSNN = LIF +
   adaptive-LIF; the "long short-term memory" *is* the slow adaptation variable, and recurrence + e-prop
   *route/compute* on top. Stripping ALIF to "isolate recurrence" (the LIF temporal-XOR setup) removes the
   field's actual memory mechanism — so the LIF-recurrence sustaining null is the expected outcome, not a
   surprise: bare recurrence at a sub-critical operating point is not what holds the trace.

**Note on §2 (criticality).** For sustaining, the operating point that matters is recurrent **gain**,
measured on the *task* drive (or by a gap-survival probe directly), not a rate on random input — a firing
rate is at best a proxy for branching-ratio ≈ 1, and on the wrong drive not even that. (An earlier
external "gain" estimator was abandoned because its analytic `Σ|w|/θ` form was blind to actual propagation
— it read "branching 16" for layers that never fire; see `experiments_results.md`.)

**What the field does — followed here.** (a) Keep **ALIF** as the delay-memory mechanism (don't strip it);
(b) `rate_reg` only as a **light, co-trained liveness/efficiency term** (modest coefficient), never a memory
mechanism and never a hard uniform target; (c) let recurrence + e-prop **compute** on top of
adaptation-held memory. The real "does recurrence earn its keep" test is **ALIF + recurrence vs ALIF-alone**
at a delay where adaptation alone is marginal — not bare-LIF recurrence vs FF.

## The side-car recurrence result vs the literature (2026-07-10)

`wave_bitnet`'s recurrence result — trained recurrence robustly beats feed-forward **only** on a backward-fed
**side-car** topology that *isolates the recurrent layer from the forward path* (the forward signal skips past
it; a separate recurrent scratchpad holds state and writes back) — is not idiosyncratic. The same design
principle, and even the same two side-findings (isolation + sparse recurrence), recur across the RNN and
e-prop literature:

- **e-prop's own successors independently found "noise separation."** *…intrinsic noise filtering / intrinsic
  noise separation in RSNNs trained with e-prop* (bioRxiv **2025-05-25**; peer-reviewed *Neuromorphic
  Computing and Engineering* **5(4), 2025**) reports (a) an intrinsic mechanism in the **input layer that
  separates noise and stabilizes the recurrent layer**, and (b) that **increased sparsity in the recurrent
  layer significantly improves learning**. Those are exactly our two findings — *isolate the recurrent layer
  from forward noise* and *keep recurrence sub-critical/sparse* — from the same algorithm family, ~a year
  before we rediscovered them from the substrate up. (We already cited this group for its firing-rate figures;
  its noise-separation result is the load-bearing one for us.)
- **The side-car IS a "dual RNN".** *Separation of Memory and Processing in Dual Recurrent Neural Networks*
  (arXiv 2020) splits the net into a **recurrent layer that only holds the time-dependencies (memory)** and a
  **feed-forward layer that combines input + memory to answer** — architecturally our side-car (recurrent
  scratchpad + skip-past forward path), in non-spiking RNNs.
- **Dual memory pathways in SNNs.** *Algorithm–hardware co-design of neuromorphic networks with dual memory
  pathways* (Nature Machine Intelligence 2026) adds an explicit **slow memory pathway alongside fast spiking
  activity** — the same "separate the memory subsystem" move on a spiking substrate.
- **The general principle is canonical:** skip/residual connections that *bypass* recurrent layers (residual
  RNNs, Residual Memory Networks, identity skips between LSTM layers); **gated memory** (the LSTM/GRU *cell
  state* is a protected highway the gates isolate from the input transformation); and **reservoir computing**
  (recurrence kept out of the trained forward path).
- **Where we depart from vanilla LSNN.** The original LSNN (Bellec 2018/2020) uses a **single fully-recurrent
  hidden layer** — recurrence *mixed into* the forward path, the opposite of isolation. Our data says that on
  this substrate, training recurrence *in* the forward path craters the reservoir, while isolating it
  (side-car) works — putting us on the **dual-pathway / residual-RNN** branch of the field, not the
  monolithic-recurrent-hidden-layer design.

**Takeaway:** the side-car is an integer-substrate instance of the dual-pathway / memory-isolation principle
that both classic RNN engineering (gating, skips, reservoirs) and the newest e-prop work converge on. It also
gives the open "why does isolation help?" question a literature-backed hypothesis to test: training recurrence
*in* the forward path injects reverberation that corrupts the class projection; a side-car lets the loop hold
temporal state without polluting it.

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
- *Efficient connectivity and intrinsic noise separation in recurrent spiking neural networks trained with e-prop* (e-prop firing-rate figures, ~16 Hz / <12 Hz) — Neuromorphic Computing and Engineering 2025 — [iop](https://iopscience.iop.org/article/10.1088/2634-4386/ae0826)
- *Sparse-firing regularization methods for spiking neural networks with time-to-first-spike coding* (fixed-uniform-rate-target pitfall) — Scientific Reports 2023 — [nature](https://www.nature.com/articles/s41598-023-50201-5)
- *Long Short-Term Memory Spiking Networks and Their Applications* (LSNN net sizes: TIMIT 400, N-MNIST 120+84) — arXiv 2020 — [arxiv](https://arxiv.org/pdf/2007.04779)
- *Effective and Efficient Computation with Multiple-timescale Spiking Recurrent Neural Networks* (fully-recurrent hidden layer, temporal-task sizes) — arXiv 2020 — [arxiv](https://arxiv.org/pdf/2005.11633)
- Higuchi et al., *Balanced Resonate-and-Fire Neurons* (BRF — memory in the neuron's oscillatory dynamics, not the trained loop) — 2024 — [code](https://github.com/AdaptiveAILab/brf-neurons) (MIT)
- *Efficient learning and intrinsic noise filtering in RSNNs trained with e-prop* (input layer separates noise → stabilizes the recurrent layer; recurrent sparsity helps — matches our side-car isolation + sub-critical density) — bioRxiv 2025-05-25 — [biorxiv](https://www.biorxiv.org/content/10.1101/2025.05.25.656058v1); published as *Efficient connectivity and intrinsic noise separation…* — Neuromorphic Computing and Engineering 5(4), 2025 — [iop](https://iopscience.iop.org/article/10.1088/2634-4386/ae0826)
- *Separation of Memory and Processing in Dual Recurrent Neural Networks* (a recurrent memory layer + a feed-forward layer that combines input + memory — the side-car pattern, non-spiking) — arXiv 2020 — [arxiv](https://arxiv.org/pdf/2005.13971)
- *Algorithm–hardware co-design of neuromorphic networks with dual memory pathways* (explicit slow-memory pathway alongside fast spiking) — Nature Machine Intelligence 2026 — [nature](https://www.nature.com/articles/s42256-026-01255-3)
- *Can Biologically Plausible Temporal Credit Assignment Rules Match BPTT? E-prop as an Example* (e-prop only approaches BPTT — open question) — ICML 2025 — [arxiv](https://arxiv.org/abs/2506.06904)
