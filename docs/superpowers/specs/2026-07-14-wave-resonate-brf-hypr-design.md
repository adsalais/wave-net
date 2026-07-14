# wave_resonate — BRF (complex Resonate-and-Fire) neurons + HYPR online training

- **Date:** 2026-07-14
- **Status:** design approved; ready for an implementation plan
- **Scope:** a **new, independent engine** `wave_resonate` (an island duplicated from `wave_driven`) whose
  neuron is the **Balanced Resonate-and-Fire (BRF)** complex-membrane oscillator of Higuchi et al.,
  *Balanced Resonate-and-Fire Neurons*, ICML 2024 (`higuchi24a`; arxiv 2402.14603), trained **online,
  without BPTT** by a HYPR-style forward eligibility (Baronig et al., *A scalable hybrid training
  approach for recurrent SNNs*, NCE 2026; arxiv 2506.14464). The **ternary ±1/0 BitNet weight substrate
  is preserved** (we are still memory-constrained); only the neuron's internal integration and the credit
  rule change. **Capability-first**: the deliverable is a correct, faithful, converging engine, not an
  experiment.

## Why a new engine (not a change to wave_driven)

`wave_driven` is a LIF/ALIF integer engine whose value is **activity-bound** sweeps over a frontier of
non-quiescent neurons. BRF replaces LIF integration with a **continuously resonating** complex oscillator
(per-neuron frequency `ω`, dampening `b`), which rings for many waves after input — it is rarely
quiescent, so the LIF frontier assumption does not hold. This is a **different dynamics family and a
different credit rule**, so per the user's instruction it lives in its own folder `src/wave_resonate/`,
duplicating whatever it needs from `wave_driven`. **No shared code** — the two engines are islands.

**BPTT stays permanently out of scope.** The reference BRF impl trains with surrogate-gradient BPTT;
we do not. HYPR is the substitute: because the sub-threshold `(x,y)` dynamics are **linear**, RTRL
(real-time recurrent learning) is exact and cheap — a forward-propagated per-synapse eligibility, the
same online credit shape `wave_driven` already uses, generalized from a scalar trace to the resonator's
2-state trace.

## The one idea

> Each neuron is a **damped complex oscillator** `u = x + i·y` with per-neuron `ω` and `b`. It integrates
> the same **integer ternary spike-deliveries** as before (`I = Σ W·z`), spikes when its **real part**
> crosses an **adaptive threshold** (`ϑ_c + q`), and never hard-resets — refractoriness comes only from
> `q` (which both raises the threshold and increases the dampening: the *smooth reset*). Because
> `(x,y)` evolve **linearly** below threshold, the gradient of the state w.r.t. a weight is itself a cheap
> **2-state linear recursion driven by the presynaptic spike** — HYPR's forward eligibility — which we
> multiply by a local surrogate `ψ` and a DFA/readout learning signal to update the ternary shadow.

## The neuron — BRF (complex Izhikevich), exact

Per neuron `j`, per wave `t`. State: real part `x_j`, imaginary part `y_j`, refractory `q_j` — **all
f32**. Trainable per-neuron params: `ω_j` (angular frequency), `b′_j` (dampening offset) — **f32**.
Constants: time step `δ`, refractory decay `γ = 0.9`, base threshold `ϑ_c = 1`.

Input current (drained integer ternary deliveries → f32; the deferred one-hop routing supplies the
`t−1`):

```
I_j^t = Σ_i W_ji · z_i^{t−1}          W_ji ∈ {−1, 0, +1} (2-bit code); z ∈ {0,1}
```

Dynamics (paper Eq. 4 + Alg. 1; reference `snn/modules/rf.py::BRFCell`):

```
p(ω)  = (−1 + √(1 − (δ·ω)²)) / δ            divergence boundary; requires δ·ω ≤ 1  (ω ≤ 1/δ)
b_j^t = p(ω_j) − |b′_j| − q_j^{t−1}          balanced adaptive dampening (smooth reset via q)
x_j^t = x_j^{t−1} + δ·( b_j^t·x_j^{t−1} − ω_j·y_j^{t−1} + I_j^t )
y_j^t = y_j^{t−1} + δ·( ω_j·x_j^{t−1} + b_j^t·y_j^{t−1} )
z_j^t = Θ( x_j^t − ϑ_c − q_j^{t−1} )         spike on the REAL part; Θ = Heaviside; NO membrane reset
q_j^t = γ·q_j^{t−1} + z_j^t
```

- **Stability:** `p(ω)` places `b_c = p(ω) − b′` so the discrete update matrix has spectral radius ≤ 1
  (`(1 + δ·b_c)² + (δ·ω)² ≤ 1`), and `q` transiently deepens the dampening after a spike. `Config::validate`
  **must enforce `δ·ω ≤ 1`** for every neuron's initial `ω` (and clamp during training — see Non-goals).
- **Initialization (from seed, per reference Tables 3–6):** `ω_j ∼ U(ω_lo, ω_hi)`, `b′_j ∼ U(b_lo, b_hi)`
  (paper defaults `b′ ∈ [0.1, 1.0]`; `ω` range is task/timescale dependent — see Timescale mapping).
  `x = y = q = 0` at rest. Weights init to the procedural ±1 sign, exactly as `wave_driven`.

### L0 transducer and readout (unchanged roles, BRF-appropriate)

- **L0** is a **pass-through spike source**: `z_j^t = 1` iff `j ∈ input` this wave, else `0`; it runs **no
  oscillator dynamics and never adapts** (input encoding stays decoupled). Mirrors `wave_driven`'s forced
  transducer.
- **Readout** (last layer, `new_with_readout`) is a **leaky integrator**: `o_j^t = κ·o_j^{t−1} + I_j^t`,
  `κ = exp(−δ/τ_out)` (reference output-neuron time constant `τ_out`), never fires. The clean cumulative
  signal a trained readout reads. Mirror of L0.

## The engine — dense membrane, sparse delivery

Per wave, per layer, `process_layer`:

1. **drain** — fold `pending` (this wave's incoming ternary deliveries) into `I_j^t` (i32 sum → f32),
   clear `pending`. **Dense** over all neurons (the oscillator update is unconditional).
2. **inject** (L0 only) — mark injected locals as firing this wave.
3. **readout** — if `readout`, integrate `o ← κ·o + I` and return (no decide/generate).
4. **integrate + decide** — compute `b_j^t`, update `(x_j, y_j)`, test `z_j^t = Θ(x_j − ϑ_c − q_j^{t−1})`,
   update `q_j`. Push firers to `fired`. **Dense** (every neuron oscillates every wave).
5. **generate** — for each firer, word-scan its occupancy bitset, decode each wired cell to its target,
   scatter the packed 2-bit ±1/0 weight into the target layer's accumulator. **Sparse** (firer-gated) —
   the unchanged `wave_driven` delivery, and the dominant cost. BRF is deliberately spike-sparse.

`Network::wave` orchestrates exactly as `wave_driven` (process each layer, route synapse groups into
target accumulators, deferred one-hop swap of `pending`/`deliv` at wave end).

**No frontier for the membrane** (resonators ring — a frontier would stay near-full and buy little; see
Non-goals). The dense-vs-online equivalence oracle therefore compares dense-membrane forward state, and
(Phase 2) the online eligibility accrual against a dense offline recomputation.

### Timescale mapping (a first-class hyperparameter)

The complex neuron rotates ≈ `δ·ω` rad/wave, so an oscillation period is ≈ `2π/(δ·ω)` waves. The
reference `δ=0.01` with small `ω` gives 60–200-wave periods — far longer than our tasks (~tens of waves).
`δ` and the `ω`-init range are therefore **explicit swept hyperparameters**; a healthy `δ·ω` (e.g. ~0.5,
period ≈ 13 waves) is expected to fit our task windows. `Config` carries `dt` (δ) and `omega_init:
(f32,f32)`, `b_offset_init: (f32,f32)`; benchmarks add an **`ω`-init-range axis** (AGENTS.md: sweep every
axis).

## Training — HYPR forward eligibility

The sub-threshold `(x,y)` recursion is **linear** in the inputs, so the derivative of the state w.r.t. a
weight (RTRL) is a 2-state linear recursion through the **same Jacobian** as the forward pass — exact and
`O(1)`-in-sequence-length (HYPR). Per synapse `i→j` keep `(ε^x_ji, ε^y_ji)`; the spike depends on `x`:

```
ε^x_ji(t) = (1 + δ·b_j^t)·ε^x_ji(t−1) − δ·ω_j·ε^y_ji(t−1) + δ·z_i^{t−1}
ε^y_ji(t) = δ·ω_j·ε^x_ji(t−1) + (1 + δ·b_j^t)·ε^y_ji(t−1)
elig_ji  += ψ_j^t · ε^x_ji(t)
```

- **`ψ_j^t`** — the local surrogate `∂z_j/∂x_j`, a **double-Gaussian** of `(x_j^t − ϑ_c − q_j^{t−1})`
  (reference `functional/autograd.py::StepDoubleGaussianGrad`; a boxcar/triangle is a simpler fallback for
  bring-up). This replaces `wave_driven`'s spike-ψ.
- **`b_j^t` in the recursion is the instantaneous forward value** (the same one used in step 4). The
  second-order feedback of `q → b` into the eligibility is dropped in the primary rule (standard e-prop /
  HYPR linearization); a `q`-coupling refinement is deferred (see Open questions).
- **Weight update (multi-layer DFA, unchanged from `wave_driven`):** for each trainable edge
  `tz = z + edge.level ∈ [1, L)`, `shadow_ji += −lr · signal_tz[j] · elig_ji` over dirty rows, then
  `repack_row(i)`. Top layer gets symmetric readout feedback; deeper layers get fixed random DFA feedback.

### Trainable ω and b′ (per-neuron eligibility — default on)

`ω_j`, `b′_j` are trained by their own **per-neuron** forward eligibilities (O(neurons), cheap). For a
param `θ ∈ {ω_j, b′_j}` keep `(g^x_θ, g^y_θ)` recursed with the extra source terms from
`∂(step)/∂θ`:

```
g^x_θ(t) = g^x_θ(t−1) + δ·[ (∂b/∂θ)·x^{t−1} + b^t·g^x_θ(t−1) − (∂ω/∂θ)·y^{t−1} − ω·g^y_θ(t−1) ]
g^y_θ(t) = g^y_θ(t−1) + δ·[ (∂ω/∂θ)·x^{t−1} + ω·g^x_θ(t−1) + (∂b/∂θ)·y^{t−1} + b^t·g^y_θ(t−1) ]
Δθ_j    = −lr_θ · signal_j · Σ_t ψ_j^t · g^x_θ(t)
```

with `∂b/∂b′ = −1` (b′>0), `∂ω/∂b′ = 0`; `∂ω/∂ω = 1`, `∂b/∂ω = p′(ω) = −δω / √(1 − (δω)²)`. A
`train_omega_b: bool` flag (**default true**) gates these; when false they stay at their seed init (the
weights-only regression path). After each update, **clamp `ω` to keep `δ·ω ≤ 1`** and `b′ ≥ 0`.

## Module layout — `src/wave_resonate/` (island)

| file | responsibility | vs wave_driven |
|---|---|---|
| `mod.rs` | module wiring | new |
| `synapse.rs` | hash/topology/`sample_distinct_cells`/`decode`/`random_l0_input` | **copied verbatim** |
| `config.rs` | `Config`, BRF `LayerConfig` (`dt`, `gamma`, `theta_c`, `omega_init`, `b_offset_init`, `tau_out`; **drops** `leak`/`adapt_bump`/`adapt_decay`); `validate` enforces `δ·ω ≤ 1` and `count ≤ (2r+1)²`. `dt` is shared by the forward dynamics and the eligibility (`EligParams` reads it) | rewritten |
| `neurons.rs` | `Layer` SoA (f32 `x,y,q,omega,b_off` + occupancy bitset + 2-bit `codes` + optional `TrainState`); `new`, `enable/disable_training`, `repack_row`, `decode`, `for_wired`, `weight_at` | structure copied, LIF/adapt state replaced by BRF state |
| `wave.rs` | `process_layer` (dense oscillator + sparse delivery) + dense oracle | rewritten dynamics |
| `network.rs` | orchestration, deferred one-hop swap, `wave`, `reset_state`, eligibility accrual, `dfa_update`, listeners, readout | copied skeleton, BRF/HYPR bodies |
| `training.rs` | HYPR `EligParams` (`dt`, `surrogate` params, `train_omega_b`, cutoffs), `Edge`, online-mirroring **dense eligibility oracle** | rewritten rule |
| `equivalence_tests.rs` | online == dense eligibility (bit-exact); forward determinism | rewritten |

Registered in `src/lib.rs` next to `wave_driven`. A `bench/wave_resonate_bench.rs` (test-only) is added in
Phase 2 for the training harness (public-API only, mirrors `wave_driven_bench.rs`).

## Data structures & data flow

- **`Layer`** per-neuron SoA: `x, y, q: Vec<f32>`, `omega, b_off: Vec<f32>` (`ls`), `pending: Vec<i32>`
  (integer delivery accumulator, drained each wave), plus the copied occupancy substrate (`occ`,
  `offsets`, `off_flat`, `slot_bases`, `codes`) and `readout: bool`.
- **`TrainState`** (present only while training): `shadow: Vec<f32>` (ternary master), `elig: Vec<f32>`
  and `eps_x, eps_y: Vec<f32>` (per-synapse HYPR traces — 2 f32/synapse, layout == `shadow`); when
  `train_omega_b`: per-neuron `g_omega_x, g_omega_y, g_bo_x, g_bo_y: Vec<f32>` (`ls` each) and
  `omega_grad, bo_grad: Vec<f32>`; `spike_count: Vec<u32>` (rate_reg). Freed by `disable_training`.
- **Active sets** (sparse accrual, mirrors `wave_driven`): `elig_active[z]` — sources whose eligibility
  row is still live (nonzero `eps_x/eps_y` or a firing this wave); pruned when a row fully decays (a
  fully-decayed 2-state trace stays 0, so it can be dropped, re-added on the next presynaptic spike).
  `dirty_rows[z]` — rows that received `elig` accrual (drive `dfa_update` + `reset_eligibility`).
- **Determinism:** pure function of `(seed, config, input)`; single-threaded f32, no nondeterministic ops.

## Phasing

- **Phase 1 — BRF inference.** `config.rs` + `neurons.rs` (BRF state, no `TrainState`) + `wave.rs`
  (dense oscillator + sparse delivery) + `network.rs` (orchestration, `reset_state`, readout, listeners).
  Deliverable: a correct, deterministic forward engine + dense oracle. **No training code.**
- **Phase 2 — HYPR training.** `TrainState`, `enable/disable_training`, the online eligibility accrual +
  `dfa_update` on `Network`, `training.rs` `EligParams` + dense eligibility oracle, `bench` harness.
  Deliverable: bit-exact online-vs-dense eligibility + end-to-end FF training above chance.
- **Phase 3 — experiments (deferred, not in this spec).** Head-to-head vs the ALIF learner on the
  existing tasks; `ω`-init/`δ` sweeps; later recurrence/side-car. Persistence added here if
  best-checkpointing is needed.

## Validation (test-first, inline `#[cfg(test)]`)

**Phase 1**
- **Single-neuron oracle.** A hand-rolled reference implementation of the BRF equations (plain f32 loop)
  vs the engine's neuron on identical `I` sequences — bit-exact `x, y, q, z` over N waves.
- **Resonance / frequency selectivity.** A neuron tuned to `ω` responds preferentially (more spikes /
  larger `x` amplitude) to input delivered at its resonant rhythm than to off-frequency drive.
- **Divergence-free stability.** Under sustained strong drive, `|x|,|y|` stay bounded (no runaway) —
  the property `p(ω)` guarantees; a vanilla-RF control (fixed `b`, no `p(ω)`) is allowed to diverge in a
  documenting test.
- **Determinism.** Two builds, same `(seed,config,input)` → identical `x,y,q` and fired sets.
- **L0 transducer / readout.** L0 fires iff injected and never oscillates; readout integrates and never
  fires.

**Phase 2**
- **Bit-exact online == dense eligibility.** `dense_eligibility` recomputes `elig` (and `eps_x/eps_y`,
  and the `ω/b′` grads) offline from recorded per-wave `(spike, x, y, q, b)` records; must equal the
  online accrual bit-for-bit (the `wave_driven` pattern). Includes the pruned-active-set variant.
- **`train_omega_b=false` regression path.** Weights-only training leaves `ω,b′` at init and still
  accrues the correct weight eligibility (a regression gate).
- **End-to-end FF training above chance** on a temporal task (pure-ternary), read over a multi-wave
  window (AGENTS.md multi-wave rule).

Benchmarks (Phase 3) follow AGENTS.md: every axis incl. `ω`-init range + `δ`, multiple seeds, worst+mean,
read the top spiking layer, report fan-in density / σ / per-layer spiking profile / held-out accuracy.

## Non-goals / deferred

- **Fixed-point membrane** — f32 for now (weights stay ternary; per-neuron f32 state is O(neurons)).
  A fixed-point port is a later option only if perf demands and after correctness (defer-until-it-bites).
- **A per-neuron frontier/active-set for the membrane** — dense now; revisit only if measured ringing
  sparsity justifies it.
- **The BHRF (harmonic) and vanilla RF variants** — BRF complex only.
- **Persistence (`.wbm`/`.wbr`) for wave_resonate** — deferred to Phase 3.
- **`q → b` second-order eligibility term** — the primary rule uses the linear-Jacobian eligibility with
  `b^t` instantaneous; a `q`-coupling refinement (analogous to `wave_driven`'s `εᵃ`) is a fast-follow,
  taken only if training underperforms (see Open questions).
- **BPTT** — permanently out of scope.

## Open questions / risks

1. **Does the linear-Jacobian eligibility (no `q`-coupling) train well enough?** e-prop/HYPR typically
   suffices for the linear part; if recurrence/side-car underperforms later, add the `q`-coupling term.
2. **Surrogate choice** (double-Gaussian vs boxcar) and its width — matters for gradient quality; a
   bring-up knob, swept in Phase 3.
3. **`ω` clamping during training** — keeping `δ·ω ≤ 1` after updates must not distort the gradient
   badly; monitor how often the clamp binds.
4. **Timescale** — whether a `δ·ω` that fits our short task windows still gives BRF its frequency-
   selectivity advantage is itself a Phase-3 finding, not an assumption.

## References

- Higuchi, Kairat, Bohté, Otte. *Balanced Resonate-and-Fire Neurons.* ICML 2024 (`higuchi24a`;
  arxiv 2402.14603). Reference impl: `github.com/AdaptiveAILab/brf-neurons` (`snn/modules/rf.py`,
  `functional/autograd.py`).
- Baronig et al. *A scalable hybrid training approach for recurrent SNNs (HYPR).* NCE 2026
  (doi 10.1088/2634-4386/ae46d4; arxiv 2506.14464).
- Bellec et al. *A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop).*
  Nat. Commun. 2020 — the online forward-eligibility framing HYPR generalizes.
- `AGENTS.md` (engine model, ternary substrate, multi-wave rule, benchmark conventions);
  `docs/superpowers/specs/2026-07-13-wave-driven-*` (the island this duplicates).
