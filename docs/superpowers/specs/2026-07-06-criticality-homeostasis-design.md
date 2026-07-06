# Criticality + Homeostasis Conditioning — Design

**Date:** 2026-07-06
**Status:** Approved design, pending implementation plan
**Scope:** Spec 1 of 3 in the learning-layer program (see *Program context* below).

## Program context

`wave-net` is turning a fixed, procedurally-wired **wave reservoir** into a trained RSNN. Two
facts about the engine reorganize the whole learning problem and motivate this spec:

1. **There are no weights to train.** Synapses are a pure function of the hash; the ±1 sign is
   fixed. Every classical rule (BPTT, e-prop, R-STDP) is a *per-synapse weight* rule, so each
   must be re-expressed as **per-neuron** learning. The trainable state is a per-neuron vector
   (threshold, an additive *field*, optionally an efferent *gain*) — O(N), not O(synapses).
2. **The integer engine is non-differentiable.** Hard threshold + saturating integer math means
   no gradients exist. Gradient methods (BPTT, surrogate e-prop) need a differentiable *shadow*
   model; gradient-free methods (node perturbation, reward-modulated / homeostatic plasticity)
   run directly on the integer spikes and are the natural fit for an integer/FPGA-clean engine.

The program is decomposed into three independently buildable specs:

- **Spec 1 (this doc): Criticality + homeostasis** — unsupervised conditioning of the substrate
  into a good dynamical regime. Prerequisite that makes everything downstream sharper.
- **Spec 2: Online readout** — replace the offline ridge with a live task-output layer.
- **Spec 3: Task-driven per-neuron learning** — node perturbation vs reward-modulated plasticity
  vs (ceiling) surrogate-gradient, compared on the same task and readout.

## Goal of this spec

Take the demo reservoir and produce a **conditioned** one — near-critical, with well-distributed
per-neuron participation — by tuning per-layer thresholds (offline) and a frozen per-neuron field
(online), then prove it helped with an evaluation harness.

All new code lives in `src/wave_net/`. **`wave_reservoir/` is not modified** — it is the frozen
reference engine and is used only as a dependency.

## Scope

**In scope**
- A measurement layer (σ estimators + per-neuron rate meter) over the existing spike listeners.
- Offline coarse tuning of per-layer `threshold_base` to a target spatial σ.
- Online fine per-neuron homeostasis via a field injected through the `drive` vector.
- Transient conditioning then **freeze**; the controller is re-invokable for later housekeeping.
- An evaluation harness (memory capacity primary, temporal-XOR secondary) and an ablation study.

**Out of scope** (deferred to later specs; the design must not preclude them)
- Task-driven / reward-driven learning (Spec 3).
- Perpetual online homeostasis during deployment (the controller is built to allow it later, but
  this spec only ships transient-then-freeze).
- Trainable efferent gain or any new mutable-parameter engine.
- Avalanche / power-law criticality rigor (occasional validation, a later pass).
- Any change to the readout — it stays the offline ridge, here only a measuring stick.

## Background: the two branching ratios

The topology mixes forward levels (+1, +2) and recurrent levels (0, −1). That gives **two**
distinct criticalities, controlled by the same excitability knob but measuring different things:

- **Spatial σ** — within one wave, does a spike in layer *z* trigger ~1 spike in layer *z+1*?
  Governs whether an input burst survives the climb up the stack. Set by the forward levels.
- **Temporal σ** — from one wave to the next, does activity trigger ~1 spike a wave later (through
  the recurrent 0/−1 synapses and leak carryover)? Governs how long information persists — memory.

The chosen primary metric, **memory capacity**, is most sensitive to the *temporal* one. The knob
we tune (per-layer threshold) moves both at once, since it is overall excitability. So we measure
both and let memory capacity be the final arbiter.

Note: **firing-rate calibration is not criticality calibration.** The existing `calibrate_on_stream`
hits a target firing rate; a network can sit at 12% firing while being sub- or super-critical.
This spec targets σ directly.

## Architecture

### Module layout

```
src/wave_net/
  mod.rs            # declarations + the ConditionedReservoir artifact
  meter.rs          # measurement: per-layer counts, per-neuron rate EMA, spatial σ, temporal σ
  criticality.rs    # offline coarse: iterate per-layer threshold → spatial σ ≈ target
  homeostasis.rs    # online fine: per-neuron field controller (re-invokable), then freeze
  conditioning.rs   # orchestrator: offline → online → freeze → ConditionedReservoir
  evaluate.rs       # memory capacity (primary) + temporal-XOR (secondary) + shared ridge/Cholesky
```

`lib.rs` gains `pub mod wave_net;` alongside `pub mod wave_reservoir;`.

### Two execution modes (the load-bearing decision)

- **Conditioning uses the sequential `wave()` in an external loop.** Online homeostasis needs
  per-wave feedback (run a wave → read spikes → adjust field → next wave), and `run_stream` owns
  its wave loop internally. Driving `wave()` ourselves is single-threaded, fully deterministic,
  and makes the `&mut` field update trivial. σ measurement also lives here (cross-layer listener
  order is only guaranteed single-threaded).
- **Deployment uses `run_stream(threads)` with the frozen field.** Once frozen, the field is just
  constant per-neuron drive, so evaluation and all later use run threaded and bit-identical,
  exactly like today.

### Data flow

```
IntConfig::demo()
   │
   ▼  criticality::tune_thresholds(cfg, input, target_σ, passes)      [offline, single-thread]
IntConfig'   (per-layer thresholds set so spatial σ ≈ target)
   │
   ▼  homeostasis: build net, attach meter, drive wave() loop,        [online, single-thread]
   │              update per-neuron field from rates, then freeze
ConditionedReservoir { cfg', field: Vec<i16> }
   │
   ▼  deploy: net = cr.build();                                        [threaded, deterministic]
   │          net.run_stream(waves, threads,
   │              |w, buf| cr.drive_into(buf, task_input(w)))
   ▼  evaluate::run(cr, seed) → { MC, XOR, spatial σ, temporal σ, rate stats }
```

### Integer boundary

The meter, σ estimators, controller internals, and ridge may use f32/f64 (the existing ridge
already does). The **only** value crossing back into the engine is the i16 field added to `drive`,
so the runtime stays integer/FPGA-clean. The controller keeps a higher-resolution internal field
(`Vec<f32>`) and injects the rounded i16, so sub-unit adjustments are not lost to quantization.

## Components

### `meter.rs` — measurement

Subscribes to every layer via `on_layer` and turns the spike stream into:

- **Per-layer counts** Aᵤ per wave (from fired-locals).
- **Per-neuron rate** — EMA of did-fire, one f32 per neuron: `rate_i ← (1-α)·rate_i + α·fired_i`.
- **Spatial σ** — per-layer ratios computed from counts **summed over the whole measurement
  window** (ΣA₍z₊₁₎ / ΣA₍z₎), not per-wave ratios, so a momentarily silent layer never causes a
  divide-by-zero or dominates the average. The reported spatial σ is the geometric mean of these
  ratios over layers 1..L−1; layer 0 is drive-dominated and excluded. The per-layer ratio vector
  (indexed by the lower layer z = 1..L−1) is exposed so the tuner knows which layer runs hot.
- **Temporal σ** — Wilting–Priesemann multistep regression on the population series A(w) = Σᵤ Aᵤ:
  fit E[A(w+k)] ≈ mᵏ·A(w) + b over lags k=1..k_max, recover m. Subsampling-robust.

```rust
pub struct Meter { /* Arc<Mutex<Inner>> internally */ }

impl Meter {
    pub fn new(dims: Dims, alpha: f32) -> Meter;
    pub fn attach(&self, net: &mut LayerNet);        // registers listeners on all layers
    pub fn end_wave(&self);                           // roll per-wave accumulators; update rate EMA
    pub fn per_neuron_rates(&self) -> Vec<f32>;
    pub fn spatial_sigma(&self) -> (f32, Vec<f32>);   // (geomean, per-layer ratios)
    pub fn temporal_sigma(&self, k_max: usize) -> f32;
    pub fn reset(&self);
}
```

Constraint: σ methods assume the sequential `wave()` driver (single-threaded). Per-neuron *feature*
collection for evaluation is order-independent and stays thread-safe (same pattern `params_study`
already uses).

### `criticality.rs` — offline coarse

Same iterate-to-target shape as `calibrate_on_stream`, targeting σ instead of firing rate:

```rust
pub fn tune_thresholds(
    cfg: IntConfig,
    target_spatial_sigma: f32,
    passes: usize,
    input: &impl Fn(usize, &mut [i16]),   // random-bit stream at input sites
) -> IntConfig;
```

```
repeat `passes` times:
    build net, attach meter, run CAL_WAVES via wave() loop over the input stream
    (_, ratio) = meter.spatial_sigma()          // ratio[z] = ΣA_{z+1}/ΣA_z, for z = 1..L-1
    for z in 2..L:                                # tune each layer from the ratio feeding it;
        r = ratio[z-1]                            # skip layers 0 (drive) and 1 (no clean feeder)
        step = (threshold_base[z] >> 2).max(1)
        if r > target: threshold_base[z] += step
        else:          threshold_base[z] = (threshold_base[z] - step).max(1)
        spread_log2[z] = spread_log2_for(threshold_base[z])
    set saturation from max threshold (as today)
```

Layer 0 receives the drive and layer 1 has no clean non-drive feeder, so both keep their demo
threshold (or a firing-rate fallback); tuning shapes layers 2..L from the spatial ratio feeding
each. Layers couple (raising *z* changes *z+1*'s input); repetition over `passes` absorbs the
coupling, exactly as the existing calibrator does. Tuned on the **same random-bit input the task
uses**, so criticality is matched to operating statistics, not to silence.

### `homeostasis.rs` — online fine + field substrate

```rust
pub struct FieldSubstrate {
    field_hi: Vec<f32>,    // high-res internal accumulator
    field_i16: Vec<i16>,   // quantized, injected into drive
    input_sites: Vec<u32>, // excluded from adaptation
    target_rate: f32,
    eta: f32,
    clamp: i16,
}

impl FieldSubstrate {
    pub fn new(n_total: usize, input_sites: &[u32], params: HomeostasisParams) -> Self;
    pub fn inject(&self, buf: &mut [i16]);   // buf[i] += field_i16[i]
    pub fn update(&mut self, rates: &[f32]); // integral control; skips input_sites; re-quantize
    pub fn field(&self) -> &[i16];           // borrow current field
    pub fn freeze(self) -> Vec<i16>;         // hand off the i16 field
}
```

Per-neuron integral controller (skips input sites — their rate is the input signal's job):

```
update(rates):
  for i not in input_sites:
      field_hi[i] += eta * (target_rate - rates[i])   // fire too little → field rises
      field_hi[i]  = field_hi[i].clamp(-clamp, clamp)
      field_i16[i] = field_hi[i].round() as i16
```

The controller is a standalone, re-invokable object — later housekeeping is "call `update` again
for K waves, refreeze." No one-shot assumption baked in.

### `conditioning.rs` / `mod.rs` — orchestrator + artifact

```rust
pub struct ConditioningParams {
    pub target_spatial_sigma: f32,   // default 0.9
    pub criticality_passes: usize,   // default 10
    pub homeostasis: HomeostasisParams,
    pub warmup_waves: usize,         // adapt only after this many
    pub settle_waves: usize,         // fixed conditioning length
    pub input_sites: Vec<u32>,
}

pub struct ConditionedReservoir {
    pub cfg: IntConfig,     // tuned thresholds
    pub field: Vec<i16>,    // frozen per-neuron field
}

impl ConditionedReservoir {
    pub fn condition(base: IntConfig, params: &ConditioningParams,
                     input: &impl Fn(usize, &mut [i16])) -> Self;   // offline → online → freeze
    pub fn build(&self) -> LayerNet;                                 // fresh net from cfg
    pub fn drive_into(&self, buf: &mut [i16], task_input: &[i16]);   // task_input + field, for drive_fn
}
```

Conditioning loop (inside `condition`, after offline tuning):

```
net = LayerNet::new(cfg'); meter.attach(&mut net); field = FieldSubstrate::new(...)
for w in 0..settle_waves:
    buf = zeros(n); add task_input(w) at input sites; field.inject(&mut buf)
    net.wave(&buf); meter.end_wave()
    if w >= warmup_waves: field.update(&meter.per_neuron_rates())
return ConditionedReservoir { cfg', field.freeze() }
```

Stop condition: fixed `settle_waves` (deterministic, predictable), with the final per-neuron
rate-error reported so we can confirm it settled. Early-stop on a rate-error EMA is a possible
later refinement, deliberately not in this spec.

### `evaluate.rs` — metrics + ablation

One streaming run produces the features once; both task metrics read from them.

```rust
pub struct EvalReport {
    pub memory_capacity: f32,      // Σ_k r²_k, primary
    pub xor_accuracy: f32,         // secondary
    pub xor_control: f32,
    pub spatial_sigma: f32,
    pub temporal_sigma: f32,
    pub rate_mean: f32,
    pub rate_std: f32,             // participation spread (dead/saturated detector)
    pub dead_fraction: f32,        // neurons with rate ≈ 0
}

pub fn run(cr: &ConditionedReservoir, seed: u64) -> EvalReport;
```

- **Memory capacity (primary):** stream random bits; per delay k=1..K ridge-fit a linear readout
  to reconstruct u(t−k) from the features; MC = Σₖ r²ₖ on the held-out split.
- **Temporal-XOR (secondary):** u(t)⊕u(t−τ) accuracy vs the input-only control, from the same
  features.
- σ and rate stats from a short single-threaded measurement pass.

`wave_net` carries its own small ridge/Cholesky (a Rust lib cannot depend on an example; the
`params_study` copy stays as the reference). Eval constants reuse `params_study` where sensible
(128 sampled neurons, WASHOUT/TRAIN/TEST, λ=1.0, PER_CHANNEL=24, INPUT_LEVEL=4).

## Defaults / starting parameters

These are starting values, expected to be tuned during implementation:

| Parameter | Start | Rationale |
|---|---|---|
| `target_spatial_sigma` | 0.9 | slightly subcritical — a driven net at exactly 1.0 runs away |
| rate EMA `alpha` | ≈ 1/64 | ~64-wave participation window |
| homeostasis `target_rate` | 0.12 | near the inverted-U firing peak `params_study` found |
| homeostasis `eta` | small (≈0.5 field-units per unit rate error) | stable integral control |
| field `clamp` | ± a few × threshold_base | keep the field from dominating dynamics |
| `warmup_waves` | 50 | let rates settle before adapting |
| `settle_waves` | 500–2000 | fixed conditioning length |
| `criticality_passes` | 10 | matches existing calibrator |
| MC delays `K` | 20 | a few × depth |
| temporal σ `k_max` | 30 | lags for the WP regression |

## Evaluation & ablation

Headline deliverable: `examples/conditioning_study.rs` (parallel to `params_study`), printing
MC / XOR / spatial σ / temporal σ / rate-spread for three substrates across a few seeds —

1. **raw demo** (no conditioning),
2. **criticality-tuned only** (offline),
3. **criticality + homeostasis** (full).

This is both the "did it help?" evidence and the direct answer to the open question of whether
homeostasis adds anything over criticality alone. Expected signatures of success: full ≥ raw on
MC; spatial σ moves toward ~1; rate_std and dead_fraction shrink (participation evens out).

## Testing & determinism

Inline `#[cfg(test)]` per module, test-first where practical.

- **meter:** spatial σ on a hand-built config with a known branching profile; temporal-σ estimator
  on a synthetic count series (no engine); rate-EMA converges to a known firing fraction.
- **criticality:** post-tuning spatial σ is closer to target than the untuned demo; tuning is
  reproducible (same seed/input → same config).
- **homeostasis:** an over-hot neuron's field goes negative and its rate falls toward target;
  identical frozen field across identical runs; input sites keep field 0.
- **conditioning:** the deployed product is **1-vs-N-thread bit-identical** (new injection path —
  tested even though it inherits the engine guarantee); `drive_into` composes field + input.
- **evaluate:** MC of a normal reservoir > MC of a memoryless (hard-leak) control — proves the
  harness measures memory, not noise; ridge/Cholesky unit-tested on a known linear system.

**Determinism guardrails** (project hard requirement): conditioning is single-threaded by
construction; the frozen product is constant-drive, so it stays bit-identical across thread counts;
all randomness flows from the seed through the existing hash (no `Date`/`rand`/hash-map iteration).

## Conventions

Standard library only; no `unsafe`; warning-free build; inline TDD tests; one commit per task with
conventional-commit messages; no `Co-Authored-By` trailer. Branch off `main` for the work.

## Open questions carried forward

- Whether homeostasis meaningfully improves on criticality-only — answered empirically by the
  ablation, not assumed.
- Exact `eta` / `settle_waves` / `clamp` — tuned during implementation against the ablation.
- Whether temporal σ should later join the tuning loop (currently a reported diagnostic only).
