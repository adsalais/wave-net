# Non-spiking potential readout — design (Spec 3, V2a)

**Date:** 2026-07-08
**Status:** approved (design), pre-plan
**Scope:** one iteration — a **non-spiking readout layer** (the first engine change since the floored leak)
plus the e-prop learning variant that reads graded **membrane potential** as its output. Builds on V1
(`bench::eprop`). Still v1-rule otherwise (global reward × spike-count eligibility, f64 shadow).

## Why this

V1 learned (~770‰ vs frozen ~271‰) but needed a **population-coding hack** because two single spiking
output neurons were usually silent → `R = 0`. A **non-spiking leaky readout** (the standard e-prop / LSNN
output) gives a graded, always-defined output — cleaner signal, no hack, and the elegant **L0-input /
L_top-output symmetry**. The trade: a non-spiking readout has no threshold to train, so *all* learning moves
into the internal thresholds (feedback-alignment regime) — genuinely harder, so this is also a test of
whether the cleaner signal outweighs losing the trainable output layer.

## The engine change (minimal, no config churn)

- **`neurons.rs`:** add `pub readout: bool` to `Layer`; `Layer::new` sets it `false`.
- **`network.rs`:** add `Network::new_with_readout(config)` — constructs like `new` (which already forces
  L0 to a transducer) and additionally flags the **last** layer `readout = true`. Refactor `new` to
  delegate to a private builder taking a `readout_last: bool` so there is no duplication.
- **`wave.rs`:** in `process_layer`, after the inbox drain (step 2), if `layer.readout`: clear `fired` and
  **return** — skipping inject/decide/fire/generate/**leak**/adapt. The readout is a **drain-only perfect
  integrator sink** (per trial): its potential is the clean cumulative sum of its ±1 input, never reset by
  firing. *(Not leaky: our floored leak would eat weak ±1 input to ~0; reset-per-trial keeps the
  accumulation bounded, and the drain's i16 clamp is the overflow guard.)*
- `LayerConfig` / `Config` structs are **untouched** — no struct-literal churn across the codebase. TDD the
  readout mode as its own engine unit.

Determinism and integer purity are preserved (the readout is drain + leak, both integer).

## Architecture

Stack: **L0 transducer** + computational layers `1..L−1` (trained) + **readout layer `L`** (non-spiking).
The computational top layer projects `level+1` into the readout (fixed ±1); the readout layer has empty
outgoing topology (a sink), and integrates its input over the trial (reset per trial). Built with
`new_with_readout`.

## Output — reuse the existing accessor

Read the readout layer's neuron **potentials** with the existing `Network::potential(layer, local)` at the
end of the trial (no new API). Split the readout layer (`size×size`) into `K` contiguous **population
groups**; class score `c` = **sum of potentials** in group `c` (`i64`, may be negative). Always defined
even when nothing "fires".

## Reward & learning (the V1 rule, cleaner signal)

- **Reward:** `R = score[correct] − max(score[rivals])` — a graded potential margin; `R̄`-modulated
  (reward-prediction-error); global.
- **Learning:** internal computational thresholds `1..L−1` via spike-count eligibility × reward, through
  the f64 shadow. The readout layer has zero spike eligibility, so it is naturally excluded from updates —
  the V1 shadow/update loop is reused **unchanged**; only the *output scoring* differs (readout potentials
  instead of top-layer spike populations), plus the `new_with_readout` constructor and the appended layer.

## Success criterion

- **V2a learns:** `train_readout(cfg, lr)` late-half accuracy > chance (500) and clearly beats the
  frozen-threshold control (`lr = 0`), deterministically.
- **V1-vs-V2a comparison printed:** whether graded-signal + all-internal learning does **better or worse**
  than V1's ~770‰. Either is a real result.
- `eprop_readout_is_deterministic`.

**Honesty gate:** all-internal (feedback-alignment) learning is harder than V1's. If V2a cannot beat its
frozen control after reasonable tuning, that is the finding (the readout is cleaner but the reservoir can't
be shaped to a fixed projection by thresholds alone) — reported, pointing to broadcast-error alignment /
potential-based *internal* eligibility. Never a faked curve. The engine unit (`readout` integrates, never
fires) must pass regardless.

## Module

- Engine: `src/wave_net/{neurons,wave,network}.rs`.
- Bench: extend `src/bench/eprop.rs` — an `EpropConfig { readout: bool, ... }` flag (or a parallel
  `train_readout`) that appends+flags the readout layer, uses `new_with_readout`, and scores from readout
  potentials. **V1's path and tests stay intact.**

## Testing

- **Engine:** `readout_layer_integrates_and_never_fires` — a 2-layer net built with `new_with_readout`;
  drive the input; assert the readout layer never appears in `fired` (listener) and its potential
  integrates (non-zero, unreset) while a normal layer would have fired/reset.
- **Bench:** `eprop_readout_learns_and_beats_frozen` (headline + V1-vs-V2a print); `eprop_readout_is_deterministic`.
- **Regression:** all V1 `eprop` tests and the whole suite stay green (the engine change must not alter
  non-readout behavior).

## Determinism & constraints

- Std-only, no new deps. Engine change is integer + deterministic; `Layer::new` default keeps every
  existing config building identically (readout off unless `new_with_readout`).
- `f64` only in the bench (shadow, reward). Pure function of `(seed, config, params)`.

## Parameters (starting points, tunable)

Reuse V1's demo but add a readout layer: `size 8`, computational layers `3` + 1 readout, `K=2`,
`present/delay/read` as V1, dense `level+1` count 16, `trials ≈ 2400`, `block 200`, `lr` tuned (the graded
potential margin has a different scale than V1's spike margin, so `lr` and `reward_rate` re-tune).

## Revision (post-implementation) — engine works, global-reward learning is a null

**The readout engine change works** (`readout_layer_integrates_and_never_fires` passes; regression clean).
**The learning does not** with a global scalar reward: readout accuracy stays at chance (~490‰) versus a
frozen control at ~510‰ at *every* `lr`, while V1's spiking-trainable-output path learns to ~770‰. The
headline test was reframed to `eprop_readout_global_reward_does_not_learn`, which documents the null.

**Why (the informative part).** A non-spiking readout has no trainable output, so learning is entirely
*internal* (feedback-alignment). The fixed ±1 readout projection doesn't separate the classes, so the
class-score margin `R` is class-uninformative → `(R − R̄) → 0` → no threshold updates track the class. **A
global *scalar* reward is too weak to shape the reservoir to a fixed projection.** This also explains why
V1 learned: its output populations were spiking with *trainable* thresholds, so the scalar reward could
directly shape the output layer.

**Conclusion → V2b.** The potential readout must pair with **per-output broadcast-error credit**, not a
scalar reward. The readout engine layer is now committed infrastructure for that. (So V2b = readout +
broadcast-error alignment, not the "potential-based internal eligibility" originally penciled as V2b.)

## Deferred (updated roadmap)

- **V2b = readout + broadcast-error alignment:** per-output error `(target − score)ᵢ` fed to internal
  neurons via fixed random feedback weights — the credit signal a scalar reward can't provide.
- Potential-based *internal* eligibility (wake silent neurons), high-drive float-free trainer, per-wave/TD
  credit, `K > 2`, training adaptation params — later rungs.
