# ALIF neurons — design

**Date:** 2026-07-08
**Status:** implemented (see the *Revision* note at the end — `adapt` is Q8 fixed point, not `i16`)
**Scope:** one iteration — make every neuron an adaptive leaky integrate-and-fire (ALIF)
neuron, start the baseline threshold low, and retarget calibration accordingly. No learning
rule, no online homeostasis (both noted as later phases in `docs/related-work.md`).

## Motivation

`docs/related-work.md` ranks *"adaptive threshold (ALIF) as the trainable slow state"* as the
highest-leverage, lowest-cost next step: it gives long-range memory, is per-neuron (so it fits the
procedural-connectivity constraint that only per-neuron state is trainable), and is exactly the slow
variable e-prop/FPTT know how to train. It bridges today's static-threshold *calibration* to future
dynamic-threshold *learning*.

An ALIF neuron splits the firing threshold into a static **baseline** plus a **dynamic adaptation**
term that jumps up on each spike and decays back at rest — "more resistive after firing." That
adaptation is a local negative-feedback controller on firing rate, so the network self-regulates.

## Chosen approach — Hybrid

Two timescales coexist:

- **Adaptation (`adapt`) is always live** — part of the neuron model on every wave, including during
  the calibration pass itself. It supplies the fast, moment-to-moment self-regulation and memory.
- **Calibration is a one-time offline pass that sets the static `baseline`** — the floor under
  adaptation. It runs waves *with adaptation active*, measures the resulting (self-regulated) rate,
  and nudges `baseline` so the average rate lands near target. `baseline` is then frozen.

Rationale for hybrid over the two alternatives:

- **Pure self-regulation** (adaptation only, no calibration) compresses and stabilizes rates but does
  **not** pin them to a target — the equilibrium rate still depends on input drive. Precise rate
  control needs something that sets the operating point.
- **Keep silent start** (high baseline, adaptation as pure add-on) preserves the old invariant but
  discards the low-baseline idea and the self-regulation benefit.
- **Hybrid** keeps a working, precise rate control (reusing the existing calibration machinery almost
  verbatim) while adopting the low baseline and letting adaptation do the fast dynamics + memory. It
  also cleanly separates a *set* parameter (`baseline`, calibrated, later trainable) from *dynamic
  state* (`adapt`) — the exact static→dynamic bridge related-work calls for.

### The self-regulation loop (reference)

`adapt` is a per-neuron slow variable next to `potential`. Two events touch it, both integer:

- **On fire:** `adapt += adapt_bump` — firing raises the effective threshold.
- **Every wave:** `adapt -= adapt >> adapt_decay` — geometric decay toward 0, like the potential leak.

Fire test changes from `potential >= threshold` to `potential >= baseline + adapt`. Negative feedback:
fire a lot → `adapt` piles up → effective threshold climbs → firing gets harder → rate falls; fire
rarely → `adapt` decays to 0 → threshold returns to `baseline` → rate rises. Equilibrium:
`adapt ≈ rate · adapt_bump · 2^adapt_decay`, so effective threshold ≈ `baseline + rate·bump·2^decay`.

Knobs: `adapt_bump` (β) = adaptation strength (bigger → lower, tighter equilibrium rate);
`adapt_decay` = memory constant (loses ~`1/2^decay` per wave, remembers ≈ `2^decay` waves).

## Detailed design

### 1. Neuron state (`neurons.rs`)

Add one field to `Layer`, mirroring `potential`:

```rust
pub adapt: Vec<i32>,   // wave-mutable; rest 0, non-negative — adaptive threshold contribution (Q8 fixed point)
```

- `i32` in **Q8 fixed point** (scaled by `2^ADAPT_SHIFT`, `ADAPT_SHIFT = 8`). See the *Revision* note
  at the end: an earlier `i16` design held `adapt` in raw units, but the integer right-shift decay
  `adapt >> k` has a dead zone (`== 0` while `adapt < 2^k`) that made small adaptation ratchet up and
  lock out weakly-driven neurons. The fixed-point scale pushes that dead zone below ~1/256 of a
  threshold unit, so decay stays exponential (τ ≈ `2^adapt_decay` waves, independent of magnitude)
  and always relaxes.
- Rests at 0; only ever ≥ 0 (bump positive, decay pulls toward 0).
- The existing `threshold` field keeps its name but is **now semantically the baseline** (the static
  floor). Doc comments updated to say so.

The bump is a **saturating add** clamped to `ADAPT_MAX = (i16::MAX as i32) << ADAPT_SHIFT`, so the
effective contribution `adapt >> ADAPT_SHIFT` never exceeds `i16::MAX` — overflow protection in the
same spirit as the existing drain clamp, not a saturation/membrane concept.

### 2. Config (`config.rs`)

Add to `LayerConfig`:

```rust
pub adapt_bump: i16,    // added to `adapt` on each fire (β, in threshold units)
pub adapt_decay: u8,    // right-shift: adapt -= adapt >> adapt_decay, each wave (like leak)
pub baseline_init: i16, // construction center for the baseline threshold (low, not i16::MAX)
```

- `adapt_bump = 0` ⇒ `adapt` stays 0 forever ⇒ effective threshold = baseline always ⇒ **plain LIF
  dynamics** (backward-compatibility escape hatch).
- `adapt_decay` mirrors `leak`'s shift; `validate()` requires it `>= 1` (as leak shifts do) so decay
  always makes progress.
- `baseline_init` replaces the hard-wired `i16::MAX`. `threshold_jitter` keeps its meaning (span) but
  is now **added** above the low baseline instead of subtracted from the max.
- Demo starting points (all calibration-tunable): `baseline_init: 12`, `adapt_bump: 16`,
  `adapt_decay: 5` (τ ≈ 32 waves, matching the multi-wave read/train window).

### 3. Init (`Layer::new`)

```rust
let jitter = map_range(h, cfg.threshold_jitter) as i32;   // [0, jitter)
*th = (cfg.baseline_init as i32 + jitter).clamp(1, i16::MAX as i32) as i16;
// adapt[i] = 0
```

The floor of **1** (already the clamp in `shift_threshold`) makes "around zero" safe: baseline ≥ 1
with `adapt = 0` gives effective threshold ≥ 1, and potential rests at 0 < 1, so a resting neuron
never spuriously fires. This closes the degenerate "threshold == 0 fires every wave" case.

### 4. Wave step (`wave.rs` `process_layer`)

Two edits to the existing six-step body.

**Step 4 (decide)** — effective threshold in `i32` (overflow-safe), bump on fire:

```rust
let eff = layer.threshold[i] as i32 + (layer.adapt[i] >> ADAPT_SHIFT);
if layer.cooldown[i] == 0 && (layer.potential[i] as i32) >= eff {
    layer.potential[i] = 0;
    layer.cooldown[i] = layer.cooldown_base;
    let bumped = layer.adapt[i] + ((layer.adapt_bump as i32) << ADAPT_SHIFT);
    layer.adapt[i] = bumped.clamp(0, ADAPT_MAX);
    fired.push(i as u32);
}
```

**New, alongside step 6 (leak)** — decay adaptation every wave:

```rust
let d = layer.adapt_decay;
for a in layer.adapt.iter_mut() {
    *a -= *a >> d;
}
```

Ordering is deliberate and matches the potential discipline: `decide` reads `adapt` *before* this
wave's bump (the neuron had not spiked yet when it crossed threshold), then bump, then decay.
Injection (forced fire) flows through `decide` unchanged, so an injected spike bumps `adapt` like any
other fire — correct.

### 5. Calibration reconciliation (`calibrate.rs`)

Key finding: **the algorithm barely changes.** `calibrate_step` is already symmetric — it *raises*
thresholds when a layer is too hot and *lowers* when too cold. Today the net boots silent so it only
ever lowers; with a low baseline the net **boots hot** and the same step moves the other way.

- `calibrate()` orchestration: unchanged.
- `calibrate_step` / `shift_threshold`: unchanged — they now tune the *baseline*, with adaptation live.
- `measure_layer_rates` calls `reset_state` (which now also zeros `adapt`) then runs `warmup` waves
  before counting; that warmup lets adaptation reach equilibrium before measurement. The measured
  rate is therefore the *self-regulated* rate, so calibration converges `baseline` to the point where
  adaptation-live rate ≈ target — precisely the hybrid intent, with no new calibration code.

The only real change is conceptual (what "before calibration" looks like), surfacing as test churn.

### 6. Network plumbing (`network.rs`)

- `reset_state`: also zero `adapt` (wave-mutable state — required for determinism and clean
  measurement).
- New accessor `adaptation(layer, local) -> i16`, mirroring `potential(...)`, for tests/observability.
- Determinism untouched — every new op is a pure integer function of existing deterministic state.

### 7. Docs

Update the **silent-start** language in `AGENTS.md` (§"The engine model", §"Calibration"/"Silent
start", and the Invariants list) and the `neurons.rs`/`wave.rs` module doc comments to describe the
low-baseline, boots-hot, self-regulating-plus-calibrated model. These docs currently assert the exact
invariant this change inverts, so updating them is part of the work.

## Testing

**New tests:**

- `adapt_decays_toward_zero` — set `adapt`, run a wave, assert geometric drop `a -= a >> decay`.
- `fire_bumps_adaptation_and_raises_effective_threshold` — fire once; assert `adapt += bump` and that a
  just-fired neuron needs more drive to fire again.
- `adaptation_self_limits_rate` — strong constant drive: instantaneous rate falls over waves as
  `adapt` climbs, then settles (the negative-feedback fixed point).
- `bump_zero_is_plain_lif` — with `adapt_bump = 0`, dynamics match a static-threshold run.
- `determinism_includes_adaptation` — two runs produce identical `adapt` trajectories.
- `calibrate_hits_target_with_adaptation_live` — after calibrate, top-layer self-regulated rate ≈ target.

**Existing tests that change:**

- `neurons.rs`: `thresholds_near_i16_max_within_jitter` → `thresholds_near_baseline_within_jitter`
  (init is now low). Any assertion of `threshold == i16::MAX` at construction updates.
- `calibrate.rs`: `calibrate_warms_silent_upper_layers` → precondition inverts (top boots *hot*);
  becomes `calibrate_settles_upper_layers`. `calibrate_lowers_every_upper_layer` may now raise some
  layers — relax to "moves toward target."

## Scope guard (YAGNI)

Explicitly **out** of this iteration, all noted as later in related-work:

- Any learning rule (e-prop / three-factor eligibility traces).
- Per-neuron `adapt_bump` / `adapt_decay` (per-layer only for now).
- Online baseline homeostasis (baseline stays offline-calibrated then frozen).
- Per-neuron / adaptive `leak`.

## Backward compatibility

`adapt_bump = 0` recovers plain LIF *dynamics* exactly (adaptation term is identically 0). Init is
no longer byte-identical to the old `i16::MAX - jitter` (it is now `baseline_init + jitter`), so
construction-time threshold values change by design; tests assert the new low-baseline init.

## Revision (2026-07-08): fixed-point adaptation, and why the potential leak is left alone

The first implementation stored `adapt` as raw `i16` and decayed it with `adapt -= adapt >> adapt_decay`.
That exposed a real integer-dynamics flaw: the right-shift **dead zone**. For any `adapt < 2^adapt_decay`
the shift is 0, so small adaptation never decays — it ratchets up in `adapt_bump` steps and *sticks*.
With the thin recurrent drive to upper layers, a single spike could then lock a neuron out for tens of
waves; calibration drove those baselines to the floor and the layers still stayed silent. It also forced
a uselessly small `adapt_decay` (τ ≈ 1.4 waves) just to keep adaptation from sticking — far too short to
be real memory.

**Fix:** store `adapt` in **Q8 fixed point** (`i32`, scaled by `2^ADAPT_SHIFT`, `ADAPT_SHIFT = 8`). The
decay formula is unchanged, but the dead zone now sits below ~1/256 of a threshold unit, so decay is
genuinely exponential with τ ≈ `2^adapt_decay` waves **independent of adaptation strength** — long,
useful memory is now safe (the calibrate config uses `adapt_decay = 5`, τ ≈ 32 waves). Effective
threshold is `threshold + (adapt >> ADAPT_SHIFT)`; a fire adds `adapt_bump << ADAPT_SHIFT`, clamped to
`ADAPT_MAX`. `Network::adaptation` returns the raw Q8 state (`i32`).

**Does calibration still help?** Yes. Adaptation *compresses* the reachable rate band (so calibration's
baseline lever loses leverage as adaptation strengthens) but never *harms* — worst case it floors the
baseline and stops. The silent upper layers were the dead-zone lock-out, not calibration; with the dead
zone fixed, calibration and adaptation are complementary on two timescales (DC operating point vs. AC
dynamics). If strong+long adaptation ever starves upper layers, the lever is **more drive** (denser
recurrence / downward `level: -1` coupling), not removing calibration. *(Follow-up, not done here.)*

**Why not apply the same fix to the `potential` leak?** The potential leak has the identical low-end
dead zone, but there it is **desirable and must stay**. `adapt` should *forget* (a non-relaxing value is
a bug); `potential` should *integrate* (holding small sub-threshold input lets sparse ±1 spikes
accumulate toward threshold). Flooring the potential leak would make a lone `+1` evaporate next wave,
gutting sub-threshold integration (and breaking `deferred_delivery_is_one_hop`). Potential also cannot
lock out — firing resets it and negative potentials always relax toward 0. So: fixed-point exponential
decay for `adapt`; `potential` leak unchanged.

## Revision (post-review): L0 transducer, and the `ADAPT_SHIFT` guard

Two issues surfaced in a critical review of the branch:

**L0 is now an explicit transducer.** With low baselines, L0 (the input surface) could fire on
recurrent drive rather than injection alone, and injected fires bumped its adaptation — coupling input
history to input excitability. Worse, because the ALIF effective threshold `baseline + (adapt >>
ADAPT_SHIFT)` can exceed `i16::MAX` while injection only sets `potential = i16::MAX`, a saturated L0
neuron could *silently swallow an injected input spike*. Fix: `Network::new` forces layer 0 to baseline
`i16::MAX` with `adapt_bump = 0`, so it fires only on injection, never adapts, and its effective
threshold stays `== baseline <= i16::MAX` (injection always fires). The ALIF dynamics apply to layers
`1..L`; calibration already skips L0. The stale `wave.rs` comment claiming `i16::MAX >= every possible
threshold` is corrected.

**The fixed-point fix has a bounded range, now guarded.** `ADAPT_SHIFT` gives dead-zone-free decay only
for `adapt_decay <= ADAPT_SHIFT`; beyond that the dead zone returns at the real scale. `ADAPT_SHIFT` was
raised from 8 to **12** (the i32 overflow bound is `SHIFT <= 14`, since worst-case `2·ADAPT_MAX =
i16::MAX << (SHIFT+1)` must fit `i32`; 12 leaves ~8× margin and allows τ up to ~4096 waves), and
`Config::validate` now rejects `adapt_decay == 0 || adapt_decay > ADAPT_SHIFT`.
