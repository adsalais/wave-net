# Rate-reg-sustained recurrence — keep the loop alive through the gap — design

**Date:** 2026-07-09
**Status:** approved (design), pre-plan
**Scope:** test whether firing-rate regularization (merged in `train_eprop`) unblocks **recurrence** by
keeping the recurrent loop alive *through* the silent gap, so temporal e-prop finally has activity to shape
into a persistent trace of the cue. Wire rate reg into the two recurrent trainers — `recurrent_update`
(level-0 lateral, via `train_xor`) and `train_recurrent` (backward level −1/−2) — and, on temporal XOR (LIF,
delay 20), run a **mechanism check** (does the loop stay alive through the gap?) then the **payoff** (does
live recurrence beat the feed-forward baseline?), for both substrates. Reuses the merged `rate_reg` /
`rate_target_permille` config; **no engine change**. All work in `bench::rsnn`.

## Why

Recurrence is a standing null: on LIF, after cue A the recurrent layer goes silent in ~6 waves, so across a
20-wave gap **no ψ has a trace to credit** — the blocker is the *sustaining dynamics*, not the credit rule
(established with the dead-readout and credit-rule confounds removed). At root that is a **liveness** problem,
and we just built and verified the field-standard fix for liveness: rate regularization
(`c_reg·(r_j − r_target)` in the e-prop learning signal) revived every dead layer in a depth-20 feed-forward
stack. Applied to the **recurrent** layer, it should keep the loop firing through the gap — turning "no
activity to credit" into "live activity the task can shape."

**The bootstrapping path:** the reg strengthens whatever recurrent connections fired during the cue → raises
recurrent gain → activity persists further into the gap → over training the sustain horizon grows. This
replaces the old `rec_init` bootstrap hack (a constant recurrent weight) with a *learned* sustain, so we set
`rec_init = 0`.

**The critical risk (carried from the depth finding):** rate reg adds **class-agnostic** activity, not
information. Keeping the loop firing does not by itself mean it *holds A* — the task e-prop signal must shape
the live activity to be A-specific. This experiment tests whether the two can coexist. It is the same
confound-removal move that isolated the depth wall: remove the liveness confound, then see what remains.

## Mechanism — rate reg on the recurrent learning signal (Approach A)

Both recurrent trainers already record per-wave fired-sets and build a per-neuron temporal eligibility
`e_ij = Σ_t pre_trace_i(t)·ψ_j(t)` with a per-target-neuron learning signal `L_j`. Add the reg term to that
signal, carried by the *same* eligibility:

```
r_j     = (Σ_t fired_j(t)) / n_waves          (target neuron j's firing rate over the recorded trial)
L_j    += c_reg · (r_j − r_target)
Δw_ij   = −lr · L_j · e_ij
```

- **`recurrent_update` (level-0 lateral):** `L_j` is the symmetric readout feedback for the L1 recurrent
  neurons; `r_j` from the recorded `waves` (L1 per-wave fired-sets it already builds). Reg pushes L1 toward
  `r_target`, keeping the lateral loop alive across the gap.
- **`train_recurrent` (backward):** `L_j` is the symmetric/DFA signal per target layer; `r_j` from the
  recorded `fired[tz]`. Reg pushes every recurrent layer toward `r_target`.
- **Guarded** by `rate_reg != 0.0`, so the existing recurrence results are byte-identical when off.
- Reuses `RsnnConfig.rate_reg` and `rate_target_permille` (already merged); no new config. `n_waves` is the
  recorded trial length (`present + delay + present + read` for the XOR trials).

Alternatives, deferred unless A stalls: **B** — a decoupled homeostatic weight update (synaptic-scaling /
intrinsic-plasticity) not gated by the task eligibility, if A's eligibility gating starves the gap;
**C** — a gap-windowed rate target (only count the silent gap in `r_j`) if the whole-trial average is too
diffuse.

## Experiment structure

For **each** substrate, temporal XOR, **LIF** (`adapt_bump = 0`, to isolate recurrence from the ALIF
incumbent), delay 20 (where FF is at chance), `rec_init = 0`:

1. **Mechanism check first** — a per-wave activity probe: does rate reg keep the recurrent layer firing
   *through* the 20-wave gap (vs dying in ~6 without)? The precondition, analogous to the depth-work revive
   probe. Print per-wave recurrent-layer spike counts across present → gap → present → read.
2. **Payoff** — held-out, multi-seed: FF baseline vs recurrence + rate reg, worst-seed.

Then **compare** the two substrates: does backward recurrence add sustain over lateral once both are kept
alive?

## Success criterion

- **Mechanism:** with rate reg, the recurrent layer's per-wave activity **survives the full gap** (nonzero
  spikes through wave ~present+delay), versus dying in ~6 waves without.
- **Payoff:** worst-seed held-out temporal-XOR accuracy with recurrence + rate reg is clearly above the
  (verified ~chance) FF baseline, multi-seed.
- Determinism: pure function of `(seed, task_seed, config)`.

**Honesty gate** — report which, never a single seed:
1. **Loop survives the gap AND XOR lifts above FF** ⇒ recurrence finally earns its keep (with liveness the
   enabler). A real result.
2. **Loop survives the gap but XOR still nulls** ⇒ the blocker is the **temporal credit rule** — the crude
   spike-timing eligibility can't shape live activity into an A-specific trace even when the substrate stays
   alive → surrogate-gradient BPTT, with the liveness confound now removed (the clean analog of the depth
   finding).
3. **Loop does NOT survive the gap even with reg** ⇒ Approach A's eligibility gating starves the gap → try
   Approach B (decoupled homeostatic sustain).

**Framing honesty:** LIF strips adaptation to isolate recurrence, so beating FF-LIF demonstrates the
*mechanism*; it does **not** beat the ALIF incumbent (which already solves XOR feed-forward). An
ALIF-vs-ALIF+recurrence arm at a longer delay (the real "beats the incumbent" test) is **deferred**.

## Determinism & constraints

- Engine untouched; the reg term is bench-side `f32`. Single-threaded, deterministic.
- **`rate_reg = 0.0` must be byte-identical** to the current `train_xor` / `train_recurrent` — the existing
  recurrence tests (`temporal_xor_ff_is_near_chance`, `recurrence_does_not_yet_beat_ff_on_temporal_xor`,
  `subthreshold_psi_is_deterministic`) stay green.
- `wave_state_machine` frozen; held-out + multi-seed from the start.

## Testing

- `rate_reg_recurrent_path_is_deterministic` — `rate_reg > 0` in `train_xor` / `train_recurrent` is a pure
  function of `(seed, config)`.
- `rate_reg_off_recurrent_is_identity` — covered by the existing recurrence tests staying green (guarded
  no-op).
- `lateral_gap_survival` (`#[ignore]`, release) — per-wave L1 activity through the gap, reg off vs on.
- `lateral_recurrence_vs_ff` (`#[ignore]`, release) — level-0 + rate reg vs FF on temporal XOR, worst-seed.
- `backward_recurrence_vs_ff` (`#[ignore]`, release) — backward + rate reg vs FF, and vs the lateral result.

## Deferred

- **Approach B** (decoupled homeostatic sustain) if A can't keep the loop alive.
- **Approach C** (gap-windowed rate target) if the whole-trial rate is too diffuse.
- **ALIF + recurrence vs ALIF-alone** at a longer delay — the real "recurrence beats the incumbent" test.
- **Surrogate-gradient BPTT** if the substrate stays alive but the temporal credit rule is the wall
  (honesty-gate case 2).
- Consolidate the result into `docs/experiments_results.md` once the experiments have run.
