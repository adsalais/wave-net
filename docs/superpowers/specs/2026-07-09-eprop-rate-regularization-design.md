# Firing-rate regularization in the e-prop learning signal — design

**Date:** 2026-07-09
**Status:** approved (approach), pre-plan
**Scope:** keep every trained layer *alive* (near a target firing rate) by folding a **firing-rate
regularization term into the e-prop learning signal** — the standard LSNN / e-prop move — so the *same*
training that learns the task also prevents dead/saturated layers. Replaces the abandoned external
"criticality gain calibration" approach (below). Integrated into `train_eprop` (the multi-layer
feed-forward path, where the dead-deep-layer failure lives); tested against the depth-16/20 wall. All work
in `bench::rsnn`; **no engine change** (the per-neuron spike count needed for the rate already exists as
`Layer.elig_pre`).

## Why (and what we abandoned)

A prior branch tried to reach criticality with a **separate, external gain calibration**: a per-(layer,
level) scalar `β` outside weight storage, tuned by measuring a branching ratio and inverting it. It was
discarded. Three findings carry forward (so we do not repeat them):

- **The analytic branching estimator (`Σ|w|·β / θ`) is invalid** — it is a *static* weight/threshold ratio,
  blind to whether spikes actually propagate. It reported branching **16** (wildly super-critical) for deep
  layers that *never fire* (measured rate 0.00). Driving a controller with it *cut* forward gain 16× and
  killed the reservoir → uniform chance across all depths/targets.
- **A separate measure-and-invert gain calibration is a non-standard bolt-on.** The field does **not**
  calibrate a trained RSNN to criticality as a distinct phase. It (a) initializes sensibly, and (b) folds a
  **firing-rate regularization** into the loss, co-trained by the same rule (LSNN, Bellec et al. 2018/2020);
  memory comes from **adaptation** (ALIF) or learned structure (BPTT), not from sitting at σ = 1.
- **Positive control:** with the gain machinery *off*, `train_eprop` cleanly reproduced the documented
  depth ceiling (size 16, multi-layer: depth 16 → 1000, depth 20 → 485). The harness is trustworthy; only
  the external-calibration idea was wrong.

So we adopt the field-standard mechanism: a rate regularizer **inside** the e-prop learning signal. It
attacks the actual failure mode (dead deep layers) directly, with no external estimator and no `β`.

## Mechanism

`train_eprop` already computes, per trained source layer `z → tz = z+1`, a **task** learning signal per
target neuron `j`:

```
L_j^task = Σ_c b_{jc} · err_c        (b = symmetric readout feedback for the top, DFA for deeper layers)
Δw_ij   = −lr · L_j^task · e_ij ,     e_ij = pre_i · ψ_j     (pre = elig_pre, ψ = elig_post)
```

Add a **rate** learning signal to the *same* per-neuron signal, applied through the *same* eligibility:

```
r_j       = elig_pre_tz[j] / n_waves            (neuron j's mean firing rate this trial)
L_j^reg   = c_reg · (r_j − r_target)
Δw_ij     = −lr · (L_j^task + L_j^reg) · e_ij
```

- **Direction is self-correcting.** A too-quiet neuron (`r_j < r_target`) makes `L_j^reg < 0`, so
  `−lr·L_j^reg·e_ij > 0` for excitatory eligibility → its incoming weights *rise* → it fires more. A
  too-active neuron is pushed down. It relaxes each trained neuron toward `r_target`.
- **Revival cascades bottom-up.** The eligibility gates the signal by `pre_i` (the feeder's spikes), so a
  stone-dead layer is revived only once its *feeder* is alive — but L1 is always alive (fed by the L0
  transducer), so reviving `L1→L2` wakes L2, which lets `L2→L3` credit flow, and the liveness climbs the
  stack. This is exactly the propagation the invalid static estimator could not see. (The near-threshold
  box-ψ is already nonzero for the deep layers, because calibration drives their thresholds to ~1 and the
  `PSI_BAND` is wide, so `ψ_j > 0` even at rest — credit *can* reach them once `pre_i > 0`.)
- **`r_j` from existing state.** `elig_pre[j]` is neuron `j`'s spike count over the trial (accrued in
  `wave.rs` decide, zeroed in `reset_state`). No engine change; the training loop already clones `elig_pre`
  for the source layer — it additionally clones the *target* layer's `elig_pre` for `r_j`.

## Config (`RsnnConfig`)

- `rate_reg: f32` — the coefficient `c_reg` (**0 = off**, default → `train_eprop` byte-identical to today).
- `rate_target_permille: u32` — `r_target` in permille (e.g. 100 = 10%), matched to the calibration target.

`n_waves = present_waves + delay + read_waves` (the trial length the rate normalizes by).

## Integration

`train_eprop` only: inside the `if cfg.hidden_lr != 0.0` block, for each trained layer, after fetching
`pre`/`psi`, also fetch the target layer's `elig_pre` and add `L_j^reg` to the existing `l_sig[j]` before
the weight update. `rate_reg = 0.0` skips the term entirely (guard), so all current results are unchanged.
(`train_recurrent` deferred — feed-forward depth is the clean testbed.)

## Success criterion

- **Revives the dead deep layers:** with `rate_reg > 0`, a per-layer firing-rate probe over a held-out trial
  shows layers that were silent (rate ~0) now fire near `r_target` through the full depth — the direct test
  that liveness propagated.
- **Pushes the depth wall (the payoff):** worst-seed held-out at depth 20 (where the doc's ceiling is ~485)
  is clearly above chance + margin with `rate_reg > 0`, versus ~485 without — held-out, multi-seed, trial
  length scaled to depth.
- Determinism: pure function of `(seed, task_seed, config)`.

**Honesty gate.** Two failure readings, each pointing somewhere specific — report which, never a single
seed:
1. **Layers revive (fire near `r_target`) but depth-20 accuracy stays at chance** ⇒ liveness was *not* the
   wall; the blocker is the **credit rule** (DFA feedback noise credits deep layers too weakly) → the lever
   is symmetric feedback / surrogate-gradient BPTT, as the doc has long suspected. A clean, informative
   result.
2. **`c_reg` large enough to revive layers also homogenizes activity → accuracy collapses to chance** ⇒ the
   regularizer is overpowering the task signal (the failure mode the external `β` solve hit). Report the
   `c_reg` sweep; the usable window (if any) is where layers are alive *and* class structure survives.

## Determinism & constraints

- Engine untouched and integer/deterministic; the regularizer is bench-side `f32`. Single-threaded.
- **`rate_reg = 0.0` must be byte-identical to current `train_eprop`** — the whole suite (incl.
  `wave_state_machine`) stays green.
- `wave_state_machine` frozen; held-out + multi-seed from the start.

## Testing

- `rate_reg_off_is_identity` — `rate_reg = 0.0` gives the same held-out permille as the current
  `train_eprop` on a small config (regression / byte-identity of the default path).
- `rate_reg_path_is_deterministic` — `rate_reg > 0` is a pure function of `(seed, config)`.
- `rate_reg_revives_dead_layers` (`#[ignore]`, release) — per-layer rate probe: deep layers silent without
  reg fire near `r_target` with it.
- `rate_reg_depth_wall` (`#[ignore]`, release) — the depth-20 headline, worst-seed multi-seed, with a
  `c_reg` sweep.

## Deferred

- **EMA rate estimate** (running per-neuron rate across trials) if the per-trial rate proves too noisy.
- **Rate regularization on `train_recurrent`** (the recurrent path) once the feed-forward result is in.
- **Per-neuron intrinsic-plasticity / threshold homeostasis** as an alternative liveness mechanism (adjust
  excitability instead of incoming weights) — a different, also-field-standard route if the weight-side reg
  fights the task.
- **Refractory tuned to the loop delay (not depth)** for the recurrent path, if self-echo double-counts
  eligibility — a small constant (~loop delay). Never depth-scaled: refractory `R` caps the max firing rate
  at `1/R`, so `R ≥ 10` makes the ~10% calibration target unreachable and recreates the cold-layer
  pathology. Adaptation (ALIF) is the graded first resort.
- Consolidate the result (and the abandoned-criticality-calibration findings above) into
  `docs/experiments_results.md` once the experiments have run.
