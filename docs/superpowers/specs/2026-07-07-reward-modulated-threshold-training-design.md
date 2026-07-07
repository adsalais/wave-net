# Reward-modulated per-neuron threshold training — design

**Date:** 2026-07-07
**Status:** approved (brainstorming complete)
**Scope:** Spec 3 continuation — a second per-neuron learning rule on the `wave_net` island.

## Program context

`wave-net` turns a fixed, hash-wired **wave reservoir** into a trained RSNN. Two engine facts
shape every learning design (see AGENTS.md and the criticality spec):

1. **No weights to train.** Synapses are a pure function of the hash; the ±1 sign is fixed.
   Trainable state is a per-neuron vector (threshold, additive field), O(N) not O(synapses).
2. **Non-differentiable integer engine.** Hard threshold + saturating i16 math. Gradient-free
   rules (node perturbation, reward-modulated plasticity) run directly on the integer spikes.

First Spec-3 result already landed: node perturbation on a top-layer additive **field** lifted
held-out temporal-XOR from ~0.62 to ~0.88. This spec adds the **second** trainable parameter
(per-neuron **threshold**) and a **second trainer** (reward-modulated plasticity with an eligibility
trace). Its sample-efficiency promise is realized by the fully-online v2; v1 first *validates the
credit signal* under a controlled harness (see #3). Both run on the `wave_net` island — the
self-contained engine fork that exists so training experiments can freely modify the engine while
`wave_reservoir` stays frozen.

An inhibition sweep (`examples/inhibition_sweep.rs`) established that **layer size matters**:
32×32×6 beats the 16×16 demo at every inhibition level, on both baseline and trained accuracy.
This spec's experiment therefore runs at 32×32.

## Goal

Make per-neuron thresholds trainable (they are currently frozen hash jitter), and train them with a
**reward-modulated rule** whose per-neuron credit is a **centered global scalar reward gated by a
near-threshold eligibility trace**. Raise held-out temporal-XOR above baseline — and, crucially,
show the gain survives the controls (*Controls and interpretation*), not just the baseline. Ship v1
as the global-scalar rule, with interfaces shaped so a spatially-**propagated** reward wave can drop
in later without an engine change. Layer scope (full-depth vs top-first) is an experimental axis,
not a fixed assumption (see reopened #4).

## Decisions locked in brainstorming

- **Reward signal:** start with a **global scalar** (readout correct/wrong per bit), broadcast to
  eligible neurons; design the interfaces so a **propagated reward wave** is a later swap.
- **Reward baseline (centering):** the per-neuron gradient uses the **centered** reward
  `r(t) − r̄`, not the raw reward. Without this, the constant `r̄·ē_i` term makes the rule a global
  excitability knob rather than a credit-assignment rule (see Component 3). Added in review.
- **Controls are mandatory:** the experiment is only interpretable against the **random-proposal**,
  **masked-random**, and **shuffled-reward** controls (see *Controls and interpretation*). Added in
  review.
- **Threshold hook:** fold the trainable delta into the **stored** threshold between trials (zero
  hot-path cost), *not* an extra add in the decide loop.
- **Layer size:** 32×32×6.

**Resolved in review (round 2):**

- **Trainer loop (#3): keep-if-better harness now, fully-online deferred to v2.** v1 fixes the
  keep-if-better outer loop as a *measurement instrument* and varies only the proposal strategy
  across arms (see *Controls and interpretation*). The fully-online rule — where the real
  sample-efficiency win lives — is premature until the arms show the gradient beats its controls,
  and it needs the deferred rate-monitoring; it becomes v2.
- **Layer scope (#4): an experimental axis, not a fixed choice.** Each arm runs at **top-only** and
  **full-depth**. The "deep credit is noise" vs "deep layers have unique leverage" tension is
  settled empirically: if full-depth's edge over top-only vanishes under the shuffled-reward
  control, the deep gradient is noise; if it survives, deep training has real unique leverage.

## Out of scope (this spec)

- **Leak-last dynamics** (leak on the trailing edge instead of the leading edge). Deferred to a
  later, standalone experiment.
- **Propagated reward wave** (spatial credit through the hash topology). Interfaces are built
  forward-compatible; the mechanism is not.
- **Harder-task sweep** (τ=2-3, parity-3, NARMA-10) and the **ES optimizer** upgrade — the
  follow-on program once this rule is validated.
- Any change to `wave_reservoir` (frozen reference).

## Reuse (unchanged)

From `wave_net`'s own engine copy: `config::IntConfig`, `hash::{key, mix, P_THRESHOLD}`,
`index::Dims`, `wiring::for_each_layered`, `pipeline::LayerNet`, plus the toolkit
(`calibrate`, `stream`, `readout`, `train::{add_field, hill_climb, Outcome}`). The reward trainer
reuses the `Outcome` type and the honest TRAIN/VAL/TEST protocol from `examples/field_training.rs`.

## Component 1 — trainable thresholds (engine, `pipeline.rs`)

Today `LayerCfg::threshold: Vec<i16>` is computed once in `LayerNet::new`
(`threshold[i] = threshold_base + (hash_jitter(i) − offset)`, pipeline.rs:108-115) and frozen; the
decide step reads `potential[i] >= threshold[i]` (pipeline.rs:226).

**Change:**
- Keep the frozen hash-jittered values as an immutable `threshold_frozen: Vec<i16>` per layer.
- Add a per-neuron trainable delta `theta`. New method:

  ```rust
  pub fn set_threshold_delta(&mut self, theta: &[i16])   // length n_total()
  ```

  recomputes each layer's effective threshold: `threshold[i] = (threshold_frozen[i] as i32 +
  theta[i] as i32).clamp(1, i16::MAX as i32) as i16`.

- The decide loop is **unchanged** — it keeps reading `threshold[i]`. Zero hot-path cost;
  reconfiguration is one `&mut self` pass between trials (single-threaded, no locks). `theta`
  all-zero reproduces today's behavior bit-for-bit.

**Rationale:** folding the delta into the stored threshold keeps the hottest loop byte-identical
and makes reconfiguration trivially cheap, versus an extra `+ theta[i]` per neuron per decide.

## Component 2 — eligibility capture (engine, `pipeline.rs`)

The "near-threshold" boolean, exposed as a hook parallel to the existing `on_layer`:

```rust
pub fn on_layer_eligibility(
    &mut self,
    layer: usize,
    margin: i16,
    listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>,
)
```

At decide, when a layer has an eligibility hook, the **same loop** that finds firers also collects
locals where `(potential[i] − threshold[i]).abs() <= margin`, captured **before** the fire-reset
zeroes `potential`, and emits them as `listener(wave_id, &eligible_locals)`. Eligible = the pivotal
band around threshold (marginal firers **and** marginal near-misses); neurons that fired hard
(`potential ≫ threshold`) or sat far below are excluded — nudging their threshold would not change
their behavior. This is a one-bit integer surrogate gradient.

**Cost:** zero when no eligibility hook is registered (lazy, exactly like the current listener:
"nothing assembled if unsubscribed"). When registered, one `abs`+compare per neuron in the decide
loop for that layer.

**Storage note:** eligibility is computed from the pre-reset `potential` and the neuron's
`threshold`; both are already in hand in the decide loop, so no new per-neuron state is stored in
the engine — the hook streams the eligible set out, and the *trainer* accumulates it per bit (just
as the field experiment accumulates firing features per bit).

## Component 3 — the reward-modulated trainer (`train.rs`)

New function beside `hill_climb`, reusing `Outcome`:

```rust
pub struct RewardParams {
    pub iters: usize,
    pub lr: i16,        // threshold-delta step magnitude per update
    pub clamp: i16,     // bound on |theta[i]|
    pub margin: i16,    // eligibility band (passed through to on_layer_eligibility)
}

/// evaluate(theta) runs the reservoir with that threshold delta and returns
/// (selection_reward, per_neuron_gradient) where the gradient uses the CENTERED reward:
///   gradient[i] = Σ_t (reward(t) − r̄) · eligible_i(t),   r̄ = mean reward over the scored bits.
pub fn reward_modulated(
    init: Vec<i16>,
    cfg: &RewardParams,
    evaluate: impl Fn(&[i16]) -> (f64, Vec<f64>),
) -> Outcome
```

Per iteration:
1. `(r0, g) = evaluate(&theta)` — selection reward + per-neuron centered-reward gradient at the
   current point.
2. Candidate: `theta' = clamp(theta − lr · sign(g[i]), −clamp, clamp)` per neuron.
   **Sign:** a neuron whose eligibility correlates with *above-average* reward (g[i] > 0) gets its
   threshold *lowered* so it fires more readily (reinforce the useful pattern); anti-correlated
   (g[i] < 0) gets raised. Hence `−sign(g)`. `g[i] = 0` (uncorrelated) → no move.
3. `(r1, _) = evaluate(&theta')` — "replay and measure the effect".
4. **Keep-if-better:** accept `theta'` iff `r1 > best`. Record best-so-far in `Outcome.history`
   (non-decreasing, like `hill_climb`).

### Why the reward must be centered (review #1)

Decompose the raw-reward gradient: `Σ_t r(t)·eligible_i(t) ≈ N·r̄·ē_i + N·Cov(r, eligible_i)`.
At a ~62% baseline `r̄ > 0`, so the first term is **positive for every eligible neuron** — it
lowers all their thresholds roughly uniformly, which is a global excitability increase, not
learning. The real credit signal is entirely in `Cov(r, eligible_i)`: does neuron i's
near-threshold-ness *differ* between correct and incorrect bits? Centering (`r(t) − r̄`) cancels
the constant term so only the covariance drives updates. This is the REINFORCE baseline /
variance-reduction trick; v1 uses the **batch mean** `r̄` over the scored bits (a per-neuron or
moving baseline is a possible later refinement).

The `evaluate → (reward, Vec<f64>)` signature carries a **per-neuron** vector, so replacing the
global-scalar-derived gradient with a propagated-credit-derived gradient later is a change to the
*experiment's closure*, not to the trainer or the engine. That is the concrete meaning of
"design for both".

## Component 4 — the experiment (`examples/reward_threshold.rs`)

Reward-modulated threshold training as a controlled **arm sweep**, mirroring `field_training.rs`'s
honest harness. The shared substrate and task:

- **Net:** `IntConfig::demo()` with `w = h = 32` (32×32×6, N = 6144). Calibrate the substrate once
  (fixed), then train only the threshold delta.
- **Task:** temporal-XOR τ=1 (comparable to the field result). Same WPB / WASHOUT / TRAIN / VAL /
  TEST split constants.
- **Readout:** the existing `OnlineReadout` on the top layer's per-bit firing features.
- **Eligibility:** `on_layer_eligibility(z, margin, …)` on the trained layers, accumulating a
  per-neuron, per-bit eligibility count (length = trained-neuron count per bit).
- **`evaluate(theta)`:** `set_threshold_delta(theta)`, `reset_state`, run the stream; the readout
  trains on TRAIN; reward per bit `r = +1` if `predict>=0.5` matches target else `−1`; centered
  gradient `g[i] = Σ_{t∈TRAIN} (r(t) − r̄) · eligibility_i(t)` with `r̄` the mean reward over TRAIN
  bits; the returned selection reward is VAL accuracy (the keep-if-better selector). TEST is never
  read inside `evaluate`.

### The arm matrix

`{proposal strategy} × {layer scope} × {seed}`, every cell sharing the keep-if-better harness and
the honest split:

- **Proposal strategies (4):** `gradient` (the real centered-reward rule) · `random` (node
  perturbation — kick random neurons, ignore eligibility+reward) · `masked-random` (kick random
  neurons *among the eligible* — isolates targeting from reward direction) · `shuffled` (gradient
  with `r(t)` permuted across bits — isolates reward correlation from the mask).
- **Layer scopes (2):** `top-only` (train the top layer's θ; others frozen) · `full-depth` (train
  all N). The θ vector is length-N throughout; scope just zeroes and freezes the untrained entries.
- **Seeds (≥3):** independent task/input/perturbation seed triples, as in `inhibition_sweep.rs`.

= 4 × 2 × 3 = **24 runs**, cells parallelized (one thread each), each a `reward_modulated`
(or `hill_climb` for the random arms) call. Rough cost ~15–30 min at 32×32; `random`/`top-only`
cells are cheaper (no gradient, fewer eligible layers).

- **Report:** a mean-over-seeds table of baseline vs trained TEST per (strategy, scope), plus the
  VAL–TEST gap; θ range and biased-neuron count for the `gradient` arms.

**Honesty:** readout trains on TRAIN; the eligibility gradient is computed on TRAIN; keep-if-better
selection uses VAL; the headline number is TEST, never selected on — identical discipline to the
field experiment, so any reported gain is real generalization.

## Controls and interpretation (mandatory) — review #2

A single trained-vs-baseline number is **uninterpretable**: a gain could come from the
reward-modulated credit, from the global excitability change centering is meant to remove, or from
the keep-if-better search improving a weak baseline *regardless of proposal quality*. The
experiment therefore runs the real rule alongside three controls under the **same keep-if-better
harness**, differing only in how the candidate θ' is proposed / scored (the four proposal
strategies of the arm matrix above):

- **`random`** = node perturbation on thresholds: kick random neurons (ignore eligibility and
  reward), same accept/reject. Isolates whether targeting + reward beat blind search at all.
- **`masked-random`** = kick random neurons *among the eligible* (near-threshold), ignoring the
  reward direction. Sits between `random` and `gradient`: it separates the benefit of **targeting
  the pivotal neurons** from the benefit of the **reward direction**.
- **`shuffled`** = compute `g` with the per-bit reward `r(t)` **randomly permuted across bits**,
  destroying the true credit while preserving the eligibility mask and the reward distribution.
  Isolates whether the *reward correlation* (not the mask alone) is doing the work.

**Interpretation gate.** The mechanism is credited only if, on TEST, by more than the ≥3-seed
spread:

- `gradient > random` — the rule beats blind search;
- `gradient > shuffled` — the *reward correlation* (not just the eligibility mask) carries signal;
- and, informatively, `gradient` vs `masked-random` attributes the gain between **targeting** and
  **reward direction**.

`gradient ≈ shuffled` means the credit signal is noise; `gradient ≈ masked-random` means the reward
direction adds nothing over just targeting near-threshold neurons; `gradient ≈ random` means
eligibility itself adds nothing. Each is a publishable *negative* result that would redirect the
program — which is why the controls run.

**Layer-scope readout (#4).** Comparing `top-only` vs `full-depth` within the `gradient` arm, net
of the `shuffled` arm at each scope, settles the deep-credit question: if full-depth's edge over
top-only survives the shuffled control, deep-layer training has real unique leverage; if it
vanishes, the deep gradient is noise and top-only is the right operating point.

## Verification

**Engine (`pipeline.rs` tests):**
1. `set_threshold_delta(vec![0; n])` leaves every neuron's firing bit-identical to a net with no
   delta call (same golden trajectory).
2. A uniform negative delta raises firing rate, a uniform positive delta lowers it (monotone in
   the expected direction) on a fixed drive.
3. Eligibility hook on a hand-constructed tiny net emits exactly the locals within
   `|potential − threshold| ≤ margin` at decide, and nothing when unsubscribed.
4. The eligibility stream is **deterministic across thread counts** (emitted under the layer lock,
   in wave order, like the spike listener) — a differential check at threads `[1,2,4]`.
5. All existing pipeline tests still pass (`threaded_matches_sequential_all_thread_counts`,
   `top_layer_trajectory_golden`, etc.) — the hot path is unchanged.

**Trainer (`train.rs` tests):**
6. Centering: on a fixed eligibility matrix with a reward that is constant across bits, the
   gradient is **all-zero** (the baseline cancels the constant) — the direct guard for review #1.
7. On a toy reward whose *centered* gradient points at a known target θ*, `reward_modulated` drives
   θ toward θ* and `history` is non-decreasing (mirrors `hill_climb_improves_on_a_quadratic`).

**Experiment:**
8. The `gradient` arm beats **both** the `random` and `shuffled` arms on TEST by more than the
   ≥3-seed spread (the interpretation gate; reported, not asserted as a unit test). A bare
   trained-vs-baseline gain is explicitly *not* sufficient.

## The trainable-neuron mask (shared by scope + masked-random)

Both the layer-scope axis and the `masked-random` arm need "only these neurons may move":

- **Scope mask** — `top-only` trains only top-layer θ, `full-depth` trains all N. Implemented by
  registering eligibility only on the trained layers (so untrained neurons get `g[i] = 0`
  naturally) and, for the random arms, restricting perturbation to the scope's index set.
- **Eligibility mask (`masked-random`)** — the static set of neurons eligible at least once during
  the baseline (θ=0) run; random kicks are restricted to it.

So the random-proposal path needs a **masked** perturbation (kick only indices in a given set).
This is a small extension of `train.rs`'s existing `perturb` — an optional index mask, `None`
reproducing today's full-net behavior. The `gradient` path needs no mask beyond where eligibility
is registered.

## Files touched

- **Edit:** `src/wave_net/pipeline.rs` — split `threshold_frozen` from effective `threshold`; add
  `set_threshold_delta`; add `on_layer_eligibility` + the decide-loop eligibility scan; tests.
- **Edit:** `src/wave_net/train.rs` — add `RewardParams` + `reward_modulated`; add an optional
  index mask to the perturbation (for the `random` / `masked-random` arms); tests.
- **New:** `examples/reward_threshold.rs` — the 24-cell arm sweep (4 strategies × 2 scopes × 3
  seeds) at 32×32, parallelized like `inhibition_sweep.rs`, with the mean-over-seeds table.
- **Edit:** `src/wave_net/mod.rs` — doc line noting thresholds are now trainable, if warranted.
- Nothing in `wave_reservoir` changes.

## Performance / risk notes

- Threshold delta folded into stored threshold → decide loop byte-identical; no runtime regression.
- Eligibility scan is opt-in and only on subscribed layers; `full-depth` arms subscribe every
  layer, `top-only` arms just one — adding one `abs`+compare per neuron per decide on subscribed
  layers during training runs (acceptable for experiments).
- The 24-cell arm sweep runs `random`/`masked-random` via `hill_climb` (1 eval/iter) and
  `gradient`/`shuffled` via `reward_modulated` (2 evals/iter); parallelizing cells keeps wall-clock
  to ~15–30 min at 32×32.
- **Stability:** reward-modulated threshold lowering can run away (lower threshold → fire more →
  lower more). The near-threshold eligibility gate damps this (only pivotal neurons move), and the
  per-neuron `clamp` on θ bounds it. If rates still drift, the substrate calibration can be re-run
  or a homeostatic renormalization added — noted as a fallback, not built in v1.
