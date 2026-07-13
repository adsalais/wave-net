# wave_driven Phase 2b — ALIF adaptation eligibility (εᵃ), spike-ψ

- **Date:** 2026-07-13
- **Status:** design approved; ready for an implementation plan
- **Scope:** Phase 2b — add the **ALIF adaptation-eligibility term `εᵃ`** (Bellec 2020) to `wave_driven`'s
  online trainer, using **spike-ψ** (the activity-scaled form), to unlock **recurrence / side-car**
  training. Builds on Phase 2a (online membrane e-prop, spike-ψ), merged on `main`. **bump-ψ (with
  decide-time snapshots) is a deferred fast-follow** (Phase 2b-2), taken only if the experiment shows
  spike-ψ + a width sweep cannot unlock recurrence.

## Motivation

Phase 2a's membrane eligibility (`e_ij = Σ_{t: j fires} pretr_i`) is a reliable **feed-forward**
learner but does not carry the slow adaptation state. The project's headline recurrence result —
**trained recurrence robustly beats feed-forward** on the backward-fed side-car (temporal XOR 990→1000,
flip-flop 985→1000, distractor-XOR 700→**995**, parity N=4 587→**837**) — is trained by the *completed
ALIF eligibility* `e = ψ·(εᵛ − β·εᵃ)`, where `εᵃ` is a slow per-synapse adaptation-eligibility recursed
at the adaptation rate `ρ`. Phase 2b brings that `εᵃ` term online on the activity-scaled engine.

**One caveat is load-bearing (see Open question):** the historical win used **bump-ψ**, and the recurrent
hub operates **near threshold** — the regime bump-ψ is tuned for (bump-ψ *collapses* only under strong
forward drive, where potentials overshoot `eff` by more than `W`). Spike-ψ has no such gap and is robust,
but whether spike-ψ + `εᵃ` reproduces the recurrence win is an **open experiment**. This phase runs it.

## The one idea

> Add a per-synapse **`εᵃ_ij`** trace recursed at the layer's adaptation rate `ρ`. With spike-ψ it
> updates only from spikes: on a target spike `εᵃ` takes an injection, otherwise it just decays at `ρ`.
> The eligibility gains a **silent-source coupling** term `−β·εᵃ_ij` that fires even when the source is
> long silent — so the accrual scans a slightly larger active set. `β = 0` reduces **exactly** to Phase
> 2a.

## The rule (spike-ψ, exact)

Per synapse `i→j` (source `i` in layer `z`, target `j` in `tz = z + edge.level`), each wave, with
`ρ = 1 − 2^(−adapt_decay)` taken from the **target layer `tz`** (since `εᵃ` tracks target-`j`'s
adaptation eligibility; see Data flow), and `ψ_j = [j fires this wave] ∈ {0,1}`:

```
j fires:   e_ij += (pretr_i − β·εᵃ_ij);   εᵃ_ij := pretr_i + (ρ − β)·εᵃ_ij
j silent:  εᵃ_ij := ρ·εᵃ_ij               (no contribution to e_ij)
then:      if |εᵃ_ij| < ε_a  →  εᵃ_ij := 0   (ε_a cutoff: bounds εᵃ + keeps the offline oracle exact)
```

This is the spike-ψ specialization of `elig_adapt_sum` (`multilayer_dfa::elig_adapt_sum`): with
`ψ ∈ {0,1}`, `ψ·pretr` is the injection and `(ρ − β·ψ)` is the decay (`ρ` when silent, `ρ−β` on a
spike). `pretr_i` is Phase 2a's ε-thresholded presynaptic trace. **`β = 0`**: the contribution is
`ψ·pretr_i` and `εᵃ` never enters `e_ij` — identical to Phase 2a (a regression gate), so `εᵃ` upkeep is
**gated off entirely when `β = 0`**.

**Silent-source coupling.** When `j` fires, `e_ij` receives `−β·εᵃ_ij` **even if `pretr_i = 0`** (the
source fired long ago; its `pretr` decayed but `εᵃ` has not). This is the whole point of `εᵃ` — it
carries credit across the adaptation timescale — and it dictates the scan set below.

## Data structures & data flow

- **`TrainState.eps_a: Vec<f32>`** — per-synapse `εᵃ`, layout identical to `shadow`/`elig`
  (`ls·total_slots`). Allocated by `enable_training` **only when the active `EligParams.elig_beta ≠ 0`**;
  otherwise absent (Phase 2a footprint). Doubles-again the training memory when present; freed by
  `disable_training`.
- **`elig_active[z]: Frontier`** — the accrual scan set: **sources that have fired since the last
  reset**. A superset of Phase 2a's `pretr_active` (which decays in `~rec_tau`), because a source with a
  live `εᵃ` tail must keep being scanned for the `−β·εᵃ` coupling even after its `pretr` hits 0. Firers
  are added each wave; the set is cleared by `reset_eligibility`. **Because training resets every trial,
  this is naturally bounded by trial length** — no horizon-expiry bookkeeping is needed (that is only for
  continuous streaming; deferred — see Non-goals).
- **`EligParams`** gains `elig_beta: f32` (β) and `epsilon_a: f32` (the `εᵃ` cutoff). `ρ` is **not** a
  param — it is derived per layer from that layer's `adapt_decay` (`ρ = 1 − 2^(−adapt_decay)`), matching
  the engine's own adaptation decay so `εᵃ` tracks the same timescale.

**Which layer's `ρ`?** `εᵃ_ij` tracks the *target* `j`'s adaptation eligibility, so it decays at the
**target layer `tz`'s** `ρ`. In the FF/side-car configs all computational layers share one `adapt_decay`,
so this is unambiguous, but the implementation reads `ρ` from layer `tz`, not `z`, to be correct under
heterogeneous `adapt_decay`.

## Online accrual (extends Phase 2a's source-driven scan)

Per wave, when training and the active `β ≠ 0` (the `β = 0` path is unchanged Phase 2a):

1. Same as 2a: fired-bitsets, `spike_count`, `pretr` update (decay → ε-drop → bump firers), and add
   firers to `elig_active[z]`.
2. **εᵃ accrual** — for each source `i` in `elig_active[z]`, scan its fan-out (the Phase-1 word-scan);
   per synapse `i→j` in target layer `tz`: read `pretr_i` (0 if its trace decayed), apply the
   spike-ψ recursion above using `ρ` from layer `tz`, ε_a-cutoff, and accrue `e_ij` (marking `dirty_rows`)
   when `j` fired. **Eager, not lazy:** outgoing-only topology forces visiting each synapse every wave
   anyway (to test whether `j` fired), so there is nothing to skip by deferring the `ρ`-decay — eager is
   simpler and matches the oracle exactly.

Cost per wave ≈ `O(fired-since-reset sources × fan_in)` — activity-scaled, with a longer tail than 2a's
`pretr`-active set (bounded by the trial). `reset_eligibility` also zeroes `eps_a` over `elig_active` /
`dirty_rows`.

## Validation

1. **`online ≡ dense` with `εᵃ`, bit-for-bit (primary oracle).** Extend `dense_eligibility` with the
   `εᵃ` recursion (spike-ψ) using the **identical** `ρ`, `β`, and `ε_a`, summed in wave order. The
   engine's online `elig` must equal it exactly (float ops in the same order; the `ε_a` cutoff is applied
   identically on both sides). **Hard gate.**
2. **`β = 0` ≡ Phase 2a, exactly.** With `β = 0`, `elig` and `codes` after a training run must match a
   Phase-2a run bit-for-bit (regression gate that the `εᵃ` path is truly off). **Hard gate.**
3. **Determinism.** A fixed `(seed, config, task-seed)` training run reproduces identical `shadow`/`codes`.
   **Hard gate.**
4. **Side-car FF-vs-recurrence experiment** (`#[ignore]`, `--release`) — the research finding. Port the
   backward-fed side-car builder and the benchmark tasks (temporal XOR, flip-flop, distractor-XOR, parity
   N=4), train with spike-ψ `εᵃ` (β ≈ 0.4), and **report FF vs side-car per task, worst + mean over
   seeds**. The experiment sweeps two recurrent axes — **width** (size 16→32→64) and **`rec_count`
   density** (into and beyond the historical bump-ψ cliff ~12) — and instruments each config with the **σ
   branching ratio** and **per-layer spiking profile** (per the benchmark convention), so a collapse can
   be classified as **dynamics** (σ → super-critical; activity explodes/dies) vs **credit** (activity
   healthy, accuracy poor). The engine is proven correct by the oracle regardless of outcome; this
   experiment answers *whether spike-ψ + `εᵃ` unlocks recurrence, and at what recurrent operating point*.

## Open question and the convergence ladder (negative-result handling)

The recurrence win was a **bump-ψ** result at the near-threshold hub; spike-ψ is a **coarser, sparser**
surrogate gradient. If the side-car does **not** beat FF, do **not** conclude spike-ψ is insufficient —
converge first (project rule: don't dismiss on a weak implementation). The ladder, in order:

1. **Re-locate the recurrent operating envelope under spike-ψ (width AND density).** The historical
   `rec_count ≈ 12` collapse cliff was measured **under bump-ψ**, so it is *not* a proven property of the
   recurrence. Two mechanisms could produce it, and they predict opposite things here:
   - **Credit starvation (bump-ψ-specific):** higher `rec_count` → stronger drive → potentials overshoot
     `eff` by > W → **bump-ψ → 0 → eligibility starved**. spike-ψ has no overshoot gap, so **spike-ψ
     should tolerate higher `rec_count`** — and a denser recurrent hub gives a richer state that offsets
     spike-ψ's coarser per-unit credit. So **test `rec_count` above the old cliff**, don't pin it low.
   - **Dynamics collapse (σ):** higher `rec_count` → super-critical branching → activity runs away/dies,
     regardless of credit rule. ALIF adaptation holds σ near 1, so the *true* ceiling is the density that
     defeats adaptation. This is the real limit; the σ + spiking-profile instrumentation locates it.

   In parallel, **scale recurrent *width*** (size 16 → 32 → 64): spike-ψ's binary credit benefits from
   more spike events (denser accrual) and population-averaged DFA credit (lower variance), and width is a
   known capacity floor (≥ size 32 for parity N=4). So the side-car experiment sweeps **both** width and
   `rec_count`, instrumented with σ + per-layer spiking profile to distinguish the two collapse modes.
2. **Retune `β` / `rec_tau`** within reason (the plain side-car used β 0.4; topology ≫ hyperparameters,
   so keep this light).
3. **Only then, bump-ψ (Phase 2b-2).** If a re-located width/density envelope + light tuning still can't
   unlock recurrence, the conclusion is that the graded near-threshold signal is load-bearing — hand off
   to the bump-ψ fast-follow (re-add decide-time pot/eff snapshots for frontier neurons + `use_bump`),
   which is a separate spec. Report the spike-ψ finding either way.

## Module & API touch-points

```
src/wave_driven/
  neurons.rs    + TrainState.eps_a (allocated only when β≠0); enable_training takes/uses the β to decide
  training.rs   + EligParams { .., elig_beta, epsilon_a }; dense_eligibility extended with the εᵃ recursion
  network.rs    + elig_active work-set; accrue_eligibility extended (εᵃ path gated on β≠0); reset zeroes eps_a
src/bench/
  wave_driven_bench.rs  + side-car builder + tasks (temporal XOR / flip-flop / distractor-XOR / parity N=4)
                          + FF-vs-side-car experiment: width × rec_count axes, σ + spiking-profile instrumented (#[ignore])
```

`EligParams` gains fields (existing 2a call sites set `elig_beta: 0.0`, `epsilon_a: <default>` — or rely
on `Default`). `enable_training` learns the active `β` (via the already-set `elig_params`) to decide
whether to allocate `eps_a`; `set_elig_params` before `enable_training` in the β>0 path.

## Non-goals (Phase 2b)

- **bump-ψ / decide-time snapshots** (Phase 2b-2, only if the ladder needs it).
- **Continuous-streaming `εᵃ` horizon expiry** — not needed while training resets per trial; a
  fixed-window expiry (reusing `fire_wave`) is the streaming fix, deferred.
- **Lazy `εᵃ` reconstruction** — pointless under outgoing-only topology (see accrual); revisit only with
  reverse adjacency (Approach B, itself deferred).
- Shadow persistence; GPU.

## Risks

- **Research risk (primary):** spike-ψ + `εᵃ` may not reproduce the recurrence win. Mitigated by the
  convergence ladder (width first); the outcome is reported, not hidden.
- **Memory:** `eps_a` adds another `O(size²·count)` f32 when `β>0` (shadow + elig + eps_a). Freed on
  `disable_training`; expected and bounded.
- **Longer scan tail:** `elig_active` (fired-since-reset) is larger than 2a's `pretr_active`; within a
  short training trial it approaches "all sources that fired this trial." Still `≪ size²·count` for a
  sparse net, and bounded by trial length; the training-throughput experiment measures the real cost.

## Appendix — copied vs new

- **Ported from `wave_bitnet` (adapted):** the `εᵃ` recursion (`elig_adapt_sum`, spike-ψ specialization);
  the side-car builder + benchmark tasks (from `benches/throughput_bitnet.rs` / `wave_bitnet_bench`).
- **New:** `TrainState.eps_a`; `elig_active` work-set; the online `εᵃ` accrual (spike-ψ, eager,
  ε_a-cutoff, silent-source coupling); the `εᵃ`-extended `dense_eligibility` oracle; `EligParams` fields.
- **Deliberately NOT built:** bump-ψ + decide snapshots; streaming horizon expiry; lazy `εᵃ`.
