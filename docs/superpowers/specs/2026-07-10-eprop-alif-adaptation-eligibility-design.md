# Complete the ALIF adaptation eligibility in e-prop — design

**Date:** 2026-07-10
**Status:** approved (design), pre-plan
**Scope:** the recurrence experiments established their null with an **incomplete** e-prop eligibility —
only the fast **membrane** term `e_ij = Σ_t εᵛ_i(t)·ψ_j(t)`. The slow **ALIF adaptation eligibility**
component (`εᵃ`, the `−β·εᵃ` term that carries credit over the ~64-wave adaptation horizon) was **never
implemented**, even though the substrate's memory *lives in* that adaptation variable. This adds it — the
faithful Bellec 2020 ALIF eligibility (Eq. 24–25) — plus the normalized bump pseudo-derivative `ψ` and
scaled `β` the recursion requires, then re-runs the ALIF recurrence benchmarks. All work in `bench::rsnn`
plus one **read-only** engine accessor; **no wave-dynamics change**, determinism preserved.

## Why

The standing conclusion in `experiments_results.md` is that recurrence is *"an airtight null … every
substrate/stabilizer/topology/neuron-model confound is now ruled out … surrogate-gradient BPTT is the sole
remaining lever."* That conclusion is **premature**: the one thing never ruled out is the **credit rule
itself**, because the credit rule was never actually complete.

e-prop's ability to assign credit across a long delay comes almost entirely from the **adaptation eligibility
vector** `εᵃ`, which recurses with the slow adaptation time constant `ρ` (≈64 waves here) rather than the
fast membrane `α` (≈6 waves). The implemented eligibility has only the membrane term, whose trace has decayed
to ~0 by the time the readout error arrives across a 20-wave gap — so recurrent synapses receive only
short-horizon, spike-coincidence credit, which (as documented) *scrambles* the signal. The project already
knew this piece existed and was missing:

- `related-work.md`: *"e-prop's ALIF eligibility has a threshold-adaptation component"* (ranked-list item 1);
  *"e-prop's eligibility trace on the per-neuron threshold is precisely the machinery that credits this slow
  held-state."*
- `AGENTS.md` (working notes): *"(A) Complete the e-prop eligibility (the ALIF adaptation trace) — fixes the
  learning rule for trained loops."*

Because the memory mechanism (ALIF adaptation) and the missing credit term (`εᵃ`) are the *same* slow
variable, this is the highest-leverage single change before conceding "BPTT only." Two outcomes, both
publishable against the standing record:

- **(a)** recurrence now earns its keep → a real result overturning the null.
- **(b)** it still fails *with a faithful rule* → the null becomes **credible** (the BPTT-only claim is then
  earned, not premature). We report whichever occurs.

## The eligibility, verified against the literature

Bellec et al. 2020 (Nature Communications), ALIF neuron, Eq. 24–25 — cross-checked from first principles
(the transition Jacobian `D_j = [[α,0],[ψ, ρ−β·ψ]]` projected onto the observable `∂z/∂s = [ψ, −β·ψ]`) and
against the official autodiff implementation (`IGITUGraz/eligibility_propagation`, `alif_eligibility_
propagation.py`), which confirms `ψ` multiplies **both** terms:

```
εᵛ_i(t)  = α·εᵛ_i(t−1) + z_i(t−1)                              (membrane elig. = filtered pre-trace)
εᵃ_ij(t) = ψ_j(t−1)·εᵛ_i(t−1) + (ρ − β·ψ_j(t−1))·εᵃ_ij(t−1)     (adaptation elig. — the missing piece)
e_ij(t)  = ψ_j(t)·( εᵛ_i(t) − β·εᵃ_ij(t) )                     (Eq. 25)
Δw_ij    = −η · L_j · Σ_t e_ij(t)                              (L_j unchanged)
```

`εᵛ` is presynaptic-only (independent of `j`), which is why today's code factors it as `pretr_i · post_j`.
`εᵃ` depends on **both** `i` and `j` through `ψ_j`, so it is a genuine per-synapse running scalar — but it
folds into the `tt` loop that already exists, at O(1) extra state per synapse and no asymptotic cost.

### Mapping to the integer substrate

- **ρ (slow adaptation decay)** `= 1 − 2^(−adapt_decay)` (≈0.984 at `adapt_decay=6`; τ≈64 waves). Derived
  from `cfg.adapt_decay`, no new knob. This is the term that makes `εᵃ` persist across the delay gap after
  `εᵛ` has decayed to 0 — the entire point of the change.
- **α (membrane decay)** — `εᵛ` keeps using the existing presynaptic trace `pretr` (decay `1 − 1/rec_tau`).
  We deliberately **do not** re-tie it to the true membrane α; keeping `εᵛ` exactly as-is means the *only*
  new mechanism introduced is `εᵃ` (minimal confound). `rec_tau` stays the tuning surface it already is.
- **ψ_j(t) (bump pseudo-derivative)** `= γ · max(0, 1 − |v_j(t) − eff_j(t)| / θ_j)`, with `v_j` the recorded
  decide-time potential, `eff_j(t) = baseline_j + (adapt_j(t) ≫ ADAPT_SHIFT)` the **effective** (adaptive)
  firing threshold, `θ_j = max(1, baseline_j)` the normalizer (v_th), and `γ` a dampening constant
  (default 0.3, matching LSNN). Centering the bump at `eff` (not `baseline`) is why we need the new accessor
  in the next section — with large adaptation, `eff ≫ baseline` and a bump centered at `baseline` would read
  ~0 exactly when the neuron is about to fire.
- **β (adaptation coupling)** — a **small, tunable** coefficient, *not* `adapt_bump`. LSNN uses β≈0.07–0.16;
  a naive `β = adapt_bump = 20` makes `(ρ − β·ψ)` swing strongly negative and `εᵃ` explode. Exposed as a
  knob (`elig_beta`) with a conservative default; expect to sweep it.

### Update loop (drop-in, per existing (i, k→j) synapse loop)

```
eps_a = 0.0                                   // εᵃ_ij(0)
e     = 0.0
for tt in 0..ttot {
    let psi = psi_arr[tz][tt][j];             // bump ψ_j(tt)
    let ev  = pretr[z][tt][i];                // εᵛ_i(tt)
    e      += psi * (ev - beta * eps_a);      // e_ij(tt)  uses εᵃ_ij(tt)
    eps_a   = psi * ev + (rho - beta * psi) * eps_a;   // εᵃ_ij(tt+1)
}
// Δw as today: out_shadow[..] += -hidden_lr * L_sig(tz, j) * e
```

When the feature is off (`elig_beta == 0.0 && !elig_bump_psi`) the code takes the **existing** branch
verbatim (`e += pretr·post`), so all current results are **byte-identical**.

## Read-only engine accessor (the only non-bench change)

```rust
/// Per-neuron effective firing threshold `baseline + (adapt >> ADAPT_SHIFT)` (the ALIF decide threshold).
pub fn layer_effective_threshold(&self, z: usize) -> Vec<i32>
```

Mirrors the `eff` computed in `wave::process_layer`'s decide step. It **reads existing state** and changes
no dynamics — the same category as the already-present `layer_decide_potential` / `adaptation` accessors.
Recorded per wave in `xor_trial_layers` (alongside `decide_potential`) so `ψ` is centered at the true
adaptive firing point on every recorded wave. (`train_xor`/`sequence_trial` path via `recurrent_update`
records the top recurrent layer's `eff` the same way.)

## Config knobs (`RsnnConfig`) — both guarded

- `elig_beta: f32` (default `0.0`) — β for the ALIF adaptation eligibility term. `0.0` ⇒ term off.
- `elig_bump_psi: bool` (default `false`) — use the normalized bump ψ instead of the spike/ramp `post`
  factor. Enables the clean **ablation**: bump-ψ + β=0 (isolates the ψ change) vs. bump-ψ + β>0 (adds εᵃ).

`elig_beta > 0` implies the bump ψ internally (the recursion needs a real ψ); `elig_bump_psi` exposes
bump-ψ *without* the adaptation term for the ablation. Both at their defaults ⇒ old path.

Threaded through the three temporal trainers: `recurrent_update` (level-0 lateral, via `train_xor` /
`train_sequence`), `train_recurrent` (uniform +1/−1/−2), and `train_multilayer` (per-layer topologies, via
`train_hidden_rec` / `train_sidecar` / `train_l2l3loop`).

## Testing (TDD)

Fast inline tests (must stay green under `cargo test`):

1. **Guard / byte-identical:** with `elig_beta=0, elig_bump_psi=false`, `train_hidden_rec` (and
   `train_recurrent`) return exactly the current value — the feature is provably off by default.
2. **Determinism:** the new path (`elig_beta>0`) is a pure function of `(seed, config)` — equal across two
   runs.
3. **εᵃ recursion unit test:** a hand-built 2-neuron, few-wave `(ψ, εᵛ)` trace produces the closed-form
   `εᵃ` and `e` (validates the ρ/β/ψ wiring and the `−β·εᵃ` sign).
4. **ψ bump unit test:** `layer_effective_threshold` + the bump formula give a nonzero, correctly-centered
   ψ for a charged-near-eff neuron and 0 far from eff, with adaptation raising the center.

Benchmark re-runs (the scientific verdict; `#[ignore]`d, run in `--release`):

- **`fair_recurrence_test`** — the airtight ALIF null (deep-FF 986 → +recurrence 498/chance, 5 seeds). Does
  the completed eligibility stop recurrence from destroying the working baseline / does it help? *Primary.*
- **`parity_recurrence_sweep`** — purpose-built recurrent-computation suite (ALIF, FF vs +rec, N=2..5).
- **`alif_recurrence_vs_ff`** — does recurrence now *extend* ALIF's horizon (delays 40/80/120)?
- Ablation on the primary config: `(bump-ψ, β=0)` vs `(bump-ψ, β>0)` and a short β sweep, to attribute any
  movement to the adaptation term specifically.

LIF configs (`fair_recurrence_lif`, etc.) are unaffected by construction (`adapt_bump=0 ⇒ εᵃ≡0`); noted, not
re-run for a result.

## Docs

- `experiments_results.md`: add the outcome and **correct** the "credit rule fully ruled out / BPTT the sole
  remaining lever / every confound removed" framing — the credit rule was incomplete when that was written.
- `AGENTS.md`: update the recurrence bullet to reflect the completed eligibility and the new result; and
  remove the stray uncommitted RF-neuron text currently appended to the file.

## Non-goals (YAGNI)

- **No engine-side online eligibility.** Bench-side post-hoc recomputation from recorded traces is enough at
  this scale and keeps the change reversible and guarded (the "full engine-side" option was declined).
- **No change to `εᵛ`/`rec_tau`, the learning signal `L_j`, `rate_reg`, or `rec_stab`.** Only the eligibility
  gains the adaptation term.
- **No surrogate-gradient BPTT.** This change is precisely the test of whether BPTT is *actually* the only
  remaining lever.
