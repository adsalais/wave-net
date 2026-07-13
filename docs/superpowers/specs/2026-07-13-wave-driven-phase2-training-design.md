# wave_driven Phase 2 — activity-scaled training (membrane e-prop, event-driven)

- **Date:** 2026-07-13
- **Status:** design approved; ready for an implementation plan
- **Scope:** Phase 2a — **online, activity-scaled eligibility** for multi-layer-DFA training on the
  `wave_driven` engine, using the **membrane-only** e-prop trace with **spike-ψ** (the event-driven
  form). The ALIF adaptation-eligibility term (`εᵃ`), bump-ψ, and the recurrence-beats-FF result are a
  later phase (2b). Builds on Phase 1 (inference), merged on `main`.

## Motivation

`wave_bitnet` trains **offline**: `run_trial` records the *entire* trial (`TrialRecords` — per-wave
spikes + `decide_potential` + `decide_eff` snapshots for **every** neuron), then `temporal_eligibility`
computes per-synapse eligibility in a post-hoc triple loop `O(L · size²·count · T)`. That is the
opposite of Phase 1's promise: it is size-bound in both memory (full per-wave records) and compute.

Phase 1 gave `wave_driven` an inference engine whose cost scales with **activity**. Phase 2 makes its
**training** scale with activity too: maintain the e-prop eligibility trace **online, during `wave()`,
touching only active synapses** — no whole-trial recording, no `O(size²·T)` post-hoc pass. This is the
piece that actually unblocks multi-seed scaling sweeps at size ≥ 64.

## The one idea

> With **spike-ψ**, the membrane e-prop eligibility collapses to
> **`e_ij = Σ_{t : j fires} pretr_i(t)`** — at every wave a target `j` fires, add source `i`'s
> presynaptic trace to synapse `i→j`. Accrue this **online** by scanning only the fan-out of sources
> with a live presynaptic trace (Approach A). At trial end, `Δshadow_ij = −lr · signal_j · e_ij`,
> repack, reset. Cost scales with activity, not size.

## Scope decision — membrane-only, spike-ψ (event-driven), and why

The e-prop eligibility has two parts (`multilayer_dfa::elig_adapt_sum`):

- **membrane term** `Σ_t pretr_i(t)·ψ_j(t)` — factors into a per-source trace × a per-target
  pseudo-derivative;
- **ALIF adaptation term** `εᵃ_ij` — **per-synapse recursive** state (`εᵃ_ij ← ψ_j·pretr_i +
  (ρ−β·ψ_j)·εᵃ_ij`) that keeps contributing even from **silent** sources.

**This phase does the membrane term only (`elig_beta = 0`), with spike-ψ.** Justification:

1. **It is already the validated FF rule.** `bench::wave_bitnet_bench::ff_cfg` sets
   `use_bump: elig_beta != 0.0`, so the feed-forward learner (`elig_beta = 0`) already uses **raw
   spike-ψ** (`ψ_j = 1` iff `j` fires). Both `wave_bitnet_trains_above_chance` and the depth-8 FF smoke
   run this. So membrane + spike-ψ is not a simplification of the FF rule — it *is* the FF rule.
2. **spike-ψ makes it event-driven.** `ψ_j ≠ 0` only at a target spike (a sparse event), so eligibility
   is pure event accrual — the ETLP shape (`docs/related-work.md`: ETLP is *"e-prop made
   hardware-local; the target shape for a rule that runs inside the substrate"*). The lazy trace-decay
   trick (Phase 1's `ρ^gap`, and NEST/Loihi synapse "archiving": Morrison–Diesmann–Gerstner 2008) is
   the standard way to keep such traces cheap.
3. **`εᵃ` is the hard, recurrence-only extra.** Its per-synapse recursive state, silent-source
   coupling, and (with bump-ψ) dense-ψ cost make it a distinct build. It is deferred to Phase 2b, where
   moving recurrence from its validated **bump-ψ** to spike-ψ is an *empirical* change to re-validate
   (spike-ψ is a coarser surrogate gradient; cf. EventProp, Wunderlich & Pehle 2021, for why
   spike-time-only computation can still be sufficient).

## The eligibility rule (exact)

Per synapse `i→j` (source `i` in layer `z`, target `j` in layer `tz = z + edge.level`), over a trial:

```
e_ij       = Σ_t pretr_i(t) · [ j fires at wave t ]          # spike-ψ membrane eligibility
pretr_i(t) = clamp0( pretr_i(t−1)·decay + [ i fires at t ] ) # presynaptic trace, decay = 1 − 1/rec_tau
clamp0(x)  = if x < ε { 0 } else { x }                        # ε-threshold → exact + activity-scaled
```

Both factors are read at the **same wave `t`** — the deferred one-hop delay (`i` fires at `t`, `j`
fires at `t+1`) is carried by the *persistence* of `pretr_i`, exactly as in `wave_bitnet`'s offline
`temporal_eligibility` (which indexes `pretr[z][t][i] · psi[tz][t][j]` at the same `t`). No
`decide_potential` / `decide_eff` snapshots are needed (spike-ψ needs only the fired set) — a genuine
simplification over `wave_bitnet`'s bump-ψ path.

The **ε-threshold** is the key to two properties at once: float decay is asymptotic (never reaches 0),
so a hard `pretr < ε → 0` cutoff (a) bounds the presynaptic-active set (→ activity-scaled) and (b) lets
the offline oracle apply the **identical ε** and thus match the online accrual **bit-for-bit**.

## Online accrual — Approach A (source-driven scan)

Our topology is stored **outgoing** (per-source occupancy bitset). `e_ij`'s trigger (`j` fires) is a
*target* property, so rather than materialize incoming adjacency (Approach B — doubles topology memory;
rejected), we drive by **source**:

Per wave `t`, when training is enabled, **after** the Phase-1 layer step and swaps:

1. **Capture fired sets** for this wave, per layer, into a per-target-layer **fired-bitset** (O(firers)
   to set/clear).
2. **Update `pretr`** for every layer: decay the presynaptic-active set, `ε`-drop those that hit 0, add
   this wave's firers (`pretr := ...·decay + 1`). O(presynaptic-active + firers).
3. **Accrue** (the eligibility scan): for each source layer `z`, for each source `i` in
   `pretr_active[z]`, for each edge, word-scan `i`'s outgoing occupancy (the Phase-1 scan) → target `j`
   in `tz`; **if `fired_bitset[tz][j]`** then `elig[i, edge, r] += pretr_i` and mark row `i` **dirty**.
   O(presynaptic-active × fan_in).
4. **Bump `spike_count`** per neuron (for `rate_reg`).

Cost per wave = `O(presynaptic-active_sources × fan_in + firers)` — activity-scaled (zero when nothing
fired recently), with a short tail set by `rec_tau` (≈ 6). No `0..size²` sweep. (Sources with a live
trace whose targets are quiescent still get scanned but accrue nothing; that scan is the price of
outgoing-only topology, and is bounded by the short `rec_tau` tail.)

## Data structures

**Optional per-`Layer` `TrainState`** (re-added to `wave_driven`; allocated by `enable_training()`,
freed by `disable_training()` — the same lean/train toggle Phase 1 established, so inference pays
nothing):

```
shadow      : Vec<f32>   // ls·total_slots — differentiable weight twin, requantized to `codes`
elig        : Vec<f32>   // ls·total_slots — per-synapse eligibility accumulator (SAME layout as shadow)
pretr       : Vec<f32>   // ls — per-source presynaptic trace
spike_count : Vec<u32>   // ls — per-neuron trial spike count (for rate_reg)
```

**Work sets (reuse `frontier::Frontier`):**
- `pretr_active[z]` — sources with `pretr ≠ 0` (the accrual scan set).
- `dirty_rows[z]` — source neurons whose `elig` row got accrual, so update/repack/reset are O(activity).

**Per-wave scratch (Network):** `fired_by_layer: Vec<Vec<u32>>` (this wave's fired per layer, captured
during `wave()`), and a per-layer **fired-bitset** (built from it, for O(1) "did `j` fire" checks).

`enable_training` builds `shadow = decode(codes)` (codes are the master, exactly like a `.wbm`
round-trip) and zeroes `elig`/`pretr`/`spike_count`. `disable_training` frees the state (lossy for
in-flight sub-threshold shadow — snaps back to codes on re-enable, same contract as Phase-1/`wave_bitnet`).

## Training flow

**Per wave** (engine-side, only when training enabled): the accrual above, wired into `Network::wave`.

**Per trial** (bench-side, ported from `wave_bitnet_bench`, using engine hooks):
1. `reset_state` (clears runtime + `elig`/`pretr`/`spike_count`/work-sets) and run the trial
   (present/delay/read waves), the top **spiking** layer's read-window spikes → `act` (via a listener,
   as today; the FF net's top layer spikes — it is *not* a `new_with_readout` layer).
2. Readout: linear `w[c][j]` over `act` → softmax → cross-entropy error `err[c]`.
3. `build_signal`: per computational layer `tz`, `signal[tz][j] = Σ_c b·err[c] + rate_reg·(rate_j −
   target)`, where `b` = readout weight (top) or fixed DFA hash weight (hidden), and
   `rate_j = spike_count[j]/ttot` from the engine.
4. `Network::dfa_update(entries, signal, lr)`: for each edge, `shadow[i,edge,r] += −lr · signal[tz][j]
   · elig[i,edge,r]` (`j` decoded from occupancy), then `repack_row` each **dirty** row.
5. `Network::reset_eligibility()`: zero `elig`/`pretr`/`spike_count` over dirty rows / active sets (next
   trial starts clean).

Readout weights and the periodic held-out best-checkpoint loop (`train_and_eval_best`) are ported
verbatim in spirit from `wave_bitnet_bench` — this phase reuses that proven harness shape.

## Validation plan

1. **`online ≡ dense` eligibility, bit-for-bit (primary oracle).** A ported offline
   `dense_eligibility(fired_records, rec_tau, ε)` computes `e_ij = Σ_t pretr_i(t)·[j fires]` from a
   fully-recorded trial (recording is fine *in the test*). Because both use the identical `ε`-thresholded
   `pretr` recurrence and sum in wave order, the engine's online `elig` must equal the offline result
   exactly. Proves the on-frontier accrual is a faithful, complete implementation of the rule.
2. **Determinism.** A fixed `(seed, config, task-seed)` training run reproduces identical `shadow`/`codes`.
3. **FF trains above chance (end-to-end acceptance).** Port `wave_bitnet_trains_above_chance`: a 4-layer
   FF, size 16, generous fan-in, single-cue 2-class task must reach `best > 600` permille held-out.
   This exercises the whole pipeline: forward + online eligibility + shadow update + repack + readout.
4. **`#[ignore]` stretch experiments** (run in `--release`): a depth-8 FF pure-ternary smoke (port of
   `wave_bitnet_ff_depth8_smoke`, target ~1000 worst-seed), and a **training-throughput** comparison —
   the online trainer vs an offline/dense-eligibility baseline at size 16/32 — to show the
   activity-scaling win (the reason this phase exists).

Tests are inline `#[cfg(test)]`; the smokes/throughput are `#[ignore]`d. `cargo test` green, warning-free.

## Module layout and API

```
src/wave_driven/
  neurons.rs    + TrainState { shadow, elig, pretr, spike_count } + enable/disable_training + repack_row (ported)
  training.rs   NEW: EligParams { rec_tau, epsilon }; per-wave accrual; dfa_update; reset_eligibility;
                     dense_eligibility (the offline oracle); Edge (edge descriptor, mirrors topology order)
  network.rs    + training fields (fired_by_layer, fired-bitsets, pretr_active, dirty_rows),
                  enable/disable_training, wave() accrual hook, dfa_update, reset_eligibility,
                  layer_spike_count accessor; reset_state clears training state
  frontier.rs   (reused for pretr_active / dirty_rows)
src/bench/
  wave_driven_bench.rs  NEW: run_trial / readout / build_signal / train_and_eval_best (ported) + FF-above-chance test
```

**API additions on `Network`:** `enable_training()`, `disable_training()`, `is_training()`,
`dfa_update(entries: &[Vec<Edge>], signal: &[Vec<f32>], lr: f32)`, `reset_eligibility()`,
`layer_spike_count(z) -> &[u32]`. The forward `wave()` gains an internal accrual step gated on
`is_training()`.

## Phasing

- **Phase 2a (this spec):** online membrane-only spike-ψ eligibility + DFA training + oracle + FF
  acceptance.
- **Phase 2b (separate spec):** the ALIF `εᵃ` term — per-synapse event-triggered trace with lazy
  `ρ^gap` decay between target spikes — to unlock recurrence/side-car training; includes re-validating
  recurrence under spike-ψ vs its historical bump-ψ result.
- **Phase 3 (future):** GPU.

## Non-goals (Phase 2a)

The `εᵃ` / bump-ψ path; the recurrence-beats-FF proof; a non-spiking `new_with_readout` output layer for
training (spike-ψ eligibility into a never-firing readout is identically 0 — FF training reads the top
**spiking** layer, as `wave_bitnet` does); shadow persistence to `.wbm`/`.wbr`; GPU kernels.

## Risks and open questions

- **`ε` choice.** Too large truncates real eligibility (biases the gradient); too small lengthens the
  presynaptic-active tail (less activity-scaling). Default `ε = 2^−10` with `rec_tau ≈ 6` keeps the tail
  ~a few `rec_tau`. It is a config knob; the oracle guarantees online == offline *for whatever ε is
  chosen*, so ε trades speed vs fidelity, not correctness.
- **Per-synapse `elig` memory** is `O(size²·count)` f32 — the same order as `shadow`, so training memory
  roughly doubles vs inference-lean (expected; matches `wave_bitnet`'s shadow footprint). Freed by
  `disable_training`.
- **Outgoing-only scan tail.** A source with a live trace but quiescent targets is still scanned (accrues
  nothing). Bounded by `rec_tau`; if it ever dominates, Approach B (incoming adjacency) is the escape —
  deferred.
- **Not bit-exact to `wave_bitnet` training.** Different adaptation dynamics (Phase 1) and the spike-ψ
  choice mean accuracy is validated against *chance* and the `wave_bitnet` *ballpark*, not a bit-match.

## Appendix — copied vs new

- **Ported from `wave_bitnet` (adapted):** `repack_row` (`neurons.rs`); the training-harness shape
  (`run_trial`, readout, `build_signal`, `train_and_eval_best`, task generators) into
  `bench/wave_driven_bench.rs`; the `Edge` descriptor and the `dfa_update` shadow-update math (from
  `multilayer_dfa`/`eprop_update_synaptic`), now reading engine-internal `elig`.
- **New:** the online per-wave accrual (source-driven, spike-ψ, ε-thresholded `pretr`); `TrainState`
  with `elig`/`pretr`/`spike_count`; `pretr_active`/`dirty_rows` work-sets; the `dense_eligibility`
  oracle; the `Network` training hooks.
- **Deliberately NOT ported:** `decide_potential`/`decide_eff` snapshots (spike-ψ needs only fired
  sets); `temporal_eligibility`'s bump-ψ and `elig_adapt_sum` `εᵃ` recursion (Phase 2b).
