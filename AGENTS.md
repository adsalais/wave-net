# AGENTS.md

Guidance for AI agents (Claude Code and others) working in this repository. These instructions
override default behavior — follow them exactly. (`CLAUDE.md` points here.)

## What this project is

`wave-net` is a research repo asking: **can a memory-lean, ternary-native integer spiking network
learn?** The engine (`wave_bitnet`) is a deterministic, procedurally-wired integer spiking substrate
(a Liquid State Machine) that **stores and trains** its plastic weights on top of a fixed materialized
topology. The foundational result — documented in `docs/experiments_results.md` — is:

> Training only per-neuron **thresholds** over a *fixed random `±1`* projection is **not a reliable
> learner** (it works only when the reservoir/task seed happens to align — a coin flip). Making the
> **weights stored and trainable** is what makes learning reliable across seeds. Thresholds can only
> *gate* a fixed random projection; trained weights *shape* it.

So the design keeps the procedural static **structure** (connectivity is a pure function of the seed,
materialized once) but **stores and trains the plastic weights** — the sanctioned hybrid (cf. GeNN,
Knight & Nowotny 2021: procedural connectivity is static-only; plastic weights must be stored).

**Current state:** it *works*. A **feed-forward + ALIF** network with e-prop / multi-layer-DFA credit
is a **reliable learner** (held-out, multi-seed) — a plain stack reaches ~12 layers and a half-count
+2 residual skip extends it to **depth 24+** (see Headline results) — and `rate_reg` reliably keeps
deep stacks alive. **Trained recurrence robustly beats feed-forward** — with the completed ALIF
eligibility on the backward-fed **side-car** topology (recurrent layer isolated from the forward path):
a strict improvement over FF across every benchmark and seed (XOR/flip-flop tie at ceiling;
distractor-XOR 700→995, parity N=4 587→837). The open questions are generality to larger sizes and
*why* the topology works. **BPTT is out of scope — permanently; do not propose it.**

## The two modules (read this before touching code)

The crate is `wave_bitnet` + `bench` (`src/lib.rs` wires them up), plus a second, independent
**inference-only** engine `wave_driven` (Phase 1) that trades size-bound sweeps for **activity-bound**
ones — each wave processes only a per-layer frontier of non-quiescent neurons with lazy fire-anchored
adaptation, so cost scales with activity rather than layer size. `wave_bitnet` remains the trainable
engine; `wave_driven` is a free redefinition of its dynamics (not bit-exact — but it *is* bit-exact
when `adapt_bump = 0`, which the equivalence tests assert) validated by a dense-vs-sparse oracle. Spec:
`docs/superpowers/specs/2026-07-13-wave-driven-event-active-set-design.md`.

- **`wave_bitnet/` — the engine.** A memory-lean integer spiking engine: topology is materialized once
  into a per-neuron neighborhood **occupancy bitset** (no per-wave hashing), and weights are stored as
  **2-bit packed ±1/0 codes**. The `f32` training shadow (and the decide-time snapshots) live in an
  **optional per-`Layer` `TrainState`** allocated only while training is enabled
  (`Network::enable_training()` / `disable_training()`), so inference pays nothing for them. It carries
  the on-engine hooks a learning rule needs. **This is where engine work happens.**
- **`bench/` — the experiment harness.** Integer tasks, readouts, decoders, and **the learning rules
  themselves** (multi-layer-DFA credit). Test-only; uses only the engine's public API. The engine
  exposes the *state* a learning rule needs; `bench` implements the rules on top.

## The one idea that explains the engine

**Synapse addresses are derived from the seed, materialized once, then scanned — never hashed per wave.**
At construction, each neuron draws `count` distinct neighborhood cells (a pure function of
`(seed, source, config)`) and sets them in a per-neuron **occupancy bitset** (one bit per `(2r+1)²`
cell). At fire time the forward pass scans the set bits (`trailing_zeros`) and decodes each cell to its
target local index with arithmetic (an offset LUT + toroidal wrap) — **no hashing after startup**. What
is stored per synapse is the **weight**: a 2-bit code (`0b00`→0, `0b01`→+1, `0b11`→−1); the `f32`
shadow it trains through lives in `Layer.train` and exists only while training is enabled. Weights
**init** to the procedural `±1` sign (so a fresh net is a random ±1 projection), and training moves the
shadow and requantizes it to the codes. Determinism flows from
`(seed, config, input)`. The engine's dominant cost is the fire-time bitset scan + decode + delivery.

## The engine model (how a wave works)

A stack of `L` square layers (`size × size`, `size` a power of two, toroidal wrap). Per neuron:
`i16` potential (rest 0), `u8` cooldown, per-neuron `i16` **baseline** threshold, and an `i32`
**adaptation** state (Q12 fixed point, rest 0). A **wave** advances every layer one step;
`wave::process_layer` runs, per layer:

1. decay cooldown
2. **drain inbox** — fold this wave's per-target deliveries (`pending`, scatter-added last wave) into
   potential in `i32`, narrow to `i16` (the only clamp — pure overflow protection; there is **no**
   saturation concept), then clear `pending`
3. **inject** (L0 only) — set injected locals' potential to `i16::MAX` and clear cooldown (forced fire)
4. **decide** — fire if `cooldown == 0 && potential >= baseline + (adapt >> ADAPT_SHIFT)` (the ALIF
   effective threshold, in `i32`); on fire reset potential to 0, reload cooldown, bump adaptation.
   Also, per wave: snapshot `decide_potential`/`decide_eff` (pre fire-reset) and accrue **e-prop
   eligibility** — `elig_post` (a box pseudo-derivative ψ: near-threshold count) for every neuron,
   `elig_pre` (spike count) for firers
5. **generate** — scan the firer's occupancy bitset, decode each wired cell to its target, and deliver
   the packed 2-bit ±1/0 weight (grouped by relative level)
6. **leak** — decay survivors' potential toward 0 (positive decay floored at 1, so small potentials
   relax to rest with a finite membrane time constant instead of freezing in the shift dead zone)
7. **adapt-decay** — decay every neuron's adaptation geometrically toward rest

**Propagation is deferred, one hop per wave.** A firer's deliveries land in the *target* layers'
per-target accumulator; `pending`/accumulator swap at wave end, so signal reaches the next layer *next*
wave. `Network::wave(input)` orchestrates: process each layer, route its synapse groups into the target
layers' accumulators, then swap.

**Boots hot, self-regulates.** Baselines init low (`baseline_init + jitter`), so neurons fire readily
from the first waves; each neuron's adaptation then rises with its own firing and quenches it — a
local negative-feedback controller that settles the firing rate (spike-frequency adaptation, the ALIF
mechanism). Input is a sparse `Vec<u32>` of L0 local addresses (spike injection), not graded current.
`adapt_bump = 0` recovers plain LIF dynamics.

**L0 is the input transducer.** `Network::new` forces layer 0 to baseline `i16::MAX` with no
adaptation, so it fires *only* on injection and never self-adapts — input encoding stays decoupled
from adaptation. The boots-hot ALIF dynamics apply to the computational layers `1..L`.

**Readout layers are the output symmetry.** `Network::new_with_readout` flags the **last** layer as a
non-spiking, drain-only integrator (`Layer.readout`): it folds its input into potential and never
fires, giving a clean cumulative signal for a trained readout — the mirror of L0's transducer role.

## Initialization — train from raw, no calibration

**There is no calibration and no criticality init.** Pure fixed `±1` magnitude cannot be scaled to a
target branching ratio σ, so there is no σ-tuning init to lean on — nets **train from raw**. The
contract is: **generous fan-in + a liveness check**. The clean three-regime picture by substrate
richness (deep 5-layer, width 32):

- **Generous fan-in** (e.g. `up_count ≥ 16` at width 32): the init is irrelevant — even raw training
  reaches ceiling; `rate_reg` + ALIF bring every layer to life. **This is the target regime — size the
  fan-in here.**
- **Marginal fan-in:** raw training can sit at *chance* (a layer never bootstraps). The fix is **more
  fan-in**, verified by a **liveness assert** (every computational layer must fire above a floor after
  training — *fail loud*, then bump fan-in).

**Untrained forward measurement** (profiling / throughput of the substrate) runs the net at its
self-regulated ALIF operating point under random L0 drive (`wave_bitnet::synapse::random_l0_input`); a
uniform deep `±1` stack there is naturally sub-critical (the cue dies with depth), and that is expected
— training's `rate_reg` + ALIF own the operating point during a run.

**Do not re-introduce a rate set-point / calibration.** e-prop/LSNN uses a soft firing-rate
*regularizer* (`rate_reg`) folded into the learning signal, never a separate proxy-drive calibration —
see `docs/related-work.md`.

## Learning: what is built, and what it found

The learning rules live in `bench/`, not in the engine. Treat `docs/experiments_results.md` as the
**source of truth** for findings. **Bottom line: wide + deep + feed-forward + ALIF with multi-layer-DFA
credit is the reliable learner; and trained recurrence — with the completed ALIF eligibility on the
isolated-recurrent **side-car** topology — robustly beats feed-forward across every benchmark and seed.**
Headline results:

- **The working learner — e-prop on stored weights + trained readout, feed-forward + ALIF.** A factored
  per-neuron eligibility (`e = pre-trace × ψ`, both O(neurons) engine state) × a learning signal from a
  trained readout updates the stored weights through the `f32` shadow. Held-out and multi-seed, it clears
  the bar threshold-only training failed. A trained readout on the full reservoir (classic LSM) is also a
  reliable baseline.
- **Multi-layer DFA credit trains deep stacks — the depth reach is TOPOLOGY-bound, not a fixed ~16.**
  Train *every* layer: the top gets symmetric readout feedback, deeper layers get Direct Feedback
  Alignment (fixed random hash-derived feedback of the output error), with width to match. A plain +1
  feed-forward stack tops out ~12 layers (single-cue separable, adapt_bump=5: clean to 12, marginal by 16
  — signal decays through depth), but a **half-count +2 residual SKIP roughly DOUBLES that**: pure ternary
  trains a clean 1000/1000 to **depth 24+** with `+2 = ½·count` at moderate fan-in (r3/c32 + r3/c16),
  3 seeds. So the earlier "~16 = DFA feedback-noise wall" was **topology-specific, NOT fundamental** —
  route signal around layers and it keeps going. Fan-in has a floor *and* a ceiling that narrows with
  depth (count is the lever, radius stays tight ~4; too much fan-in over-drives and collapses the stack).
- **`rate_reg` is a *conclusive* liveness rescue for feed-forward depth.** A soft per-neuron term
  `c_reg·(rate − target)` folded into the e-prop learning signal reliably revives a *liveness-starved*
  deep FF stack: chance → ~980 on temporal XOR, 5-seed robust. Hard rule: it **requires ALIF** (a LIF
  deep stack cannot be revived — adaptation is load-bearing). **Apply it across all layer types —
  `rate_reg` everywhere.** **Caveat — it over-trains: `rate_reg` is class-agnostic and always-on, so
  after the task converges it homogenizes firing rates and *erodes* the class signal → a non-monotonic
  accuracy collapse (transient at ~4 layers, permanent by ~12; ablation-confirmed). Mitigate with
  early-stopping / best-checkpoint (`train_and_eval_best` — periodic held-out eval, checkpoint the peak
  via `Network::save_model_path`); compare at the *peak* of a duration sweep, never a fixed final trial
  count. Alternatives to try: anneal c_reg→0 as task error falls, or gate rate_reg off for neurons already
  at target rate.**
- **ALIF adaptation is both a working memory *and* load-bearing for liveness.** It is a strong ~64-wave
  held-category memory (store-recall); it does **not** help linear echo (MC) or nonlinear temporal
  computation (XOR) feed-forward — LIF wins those short tasks — **but it is *necessary* for deep-FF
  propagation** (removing it kills the deep stack; `rate_reg` can't revive LIF). There is no calibration
  — nets train from raw weights; ALIF + `rate_reg` own the operating point during a run.
- **Recurrence robustly beats feed-forward — with the completed ALIF credit rule AND a topology that
  isolates the recurrent layer from the forward path.** The winner is the backward-fed **side-car**: the
  forward signal *skips past* the recurrent layer (L1→L3, +2) while a separate **recurrent scratchpad**
  (L2 self-loop + L2→L3, L3→L2 back) holds state — the loop computes without polluting the clean forward
  projection. Size 32, worst-seed / 3 seeds, it is a **strict improvement over FF on every benchmark**:
  temporal XOR 990→**1000**, flip-flop 985→**1000**, distractor-XOR 700→**995**, parity N=4 587→**837**.
  The credit rule that trains it is the completed **ALIF adaptation eligibility** `εᵃ`
  (`e = ψ·(εᵛ − β·εᵃ)`, Bellec 2020; missing from the earlier crude spike-timing rule) — built + verified
  (`EligParams.elig_beta`/`use_bump`/`elig_psi_width`; decide-time `eff` snapshot `Layer.decide_eff`;
  fixed-width bump ψ). Levers that matter: **topology (isolation) ≫ hyperparameter tuning**, **width**
  (sets the ceiling), and **sub-critical recurrence density** (collapse cliff at rec_count ≈ 12 — keep it
  below). A **scaling study** (in progress) finds a per-neuron forward-drive threshold (~up_count 32),
  width as a capacity floor (≥ size 32 for parity N=4), joint width×forward scaling above both bars, and
  that reading the recurrent layer *directly* beats a dedicated read layer. **Blocker: the
  single-threaded integer engine is too slow to run these sweeps multi-seed at size ≥ 64 — the immediate
  next work is performance optimization, then resume the systematic scaling / forward-topology study.**
- **Pure ±1/0 ternary weights, and ALIF strength is the key knob.** Weights are 2 bits: a `nonzero` mask
  and a `sign`. A tunable prune threshold `t` (`|shadow|/γ < t → 0`; default 0.5 = round, sweet spot
  ~0.7) sets sparsity. **The DFA-net computational-layer `adapt_bump` default is 5, not 20:** at 20 the
  recurrent hub adaptation-LOCKS after the first cue so pure ±1 can't drive later cues; at 5 it relays
  every cue and pure ternary solves side-car parity (3-seed 494→1000/1000/1000). Pure ±1 does BOTH
  feed-forward and recurrence; at depth it holds 1000/1000 to depth 24 with the +2 skip (fixed ±1 is
  collapse-robust). Sparsity is a **surplus-capacity dividend** — it appears only when fan-in exceeds the
  task's needs (deep nets prune ~nothing; wide/recurrent nets prune 20–30% for free at t=0.7). **Gotcha:
  L0 is a forced non-adapting input transducer (`Network::new` sets threshold i16::MAX + adapt_bump 0) —
  NEVER give L0 adaptation or it swallows cue injections and collapses the whole net.**

## Reading & training: the multi-wave rule

**A single wave does not contain the network's response to an input.** Propagation is deferred one-hop
(forward signal takes ~`L` waves to climb the stack) and the topology is recurrent (level 0/−1 feed
activity back over subsequent waves). Therefore, for any readout or training:

- **Drive the input over several consecutive waves**, not a single-wave impulse.
- **Read over a multi-wave window** — integrate spikes/state across enough waves to capture both the
  forward climb and the backward/recurrent settling.
- **Training or reading from one wave's data is an error.** An input's representation is distributed
  across a multi-wave window; a single-wave feature is incomplete and will mistrain.

## Persistence

Two independent, std-only binary formats (`wave_bitnet::persist`): a self-contained **model** (`.wbm`:
materialized structure + 2-bit codes, inference-ready — no seed dependence, no shadow) and a **runtime
overlay** (`.wbr`: only the resumable forward state — `potential`/`cooldown`/`adapt`/`pending` +
`wave_id`, applied onto a loaded model). API on `Network`: `save_model`/`load_model`,
`save_runtime`/`apply_runtime` (+ `*_path` conveniences), `model_fingerprint` (binds an overlay to its
model). Loading a `.wbm` reconstructs each `Layer` via `Layer::from_parts` **inference-lean**
(`train: None`). **Training is toggled:** `Network::enable_training()` allocates the per-`Layer`
`TrainState` (rebuilding each `shadow` as decode(codes)); `disable_training()` frees it to serve lean.
Disabling is **lossy for in-flight sub-threshold shadow** — re-enabling snaps it back to the quantized
codes, exactly like a `.wbm` round-trip (codes are the cross-checkpoint master). This is the primitive
behind best-checkpointing (see the over-training caveat).

## Commands

```bash
cargo test                       # all inline #[cfg(test)] tests; must stay green
cargo build                      # must stay warning-free
cargo test -- --ignored          # the experiments (long-running; some want --release)
cargo test --features strong_hash # test-only: swap the procedural hash for BLAKE3 (hash-quality control)
```

Experiments are written as `#[ignore]`d tests (the expensive ones note `run manually in --release`),
so `cargo test` stays fast and the experiments are reproducible on demand.

## Profiling

Baseline throughput lives in `benches/throughput_bitnet.rs` (`cargo bench --bench throughput_bitnet`):
it trains a size-32 side-car once, caches it to a `.wbm`, then measures ~waves/s of the loaded **trained**
model (so the delivery load reflects real post-training weights). To find *where* time goes, sample the
forward pass with `perf`: `examples/profile_bitnet.rs` is a ready harness (same 32×32×5 config,
eligibility off, tight wave loop under random L0 drive).

```bash
# perf needs kernel.perf_event_paranoid <= 1 for user-space sampling. ALWAYS CHECK IT FIRST:
cat /proc/sys/kernel/perf_event_paranoid
CARGO_PROFILE_RELEASE_DEBUG=true RUSTFLAGS="-C force-frame-pointers=yes" \
  cargo build --profile profiling --example profile_bitnet
perf record -g --call-graph fp -- ./target/profiling/examples/profile_bitnet 1000000
perf report --stdio --no-children -g none | head        # flat self-time by function
perf annotate --stdio -l -s wave_net::wave_bitnet::wave::process_layer   # per-instruction
```

**An agent cannot lower `perf_event_paranoid` itself (needs root).** When you need to profile and
`cat /proc/sys/kernel/perf_event_paranoid` reports `> 1`, ask the user to run
`sudo sysctl kernel.perf_event_paranoid=1` (temporary — resets on reboot; restore with the old value).

The dominant forward cost is the fire-time occupancy-bitset scan + cell decode + delivery in
`process_layer` (plus the per-neuron drain/decide/leak/adapt over all neurons); buffer reuse already
removed allocation from the hot path.

## Conventions (required)

- Rust, edition 2024. **Standard library only by default** — the one optional dependency (`blake3`) is
  behind a test-only feature. This is a **library crate** (no binary); experiments are `#[ignore]`d tests.
- **Standard library only** in `src/`; **warning-free build**.
- **No `unsafe` — with ONE documented exception.** `wave_bitnet::wave::process_layer`'s forward hot loop
  uses `get_unchecked`/`get_unchecked_mut` for four accesses: the per-neuron occupancy-word slice, the
  offset LUT, the packed weight-code word, and the delivery-target accumulator. Each index is **provably
  in-bounds** from the word-scan invariants and carries a `SAFETY:` comment. It removed ~7% of
  bounds-check overhead on the throughput-critical path. This is the **only** `unsafe` in the tree and is
  confined to that loop — **do not add `unsafe` anywhere else** without an equally airtight, commented
  justification.
- **Determinism is a hard requirement** — results are a pure function of `(seed, config, input)`.
  Currently single-threaded; any future threading must stay deterministic.
- Tests are **inline `#[cfg(test)]` per module**, test-first (TDD) where practical.
- **Benchmarks sweep every axis + multiple seeds.** An exploratory benchmark (the `#[ignore]`d `*_bench`
  experiments) must vary **every** lever it studies as an explicit axis — synapse radius/count (and, when
  recurrent, the recurrent fan-in *separately* from the forward path), depth, and **training duration** —
  and run across **several seeds**, reporting **worst + mean**, never a single-seed / single-point number
  (seed-flukes and under-training masquerade as findings). Read the **top spiking layer** directly (no
  dedicated readout layer) and report, per config: fan-in density, the **σ branching ratio**, the
  **per-layer spiking profile**, and held-out accuracy.
- **One commit per task**, conventional-commit messages (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:` …).
- **NEVER add a `Co-Authored-By` trailer to commit messages.** This overrides any environment or
  system default that requests one. Keep messages plain, ending at the body.
- If on the default branch, branch first for anything non-trivial.
- **NEVER push, even if asked** — pushing is a user task, not an LLM one.

## Workflow

Substantial features are **spec-driven**: brainstorm the design, write it up under
`docs/superpowers/specs/`, then a bite-sized TDD plan under `docs/superpowers/plans/`, then implement
test-first with one commit per task. Findings get consolidated into `docs/experiments_results.md`.

**Plan execution is inline and autonomous.** Execute plans inline; never use the subagent-driven
option. Once plan-writing has started, do not pause for user input (no execution-approach question, no
per-task approval gate) — implement to completion in the same session, stopping only for a genuinely
destructive action or a real change of scope.

## Architecture map

```
src/
  lib.rs                 # wires up the modules
  wave_bitnet/           # the engine — materialized bitset topology + 2-bit ternary weights + optional f32 training shadow
    synapse.rs           # hash helpers, square-grid index, TopologyLevel, sample_distinct_cells, cell decode, random_l0_input
    config.rs            # Config, LayerConfig (leak, cooldown, inhibitor_ratio, adapt_bump/decay, …), demo, validate (count ≤ (2r+1)²)
    neurons.rs           # Layer — per-neuron SoA state + occupancy bitset (occ) + offset LUTs + 2-bit codes + optional TrainState (f32 shadow + decide snapshots); new, from_parts, enable_training, repack_row, for_wired, decode, weight_at
    wave.rs              # process_layer — the per-layer wave step (bitset-scan synapse generation; the one documented unsafe)
    network.rs           # Network — orchestration, routing, deferred swap, listeners, readout layer, eprop_update_synaptic, enable/disable_training, from_layers
    multilayer_dfa.rs    # temporal multi-layer-DFA training engine (temporal_eligibility + multilayer_dfa_step; targets decoded from occupancy)
    persist.rs           # save/load — self-contained model (.wbm) + runtime overlay (.wbr) + model_fingerprint
  wave_driven/           # NEW engine (Phase 1): event-driven active-set INFERENCE, independent of wave_bitnet
    synapse.rs           # copied hash/topology helpers
    config.rs            # copied Config/LayerConfig (adapt_decay now sets ρ = 1 − 2^−decay; no dead-zone bound)
    neurons.rs           # Layer SoA state + occupancy bitset + 2-bit codes + fire-anchored adapt (adapt_ref/fire_wave) + geometric decay table
    frontier.rs          # Frontier: worklist Vec + dedup mark bitset (GPU unique-append primitive)
    wave.rs              # process_layer(Work::Sparse|Dense) — frontier step + the dense equivalence oracle
    network.rs           # Network: sparse/dense orchestration, injection-into-frontier, deferred one-hop swap
    equivalence_tests.rs # (test-only) sparse==dense oracle + adapt_bump==0 wave_bitnet cross-check
  bench/                 # experiment harness (public-API only, test-only) — the learning rules + tasks
    wave_bitnet_bench.rs # FF + side-car training harness (run_trial, build_signal, train_and_eval_best, tasks) + smoke benchmark
docs/
  experiments_results.md # SOURCE OF TRUTH for findings
  related-work.md        # literature framing (GeNN, e-prop, ALIF/LSNN, FPTT, …)
  superpowers/{specs,plans}/ # per-feature design specs and TDD plans
```

Invariants that bite if ignored: `size` must be a power of two (toroidal wrap is a bitmask); local
index is `y*size + x`, global neuron id is `layer*size*size + local`; per-layer state is
struct-of-arrays; the synapse **address** is materialized (occupancy bitset, decoded arithmetically at
fire time) and the **weight** is a stored 2-bit `±1/0` code (init to the `±1` sign, trained via the f32
shadow, requantized by `repack_row`); baselines init low (`baseline_init + jitter`, clamped to
`[1, i16::MAX]`) so the net boots hot and self-regulates via per-neuron adaptation (no calibration —
nets train from raw; see Initialization); `adapt` is Q12 fixed point so its geometric decay stays
exponential (no dead-zone ratchet), valid only while `adapt_decay <= ADAPT_SHIFT` (`Config::validate`
enforces it); `count <= (2r+1)²` (a per-cell occupancy bitset caps fan-in at the neighborhood size); a
`Layer` is a self-contained, persistable unit (owns its structure, thresholds, and stored weights) —
serialized via `persist.rs`.
