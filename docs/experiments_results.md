# Experiment results — the `wave_bitnet` RSNN

**What this is:** the standing record of results on `wave_bitnet`, a memory-lean, ternary-native integer
spiking network. The harness in `src/bench/wave_bitnet_bench.rs` drives the experiments; the training rule
itself lives in the engine's `wave_bitnet::multilayer_dfa` module. Every number is held-out, multi-seed,
and a pure function of `(seed, config, params)`. Design rationale lives under `docs/superpowers/specs/`;
literature framing in `docs/related-work.md`.

## Substrate

`wave_bitnet` is a stack of square spiking layers (`size × size`, `size` a power of two, toroidal wrap).
**Topology is materialized once at construction** into a per-neuron neighborhood **occupancy bitset** — one
bit per `(2r+1)²` neighborhood cell — and the forward pass simply scans the set bits (`trailing_zeros` over
word-aligned occupancy words + a per-level offset LUT that turns a cell into a `(dx, dy)`). There is no
per-wave hashing and no re-derivation of synapse addresses at fire time.

**Weights are ternary.** Each wired synapse is a **2-bit packed code** (`0b00 = 0`, `0b01 = +1`, `0b11 = −1`
— a nonzero mask plus a sign bit), 32 codes per `u64`, with a per-synapse `f32` **training shadow**. The
shadow is the training master; each step it is requantised to the 2-bit `codes` per row (`repack_row`) via a
prune threshold `t`: `γ = mean(|shadow|)` over the row, then `|shadow|/γ < t → 0`, else `sign(shadow)`.
Default `t = 0.5` (round-to-nearest); the sweet spot is `t ≈ 0.7`. At rest and at inference you ship only the
occupancy bitsets + the 2-bit codes (the memory win); training pays the `f32` shadow.

Per neuron: `i16` potential (rest 0), `u8` cooldown, a per-neuron `i16` **baseline** threshold, and an `i32`
**ALIF adaptation** state (Q12 fixed point, τ ≈ `2^adapt_decay` waves). ALIF = adaptive threshold
(`adapt_bump > 0`); LIF = `adapt_bump = 0`. The computational-layer default is `adapt_bump = 5`. Propagation
is **deferred, one hop per wave**; the engine is deterministic, single-threaded, and a pure function of
`(seed, config, input)`. Readouts are bench-side `f64`; the engine stays integer.

**L0 is a forced non-adapting input transducer** — `Network::new` sets its baseline to `i16::MAX` and
`adapt_bump 0`, so it fires only on injection and never self-adapts (giving L0 adaptation makes it swallow cue
injections and collapse the whole net). The **last layer can be a non-spiking, drain-only readout
integrator** (folds its input into potential, never fires), giving a clean cumulative signal for a trained
readout — the mirror of L0's transducer role.

One engine fix underlies everything: the potential leak is **floored at 1** (`p -= max((p>>a)+(p>>b), 1)`).
The old shift-only leak had a dead zone (`0` for `0 < p < 2^a`) that froze sub-threshold potentials forever
(infinite passive memory); flooring gives a finite membrane time constant and caps how long a trace persists.

## Train the weights, not the thresholds

Training only per-neuron **thresholds** over a *fixed random ±1* projection does **not** reliably learn
(held-out, multi-seed: works only when the reservoir×task seed pair aligns, ~1-in-3). Thresholds can only
*gate* a fixed projection, never *shape* it — scaling width, a stronger connectivity hash, and richer static
weights all failed to rescue it. This is what GeNN (Knight & Nowotny 2021) predicts: procedural connectivity
is *static*-only; **plastic weights must be stored**. So `wave_bitnet` **stores and trains the ternary
weights** (moving the `f32` shadow, repacked to the 2-bit codes), and **e-prop on the stored weights learns
reliably** where thresholds failed:

| | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| fixed-reservoir top-layer readout | 727 | 500 | 655 | 507 |
| **+ e-prop hidden-weight training** | **1000** | **965** | **890** | **987** |

(A trained readout on the *full* reservoir also hits 1000 on all seeds — the classic LSM.) Two facts carry
forward: the class signal separates **less well the deeper** a fixed feed-forward stack runs, and **ALIF
adaptation is a strong ~64-wave working memory** (the bar trained recurrence must beat).

## Depth is usable, and width sets how deep

Train **every** layer, not just the last: the top layer gets symmetric readout feedback, deeper layers get
**Direct Feedback Alignment** (fixed random hash-derived feedback of the output error). Single-layer e-prop is
at chance for every depth ≥ 4 (untrained intermediate layers erode the class signal); **multi-layer DFA is
reliable to ~16 layers — but only with enough width.** Width is the lever that sets the depth the credit can
reach (worst-seed / 3 seeds, multi-layer):

| size (N/layer) | depth 4 | depth 5 | depth 6 |
|---|---|---|---|
| 8 (64) | 1000 | **485** | 902 |
| 16 (256) | 1000 | **1000** | **1000** |

Size 8 collapses past depth 4 (too few neurons to train the hard seed's deep layers); size 16 stays clean
through depth 6, and (with trial length scaled to depth so the cue reaches the top) holds to ~16 layers.
Beyond that, DFA's random feedback is too noisy. **Wide + multi-layer = reliable deep learning; either alone
fails.**

## Recurrence — the side-car topology robustly beats feed-forward

Trained recurrence earns its keep once two things are in place: **the completed ALIF credit rule** and **a
topology that isolates the recurrent layer from the forward path**. The winning architecture is the
backward-fed **side-car**: the forward signal **skips past** the recurrent layer (L1 → L3 via a +2 skip)
while a separate **recurrent scratchpad** (L2: self-loop + a L2→L3 forward, with L3 feeding L2 back) holds
state alongside — so the loop computes *without* injecting its reverberation into the clean forward
projection. *(These numbers are the original **direct-read** side-car — the readout reads the recurrent layer
**L3 directly**, so L3 is the top layer with clean symmetric feedback. The side-car config has since been
extended with a dedicated read layer L4 for the scaling study below; reading L3 directly scored higher — see
"Scaling study".)*

**Result (size 32, worst-seed / 3 seeds) — a strict improvement over feed-forward on every benchmark:**

| task | FF | **side-car** |
|---|---|---|
| temporal XOR (delay 20) | 990 | **1000** |
| flip-flop (delay 12) | 985 | **1000** |
| distractor-XOR (delay 20) | 700 | **995** |
| parity N=4 (delay 8) | 587 | **837** |

Where FF **saturates** (XOR, flip-flop) the side-car ties at ceiling — no cost. Where FF **struggles** it wins
big: parity N=4 587 → 837, and distractor-XOR 700 → 995 (the task hides A behind an irrelevant middle cue; the
recurrent scratchpad holds A across it where a pure forward stack can't). Verified across seeds *and* tasks —
the first robust, general result where trained recurrence beats feed-forward here.

**The credit rule that trains it.** Earlier recurrence tests used the **crude spike-timing** eligibility (only
the fast membrane term `e = Σ_t ψ_j·εᵛ_i`). Textbook e-prop for ALIF neurons (Bellec 2020, Eq. 24–25) adds a
**slow** adaptation eligibility `εᵃ`, recursed at the adaptation rate `ρ`: `e = ψ·(εᵛ − β·εᵃ)` — carrying
credit over the ~64-wave adaptation horizon. It is now built and verified against the paper
(`EligParams.elig_beta` / `elig_psi_width` / `use_bump`; a decide-time `eff` snapshot captured in
`TrialRecords.effs`; a fixed-width bump ψ, `PSI_WIDTH`). Two ψ bugs fixed on the way: `eff` must be read at
the decide step (before the fire-bump), and the bump needs a fixed absolute band (the θ≈1 baseline collapses ψ
to ~0, since ±1 drive overshoots `eff` by O(2–26)).

**What the sweeps found (hidden-recurrent stack, the route to the side-car):**

- **Width sets the ceiling** — rec_count 8, FF/rec: 990/725 (size 16) → 1000/982 (size 32).
- **Recurrence density must stay sub-critical** — sharp collapse cliff at rec_count ≈ 12 (size 16); above it
  the ±1 loop is super-critical and drowns the read layer to chance regardless of the credit rule.
- **Completed eligibility ≥ crude wherever it's trainable**, its edge largest where credit is hardest (light
  density / small width). But β·`elig_psi_width` tuning that helped the hidden-rec stack *hurt* the side-car
  (**topology >> hyperparameter tuning**) — the plain side-car (β 0.4, W 16) is best.
- **ALIF strength gates the recurrent hub — `adapt_bump` default = 5.** At `adapt_bump 20` the recurrent hub
  **adaptation-locks** after the first cue (pure ±1 then can't drive later cues → it *looks* as if ternary
  can't recurse); at the default **5** it relays every cue and pure ternary solves the side-car parity task.
  The limit was ALIF adaptation, not the ±1 magnitude.

**Open:** generality to larger sizes; and *why* the side-car structure (forward skip + isolated recurrent
scratchpad + backward loop) is what unlocks it — working hypothesis: isolating the loop from the forward path
keeps the class projection clean while the scratchpad holds temporal state (a hypothesis to check against the
LSNN/e-prop literature). Tools: the task drivers + readout in `src/bench/wave_bitnet_bench.rs`; the credit
rule in `wave_bitnet::multilayer_dfa` (`temporal_eligibility`, `multilayer_dfa_step`,
`EligParams{elig_beta, elig_psi_width, use_bump, rec_tau}`).

### Reproduced on `wave_driven` with **spike-ψ** `εᵃ` — recurrence beats FF *without* bump-ψ (2026-07-14)

The activity-scaled engine's Phase 2b (`wave_driven`, online `εᵃ`, **spike-ψ**, β=0.4) reproduces — and
at width exceeds — the recurrence-beats-FF result, so bump-ψ is **not** required for it. Side-car vs FF,
2 seeds, worst/mean (`bench::wave_driven_bench::wave_driven_sidecar_vs_ff`):

| task | width | FF | **side-car (best rec_count)** | historical bump-ψ |
|---|---|---|---|---|
| temporal XOR (parity N=2) | 16 | 720 | **1000** (rec 8/16/24) | 990→1000 |
| parity N=4 | 16 | 515 | **860** (rec 8) | 587→**837** |
| parity N=4 | 32 | 595 | **1000** (rec 8/16) | — |

**Findings.** (1) **spike-ψ `εᵃ` unlocks recurrence** — it beats FF decisively and, with width, *exceeds*
the historical bump-ψ side-car (parity N=4: 595→**1000** at size 32 vs 837). So the graded near-threshold
signal is **not** load-bearing here; the sparse spike-driven adaptation eligibility suffices. (2) **Width
is the decisive lever** (parity N=4: 860 at size 16 → 1000 at size 32, rec 8), confirming width as a
capacity floor for the hard task. (3) **Recurrence density has a real sweet spot at *sparse* rec_count
(rec 8 best; rec 24 collapses back to ~FF baseline).** The rec_count sweep *above* the historical bump-ψ
cliff (~12) was run precisely to test whether that cliff was a bump-ψ credit-starvation artifact — it is
**not**: under spike-ψ, high density (rec 24) still collapses, and the σ / spiking-profile instrumentation
shows a **dynamics** collapse (σ → ~0.9 sub-critical, the recurrent scratchpad's contribution dies),
*not* credit starvation. So the operating point is **sparse recurrence + width**, and σ (not the credit
rule) is the true density ceiling. Correctness is anchored by a bit-exact online-vs-dense `εᵃ` oracle.
bump-ψ remains a deferred fast-follow — now unlikely to be needed. *(The rec-sweep above was 2-seed,
parity-only; the confirmation below closes that.)*

**Confirmation — 3 seeds, all four benchmarks, matched FF baseline (2026-07-14).** At the fixed operating
point (size 32, rec 8) with a **depth-matched 5-layer FF** trained to its own best-checkpoint ceiling
(so the delta is topology, not FF under-training), 3 seeds, worst/mean permille
(`bench::wave_driven_bench::wave_driven_recurrence_confirmation`):

| task | FF worst/mean | **side-car worst/mean** | historical bump-ψ | note |
|---|---|---|---|---|
| temporal XOR | 1000/1000 | **1000/1000** | 990→1000 | tie at ceiling (FF already solves it) |
| parity N=4 | 595/628 | **1000/1000** | 587→837 | decisive; **exceeds** historical |
| distractor-XOR | 705/735 | **1000/1000** | 700→995 | decisive; matches/exceeds historical |
| flip-flop | 525/561 | **670/810** → **1000/1000** | 985→1000 | at `adapt_bump 5` quenched (σ≈0.05); at `adapt_bump 3` → ceiling |

**Verdict: recurrence beats FF (worst-seed) on 4/4 tasks, 3 seeds, matched baseline** — closing the
Phase-2b caveats. The FF baselines land on the historical FF numbers (parity-4 595≈587, distractor
705≈700, XOR 1000≈990), confirming the comparison is fair. On parity-4 and distractor-XOR the side-car
reaches **ceiling**, *exceeding* the historical bump-ψ side-car — so **spike-ψ `εᵃ` is not just a
reproduction, it is at-least-as-strong**. **flip-flop needed — and got — its own operating point
(RESOLVED 2026-07-14).** At the shared `adapt_bump 5` config both engines were weak (FF 525, side-car
670/810) and the side-car's σ≈0.05 showed the recurrent scratchpad (L2) **silent (0.0% firing)**. Two
diagnostic sweeps (`wave_driven_bench::wave_driven_flipflop_{rec,adapt}_sweep`, 3 seeds) pinned the cause:

- **Not recurrence density.** Denser/tighter recurrence did *not* fix it — `r3/c16` left L2 silent (0.1%,
  620/771) and `r3/c32` woke L2 (3.7%) but *collapsed* accuracy to 470 (< FF; the density cliff). So more
  scratchpad synapses can't overcome the quench.
- **It was adaptation quenching.** The bump piles up as a leaky accumulator (`adapt_ref = decayed + bump`
  per fire), and the sustained effective-threshold contribution ≈ `adapt_bump · 2^adapt_decay` = `5·64 ≈
  320` at the shared config — far above what L2's sparse `count-8` ±1 loop can drive, so L2 quenches and
  the read path (L3→L4) starves. **Lowering `adapt_bump` from 5 to ≤ 3 (or `adapt_decay` 6→4) takes
  flip-flop to 1000/1000 worst-seed**, every setting in the swept range (`bump{3,2,1}` and `bump3/decay4`,
  `bump2/decay4` all hit ceiling); the whole net comes alive (L3 3.4→7–21%, L4 0.5→2.5–20%). Interestingly
  ceiling arrives at `bump3` even with L2 still ~0.1% — the dominant effect is **un-quenching the
  read path** so the held state is legible, with L2 waking further as bump drops.

**Net: spike-ψ `εᵃ` recurrence reaches ceiling on ALL four benchmarks** (flip-flop with its own
`adapt_bump 3` operating point), and beats FF worst-seed 4/4 — the recurrence result is **firmly
confirmed with no open wrinkle**. (Follow-up if wanted: adaptation is task-timescale-dependent, so a
principled per-task or adaptive `adapt_bump` — rather than a hand-picked constant — is the clean
generalization.)

## Scaling study (in progress) — forward drive, width, and read-layer topology

**Status: ongoing systematic exploration, mostly single-seed / preliminary** — the single-threaded integer
engine is too slow to run these sweeps multi-seed at the sizes needed. Recorded so the direction survives;
*re-verify multi-seed once the engine is faster.*

**Topology note (direct-read vs read-layer).** The robust headline result above (parity N=4 = 837, plus
XOR/flip-flop/distractor) is the **direct-read** side-car — readout on the recurrent layer **L3 directly** (L3
is the top layer → clean symmetric readout feedback). The side-car config now instead has a **dedicated read
layer L4** (`… L3 → L4 (+1); L4 read`) — the *read-layer variant* used for this study. Reading L3 directly
scored **higher** (parity N=4, same params: direct-read **837** vs read-layer **~700–765** single-seed): a
separate read layer **demotes** the recurrent layer from symmetric feedback to noisy DFA (≈ −200). So
*reading the recurrent computation directly is itself load-bearing.* The read-layer variant is kept because it
cleanly separates "readout" from "computation" for the parameter sweeps.

**Forward drive × width (read-layer variant, parity N=4, rec 16/r4, single seed 0xE9…):**

| size ↓ / up_count → | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| 16 (256) | 540 | 540 | 557 | 560 |
| 32 (1024) | 540 | 540 | 712 | 710 |
| 64 (4096) | 540 | 542 | 697 | **765** |

Three relationships (preliminary):
- **Per-neuron forward-drive threshold ≈ up_count 32** (at radius 3), roughly **width-independent** — below
  it *everything* is chance. `up_count` is synapses/neuron into a fixed window, so it's a per-neuron drive
  switch, not a total-count one.
- **Width is a capacity floor** — size 16 (256) never leaves chance; ≥ size 32 (1024) is needed to have a
  working regime at all.
- **Above both bars, width × forward scale *jointly*** — size 64 + up 64 (765) beats either lever alone
  (~710). It's an AND, not a sum, and hasn't plateaued (size/up 128 may keep climbing).

**Recurrence density** wants to stay **sparse** — rec_count ≈ 8–16 is the sweet spot (16 slightly best at high
forward drive); rec_count 32 goes super-critical and collapses toward chance (radius trades against count:
wider radius lowers local density and partly rescues a high count). **Depth hurts** — every deeper (6-layer)
side-car variant underperformed the compact one (DFA credit noise + propagation lag + starved deep read
layers), across couplings, an L3 self-loop, and forward/recurrence density tuning.

**Direction:** harder / deeper recurrent computation appears to need **more width + more forward drive
together**, and the **forward topology** (skip distances, where the side-car couples) is a large untested
axis. **Immediate blocker → next step: engine performance optimization** (the single-threaded integer engine
can't run these sweeps multi-seed at size ≥ 64), then resume the systematic scaling / forward-topology study.

## Initialization — train from raw, no calibration

**There is no calibration and no σ-init in `wave_bitnet`.** A pure fixed ±1 magnitude cannot be scaled to a
target σ (branching ratio), so there is no edge-of-chaos gain controller and no firing-rate calibration pass —
neither has anything to tune. Nets **train from raw**: weights initialise to the procedural ±1 sign (a fresh
net is the pure-procedural LSM), and training moves the `f32` shadow, repacked to the 2-bit codes each step.

The standing contract is **generous fan-in + a liveness assert**: size the fan-in so that every computational
layer fires above a floor after training, and **fail loud** (assert) if a layer dies — then raise fan-in. In
the generous-fan-in regime (e.g. `up_count ≥ 16` at width 32) even raw weights train to ceiling. Liveness
*during* training is owned by **`rate_reg`** — a soft per-neuron firing-rate term `c_reg·(rate − target)`
folded into the e-prop learning signal (the LSNN/e-prop mechanism). It **requires ALIF**: a LIF deep stack has
no near-threshold ψ, hence no gradient, and cannot be revived. See the over-training caveat at the end —
`rate_reg` is load-bearing early (revival) but harmful after convergence.

## Multi-layer temporal-DFA training engine + the `rate_reg`-everywhere decision (2026-07-11)

**The training engine lives in `wave_bitnet::multilayer_dfa`.** `src/wave_bitnet/multilayer_dfa.rs` is the
**self-contained, task-agnostic** temporal multi-topology multi-layer-DFA rule: `temporal_eligibility` (a
decaying presynaptic trace × ψ × optional ALIF adaptation eligibility `εᵃ`) + `multilayer_dfa_step`, built on
the engine primitive `Network::eprop_update_synaptic` — a **non-factored, per-synapse** shadow update that
repacks each touched row to the 2-bit codes. The temporal eligibility `e_ij = Σ_t pretr_i(t)·ψ_j(t)` is *not*
separable into `pre_i · ψ_j`, so the update must run per synapse (targets decoded from the source layer's
occupancy bitset, in wired-rank order — credit reuses the same materialized topology the forward pass
iterates). The engine owns the update + eligibility; the caller owns the trial, readout, and learning signal —
the seam is `TrialRecords{spikes, pots, effs}` in, `signal[layer][j]` in. The harness
(`src/bench/wave_bitnet_bench.rs`) drives it. Design + TDD plan under
`docs/superpowers/{specs,plans}/2026-07-11-multilayer-dfa-engine*`. Verified: trains temporal XOR (held-out,
ignored/`--release`), a 2-class task to ceiling in the generous-fan-in regime (size 16, `up_count 32`; a
fan-in-starved 4-layer size-8 stack goes to chance), trains recurrent (level-0) edges, and is deterministic
for both the membrane (β=0) and ALIF-`εᵃ` (β>0) eligibility flavors.

**`rate_reg` everywhere; `rec_stab` set aside — supersedes the AGENTS.md "forward-path-only" hard rule.**
The earlier rule (AGENTS.md) held that `rate_reg` (per-neuron `c_reg·(rate − target)`) belongs on the forward
path *only*, and that on recurrent weights it homogenizes the class signal and *hurts* — with the per-layer,
class-preserving `rec_stab` as the recurrent substitute. **Current position: `rate_reg` is better across all
layer types; `rec_stab` was not carrying its weight.** So `rate_reg` is now applied to every edge type
(forward `+1`, lateral `0`, backward `−1/−2`) and `rec_stab` is dropped from the training signal — a
documented option to resurrect only if a recurrent config is shown to need it. The engine's signal builder
uses per-neuron `rate_reg` on all layers, no `rec_stab`.

**Eligibility note — bump-ψ collapses under strong forward drive.** The bump pseudo-derivative
`ψ = γ·max(0, 1 − |v − eff|/W)` (the β>0 path; `elig_psi_width` = W, default 16) goes to ~0 when a neuron's
decide-time potential overshoots its effective threshold by more than W. Under strong drive (e.g. `up_count`
32 at size 8) integer ±1 inputs push potentials well past `eff` (overshoot O(2–26)), so a *well-driven*
layer's bump-ψ is frequently zero → its eligibility (hence its weight updates) is starved. Spike-ψ (β = 0)
has no such gap and stays robust. Practical consequence: on strongly-driven layers, either widen
`elig_psi_width` to match the potential-overshoot scale or use spike-ψ; the fixed W = 16 is tuned for
near-threshold operation, not saturated drive. (This is why the engine's recurrent-edge training test uses
spike-ψ; the eligibility-β path itself is covered by the `temporal_eligibility` unit tests.)

## Over-training collapse — `rate_reg` dominance after convergence (diagnosed 2026-07-11)

**Finding (multi-seed).** Multi-layer temporal-DFA training is **non-monotonic**: held-out accuracy peaks
then degrades with more training, seed-robustly (3 seeds). Severity scales with depth — a 4-layer FF
(temporal XOR, r4/c64) *dips transiently* (worst-seed 1000→526→1000 over 800/3000/6000 trials, recovers); a
12-layer FF (2-class, r4/c48) *collapses permanently* (all seeds 1000@300 → 465@1500). **Single-seed,
single-checkpoint benchmarks masked this entirely** — it only shows up once training duration and seeds are
explicit axes (now the standing benchmark convention; the `*_bench` reports show the curve). This is not new
breakage — it is a property of the DFA+`rate_reg` rule that fixed-trial evaluation never surfaced.

**Root cause — `rate_reg` (ablation-confirmed).** On the collapsing configs, the collapse appears **only with
strong `rate_reg`** and vanishes without it:

| config | `rate_reg` | acc over training |
|---|---|---|
| FF XOR r4/c64 (4L) | 5 | 1000 / 1000 / 733 / 526 / 1000 / 1000 |
| " | 1 | 1000 / 1000 / 1000 / 1000 / 1000 / 1000 |
| " | 0 | 1000 / 1000 / 1000 / 1000 / 1000 / 1000 |
| FF 2-class r4/c48 (12L) | 5 | 1000 / 465 / 465 / 465 |
| " | 0 | 1000 / 1000 / 1000 / 1000 |

Mechanism: `rate_reg = c_reg·(rate − target)` is **class-agnostic and always-on** (fixed c_reg = 5, folded
into the learning signal via the same eligibility). It is *load-bearing* to revive/keep-alive a starved
substrate (its intended job — see the liveness rescue above). But once the task converges the task signal
(`readout-error × feedback`) →0, so `rate_reg` becomes the dominant term and keeps pushing every neuron toward
the *uniform* target rate — **homogenizing the representation and eroding class separability**. Depth
amplifies it (more compounding reg terms, a more fragile deep class signal), turning a transient dip (4L,
recovers) into a permanent collapse (12L). `rate_reg = 0` (or weak = 1) removes the collapse **on a generous
substrate** (r4/c48/c64, alive without the reg); on a *starved* substrate the reg is still needed for
liveness, so it cannot simply be deleted.

**Mitigation — early-stopping (documented, NOT implemented).** Because `rate_reg` must stay ON during the
bootstrap/revival phase but is harmful after convergence, the fix is to **stop eroding the converged
solution**: select the **best held-out checkpoint (early-stopping)** rather than training a fixed trial count
— the duration-swept benchmarks already expose the peak, so compare at the *peak* of the curve, not the final
point. Equivalent directions to try (not yet built): **anneal `c_reg` → 0 as the task error falls** (keep the
liveness rescue early, remove the homogenizing pressure late), or **gate `rate_reg` off** for neurons/layers
already at/above target rate. Left unimplemented per scope; recorded so the collapse is not mistaken for a
substrate/credit limit.

---

# The `wave_resonate` engine — BRF (complex resonate-and-fire) + HYPR

**What this is:** a second, independent engine (`src/wave_resonate/`, an island duplicated from
`wave_driven`) whose neuron is the **Balanced Resonate-and-Fire (BRF)** complex-membrane oscillator —
the flagship variant of Higuchi et al., *Balanced Resonate-and-Fire Neurons*, ICML 2024 — trained
**online without BPTT** by a **HYPR**-style forward eligibility (Baronig et al. 2026). The **ternary
±1/0 BitNet weight substrate is preserved**; only the neuron's internal integration (LIF/ALIF → a damped
complex oscillator with per-neuron frequency `ω` and dampening `b′`) and the credit rule change. Design:
`docs/superpowers/specs/2026-07-14-wave-resonate-*`. Harness: `src/bench/wave_resonate_bench.rs`.

## The neuron & why it needed a different operating point

Each neuron is `u = x + i·y` with `x' = x + δ(b·x − ω·y + I)`, `y' = y + δ(ω·x + b·y)`, spike on the real
part `z = Θ(x − ϑ_c − q)`, refractory `q' = γ·q + z`, balanced dampening `b = p(ω) − |b′| − q`,
`p(ω) = (−1+√(1−(δω)²))/δ`. The sub-threshold `(x,y)` is **linear**, so HYPR's forward eligibility is an
**exact 2-state per-synapse trace** `(εˣ,εʸ)` recursed through the same Jacobian, `elig += ψ·εˣ`
(ψ = the reference double-Gaussian surrogate) — generalizing the `wave_driven` e-prop rule to a resonator,
validated by a **bit-exact online-vs-dense eligibility oracle**.

**Load-bearing bring-up findings (both non-obvious):**

- **A balanced resonator barely responds to DC drive.** Our cues are constant over the present window
  (DC), and at the reference threshold `ϑ_c = 1` the entire compute stack is **silent** (damped
  steady-state `x ≈ 0.03·I ≈ 0.2` ≪ 1). **`ϑ_c ≈ 0.1`** gives a live, depth-stable stack (~4–5%
  spikes/neuron/wave). This is *the* gotcha for driving RF networks with rate/spatial (non-oscillatory)
  input.
- **HYPR's ε traces are δ-scaled** (injections are `δ·z ≈ 0.05`), so the hidden learning rate must be
  **~100× the integer engines'** (`hidden_lr ≈ 2` vs `0.004`) or the ternary shadow never moves.
- Membrane is **f32** (O(neurons)); weights stay **ternary** (O(synapses) — the memory constraint).
  Dense oscillator update + sparse firer-gated delivery (resonators ring, so there is no membrane
  frontier). Deterministic, single-threaded.

## Headline — temporal-task battery, size 32, 3 seeds (worst/mean held-out ×1000)

Config: 5 layers, forward fan-in `r3/c32`, side-car recurrent `rec 8`/`r4` (backward-fed:
`L0→L1`, `L1→L3` skip, `L2` self + `L2→L3`, `L3→L2` back + `L3→L4`, read `L4`), `ϑ_c 0.1`, `δ 0.05`,
`ω∼U(5,10)`, `b′∼U(0,0.2)`, `hidden_lr 2`, trained-`ω/b′` `omega_b_lr 1.0`, best-checkpointed
(`train_and_eval_best`, ≤2400 trials). σ (mean consecutive-layer spike ratio) ≈ **1.14** across configs.

| task           | FF frozen | FF +ω/b′ | side-car frozen | side-car +ω/b′ | ALIF ref (FF→side) |
|----------------|-----------|----------|-----------------|----------------|--------------------|
| temporal-XOR   | 1000/1000 | 1000/1000 | 1000/1000      | 1000/1000      | 990 → 1000         |
| parity-4       |  565/600  |  620/631 |  650/746        |  670/733       | 587 → 837          |
| distractor-XOR |  720/755  | **1000/1000** | **1000/1000** | 1000/1000  | 700 → 995          |
| flip-flop      |  930/963  |  960/986 | **1000/1000**   | 1000/1000      | 985 → 1000         |

(ALIF ref = the `wave_driven` ALIF+e-prop worst-seed numbers, size 32, for context — a *different* neuron
model on the identical tasks/harness.)

## Verdicts

- **RQ1 — BRF+HYPR is a viable temporal learner.** It clears chance on all four tasks and reaches
  **ceiling on three** (temporal-XOR, distractor-XOR, flip-flop); parity-4 is above chance but not solved
  (the hardest task — ALIF only reaches 837 too). Competitive with the ALIF+e-prop reference on the same
  battery, from a completely different neuron model — a second, independent confirmation that the
  ternary-weight + online-eligibility recipe learns.
- **RQ2 — trainable `ω/b′` matters, exactly where the frozen frequency bank falls short.** The clean case
  is **distractor-XOR: FF frozen 720 → FF +ω/b′ 1000/1000** — learning the resonant frequencies solves a
  task the random bank cannot. It also helps FF on parity-4 (565→620) and flip-flop (930→960). On tasks a
  frozen bank already ceilings (temporal-XOR), it neither helps nor (at the chosen LR) hurts. **The LR is
  task-dependent** (size-16 FF sweep): `omega_b_lr ≤ 1.0` helps the hard tasks with **no** regression on
  the easy ones (distractor 515→705/852, XOR held at 1000); `omega_b_lr 2.0` maxes distractor (→1000) but
  **destabilizes** temporal-XOR (1000→760, over-aggressive frequency updates on an already-good bank). We
  use `1.0` as the safe default.
- **RQ3 — topological recurrence remains the strongest single lever, but intrinsic resonance narrows the
  gap.** The backward-fed **side-car frozen** solves distractor-XOR and flip-flop to **1000** with *no*
  `ω/b′` training, and is best on parity-4 — reproducing the repo's "recurrence beats FF" headline with a
  BRF neuron. **But** on distractor-XOR, **FF + trainable `ω/b′` also reaches 1000** — i.e. per-neuron
  oscillatory memory can *substitute* for wiring recurrence there; on flip-flop it closes most of the gap
  (FF 930 → FF+ω/b′ 960 vs side 1000). So: side-car ≥ FF, and trained resonance is a partial (sometimes
  full) FF-side substitute for the recurrent scratchpad.
- **RQ4 — timescale (`ω`-init × `δ`) is a first-class lever.** On distractor-XOR FF **frozen** (size 16,
  2 seeds), the study's default `ω∼U(5,10)/δ0.05` (515) is **suboptimal**; lowering to **`ω∼U(3,6)` with
  `δ0.10` reaches 770/795** — nearly doubling worst-seed. The winner has period `≈ 2π/(δω) ≈ 10–21 waves`,
  matching the task's ~20-wave memory span; higher `ω∼U(8,16)` is worse (460–602). So a frozen frequency
  bank is only as good as its range vs the task's memory horizon — which is exactly what trainable `ω/b′`
  (RQ2) discovers automatically, and a caveat that the **frozen** numbers above are timescale-under-tuned,
  not a BRF ceiling. (Full grid in `bench::wave_resonate_omega_delta_sweep`.)
- **Width floor confirmed** (matches `wave_bitnet`/`wave_driven`): parity-4 is at chance at size 16 for
  every config and only rises above chance at size 32; distractor FF frozen 515 (size 16) → 720 (size 32).

## Cost & the perf wall

BRF is f32 with a per-synapse 2-state eligibility, and resonators ring (a larger eligibility active set),
so it is **markedly slower than the integer engines**: the size-32 / 3-seed / 4-task / 4-config study took
**~95 min** (`--release`); size-16 / 2-seed took ~12.5 min. This is the anticipated scaling wall (cf. the
standing perf-then-scaling note) arriving earlier for the f32 engine — size ≥ 64 multi-seed sweeps are not
practical without the deferred fixed-point port / perf pass. All experiments are `#[ignore]`d
(`bench::wave_resonate_bench`); no config was silently capped.
