# wave_driven — sequence-recall memory results (2026-07-15)

- **Question:** can `wave_driven` *memorize* a branching sequence set — reproduce deterministic
  continuations, match fork marginals as calibrated readout mass, and resolve a prefix family that only
  a 3-token memory can answer?
- **Spec:** `docs/superpowers/specs/2026-07-15-wave-driven-sequence-memory-design.md`
- **Plan:** `docs/superpowers/plans/2026-07-15-wave-driven-sequence-memory.md`
- **Code:** `src/bench/wave_driven_seq_bench.rs` (`#[ignore]`d, run in `--release`)
- **Runs:** Phase A1 (39) + A2 (9) + Phase B (162) = 210, size 16, 5 layers, 3 seeds each.

## Headline

**Yes — and the side-car does it reliably while feed-forward does not.**

- **Memory is demonstrated.** On the `[·,2,3]` disambiguation family — where a Markov-2 model is
  provably capped at 1/k — the side-car reaches **family accuracy 1.000 on the worst seed of all 27
  Phase-B cells**, against ceilings of 0.500 / 0.333 / 0.250 for the 4/5/6-sequence sets. It resolves a
  **4-way** first-token disambiguation perfectly. FF fails (worst-seed family at or below the ceiling)
  in **8 of 27** cells.
- **The side-car beats FF in 27/27 cells** on fidelity. *Partly confounded — see Caveats.*
- **This task is discriminative**, unlike the existing battery where all four tasks ceiling on both
  topologies. FF spans 0.465–0.925 mean fidelity, side-car 0.552–0.966. Real headroom.
- **The central `adapt_bump` hypothesis is refuted, and inverted.** See below — this is the most
  important negative.

Chance fidelity (uniform predictor) is 1/V ≈ **0.111**.

## Setup

Operating point **selected by measurement, not assumption**: `density 1` (~32 of 256 sites/token),
`r3/c48`, side-car `rec n8/r4`, `adapt_decay 6`, `present 6 / delay 4 / read 8`, `readout_lr 0.02`,
`hidden_lr 0.004`, `rate_target 0.1`, `max_trials 10000`, `eval_every 100`, `patience 10`.

Sequences (token ids; `0→"1" 1→"2" 2→"3" 3→"4" 4→"5" 5→"6" 6→"7" 7→"8" 8→"16"`):

```
S1 1→2→3→4     S4 2→2→3→5     (set 4 = S1..S4, 9 prefixes,  family 2-way, markov-2 0.500)
S2 1→2→4→8     S5 3→2→3→6     (set 5 = S1..S5, 12 prefixes, family 3-way, markov-2 0.333)
S3 1→4→8→16    S6 4→2→3→7     (set 6 = S1..S6, 15 prefixes, family 4-way, markov-2 0.250)
```

`fidelity` = mean over all prefixes of `1 − TV(truth, softmax)`. For a deterministic prefix that is
exactly the softmax mass on the target; for a fork it is the calibration score. One scalar for both.

## Phase A1 — fan-in is the whole game

FF, 4-set, `adapt_bump 3`, `rate_reg 5`, 3 seeds, trained to peak:

| cell | fan-in/neuron | fidelity w/m | family worst | σ | dead layers |
|---|---|---|---|---|---|
| **d1/c48** | 6.0 | **0.818 / 0.880** | **1.000** | 1.094 | 0 |
| d2/c40 | 10.0 | 0.844 / 0.878 | 0.500 | 0.825 | 0 |
| d2/c48 | 12.0 | 0.798 / 0.800 | 0.500 | 0.938 | 0 |
| d2/c32 | 8.0 | 0.591 / 0.706 | 1.000 | 0.464 | 0 |
| d1/c40 | 5.0 | 0.514 / 0.580 | 0.500 | 0.459 | 0 |
| d2/c24 | 6.0 | 0.142 / 0.161 | 0.500 | 0.097 | 2 |
| d1/c32 | 4.0 | 0.263 / 0.353 | 0.500 | 0.155 | 2 |
| **d2/c16** | 4.0 | **0.136 (chance)** | 0.000 | 0.006 | 3 |
| d1/c16 | 2.0 | 0.136 (chance) | 0.000 | 0.000 | 3 |

Radius sweep at c24, density 2: r2 → σ 0.127, r3 → 0.097, r4 → 0.129, all with 2 dead layers.

**`r3/c16` — the spec's assumed operating point — cannot train at 5 layers.** Untrained profile
`[22.3, 8.1, 1.3, 0.0, 0.0]`: activity dies by L3, the read layer never spikes, `act` is all zeros, the
readout SGD multiplies by zero. It stays at chance (0.136) *after* training too.

**Count is the lever; radius is noise.** r2/r3/r4 at fixed c24 are indistinguishable (σ 0.127/0.097/0.129).
Confirms AGENTS.md's "count is the lever, radius stays tight".

**Density and `c` are not interchangeable.** At *equal* input drive (4 synapses/neuron), untrained
`d1/c32` measures σ 0.473 but `d2/c16` measures σ 0.069 — a 7× gap. Density only affects the L0→L1 hop;
`c` also sets every hidden hop, where the source is L1's firing neurons rather than the injected sites.
That hidden job dominates σ. The spec's claim that drive is set by the product `sites × c` is **wrong**.

**The coincidence floor is ~6–8 synapses/neuron, not the spec's ~2.** The spec's arithmetic ignored sign
cancellation: untrained ternary weights are random ±1/0, so of 8 synapses ~2.7 are +1 and ~2.7 are −1 and
they largely cancel — ~8 raw synapses are needed to reliably net ≥2.

**`rate_reg` cannot rescue a *fully* dead layer.** c16 stays at chance with 3 dead layers even after
training with `rate_reg 5`. Eligibility accrues only on **target fire**, so a silent layer has `e = 0` and
`shadow += -lr·signal·0` is identically zero — there is nothing for the regulariser to act on.
**AGENTS.md's "conclusive liveness rescue" has an unstated precondition: the layer must be weakly firing,
not silent.**

Peak trials: every live FF cell peaks by 2600 (most 300–1400). The plan's `max_trials 12000` was ~5×
over-provisioned; a side-car seed later peaked at 5400, so the ceiling was set to 10000.

## Phase A2 — recurrent fan-in

Side-car, 4-set, `adapt_bump 3`, `rate_reg 5`, at the A1 operating point:

| rec | fidelity w/m | det worst | family worst | profile |
|---|---|---|---|---|
| n8/r3 | 0.927 / 0.945 | 1.000 | 1.000 | `[10.9, 9.4, 0.4, 10.9, 12.7]` |
| **n8/r4** | **0.947 / 0.961** | 1.000 | 1.000 | `[10.9, 9.4, 0.4, 11.0, 12.9]` |
| n16/r4 | 0.949 / 0.956 | 1.000 | 1.000 | `[10.9, 8.9, 10.3, 13.8, 16.2]` |

n8/r4 chosen (the recorded sweet spot; n16 is statistically tied and costs more).

**σ is meaningless for the side-car.** It averages ratios over *consecutive* layers, which assumes every
layer is on the forward path. The side-car's is **L0→L1→L3→L4 with L2 an isolated scratchpad**, so a quiet
L2 (0.4%) makes L3/L2 ≈ 27 and reports σ 8.657 on a demonstrably healthy stack. **Do not compare FF and
side-car σ.** The per-layer profile stays honest for both and is the diagnostic used in Phase B.

## Phase B — the main grid (162 runs)

Mean fidelity over 3 seeds; **bold** = side-car wins the cell; `✗` = worst-seed family accuracy at or
below the Markov-2 ceiling (memory *not* demonstrated).

| set | bump | rate_reg | FF | side-car |
|---|---|---|---|---|
| 4 | 1 | 0 | 0.925 | **0.956** |
| 4 | 1 | 2 | 0.753 ✗ | **0.907** |
| 4 | 1 | 5 | 0.826 | **0.956** |
| 4 | 3 | 0 | 0.846 ✗ | **0.931** |
| 4 | 3 | 2 | 0.807 | **0.923** |
| 4 | 3 | 5 | 0.880 | **0.961** |
| 4 | 5 | 0 | 0.812 ✗ | **0.928** |
| 4 | 5 | 2 | 0.933 | **0.966** |
| 4 | 5 | 5 | 0.944 | **0.948** |
| 5 | 1 | 0 | 0.776 | **0.939** |
| 5 | 1 | 2 | 0.721 ✗ | **0.881** |
| 5 | 1 | 5 | 0.775 | **0.930** |
| 5 | 3 | 0 | 0.613 ✗ | **0.863** |
| 5 | 3 | 2 | 0.679 | **0.820** |
| 5 | 3 | 5 | 0.765 | **0.960** |
| 5 | 5 | 0 | 0.512 ✗ | **0.639** |
| 5 | 5 | 2 | 0.594 | **0.735** |
| 5 | 5 | 5 | 0.684 | **0.943** |
| 6 | 1 | 0 | 0.763 | **0.919** |
| 6 | 1 | 2 | 0.620 ✗ | **0.864** |
| 6 | 1 | 5 | 0.645 | **0.921** |
| 6 | 3 | 0 | 0.643 | **0.813** |
| 6 | 3 | 2 | 0.628 | **0.841** |
| 6 | 3 | 5 | 0.735 | **0.947** |
| 6 | 5 | 0 | 0.488 ✗ | **0.614** |
| 6 | 5 | 2 | 0.540 | **0.799** |
| 6 | 5 | 5 | 0.639 | **0.902** |

**Side-car wins 27/27.** FF fails the memory test in 8/27; the side-car in **0/27**.

### 1. Recurrence beats FF, on a task with headroom

The existing battery's recurrence result is real but unquantified — all four tasks ceiling on both
topologies, so the delta cannot be ranked. Here nothing ceilings, and the side-car wins every cell. The
gap widens with difficulty: at set 6 the mean margin is **+0.19 fidelity**, and at `bump5/rr0` FF collapses
to 0.488 while the side-car holds 0.614.

### 2. The `adapt_bump` hypothesis is REFUTED — and inverted

The spec predicted: FF stores the first token in the adaptation trace, so **lowering** `adapt_bump` should
degrade FF while leaving the side-car (which stores it in recurrent spiking) flat — a crossing.

**No crossing exists.** Mean fidelity over `rate_reg`:

| set | FF b1 → b3 → b5 | side-car b1 → b3 → b5 |
|---|---|---|
| 4 | 0.835 → 0.844 → **0.896** (rises) | 0.940 → 0.938 → 0.947 (flat) |
| 5 | **0.757** → 0.686 → 0.597 (falls) | 0.917 → 0.881 → 0.772 (falls) |
| 6 | **0.676** → 0.669 → 0.556 (falls) | 0.901 → 0.867 → 0.772 (falls) |

At sets 5 and 6 **both topologies degrade as bump rises** — FF by 18%, side-car by 14%. Not a crossing;
a common slope. The profiles say why: adaptation adds to the effective threshold (`eff = threshold + adapt`),
so higher bump **quenches** the stack. FF at set5/`bump5/rr0` reads `[10.9, 4.1, 2.1, 0.5, 0.0]` — dead at
L4 — versus `[10.9, 8.0, 6.6, 5.2, 6.0]` at bump 1. This is the "adaptation-quenched, not density-starved"
effect the flip-flop work already found.

**The experiment cannot separate "adaptation as memory" from "adaptation as quencher."** At high bump the
stack dies and everything fails for liveness reasons, so the memory benefit — if any — never gets a chance
to show. The direction even flips with difficulty (set 4 FF *improves* with bump), which is unexplained.

To test adaptation-as-memory properly you would need to raise bump **without** losing liveness — e.g.
sweeping `adapt_decay` (the trace's *horizon*) at fixed low bump, or compensating the threshold. That is
the natural follow-up; **`adapt_decay` was fixed at 6 here and is the untested axis.**

### 3. `rate_reg` — liveness dominates erosion

Mean fidelity over `adapt_bump`. **`rate_reg 5` is best in all 6 (set × topology) combinations:**

| set | FF rr0 → rr2 → rr5 | side-car rr0 → rr2 → rr5 |
|---|---|---|
| 4 | 0.861 → 0.831 → **0.883** | 0.938 → 0.932 → **0.955** |
| 5 | 0.634 → 0.665 → **0.741** | 0.814 → 0.812 → **0.944** |
| 6 | 0.631 → 0.596 → **0.673** | 0.782 → 0.835 → **0.923** |

**The spec's erosion worry did not materialise as a net negative** — but it is visible where predicted. In
the one regime where the stack is *already* healthy (set 4, bump 1) `rate_reg 0` wins outright:
**0.925 (rr0) > 0.826 (rr5) > 0.753 (rr2)**. Elsewhere the liveness rescue dominates: at `bump5` FF goes
from a dead `[…0.6, 0.0]` at rr0 to a live `[…2.9, 2.5]` at rr5.

So `rate_reg` is **self-limiting in the right direction**: it earns its keep exactly where adaptation has
quenched the stack, and costs accuracy exactly where the stack didn't need it. Fixing `rate_reg 5` as the
spec originally proposed would have hidden both halves — promoting it to an axis was the right call.

### 4. Capacity is not the limit at 6 sequences

Best side-car mean fidelity: set 4 **0.966** → set 5 **0.960** → set 6 **0.947**, with family accuracy
1.000 throughout. Going from a 2-way to a 4-way first-token disambiguation costs ~2% fidelity. The
capacity ceiling is somewhere beyond 6 sequences / 15 prefixes.

## Caveats — read before citing

**1. Token codes carry unequal drive; the fidelity comparison is confounded.** `token_sites` uses an
independent per-site hash draw, so counts are Binomial(256, ⅛): measured **`[28, 25, 42, 36, 32, 40, 30,
29, 30]`** at density 1 — a **1.7× spread**. At c48 that is 4.7 synapses/neuron for token 1 versus 7.9 for
token 2, straddling the ~6 coincidence floor. **Token identity is therefore correlated with drive
strength.**

Consequence, measured directly (FF, set 6, bump1/rr5, seed 1): **6 of 15 prefixes produce zero
read-window activity**, and they are exactly the ones *ending* in token 0 (28 sites) or token 1 (25 sites):

```
[0] SILENT  [0,1] SILENT  [1] SILENT  [1,1] SILENT  [2,1] SILENT  [3,1] SILENT
[2] 221     [0,3] 319     [3] 202     [0,1,2] 382   [1,1,2] 352   [3,1,2] 415
```

The side-car is silent on only **1 of 15** — its recurrence sustains activity where FF's has decayed.

- **The memory result is NOT affected.** Every family prefix `[·,1,2]` ends in token 2 — the *strongest*
  token, 42 sites — and all carry healthy activity (352–415). `family_acc` is measured on clean signal.
- **`fidelity` IS affected.** It averages over all prefixes including silent ones, so FF is penalised for
  an *encoding* weakness, not a memory one. **"Side-car wins 27/27 on fidelity" is partly measuring
  "recurrence sustains activity through the read window."** Interesting, but not what the metric claims.

Fix for a re-run: sample **exactly N distinct sites** per token (partial Fisher-Yates, as the engine's own
`sample_distinct_cells` does) instead of a per-site coin flip. Not done here.

**2. This is a memorization/capacity measurement, not generalization.** There is **no holdout, by design**:
the input universe *is* these 9/12/15 prefixes, and a held-out prefix's answer is arbitrary rather than
derivable. Train and test are the same items — that is what memorization means. The Markov-2 control does a
holdout's actual job, ruling out the answer-from-recent-context shortcut. **Do not cite these numbers as
generalization.**

**3. `rate_profile_seq` measures the wrong regime for readout prediction.** It profiles activity under
*continuous* drive, but the readout integrates a read window with *no* drive. `set6/bump1/rr5/ff` has a
healthy driven L4 (5.2%) yet is silent on both forks. A live profile does not imply a usable readout.

**4. Best-checkpoint selection.** Reported numbers are the peak over evals, which selects on the reported
set. The bias is far weaker than the battery's — this evaluation has **no sampling noise**, so the max
reads the true peak of a deterministic curve rather than the top of the noise — and both topologies are
measured identically, so the FF/side-car delta is unaffected. But the absolute figures are an upward-biased
estimate of "accuracy at a fixed sensible stopping point".

**5. Stated deviations from the AGENTS.md defaults.** Depth fixed at 5 (matched between topologies, so the
comparison isolates topology). `adapt_decay` fixed at 6. Fork calibration is reported per-cell as raw TV
rather than analysed; `forkTV 0.778` is the signature of a uniform softmax, i.e. a silent read window
(caveat 1), not a calibration failure.

## Follow-ups

1. **Fix the encoding** (exactly-N site codes) and re-run A1 + B. The memory finding should survive; the
   FF-vs-side-car fidelity gap will likely narrow. A1's operating-point choice may also change — `d1/c48`
   may have won partly *because* weak tokens dragged the denser cells down.
2. **Sweep `adapt_decay`** at fixed low `adapt_bump` to test adaptation-as-memory without the quenching
   confound. This is the experiment that would actually settle §2.
3. **Push capacity past 6 sequences** — the ceiling was not reached.
4. **Make σ topology-aware** (follow the forward path, skipping the side-car's isolated scratchpad) or
   retire it in favour of the per-layer profile for non-FF topologies.
