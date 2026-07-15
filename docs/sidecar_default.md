# Side-car — default configuration

- **Status:** the default for future `wave_driven` side-car tests, as of 2026-07-15.
- **Provenance:** selected by measurement across 210 runs (Phase A1 39, A2 9, Phase B 162), size 16,
  5 layers, 3 seeds per cell. Full results: `experiments_results_seq_memory.md`.
- **Code:** `src/bench/wave_driven_seq_bench.rs` (`OP_*` consts + `make_sidecar_seq` + `seq_cfg`).

## The default

```
adapt_bump = 3      rate_reg = 5.0
```

Start here unless you have a reason not to. It was the strongest cell on every axis in the
`adapt_bump {1,3,5} × rate_reg {0,2,5}` grid: **worst-seed fidelity 0.936 / mean 0.956**, deterministic
prefix accuracy **1.000 on every seed of every task size**, and family accuracy **1.000** (against
Markov-2 ceilings of 0.500/0.333/0.250). It is the only configuration that never drops below 0.93
anywhere.

## Layout

Forward path is **L0 → L1 → L3 → L4**. L2 is a memory loop hanging off L3, **not** a pipeline stage —
`L1→L3` is a `level: +2` edge that skips over it. This is what "recurrent layer isolated from the forward
path" means.

```
  forward path:  L0 ──► L1 ──────────► L3 ──► L4        (L2 is NOT on it)

   L0                L1                    L3                L4
   transducer        hidden                hidden            READ
  ┌──────────┐      ┌──────────┐          ┌──────────┐      ┌──────────┐
  │  16×16   │ +1   │  16×16   │   +2     │  16×16   │  +1  │  16×16   │
  │ 32 sites ├─────►│          ├─────────►│          ├─────►│ no out   │
  │  10.9%   │r3/c48│   9.4%   │  r3/c48  │  11.0%   │r3/c48│  12.9%   │
  └──────────┘      └──────────┘          └────┬─────┘      └────┬─────┘
                      (skips L2) ───────────►  │ ▲                │
                                        −1     │ │  +1            ▼
                                      r4/c8    │ │ r4/c8      9-class linear
                                               ▼ │            readout over
                                          ┌──────────┐        read-window
                                    self  │    L2    │        spike counts
                                    0 ┌──►│scratchpad│
                                 r4/c8 └──┤   0.4%   │
                                          └──────────┘
```

Percentages are the measured per-layer firing rate at the default (bump 3 / rate_reg 5).

**Edge list** (`level` is the relative layer offset — `tz = z + level`):

| from | level | to | radius/count |
|---|---|---|---|
| L0 | +1 | L1 | r3 / c48 |
| L1 | **+2** | **L3** | r3 / c48 — *skips the scratchpad* |
| L2 | 0 | L2 | r4 / c8 — self-loop |
| L2 | +1 | L3 | r4 / c8 |
| L3 | **−1** | **L2** | r4 / c8 — back-feed |
| L3 | +1 | L4 | r3 / c48 |
| L4 | — | — | empty (read directly) |

L0 is forced to a transducer by the engine (`threshold = i16::MAX`, `adapt_bump = 0`), so 5 layers is
4 computing layers. The top layer is read directly — **no dedicated readout layer** (`wave_driven` has no
`readout` flag; it was removed as a silent training killer).

## Parameters

| | value | chosen by |
|---|---|---|
| **`adapt_bump`** | **3** | Phase B — best cell |
| **`rate_reg`** | **5.0** | Phase B — best cell |
| forward radius / count | r3 / c48 | Phase A1 — count is the lever |
| recurrent radius / count | r4 / n8 | Phase A2 |
| input density | 32 sites/token (of 256) | Phase A1 — d1 beat d2 |
| `adapt_decay` | 6 (τ ≈ 64 waves) | inherited — **untested axis** |
| `elig_beta` / `rec_tau` | 0.4 / 20.0 (spike-ψ εᵃ) | inherited — makes recurrence trainable |
| `rate_target` | 0.1 | inherited |
| `readout_lr` / `hidden_lr` | 0.02 / 0.004 | inherited from the 2-class harness |
| leak / cooldown / jitter / baseline | (3,5) / 2 / 32 / 6 | inherited from the battery |
| `max_trials` / `eval_every` / `patience` | 10000 / 100 / 10 | A1+A2 measurement (peaks land 1300–5400) |

**`adapt_bump` must stay > 0.** `elig_beta 0.4` is the spike-ψ εᵃ term, and it needs an adaptation trace
to couple to. Bump 0 is not an available setting for the side-car.

## Why these values

Side-car mean fidelity across sets 4/5/6, 3 seeds (`rate_reg` × `adapt_bump`):

| bump | rr0 | rr2 | rr5 |
|---|---|---|---|
| 1 | 0.938 | 0.884 | 0.936 |
| **3** | 0.869 | 0.861 | **0.956** |
| 5 | 0.727 | 0.833 | 0.931 |

**`rate_reg` is needed in proportion to `adapt_bump`.** Adaptation adds to the effective threshold
(`eff = threshold + adapt`), so higher bump quenches the stack; `rate_reg` undoes exactly that. At bump 1
the regulariser buys nothing (rr0 0.938 ≈ rr5 0.936); at bump 5 it is the difference between collapse and
health (0.727 → 0.931).

**Viable alternative: `adapt_bump 1, rate_reg 0`** — worst 0.882 / mean 0.938, also `det 1.000`
everywhere. It is the **cheapest configuration in the grid**: no `rate_reg` term to compute and minimal
adaptation. Use it when throughput matters more than the last ~2% of fidelity. The two defaults are the
same insight from opposite ends: either keep bump low and skip the regulariser, or run bump 3 and pay for
`rate_reg` to undo the quenching.

**Avoid `adapt_bump 5, rate_reg 0`** — the only cell in the grid that fails to demonstrate memory (family
0.667 at set 5, 0.750 at set 6). Max quenching, no rescue.

**`rate_reg 2` is an anomalously bad middle** at bump 1 and 3 — worse than both 0 and 5. Unexplained.
Plausibly it homogenizes enough to erode the learned patterns without being strong enough to rescue
liveness, but that is a guess, not a finding.

## Note on `n16/r4` — recurrent fan-in

Phase A2 (side-car, 4-set, bump 3, rate_reg 5, 3 seeds):

| rec | fidelity w/m | det worst | family worst | profile |
|---|---|---|---|---|
| n8/r3 | 0.927 / 0.945 | 1.000 | 1.000 | `[10.9, 9.4, 0.4, 10.9, 12.7]` |
| **n8/r4** *(default)* | **0.947 / 0.961** | 1.000 | 1.000 | `[10.9, 9.4, 0.4, 11.0, 12.9]` |
| n16/r4 | 0.949 / 0.956 | 1.000 | 1.000 | `[10.9, 8.9, 10.3, 13.8, 16.2]` |

**`n16/r4` is statistically tied with the default** — 0.949 vs 0.947 worst, 0.956 vs 0.961 mean. It is not
better; it costs 2× the recurrent synapses. **n8/r4 is the default on cost, not on performance.** If a
future task needs more recurrent capacity, n16/r4 is the first thing to try and should be expected to
perform about the same at 2× the cost.

**The unexplained part, and why n16 is worth remembering.** The two configurations run their scratchpad at
wildly different activity levels — **L2 at 0.4% for n8 versus 10.3% for n16**, a 25× difference — and
perform identically. On a 256-neuron layer, 0.4% is roughly *one neuron firing per wave*. So whatever the
side-car's memory advantage is, **it does not appear to live in how much L2 fires**, and the current
metrics cannot settle where it does live. Anyone probing the mechanism should start from this pair: same
outcome, 25× different dynamics.

A practical consequence: **`n16/r4` is the variant whose σ is meaningful.** σ averages ratios over
*consecutive* layers, which assumes every layer is on the forward path. With L2 at 0.4%, the L3/L2 ratio
is ≈ 27 and σ reports **8.657** on a demonstrably healthy stack. At n16/r4, L2 is active and σ reads
1.198. **Do not compare FF and side-car σ, and do not read σ for n8 at all** — use the per-layer profile,
which stays honest for every topology.

## Caveats

- **`adapt_decay` (6) is the untested axis.** It sets the adaptation trace's *horizon* where `adapt_bump`
  sets its *amplitude*. The prediction that adaptation carries the FF memory was refuted by the bump
  sweep, but only because high bump quenches the stack dead before any memory benefit can show. Sweeping
  `adapt_decay` at fixed low bump is the experiment that would actually settle it.
- **Token-drive confound.** `token_sites` uses a per-site coin flip, so per-token site counts are binomial
  (`[28, 25, 42, 36, 32, 40, 30, 29, 30]` at density 1 — a 1.7× spread) and token identity correlates with
  drive strength. The side-car is silent on 1 of 15 prefixes because of it (FF: 6 of 15). Fidelity numbers
  here are therefore slightly depressed and partly measure drive-robustness. **The family/memory result is
  unaffected** — every family prefix ends in the strongest token. Fix before re-running: sample exactly N
  distinct sites per token (partial Fisher-Yates, as the engine's `sample_distinct_cells` does).
- **The operating point may itself be an artifact.** Phase A1 chose density 1 / c48 *under* the confound
  above; equalizing token drive could move it.
- **`rate_profile_seq` measures the wrong regime for readout prediction** — it profiles activity under
  continuous drive, while the readout integrates a read window with no drive. A live profile does not
  imply a usable readout.
- **These numbers are memorization, not generalization** — the task has no holdout by design.
