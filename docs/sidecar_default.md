# Side-car — default configuration

- **Status:** the default for future  side-car tests, as of 2026-07-15.

## The default

```
adapt_bump = 3      rate_reg = 5.0
```

## Side car Layout

Forward path is **L0 → L1 → L3 → L4**. L2 is a memory loop hanging off L3, **not** a pipeline stage —
`L1→L3` is a `level: +2` edge that skips over it. This is what "recurrent layer isolated from the forward
path" means.

```
  forward path:  L0 ──► L1 ──────────► L3 ──► L4        (L2 is NOT on it)

   L0                L1                    L3                L4
   transducer        hidden                hidden            CONTINUOUS_READOUT
  ┌──────────┐      ┌──────────┐          ┌──────────┐      ┌──────────┐
  │  16×16   │ +1   │  16×16   │   +2     │  16×16   │  +1  │  16×16   │
  │ 32 sites ├─────►│          ├─────────►│          ├─────►│ no out   │
  │  10.9%   │r3/c48│   9.4%   │  r3/c48  │  11.0%   │r3/c48│  12.9%   │
  └──────────┘      └──────────┘          └────┬─────┘      └────┬─────┘
                      (skips L2) ───────────►  │ ▲                │
                                        −1     │ │  +1            ▼
                                      r4/c16    │ │ r4/c16     
                                               ▼ │           
                                          ┌──────────┐        
                                    self  │    L2    │        
                                    0 ┌──►│scratchpad│
                                 r4/c16 └──┤   0.4%   │
                                          └──────────┘
```


**Edge list** (`level` is the relative layer offset — `tz = z + level`):

| from | level | to | radius/count |
|---|---|---|---|
| L0 | +1 | L1 | r3 / c48 |
| L1 | **+2** | **L3** | r3 / c48 — *skips the scratchpad* |
| L2 | 0 | L2 | r4 / c8 — self-loop |
| L2 | +1 | L3 | r4 / c8 |
| L3 | **−1** | **L2** | r4 / c8 — back-feed |
| L3 | +1 | L4 | r3 / c48 |
| L4 | — | — | continous readout |

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
