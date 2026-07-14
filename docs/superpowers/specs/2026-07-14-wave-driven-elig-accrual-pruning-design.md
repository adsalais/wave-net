# wave_driven — εᵃ eligibility-accrual pruning + unchecked indexing

**Date:** 2026-07-14 · **Status:** design → implementation · **Branch:** `perf/wave-driven-elig-accrual`

## Motivation

Profiling the `wave_driven` **training** trial (`examples/profile_driven_train.rs`, side-car parity-N=4
rec16, size 32, 172k perf samples) found the online-training path — not inference — is the bottleneck:

| symbol | self-time |
|---|---|
| `Network::wave` (incl. **`accrue_eligibility` inlined**) | 82.6% |
| `wave::process_layer` (forward) | 11.5% |
| `dfa_update` + `repack_row` + `reset_eligibility` | ~4.4% |

`perf annotate` pins the hot lines to the **εᵃ (ALIF adaptation-eligibility) per-synapse inner loop**
(`network.rs:318–333`). Root cause: the εᵃ accrual scans `elig_active` — *every source that has fired
since the last reset* — which is **push-only within a trial** (`accrue_eligibility` pushes firers,
cleared only by `reset_eligibility`). So late-trial waves re-scan nearly every neuron that ever fired,
each time re-decoding its full occupancy bitset and doing an f32 εᵃ recursion per synapse. Cost grows
with cumulative activity, ∝ size² at the fixed `rate_reg` operating point — the size-≥64 scaling blocker.

The forward `process_layer` (11.5%) is healthy activity-scaled work. `pretr_active` already **self-prunes**
dead presynaptic sources (`accrue_eligibility` step 2); `elig_active` has no such drop. Allocation is a
non-issue (~0.4% libc — buffer reuse works).

## Change 1 — prune `elig_active` when a source's εᵃ row fully decays (the structural win)

**Invariant that makes it bit-exact.** In the εᵃ recursion, a source `i` contributes on a future wave
only via (a) its presynaptic trace `pr = pretr[i]` (injected into `elig`/`eps_a` on a target fire), or
(b) an already-live `eps_a` slot (the silent-source coupling `−β·εᵃ` and the decay `ρ·εᵃ`). If
`pretr[i] == 0` **and every** `eps_a` slot in the row is `0`, then for every synapse both branches yield
exactly `0.0` (`elig += 0`, `new_ea = ρ·0 = 0`) on **every** subsequent wave — until the source fires
again, which **re-adds** it to `elig_active` (step 3) and re-bumps `pretr`. So dropping such a row from
the scan set changes no computed `elig`/`eps_a` value. The dense oracle (`dense_eligibility`, scans all
neurons every wave) computes those same zeros, so **online == dense stays bit-exact**.

**Implementation.** Restructure the εᵃ branch to the same take-and-re-push shape `pretr_active` uses:
`let scan = std::mem::take(&mut elig_active[z].list);` clear its marks, process each source, and
re-`push` a source **iff** `pr != 0.0 || any_ea_alive`, where `any_ea_alive` ORs `eps_a[widx] != 0.0`
(post-`eps_a_cut`) across the row. Using this-wave `pr` (read before next wave's decay) as the keep
condition is conservative — a kept row that then contributes nothing is a numeric no-op — so it stays
bit-exact. Source order within `elig_active` is irrelevant: each source writes a disjoint `eps_a`/`elig`
slice (`widx = i·ts + …`), so results are order-independent.

**Reset invariant preserved.** `reset_eligibility` zeroes `eps_a` over `elig_active`; every row with a
nonzero `eps_a` remains in `elig_active` (pruned rows are all-zero), so reset stays correct. `dirty_rows`
(the `elig`-zeroing set) is untouched by pruning.

## Change 2 — `get_unchecked` in the εᵃ inner loop (constant-factor win)

The inner loop indexes `offsets[e_idx][cell]`, `tr.eps_a[widx]`, `tr.elig[widx]`, and `fb[j>>6]` with
indices **provably in-bounds by the word-scan invariants** — the *same* justification AGENTS.md already
documents for the one sanctioned `unsafe` in `process_layer` (which removed ~7% there):

- `cell = wi·64 + tz(word)` is a **set** occupancy bit ⇒ `cell < offsets[e_idx].len()` (= neighborhood
  cell count; padding bits are never set).
- `widx = i·ts + slot_bases[e_idx] + rank`, `rank < entry.count`, `slot_bases[e_idx]+count ≤ ts`,
  `i < ls` ⇒ `widx < ls·ts = eps_a.len() = elig.len()`.
- `j = local_of(wrap(...), wrap(...)) < ls` ⇒ `j>>6 < fb.len() = ceil(ls/64)`.

Each access gets a `// SAFETY:` comment. This makes `unsafe` appear in a **second** location, so
AGENTS.md's "ONE documented exception" convention is updated to name both sites (still confined to the
two hot accrual/generation loops, each with an airtight, commented justification).

## Testing

Equivalence-preserving change → the dense oracle is the guard; the win is measured by the profiler.

- **`elig_active_prunes_dead_rows` (RED→GREEN — the new behavior):** drive `adapt_decay=2` (ρ=0.75, fast
  εᵃ decay) with a silent tail long enough that all traces decay; assert `elig_active_len(z) == 0` for a
  layer that fired (a `#[cfg(test)]` accessor). RED on current code (monotonic set stays populated),
  GREEN after Change 1.
- **`online_equals_dense_with_eps_a_pruning` (bit-exact guard):** same prune-heavy config; assert
  `online.elig == dense.elig`. Green before and after — guards that pruning corrupts nothing.
- Existing `online_equals_dense_eligibility_with_eps_a_bit_exact`, `sparse_equals_dense_*`, and the full
  `cargo test` suite stay green (Change 2 is behavior-preserving).
- **Measurement:** re-run `profile_driven_train 32 2000` before/after; report ms/trial and the `wave`
  self-time share.

## Out of scope

Fusing the forward-delivery and accrual scans (a bigger refactor), and `eps_a_cut`/β tuning (a
learning-quality lever, not bit-exact). Noted as follow-ups.
