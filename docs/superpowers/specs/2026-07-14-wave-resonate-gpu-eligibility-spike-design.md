# wave_resonate — GPU eligibility kernel spike (CUDA + WebGPU)

- **Date:** 2026-07-14
- **Status:** design approved; ready for an implementation plan
- **Scope:** a **de-risking spike**, not the engine. Port **only** `Network::accrue_eligibility` (the
  HYPR per-synapse eligibility recursion) to a GPU kernel on **two hand-written backends** — **CUDA**
  (cudarc + a `.cu` kernel, peak throughput on the NVIDIA RTX 3060) and **WebGPU** (wgpu + a `.wgsl`
  kernel, portability) — behind a small internal `GpuBackend` trait. Validate each against the existing
  CPU `dense_eligibility` oracle to a tolerance, and measure throughput vs the CPU baseline across a size
  sweep. The deliverable is a **GO/NO-GO decision plus real numbers** for a later full-engine spec — no
  behavior of the CPU engine changes.

## Why a spike (not the full GPU engine)

Profiling the training path (`benches/throughput_resonate.rs` + `examples/profile_resonate_train.rs`,
size 32, FF r3/c32) found training runs at **~1087 waves/s, ~13–17× slower than inference**, and that
**~85% of that time is `accrue_eligibility`** — the per-synapse 2-state RTRL recursion — while the dense
BRF membrane forward is only ~7% and `dfa_update` ~4%. That 85% is a dense, per-synapse-independent,
elementwise map: the textbook GPU-ideal kernel (confirmed by the architectural analysis — one thread per
synapse, SoA f32, no cross-thread dependency). The single largest unknown before committing to a full GPU
engine is **the achievable speedup and the per-wave host↔device data-flow**. A kernel spike answers both
cheaply. It also de-risks the **two-kernels-in-sync** architecture (CUDA C + WGSL) chosen for the
dual-backend goal by exercising it on the smallest real kernel first.

The CPU engine is currently **blocked at size ≥ 64** for multi-seed scaling sweeps (single-threaded, too
slow). Unblocking that is the *purpose* the eventual GPU build serves; this spike measures whether GPU
delivers enough to be worth building.

## The one idea

> `accrue_eligibility` updates each synapse's 2-state trace `(εˣ, εʸ)` **independently within a wave**
> from its **target** neuron's forward outputs (`b_eff, ω, ψ`) and its **source** neuron's previous-wave
> spike. The temporal recursion couples wave *t* to *t−1*, but never one synapse to another. So replace
> the CPU's per-wave occupancy-bitset word-scan + `decode` (which rediscovers each synapse's target every
> wave — divergent, GPU-hostile) with a **flat per-synapse edge list precomputed once**, and the whole
> accrual becomes a single per-element GPU kernel: **one thread per synapse, two read-only gathers, no
> atomics, deterministic per backend.**

## Data layout — the flat edge-list transform

Precompute once at spike setup, from a `wave_resonate::Network` (this builder is std-only and always
compiles, independent of the GPU features):

- **Per-synapse arrays** (length `N = Σ_z ls·total_slots`, device-resident, persistent across waves):
  `eps_x[]`, `eps_y[]`, `elig[]` (state), and the precomputed `tgt_g[]` (global target neuron id) and
  `src_g[]` (global source neuron id). The per-synapse index `e` corresponds exactly to the CPU's
  `widx = i·ts + sbase + rank`, so the arrays line up index-for-index with `TrainState.{eps_x,eps_y,elig}`.
- **Per-neuron global arrays** (length `L·ls`, re-uploaded each wave): `b_eff_g[]`, `omega_g[]`,
  `psi_g[]`, `prev_fired_g[]` (packed by global neuron id `g = z·ls + local`).

`tgt_g[e]` is filled by walking each source's wired cells once (the same `for_wired` + `decode` the CPU
uses, but at *setup*, folding the target-layer offset `z + edge.level` into a global id); `src_g[e]` is
the source's global id. Memory is trivial: the FF stack has 4 synapse-bearing layers (the top
readout-target layer has empty topology → 0 slots), so size 128 → `N ≈ 4·16384·32 ≈ 2.1M` synapses ×
(2×u32 + 3×f32 = 20 B) ≈ ~42 MB total, well within the 6 GB card.

## The kernel (identical semantics on both backends)

One thread per synapse `e ∈ 0..N`:

```
j   = tgt_g[e]                       // global target neuron
i   = src_g[e]                       // global source neuron
inj = prev_fired_g[i] ? dt : 0.0
b   = b_eff_g[j];  om = omega_g[j];  psi = psi_g[j]
ex  = eps_x[e];    ey = eps_y[e]
coef = 1.0 + dt*b
nex  = coef*ex - dt*om*ey + inj
ney  = dt*om*ex + coef*ey
nex  = (abs(nex) < cut) ? 0.0 : nex   // eps_cut, matches CPU
ney  = (abs(ney) < cut) ? 0.0 : ney
eps_x[e] = nex;  eps_y[e] = ney
if (psi != 0.0 && nex != 0.0) { elig[e] += psi * nex }
```

- **No active-set pruning.** The GPU processes *all* `N` synapses every wave (uniform, launch-friendly).
  The CPU's online path prunes dead sources, but at the ringing BRF operating point the active set is
  ≈ full, so dense-all is representative; it is also exactly what the CPU `dense_eligibility` oracle does,
  making validation apples-to-apples.
- **Determinism.** Each thread writes only its own `eps_x[e]/eps_y[e]/elig[e]`; no atomics, no contention
  → bit-exact run-to-run on a given backend. Cross-backend/CPU differences are only FMA/rounding, absorbed
  by the tolerance bar. This preserves the repo's *pure function of (seed, config, input)* rule.

## Data flow & what is measured

Per wave (forward stays on CPU in the spike):

1. CPU forward (`process_layer`) captures `b_eff/psi/omega/fired` into the global per-neuron arrays.
2. **Upload only the per-neuron arrays** (`O(L·ls)`: ~80 KB/wave at size 32, ~1.3 MB at size 128).
3. **Launch the kernel.** `eps_x/eps_y/elig` stay resident on device — never re-uploaded.
4. Nothing is downloaded per wave; `elig` is read back once per trial (what `dfa_update` consumes).

Measured **per backend × size {32, 64, 128}**:

- **Kernel-only speedup** — CPU `accrue_eligibility` time/wave vs GPU `[upload + launch]` time/wave.
- **End-to-end training speedup** — full CPU train waves/s vs `(CPU forward + GPU elig)` waves/s: the real
  win from this partial port.
- **Full-engine projection** — the spike *includes* the per-wave upload; the full engine would compute the
  forward on-device (zero upload), so the spike numbers are a **conservative lower bound**.

**GO bar (proposed):** end-to-end training at size ≥ 64 clearing **~5× CPU** materially unblocks
multi-seed scaling sweeps → GO for the full-engine spec. The measured table is the real artifact; the bar
is guidance, not a gate.

## Module layout, backend trait, feature gates

```
src/wave_resonate_gpu/            (feature-gated module; absent from the default build)
  mod.rs      — pub trait GpuBackend + re-exports
  layout.rs   — Network → flat edge list (tgt_g/src_g) + global-array packing   (std-only; always compiles)
  cuda.rs     — CudaBackend (cudarc: alloc, H2D, launch, D2H)                    #[cfg(feature = "cuda")]
  elig.cu     — CUDA C kernel (compiled via cudarc NVRTC at runtime — no build.rs)
  wgpu.rs     — WgpuBackend (wgpu device/queue/pipeline/bind groups)            #[cfg(feature = "wgpu")]
  elig.wgsl   — WGSL kernel
```

```rust
trait GpuBackend {
    fn new(layout: &Layout) -> Self;
    fn upload_neuron_arrays(&mut self, b_eff: &[f32], omega: &[f32], psi: &[f32], prev_fired: &[u32]);
    fn step(&mut self);                 // launch one wave's kernel
    fn download_elig(&self) -> Vec<f32>;
    fn reset(&mut self);                // zero eps/elig on device
}
```

The spike harness is backend-generic over `GpuBackend`. Cargo:

```toml
[dependencies]
cudarc   = { version = "…", optional = true }   # pin current at implementation
wgpu     = { version = "…", optional = true }
pollster = { version = "…", optional = true }    # block on wgpu async
bytemuck = { version = "…", optional = true }    # POD casts for wgpu buffers

[features]
cuda = ["dep:cudarc"]
wgpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]
```

Default `cargo build` / `cargo test` are **unchanged** — std-only, warning-free (the GPU module and its
deps compile only under a feature). Driver + reporting: a feature-gated **`examples/profile_resonate_gpu.rs`**
runs the size sweep and prints the CPU / CUDA / wgpu waves/s table + `max_err` per backend.

## Validation

Feed the **same** per-wave forward-captured inputs (`b_eff, psi, omega, prev_fired`) to both the existing
CPU `dense_eligibility` oracle and the GPU kernel over a wave sequence; assert **`max |Δelig| < 1e-5`
AND `max relative error < 1e-5`**, per backend, per size. Reuses the oracle (no new reference to trust);
its dense-all-sources scan matches the GPU's dense-all approach exactly. Encoded as feature-gated
`#[test]`s: `cargo test --features cuda` / `cargo test --features wgpu`.

## Risks

- **Size-32 may not beat the CPU** (kernel-launch + upload overhead > tiny work). *Expected*; the sweep to
  64/128 is the point, and a size-32 loss is **not** a NO-GO.
- **cudarc / wgpu version churn.** Pin current versions at implementation; NVRTC compiles the `.cu` at
  runtime (`nvcc` is present on the machine).
- **Target-gather divergence** (`b_eff_g[j]` scattered by target id). Read-only and L2-cached, acceptable
  for the spike; sorting edges by target is a deferred optimization, not in scope.
- **WGSL feature parity.** WebGPU mandates f32 and the kernel uses only `+ − × abs` — no intrinsics gap.

## Non-goals (deferred to a later full-engine spec, gated on this spike's GO)

The forward (membrane) on GPU; the firer-gated delivery (`generate`) on GPU; `dfa_update` and `repack_row`
on GPU; the per-neuron ω/b′ parameter eligibility; keeping the whole per-wave loop device-resident with no
transfer; integration with the `bench` training harness; sizes > 128; any change to the CPU engine's
behavior. This spec is **only** the eligibility kernel, its two backends, its validation, and its numbers.

## Implementation plan preview (for writing-plans)

1. `layout.rs` — flat edge-list builder from a `Network` + a std-only unit test (indices line up with
   `TrainState` widx; `tgt_g` matches `decode`). Compiles in the default build.
2. `GpuBackend` trait + the size-sweep harness scaffolding (CPU-only path first: drive forward, capture
   arrays, call CPU `dense_eligibility`, report baseline waves/s).
3. CUDA backend (`cuda.rs` + `elig.cu`) — alloc, upload, NVRTC-compiled launch, download; tolerance test.
4. wgpu backend (`wgpu.rs` + `elig.wgsl`) — device/pipeline/bind groups; same tolerance test.
5. `examples/profile_resonate_gpu.rs` — the CPU/CUDA/wgpu × {32,64,128} table + `max_err`; record numbers
   and the GO/NO-GO in `docs/experiments_results.md`.

One commit per task, conventional messages, no `Co-Authored-By` trailer, on branch `feat/wave-resonate-gpu`.
```
