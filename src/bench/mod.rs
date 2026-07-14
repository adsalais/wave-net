//! `bench` — the training/benchmark harness for the `wave_bitnet` engine: the temporal multi-layer-DFA
//! training loop, tasks, readout, and liveness reports, driving the engine through its public API only.
//! Test-only. Findings are consolidated in docs/experiments_results.md.

pub mod wave_bitnet_bench;
pub mod wave_driven_bench;
pub mod wave_resonate_bench;
