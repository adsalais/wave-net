//! `bench` — the multi-layer-DFA training engine (`multilayer_dfa`) and its benchmark suites
//! (`multilayer_dfa_bench`, `multilayer_dfa_bitnet_bench`). Uses only the engine's public API.
//! (The prior RSNN/LSM benchmark corpus — rsnn, store_recall, temporal_xor, memory_capacity,
//! stream, readout, regime, linalg, and the bench-local eprop/critical_init — was removed
//! 2026-07-12; recover from git history if needed. Its findings live in docs/experiments_results.md.)

pub mod multilayer_dfa;
pub mod multilayer_dfa_bench;
pub mod multilayer_dfa_bitnet_bench;
