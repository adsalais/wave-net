//! `wave_bitnet` — a memory-lean, ternary-native fork of `wave_net`. Pure ±1/0 weights stored as
//! 2-bit packed values; topology materialized once at startup into a per-neuron neighborhood
//! occupancy bitset (no per-wave hashing). `wave_net` is the frozen reference; duplication is intended.
//! Spec: docs/superpowers/specs/2026-07-12-wave-bitnet-design.md.

pub mod bits;
pub mod config;
pub mod neurons;
pub mod synapse;
pub mod wave;
