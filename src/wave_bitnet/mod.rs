//! `wave_bitnet` — a memory-lean, ternary-native integer spiking engine. Pure ±1/0 weights stored as
//! 2-bit packed values; topology materialized once at startup into a per-neuron neighborhood
//! occupancy bitset (no per-wave hashing).
//! Spec: docs/superpowers/specs/2026-07-12-wave-bitnet-design.md.

pub mod config;
pub mod multilayer_dfa;
pub mod network;
pub mod neurons;
pub mod persist;
pub mod synapse;
pub mod wave;
