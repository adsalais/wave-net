//! `wave_resonate` — an independent engine (island duplicated from `wave_driven`) whose neuron is the
//! Balanced Resonate-and-Fire (BRF) complex-membrane oscillator (Higuchi et al., ICML 2024), integrating
//! integer ternary spike-deliveries. f32 membrane + ternary ±1/0 weights. Phase 1: forward inference.
//! Spec: docs/superpowers/specs/2026-07-14-wave-resonate-brf-hypr-design.md.

pub mod config;
pub mod network;
pub mod neurons;
pub mod synapse;
pub mod training;
pub mod wave;

#[cfg(test)]
mod equivalence_tests;
