//! `wave_driven` — an event-driven, active-set spiking **inference** engine (Phase 1). Per-wave cost
//! scales with activity, not layer size: a per-layer frontier of non-quiescent neurons, a sparse
//! delivery accumulator, and lazy fire-anchored adaptation. Independent of `wave_bitnet` (topology
//! substrate is copied). Spec: docs/superpowers/specs/2026-07-13-wave-driven-event-active-set-design.md.

pub mod config;
pub mod frontier;
pub mod network;
pub mod neurons;
pub mod synapse;
pub mod wave;
