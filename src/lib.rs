//! wave-net — a memory-lean, ternary-native integer spiking-neural-network engine and the base for a
//! learning RSNN.
//!
//! The primary engine lives in [`wave_bitnet`]: a stack of square spiking layers whose topology is
//! materialized once into a per-neuron neighborhood occupancy bitset (no per-wave hashing), with 2-bit
//! packed ±1/0 weights and deferred one-hop wave propagation. Deterministic and single-threaded. This
//! crate is a library only.
//!
//! A second, independent engine [`wave_driven`] explores **event-driven active-set inference**: each
//! wave processes only a per-layer frontier of non-quiescent neurons, with lazy fire-anchored
//! adaptation, so cost scales with activity rather than layer size (Phase 1: inference only).

pub mod bench; // experiments and benchmark harness for the wave_bitnet engine
pub mod wave_bitnet; // the engine: bitset topology + 2-bit ternary weights
pub mod wave_driven; // event-driven active-set inference engine (Phase 1)
pub mod wave_resonate; // BRF resonate-and-fire engine (island); Phase 1: inference
