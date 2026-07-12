//! wave-net — an integer spiking-neural-network engine and the base for a learning RSNN.
//!
//! The whole engine lives in [`wave_net`]: a stack of square spiking layers with procedurally
//! hash-generated synapses (never stored), deferred one-hop wave propagation, and firing-rate
//! calibration. Deterministic and single-threaded. This crate is a library only.

pub mod bench; // experiments, pinned to the wave_state_machine reference
pub mod wave_bitnet; // memory-lean, ternary-native fork of wave_net (bitset topology + 2-bit weights)
pub mod wave_net; // active R&D engine — being modified to store weights (moving beyond pure procedural)
pub mod wave_state_machine; // frozen reference: the memory-efficient (pure procedural) LSM
