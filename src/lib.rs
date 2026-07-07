//! wave-net — an integer spiking-neural-network engine and the base for a learning RSNN.
//!
//! The whole engine lives in [`wave_net`]: a stack of square spiking layers with procedurally
//! hash-generated synapses (never stored), deferred one-hop wave propagation, and firing-rate
//! calibration. Deterministic and single-threaded. This crate is a library only.

pub mod wave_net;
