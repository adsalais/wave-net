//! wave-net — an integer wave-reservoir spiking engine, the base for a learning RSNN.
//!
//! `wave_reservoir` is the frozen reference engine (procedural hash-generated synapses,
//! wavefront-pipelined and deterministic). `wave_net` is a self-contained fork of that
//! engine plus the training/diagnostics toolkit — it shares no code with `wave_reservoir`
//! so training experiments can modify the engine freely.

pub mod wave_net;
pub mod wave_reservoir;
