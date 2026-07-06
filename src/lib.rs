//! wave-net — an integer wave-reservoir spiking engine, the base for a learning RSNN.
//!
//! `wave_reservoir::pipeline::LayerNet` is the reservoir engine (procedural hash-generated
//! synapses, wavefront-pipelined and deterministic). Learning is built on top of it.

pub mod wave_net;
pub mod wave_reservoir;
