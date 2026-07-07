//! wave-net — an integer wave-reservoir spiking engine, the base for a learning RSNN.
//!
//! `wave_reservoir` is the frozen reference engine (procedural hash-generated synapses,
//! wavefront-pipelined and deterministic). `legacy_net` is the previous self-contained fork
//! of that engine plus its training/diagnostics toolkit, kept for reference while `wave_net`
//! is rebuilt from scratch. `wave_net` will be re-wired here once it has content.

pub mod legacy_net;
pub mod wave_net;
pub mod wave_reservoir;
