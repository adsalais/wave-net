//! `bench` — an integer test bench for the RSNN: spike-count readouts, decoders, and temporal
//! tasks that validate the substrate (and, later, training). Uses only the engine's public API.

pub mod linalg;
pub mod memory_capacity;
pub mod readout;
pub mod store_recall;
pub mod stream;
pub mod temporal_xor;
