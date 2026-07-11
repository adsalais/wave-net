//! `bench` — an integer test bench for the RSNN: spike-count readouts, decoders, and temporal
//! tasks that validate the substrate (and, later, training). Uses only the engine's public API.

pub mod critical_init;
pub mod eprop;
pub mod linalg;
pub mod regime;
pub mod rsnn;
pub mod memory_capacity;
pub mod multilayer_dfa;
pub mod readout;
pub mod store_recall;
pub mod stream;
pub mod temporal_xor;
