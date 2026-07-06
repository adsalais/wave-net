//! Integer (i16) wave-reservoir engine — FPGA/embedded-clean (shift/add/sub only, no float in
//! the runtime). `pipeline::LayerNet` is the wavefront-pipelined, deterministic, multi-threaded
//! engine (streaming input via `run_stream` + per-layer event listeners via `on_layer`);
//! `config` holds the integer config + calibration knobs; `wiring` is the procedural synapse
//! generator; `hash`, `index`, and `input` are the shared primitives.

pub mod config;
pub mod hash;
pub mod index;
pub mod input;
pub mod pipeline;
pub mod wiring;
