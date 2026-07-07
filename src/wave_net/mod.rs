//! `wave_net` — self-contained engine fork + conditioning & diagnostics toolkit.
//!
//! `wave_net` carries its own copy of the integer wave-reservoir engine (`config`, `hash`,
//! `index`, `input`, `pipeline`, `wiring`) so training experiments can freely modify it;
//! `wave_reservoir` stays the frozen reference base. On top of the engine sit the reusable
//! pieces that prepare and inspect a reservoir before training: shared linear algebra, the
//! input/bit-stream harness, a firing-rate calibrator generic over any `IntConfig`, and a
//! depth-utilization diagnostic. Bigger networks added later are calibrated and checked with
//! the same tools. Learning rules (per-neuron training) build on top of this.

// Engine (diverged copy of `wave_reservoir`)
pub mod config;
pub mod hash;
pub mod index;
pub mod input;
pub mod pipeline;
pub mod wiring;

// Toolkit
pub mod calibrate;
pub mod depth;
pub mod linalg;
pub mod readout;
pub mod stream;
pub mod train;
