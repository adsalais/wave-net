//! `wave_net` — conditioning & diagnostics toolkit built on the `wave_reservoir` engine.
//!
//! The engine (`wave_reservoir`) is the frozen reference base; `wave_net` holds the reusable
//! pieces that prepare and inspect a reservoir before training: shared linear algebra, the
//! input/bit-stream harness, a firing-rate calibrator generic over any `IntConfig`, and a
//! depth-utilization diagnostic. Bigger networks added later are calibrated and checked with
//! the same tools. Learning rules (per-neuron training) build on top of this.

pub mod calibrate;
pub mod depth;
pub mod linalg;
pub mod readout;
pub mod stream;
pub mod train;
