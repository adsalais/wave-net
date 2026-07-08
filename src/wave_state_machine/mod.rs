//! `wave_state_machine` — a memory-efficient Liquid State Machine, forked from [`crate::wave_net`].
//!
//! Same procedural, hash-generated spiking reservoir (never-stored ±1 synapses, deferred one-hop waves,
//! firing-rate calibration), but this variant will grow a **stored, trained readout** on top of the
//! reservoir state — the GeNN-sanctioned split of *procedural static reservoir* + *stored plastic readout*
//! (Knight & Nowotny 2021). The verification arc showed in-network threshold-only learning on a purely
//! procedural net is structurally unreliable (see `docs/experiments_results.md`); the LSM readout is the
//! reliable path. `wave_net` stays untouched as the pure-procedural reference.

pub mod calibrate;
pub mod config;
pub mod network;
pub mod neurons;
pub mod synapse;
pub mod wave;
