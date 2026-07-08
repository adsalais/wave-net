//! `wave_state_machine` — the **frozen reference**: a memory-efficient Liquid State Machine built on the
//! pure procedural, hash-generated spiking reservoir (never-stored ±1 synapses, deferred one-hop waves,
//! firing-rate calibration). This is the stable baseline the `bench` experiments are pinned to, so the
//! verification-arc findings stay reproducible. Active work — teaching the network to **store weights**
//! (the GeNN split of procedural static reservoir + stored plastic part, Knight & Nowotny 2021) — happens
//! in [`crate::wave_net`], not here. Keep this module unchanged.

pub mod calibrate;
pub mod config;
pub mod network;
pub mod neurons;
pub mod synapse;
pub mod wave;
