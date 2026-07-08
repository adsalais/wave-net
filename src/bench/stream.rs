//! `stream` — the shared streaming substrate for the temporal bench tasks: a binary i.i.d. bit
//! stream, its L0 injection encoding, per-bin spike-count state collection, and the engine-config
//! builder (recurrent vs feed-forward). Used by `memory_capacity` and `temporal_xor`.

use crate::bench::readout::record_response;
use crate::wave_net::config::{Config, LayerConfig};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix, TopologyLevel};

const P_BIT: u64 = 23; // input bit per timestep
const P_STREAM: u64 = 29; // fixed L0 pattern injected on a "1" bit

/// The i.i.d. input bit for timestep `t`.
pub(crate) fn bit(bit_seed: u64, t: usize) -> bool {
    (mix(key(bit_seed, t as u32, 0, 0, P_BIT)) & 1) == 1
}

/// The fixed L0 pattern injected whenever the bit is 1 (same every timestep).
pub(crate) fn stream_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
    let ls = size * size;
    (0..ls).filter(|&s| ((mix(key(seed, s, 0, 0, P_STREAM)) & 0xFFFF) as u32) < density_q16).collect()
}

/// Engine config for the bench. `recurrent` adds level 0 / -1 coupling; feed-forward is level +1
/// only. Both use the dense drive the floored leak requires. `inhibitor_ratio` (Q16) sets the
/// inhibitory fraction — 0 for MC; the XOR bench uses inhibition, which the architecture sweep
/// showed markedly improves nonlinear separation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn engine_config(
    seed: u64,
    size: u32,
    layers: usize,
    baseline_init: i16,
    adapt_bump: i16,
    adapt_decay: u8,
    inhibitor_ratio: u32,
    recurrent: bool,
) -> Config {
    let mut topology = vec![TopologyLevel { level: 1, radius: 3, count: 16 }];
    if recurrent {
        topology.push(TopologyLevel { level: 0, radius: 1, count: 3 });
        topology.push(TopologyLevel { level: -1, radius: 1, count: 3 });
    }
    let layer = LayerConfig {
        topology,
        leak: (3, 5),
        cooldown_base: 2,
        inhibitor_ratio,
        threshold_jitter: 32,
        baseline_init,
        adapt_bump,
        adapt_decay,
    };
    Config { seed, size, layers: vec![layer; layers] }
}

/// Streaming parameters shared by the temporal tasks.
pub(crate) struct StreamParams {
    pub seed: u64,
    pub size: u32,
    pub stream_density_q16: u32,
    pub bit_seed: u64,
    pub bin_waves: usize,
    pub warmup_bins: usize,
    pub collect_bins: usize,
}

/// Drive the continuous bit stream and collect per-bin state rows (per-neuron spike counts over the
/// bin, layers 1..L, ++ a bias 1.0) and the bit sequence. Warmup bins advance the reservoir but are
/// not collected. No reset between bins.
pub(crate) fn collect_states(net: &mut Network, p: &StreamParams) -> (Vec<Vec<f64>>, Vec<f64>) {
    let pattern = stream_pattern(p.seed, p.size, p.stream_density_q16);
    for t in 0..p.warmup_bins {
        let on = bit(p.bit_seed, t);
        for _ in 0..p.bin_waves {
            net.wave(if on { &pattern } else { &[] });
        }
    }
    let mut xs = Vec::with_capacity(p.collect_bins);
    let mut us = Vec::with_capacity(p.collect_bins);
    for i in 0..p.collect_bins {
        let on = bit(p.bit_seed, p.warmup_bins + i);
        let pat = if on { pattern.clone() } else { Vec::new() };
        let counts = record_response(net, p.bin_waves, move |_w| pat.clone());
        let mut row: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        row.push(1.0); // bias
        xs.push(row);
        us.push(if on { 1.0 } else { 0.0 });
    }
    (xs, us)
}
