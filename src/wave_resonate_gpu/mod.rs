//! `wave_resonate_gpu` — a de-risking spike: GPU port of the HYPR eligibility accrual
//! (`Network::accrue_eligibility`), validated vs the CPU `dense_eligibility` oracle. The std-only parts
//! here (flat edge-list layout, the `cpu_accrue` reference, the `GpuBackend` trait + `CpuBackend`) compile
//! in the default build; the CUDA/wgpu backends are feature-gated. See
//! docs/superpowers/specs/2026-07-14-wave-resonate-gpu-eligibility-spike-design.md.

use crate::wave_resonate::network::Network;

/// Flat per-synapse edge list derived once from a `Network`. Synapse `e` corresponds to the CPU's
/// `widx = i*ts + sbase + rank` within layer `z`, offset by `syn_base[z]`. `tgt_g[e]`/`src_g[e]` are the
/// GLOBAL neuron ids (`layer*ls + local`) of the synapse's target and source.
pub struct Layout {
    pub n: usize,             // total synapses across the stack
    pub ls: usize,            // size*size
    pub l: usize,             // layer count
    pub syn_base: Vec<usize>, // len l+1; synapse-index base per layer (prefix sum of ls*ts)
    pub tgt_g: Vec<u32>,
    pub src_g: Vec<u32>,
}

/// Build the flat edge list: walk every source's wired cells once (setup-time `for_wired` + `decode`,
/// folding the target-layer offset `z+level` into a global id). Every slot of every synapse-bearing layer
/// is a wired synapse (Σ count_e == total_slots), so no sentinels are needed for the FF spike config.
pub fn build_layout(net: &Network) -> Layout {
    let size = net.size();
    let ls = (size as usize) * (size as usize);
    let l = net.layer_count();
    let ts_by: Vec<usize> = (0..l).map(|z| net.with_layer(z, |lz| lz.total_slots)).collect();
    let mut syn_base = vec![0usize; l + 1];
    for z in 0..l {
        syn_base[z + 1] = syn_base[z] + ls * ts_by[z];
    }
    let n = syn_base[l];
    let mut tgt_g = vec![0u32; n];
    let mut src_g = vec![0u32; n];
    for z in 0..l {
        let ts = ts_by[z];
        if ts == 0 {
            continue;
        }
        let base_z = syn_base[z];
        net.with_layer(z, |lz| {
            for i in 0..ls {
                for (e_idx, edge) in lz.topology.iter().enumerate() {
                    let tz_i = z as i32 + edge.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let sbase = lz.slot_bases[e_idx];
                    lz.for_wired(e_idx, i, |rank, cell| {
                        let j = lz.decode(e_idx, i as u32, cell, size) as usize;
                        let e = base_z + i * ts + sbase + rank;
                        tgt_g[e] = (tz * ls + j) as u32;
                        src_g[e] = (z * ls + i) as u32;
                    });
                }
            }
        });
    }
    Layout { n, ls, l, syn_base, tgt_g, src_g }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;

    // FF config with the TOP layer empty-topology (0 source synapses — read directly), so every
    // allocated eligibility slot is an in-range wired synapse (no inert out-of-range slots to special-case).
    fn ff_cfg(size: u32, layers: usize) -> Config {
        let up = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
            inhibitor_ratio: 0,
            omega_init: (5.0, 10.0),
            b_offset_init: (0.0, 0.2),
            tau_out: 20.0,
        };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        let layers_vec: Vec<LayerConfig> =
            (0..layers).map(|z| if z == layers - 1 { top.clone() } else { up.clone() }).collect();
        Config { seed: 7, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: layers_vec }
    }

    #[test]
    fn layout_indices_line_up_with_widx_and_decode() {
        let size = 8u32;
        let ls = (size * size) as usize;
        let net = Network::new(ff_cfg(size, 3));
        let layout = build_layout(&net);
        // top layer (empty topology) contributes 0 synapses; layers 0,1 contribute ls*32 each.
        assert_eq!(layout.syn_base[3], layout.n);
        assert_eq!(layout.n, 2 * ls * 32);
        // spot-check: for layer 0, source i, edge 0, the flat entry matches decode() and the global ids.
        net.with_layer(0, |lz| {
            let ts = lz.total_slots;
            for &i in &[0usize, 5, ls - 1] {
                lz.for_wired(0, i, |rank, cell| {
                    let e = layout.syn_base[0] + i * ts + lz.slot_bases[0] + rank;
                    let j = lz.decode(0, i as u32, cell, size) as usize;
                    assert_eq!(layout.tgt_g[e] as usize, ls + j, "target in layer 1");
                    assert_eq!(layout.src_g[e] as usize, i, "source in layer 0");
                });
            }
        });
    }
}
