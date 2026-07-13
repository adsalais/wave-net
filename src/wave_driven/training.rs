//! `training` — online, activity-scaled multi-layer-DFA training for `wave_driven`: membrane e-prop
//! eligibility with spike-ψ, accrued on the frontier during `wave()`. Types here; the accrual and
//! shadow-update live on `Network` (they need the layer stack + per-wave fired sets). The offline
//! `dense_eligibility` oracle (this file) is the bit-exact reference for the online accrual.

/// Eligibility knobs (membrane-only, spike-ψ). `rec_tau` sets the presynaptic-trace decay
/// (`decay = 1 − 1/rec_tau`); `epsilon` is the hard trace cutoff (activity-scaling + exact oracle).
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub rec_tau: f32,
    pub epsilon: f32,
}

impl Default for EligParams {
    fn default() -> Self {
        EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 }
    }
}

/// One topology edge of a source layer, in built `LayerConfig` topology order (so `entries[z][e]`
/// lines up with the layer's `e`-th level). Mirrors the DFA credit wiring.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub level: i32,
    pub count: usize,
    pub radius: u32,
}

use crate::wave_driven::network::Network;

/// Offline reference eligibility from full fired records: `e_ij = Σ_t pretr_i(t)·[j fires at t]`,
/// `pretr` maintained with the canonical decay → ε-drop → bump order. Returns per-layer `elig`-layout
/// vectors (`ls·total_slots`) for direct comparison to the engine's online `elig`.
pub fn dense_eligibility(net: &Network, entries: &[Vec<Edge>], fired: &[Vec<Vec<u32>>], p: &EligParams) -> Vec<Vec<f32>> {
    let size = net.size();
    let ls = (size as usize) * (size as usize);
    let l = net.layer_count();
    let ttot = fired.iter().map(|f| f.len()).max().unwrap_or(0);
    let decay = 1.0 - 1.0 / p.rec_tau.max(1.0);
    let eps = p.epsilon;
    let mut out: Vec<Vec<f32>> = (0..l).map(|z| net.with_layer(z, |lz| vec![0f32; ls * lz.total_slots])).collect();
    let mut pretr = vec![vec![0f32; ls]; l];
    for t in 0..ttot {
        // pretr: decay -> eps-drop -> bump firers (identical order to Network::accrue_eligibility)
        for z in 0..l {
            for i in 0..ls {
                pretr[z][i] *= decay;
                if pretr[z][i] < eps {
                    pretr[z][i] = 0.0;
                }
            }
            if t < fired[z].len() {
                for &i in &fired[z][t] {
                    pretr[z][i as usize] += 1.0;
                }
            }
        }
        // fired bitset per layer at wave t
        let mut fb = vec![vec![0u64; (ls + 63) / 64]; l];
        for z in 0..l {
            if t < fired[z].len() {
                for &j in &fired[z][t] {
                    fb[z][(j >> 6) as usize] |= 1u64 << (j & 63);
                }
            }
        }
        // accrue: source-driven, add pretr where the target fired
        for z in 0..l {
            net.with_layer(z, |lz| {
                let ts = lz.total_slots;
                for (e_idx, edge) in entries[z].iter().enumerate() {
                    let tz_i = z as i32 + edge.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let sbase = lz.slot_bases[e_idx];
                    for i in 0..ls {
                        let pr = pretr[z][i];
                        if pr == 0.0 {
                            continue;
                        }
                        lz.for_wired(e_idx, i, |r, c| {
                            let j = lz.decode(e_idx, i as u32, c, size);
                            if fb[tz][(j >> 6) as usize] & (1u64 << (j & 63)) != 0 {
                                out[z][i * ts + sbase + r] += pr;
                            }
                        });
                    }
                }
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{random_l0_input, TopologyLevel};
    use std::sync::{Arc, Mutex};

    fn deep_cfg(size: u32) -> (Config, Vec<Vec<Edge>>) {
        let mk = |topology: Vec<TopologyLevel>| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 4, adapt_bump: 5, adapt_decay: 6 };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: 0, radius: 1, count: 3 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![]),
        ];
        let entries = vec![
            vec![Edge { level: 1, count: 8, radius: 2 }],
            vec![Edge { level: 1, count: 8, radius: 2 }, Edge { level: 0, count: 3, radius: 1 }],
            vec![Edge { level: 1, count: 8, radius: 2 }],
            vec![],
        ];
        (Config { seed: 0x0E11, size, layers }, entries)
    }

    #[test]
    fn online_equals_dense_eligibility_bit_exact() {
        let size = 16u32;
        let (cfg, entries) = deep_cfg(size);
        let mut net = Network::new(cfg);
        net.enable_training();
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });

        // record fired per layer per wave via listeners, in lockstep with the online accrual
        let l = net.layer_count();
        let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
        for z in 0..l {
            let r = rec.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| r.lock().unwrap()[z].push(f.to_vec())));
        }
        net.reset_state();
        let input = random_l0_input(0x0E11, size, 15000);
        for w in 0..120 {
            net.wave(&input(w));
        }
        net.clear_listeners();
        let fired = rec.lock().unwrap().clone();

        let dense = dense_eligibility(&net, &entries, &fired, &EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });
        for z in 0..l {
            net.with_layer(z, |lz| {
                assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "layer {z} online == dense elig (bit-exact)");
            });
        }
    }
}
