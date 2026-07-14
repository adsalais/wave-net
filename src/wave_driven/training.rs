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
    pub elig_beta: f32, // β: ALIF adaptation-eligibility coupling (0 = membrane-only = Phase 2a)
    pub epsilon_a: f32, // εᵃ magnitude cutoff (bounds εᵃ + keeps the offline oracle exact)
}

impl Default for EligParams {
    fn default() -> Self {
        EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 }
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

/// Offline reference eligibility from full fired records. Membrane (β=0): `e_ij = Σ_t pretr_i·[j fires]`.
/// With `β≠0`, adds the ALIF adaptation eligibility `εᵃ` (spike-ψ): `e_ij += pretr_i − β·εᵃ_ij` on a
/// target spike, `εᵃ_ij` recursed at the target layer's `ρ`. Uses the identical `pretr`/`εᵃ` order and
/// cutoffs as `Network::accrue_eligibility`, so it matches the engine's online `elig` bit-for-bit.
/// Returns per-layer `elig`-layout vectors (`ls·total_slots`).
pub fn dense_eligibility(net: &Network, entries: &[Vec<Edge>], fired: &[Vec<Vec<u32>>], p: &EligParams) -> Vec<Vec<f32>> {
    let size = net.size();
    let ls = (size as usize) * (size as usize);
    let l = net.layer_count();
    let ttot = fired.iter().map(|f| f.len()).max().unwrap_or(0);
    let decay = 1.0 - 1.0 / p.rec_tau.max(1.0);
    let eps = p.epsilon;
    let beta = p.elig_beta;
    let eps_a_cut = p.epsilon_a;
    let use_ea = beta != 0.0;
    let rho: Vec<f32> = (0..l).map(|z| net.with_layer(z, |lz| 1.0 - 2f32.powi(-(lz.adapt_decay as i32)))).collect();
    let mut out: Vec<Vec<f32>> = (0..l).map(|z| net.with_layer(z, |lz| vec![0f32; ls * lz.total_slots])).collect();
    let mut epsa: Vec<Vec<f32>> = if use_ea { out.iter().map(|o| vec![0f32; o.len()]).collect() } else { Vec::new() };
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
        // accrue
        for z in 0..l {
            let out_z = &mut out[z];
            let epsa_z = if use_ea { Some(&mut epsa[z]) } else { None };
            net.with_layer(z, |lz| {
                let ts = lz.total_slots;
                accrue_dense_layer(lz, z, l, size, entries, &pretr[z], &fb, &rho, beta, eps_a_cut, use_ea, ts, out_z, epsa_z);
            });
        }
    }
    out
}

/// One layer's dense accrual for wave `t` — mirrors `Network::accrue_eligibility` line-for-line (both
/// membrane and `εᵃ` branches), so the offline result is bit-exact to the online one.
#[allow(clippy::too_many_arguments)]
fn accrue_dense_layer(
    lz: &crate::wave_driven::neurons::Layer,
    z: usize,
    l: usize,
    size: u32,
    entries: &[Vec<Edge>],
    pretr_z: &[f32],
    fb: &[Vec<u64>],
    rho: &[f32],
    beta: f32,
    eps_a_cut: f32,
    use_ea: bool,
    ts: usize,
    out_z: &mut [f32],
    mut epsa_z: Option<&mut Vec<f32>>,
) {
    let ls = (size as usize) * (size as usize);
    for (e_idx, edge) in entries[z].iter().enumerate() {
        let tz_i = z as i32 + edge.level;
        if tz_i < 0 || tz_i as usize >= l {
            continue;
        }
        let tz = tz_i as usize;
        let r_tz = rho[tz];
        let sbase = lz.slot_bases[e_idx];
        for i in 0..ls {
            let pr = pretr_z[i];
            if !use_ea && pr == 0.0 {
                continue;
            }
            lz.for_wired(e_idx, i, |r, c| {
                let j = lz.decode(e_idx, i as u32, c, size);
                let fired = fb[tz][(j >> 6) as usize] & (1u64 << (j & 63)) != 0;
                let widx = i * ts + sbase + r;
                if use_ea {
                    let epsa_z = epsa_z.as_deref_mut().unwrap();
                    let ea = epsa_z[widx];
                    let new_ea = if fired {
                        out_z[widx] += pr - beta * ea;
                        pr + (r_tz - beta) * ea
                    } else {
                        r_tz * ea
                    };
                    epsa_z[widx] = if new_ea.abs() < eps_a_cut { 0.0 } else { new_ea };
                } else if fired {
                    out_z[widx] += pr;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{random_l0_input, TopologyLevel};
    use std::sync::{Arc, Mutex};

    fn deep_cfg(size: u32) -> (Config, Vec<Vec<Edge>>) {
        deep_cfg_ad(size, 6)
    }

    fn deep_cfg_ad(size: u32, adapt_decay: u8) -> (Config, Vec<Vec<Edge>>) {
        let mk = |topology: Vec<TopologyLevel>| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 4, adapt_bump: 5, adapt_decay };
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
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 });

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

        let dense = dense_eligibility(&net, &entries, &fired, &EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 });
        for z in 0..l {
            net.with_layer(z, |lz| {
                assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "layer {z} online == dense elig (bit-exact)");
            });
        }
    }

    // Drive a training net for `waves`, recording fired per layer; return (net, fired-records).
    fn drive_and_record(cfg: Config, params: EligParams, waves: usize) -> (Network, Vec<Vec<Vec<u32>>>) {
        let size = cfg.size;
        let mut net = Network::new(cfg);
        net.set_elig_params(params);
        net.enable_training();
        let l = net.layer_count();
        let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
        for z in 0..l {
            let r = rec.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| r.lock().unwrap()[z].push(f.to_vec())));
        }
        net.reset_state();
        let input = random_l0_input(0x0E11, size, 15000);
        for w in 0..waves {
            net.wave(&input(w));
        }
        net.clear_listeners();
        let fired = rec.lock().unwrap().clone();
        (net, fired)
    }

    // Drive an explicit sequence of L0 inputs (e.g. an active burst then a silent tail), recording
    // fired per layer; return (net, fired-records).
    fn drive_and_record_gen(cfg: Config, params: EligParams, drives: &[Vec<u32>]) -> (Network, Vec<Vec<Vec<u32>>>) {
        let mut net = Network::new(cfg);
        net.set_elig_params(params);
        net.enable_training();
        let l = net.layer_count();
        let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
        for z in 0..l {
            let r = rec.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| r.lock().unwrap()[z].push(f.to_vec())));
        }
        net.reset_state();
        for dv in drives {
            net.wave(dv);
        }
        net.clear_listeners();
        let fired = rec.lock().unwrap().clone();
        (net, fired)
    }

    // A prune-triggering drive: an active burst, then a silent tail long enough for every εᵃ/pretr
    // trace to decay below its cut. `adapt_decay=2` (ρ=0.75) makes εᵃ decay fast.
    fn burst_then_silence(size: u32) -> Vec<Vec<u32>> {
        let l0 = random_l0_input(0x0E11, size, 15000);
        let mut drives: Vec<Vec<u32>> = (0..50usize).map(|w| l0(w)).collect();
        drives.extend((0..150).map(|_| Vec::new()));
        drives
    }

    #[test]
    fn elig_active_prunes_dead_rows() {
        use std::collections::HashSet;
        let size = 16u32;
        let (cfg, _entries) = deep_cfg_ad(size, 2);
        let params = EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 };
        let (net, fired) = drive_and_record_gen(cfg, params, &burst_then_silence(size));
        let distinct: HashSet<u32> = fired[1].iter().flatten().copied().collect();
        assert!(!distinct.is_empty(), "layer 1 must fire for the test to be meaningful");
        // After the silent tail every trace has decayed, so the εᵃ scan set must have drained. The
        // current push-only set stays at `distinct.len()` → RED until pruning lands.
        assert_eq!(net.elig_active_len(1), 0, "{} sources fired but the εᵃ scan set never drained", distinct.len());
    }

    #[test]
    fn online_equals_dense_with_eps_a_pruning() {
        // Same prune-heavy drive: pruning must not change any accrued elig value (bit-exact vs dense).
        let size = 16u32;
        let (cfg, entries) = deep_cfg_ad(size, 2);
        let params = EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 };
        let (net, fired) = drive_and_record_gen(cfg, params, &burst_then_silence(size));
        let dense = dense_eligibility(&net, &entries, &fired, &params);
        for z in 0..net.layer_count() {
            net.with_layer(z, |lz| {
                assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "layer {z} online==dense under εᵃ pruning");
            });
        }
    }

    #[test]
    fn online_equals_dense_eligibility_with_eps_a_bit_exact() {
        let size = 16u32;
        let (cfg, entries) = deep_cfg(size);
        let params = EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.4, epsilon_a: 1.0 / 1024.0 };
        let (net, fired) = drive_and_record(cfg, params, 120);
        let dense = dense_eligibility(&net, &entries, &fired, &params);
        for z in 0..net.layer_count() {
            net.with_layer(z, |lz| {
                assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "layer {z} online == dense elig with εᵃ (bit-exact)");
            });
        }
    }

    #[test]
    fn beta_zero_dense_matches_membrane() {
        // β=0 dense_eligibility must equal the Phase-2a membrane result (the regression gate).
        let size = 16u32;
        let (cfg, entries) = deep_cfg(size);
        let membrane = EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: 0.0, epsilon_a: 1.0 / 1024.0 };
        let (net, fired) = drive_and_record(cfg, membrane, 120);
        let dense = dense_eligibility(&net, &entries, &fired, &membrane);
        for z in 0..net.layer_count() {
            net.with_layer(z, |lz| {
                assert_eq!(lz.train.as_ref().unwrap().elig, &dense[z][..], "β=0 online==dense (membrane)");
            });
        }
    }
}
