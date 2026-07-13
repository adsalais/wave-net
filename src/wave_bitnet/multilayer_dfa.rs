//! `multilayer_dfa` — the temporal multi-topology multi-layer-DFA training engine for `wave_bitnet`.
//! A synapse's target is DECODED from the source layer's occupancy bitset (wired-rank order), so credit
//! assignment reuses the same materialized topology the forward pass iterates.

use crate::wave_bitnet::network::Network;

/// One topology edge of a source layer, in the SAME order as the built `LayerConfig` topology, so
/// `entries[z][e]` lines up with the layer's `e`-th topology level.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub level: i32,
    pub count: usize,
    pub radius: u32,
}

/// Per-wave records for every layer over one trial (produced by the bench trial runner).
pub struct TrialRecords {
    pub spikes: Vec<Vec<Vec<u32>>>, // [z][wave] = fired local ids
    pub pots: Vec<Vec<Vec<i16>>>,   // [z][wave][local] = decide_potential
    pub effs: Vec<Vec<Vec<i32>>>,   // [z][wave][local] = decide_eff threshold
}

/// Temporal-eligibility knobs (the engine's own — NOT task/readout).
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub rec_tau: f32,        // presynaptic-trace decay time constant (waves)
    pub elig_beta: f32,      // ALIF adaptation coupling β (0 = membrane-only)
    pub elig_psi_width: f32, // bump-ψ half-width W
    pub use_bump: bool,      // bump-ψ (centered at decide_eff) vs raw spike ψ
    pub adapt_decay: u8,     // ALIF adaptation decay shift → ρ = 1 − 2^(−adapt_decay)
}

/// Sane default bump-ψ half-width in i16 potential units.
pub const PSI_WIDTH: f32 = 16.0;

/// Dampening γ for the bump pseudo-derivative ψ = γ·max(0, 1−|v−eff|/W).
const PSI_GAMMA: f32 = 0.3;

/// Σ_t of the ALIF eligibility trace e_ij(t) = ψ_j(t)·(εᵛ_i(t) − β·εᵃ_ij(t)), εᵃ recursed at ρ.
/// β = 0 reduces to the plain membrane trace Σ_t ψ·εᵛ. (Bellec et al. 2020, Eq. 24–25.)
fn elig_adapt_sum(ttot: usize, beta: f32, rho: f32, psi: impl Fn(usize) -> f32, ev: impl Fn(usize) -> f32) -> f32 {
    let mut eps_a = 0.0f32;
    let mut e = 0.0f32;
    for tt in 0..ttot {
        let p = psi(tt);
        let v = ev(tt);
        e += p * (v - beta * eps_a);
        eps_a = p * v + (rho - beta * p) * eps_a;
    }
    e
}

/// Temporal per-synapse eligibility for every layer/edge from one trial's per-wave records.
/// Returns `e[z][edge_idx][i*count + r]` (r = wired-synapse rank); off-stack / into-L0 targets are 0.
pub fn temporal_eligibility(net: &Network, entries: &[Vec<Edge>], rec: &TrialRecords, p: &EligParams) -> Vec<Vec<Vec<f32>>> {
    let size = net.size();
    let l = net.layer_count();
    let ls = (size as usize) * (size as usize);
    let ttot = rec.spikes[l - 1].len();
    // fired[z][t][j] ∈ {0,1}
    let mut fired = vec![vec![vec![0f32; ls]; ttot]; l];
    for z in 0..l {
        for (t, wv) in rec.spikes[z].iter().enumerate() {
            for &loc in wv {
                fired[z][t][loc as usize] = 1.0;
            }
        }
    }
    // pretr[z][t][i]: decaying presynaptic trace
    let decay = 1.0 - 1.0 / p.rec_tau.max(1.0);
    let mut pretr = vec![vec![vec![0f32; ls]; ttot]; l];
    for z in 0..l {
        for i in 0..ls {
            let mut tr = 0.0f32;
            for t in 0..ttot {
                tr = tr * decay + fired[z][t][i];
                pretr[z][t][i] = tr;
            }
        }
    }
    let use_adapt = p.elig_beta != 0.0;
    let use_bump = p.use_bump || use_adapt;
    let rho = 1.0 - (2.0f32).powi(-(p.adapt_decay as i32));
    // ψ[z][t][j]: bump centered on decide_eff, else raw spike
    let mut psi = vec![vec![vec![0f32; ls]; ttot]; l];
    for z in 0..l {
        for t in 0..ttot {
            for j in 0..ls {
                psi[z][t][j] = if use_bump {
                    (PSI_GAMMA * (1.0 - (rec.pots[z][t][j] as f32 - rec.effs[z][t][j] as f32).abs() / p.elig_psi_width.max(1.0))).max(0.0)
                } else {
                    fired[z][t][j]
                };
            }
        }
    }
    // per (layer, edge): e_ij correlation, target decoded from occupancy
    let mut out: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l);
    for z in 0..l {
        let mut layer_out: Vec<Vec<f32>> = Vec::with_capacity(entries[z].len());
        for (lvl, edge) in entries[z].iter().enumerate() {
            let count = edge.count;
            let mut e_entry = vec![0f32; ls * count];
            let tz_i = z as i32 + edge.level;
            if tz_i >= 1 && (tz_i as usize) < l {
                let tz = tz_i as usize;
                // targets for (layer z, level lvl) decoded from occupancy, wired-rank order (length ls*count)
                let targets: Vec<usize> = net.with_layer(z, |lz| {
                    let mut ts = Vec::with_capacity(ls * count);
                    for ii in 0..ls {
                        lz.for_wired(lvl, ii, |_r, c| ts.push(lz.decode(lvl, ii as u32, c, size) as usize));
                    }
                    ts
                });
                for i in 0..ls {
                    for r in 0..count {
                        let j = targets[i * count + r];
                        e_entry[i * count + r] = if use_adapt {
                            elig_adapt_sum(ttot, p.elig_beta, rho, |t| psi[tz][t][j], |t| pretr[z][t][i])
                        } else {
                            let mut s = 0f32;
                            for t in 0..ttot {
                                s += pretr[z][t][i] * psi[tz][t][j];
                            }
                            s
                        };
                    }
                }
            }
            layer_out.push(e_entry);
        }
        out.push(layer_out);
    }
    out
}

/// One training step: build the temporal eligibility from `rec`, then update **every** trainable edge via
/// `Network::eprop_update_synaptic` with the per-target-layer `signal` (`signal[tz][j]`). Edges whose
/// target is off-stack or into L0 (`tz ∉ [1, L−1]`) are skipped. `e_idx` is the level index (entries
/// mirror topology order).
pub fn multilayer_dfa_step(net: &mut Network, entries: &[Vec<Edge>], rec: &TrialRecords, signal: &[Vec<f32>], lr: f32, p: &EligParams) {
    let l = net.layer_count();
    let elig = temporal_eligibility(net, entries, rec, p);
    for z in 0..l {
        for (e_idx, edge) in entries[z].iter().enumerate() {
            let tz_i = z as i32 + edge.level;
            if tz_i < 1 || tz_i as usize >= l {
                continue;
            }
            let tz = tz_i as usize;
            net.eprop_update_synaptic(z, e_idx, &elig[z][e_idx], &signal[tz], lr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::{Config, LayerConfig};
    use crate::wave_bitnet::network::Network;
    use crate::wave_bitnet::synapse::TopologyLevel;

    fn net2(size: u32) -> (Network, Vec<Vec<Edge>>) {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        let net = Network::new(Config { seed: 3, size, layers: vec![up, top] });
        let entries = vec![vec![Edge { level: 1, count: 8, radius: 2 }], vec![]];
        (net, entries)
    }

    /// All neurons fire every wave, so eligibility is nonzero for every synapse (no self-synapse dependency).
    fn dense_records(ls: usize, l: usize, ttot: usize) -> TrialRecords {
        let all: Vec<u32> = (0..ls as u32).collect();
        TrialRecords {
            spikes: (0..l).map(|_| (0..ttot).map(|_| all.clone()).collect()).collect(),
            pots: (0..l).map(|_| (0..ttot).map(|_| vec![7i16; ls]).collect()).collect(),
            effs: (0..l).map(|_| (0..ttot).map(|_| vec![8i32; ls]).collect()).collect(),
        }
    }

    #[test]
    fn eligibility_is_shaped_and_deterministic() {
        let (net, entries) = net2(8);
        let ls = 64;
        let rec = dense_records(ls, 2, 6);
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let a = temporal_eligibility(&net, &entries, &rec, &p);
        let b = temporal_eligibility(&net, &entries, &rec, &p);
        assert_eq!(a[0][0].len(), ls * 8, "elig[layer0][edge0] length = ls*count");
        assert_eq!(a, b, "deterministic");
    }

    #[test]
    fn step_raises_weights_on_negative_signal() {
        let (mut net, entries) = net2(8);
        let ls = 64;
        let rec = dense_records(ls, 2, 6);
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]]; // negative on layer 1 (the target)
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let before: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow.iter().sum());
        multilayer_dfa_step(&mut net, &entries, &rec, &signal, 0.02, &p);
        let after: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow.iter().sum());
        assert!(after > before, "negative target signal + positive eligibility raises L0 shadow: {before}->{after}");
    }
}
