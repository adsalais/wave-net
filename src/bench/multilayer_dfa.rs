//! `multilayer_dfa` — a self-contained temporal multi-topology multi-layer-DFA training engine, staged in
//! `bench` until proven, then lifted into `wave_net`. Depends ONLY on `wave_net` (no other bench file).
//! Engine-to-be (this module): temporal eligibility + the multi-layer update step over
//! `Network::eprop_update_synaptic`. Bench-owned (the `#[cfg(test)]` block): the trial, readout, DFA
//! signal, tasks, and loop. Spec: docs/superpowers/specs/2026-07-11-multilayer-dfa-engine-design.md.

use crate::wave_net::network::Network;
use crate::wave_net::synapse::target_of;

/// One topology edge of a source layer, in the SAME order as the built `LayerConfig` topology, so slot
/// indices align with `out_weights` (the invariant `rsnn::train_multilayer`'s `layer_entries` keeps by hand).
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

/// Sane default bump-ψ half-width in i16 potential units. (Copied from rsnn to keep this file free of
/// bench-file dependencies.)
pub const PSI_WIDTH: f32 = 16.0;

/// Dampening γ for the bump pseudo-derivative ψ = γ·max(0, 1−|v−eff|/W). (Copied from rsnn.)
const PSI_GAMMA: f32 = 0.3;

/// Σ_t of the ALIF eligibility trace e_ij(t) = ψ_j(t)·(εᵛ_i(t) − β·εᵃ_ij(t)), εᵃ recursed at ρ.
/// β = 0 reduces to the plain membrane trace Σ_t ψ·εᵛ. (Copied from rsnn — Bellec et al. 2020, Eq. 24–25.)
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
/// Returns `e[z][entry_idx][i*count + k]`; off-stack / into-L0 targets are 0 (untrainable).
pub fn temporal_eligibility(net: &Network, entries: &[Vec<Edge>], rec: &TrialRecords, p: &EligParams) -> Vec<Vec<Vec<f32>>> {
    let seed = net.seed_val();
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
    // per (layer, edge): e_ij correlation
    let mut out: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l);
    for z in 0..l {
        let mut layer_out: Vec<Vec<f32>> = Vec::with_capacity(entries[z].len());
        for edge in &entries[z] {
            let count = edge.count;
            let mut e_entry = vec![0f32; ls * count];
            let tz_i = z as i32 + edge.level;
            if tz_i >= 1 && (tz_i as usize) < l {
                let tz = tz_i as usize;
                for i in 0..ls {
                    let sg = (z * ls + i) as u32;
                    for k in 0..count {
                        let j = target_of(seed, sg, i as u32, edge.level, k as u32, edge.radius, size) as usize;
                        e_entry[i * count + k] = if use_adapt {
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
/// `Network::eprop_update_synaptic` with the caller-supplied per-target-layer `signal` (`signal[tz][j]`).
/// Edges whose target is off-stack or into L0 (`tz ∉ [1, L−1]`) are skipped (untrainable). Requantising the
/// source layer once per edge is equivalent to accumulating then requantising once.
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
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    // A 2-layer, radius-0, count-1 up net: target of source local i is local i above.
    fn tiny_net() -> Network {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 0, count: 1 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 6,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        Network::new(Config { seed: 1, size: 4, layers: vec![lc; 2] })
    }

    // Records where every neuron of both layers fires on each of `ttot` waves.
    fn dense_records(ls: usize, l: usize, ttot: usize) -> TrialRecords {
        let all: Vec<u32> = (0..ls as u32).collect();
        TrialRecords {
            spikes: vec![vec![all.clone(); ttot]; l],
            pots: vec![vec![vec![0i16; ls]; ttot]; l],
            effs: vec![vec![vec![1i32; ls]; ttot]; l],
        }
    }

    #[test]
    fn temporal_eligibility_membrane_matches_hand_computed() {
        let net = tiny_net();
        let ls = 16;
        let ttot = 3;
        let rec = dense_records(ls, 2, ttot);
        let entries = vec![vec![Edge { level: 1, count: 1, radius: 0 }], vec![]];
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let e = temporal_eligibility(&net, &entries, &rec, &p);
        // membrane, spike-ψ (fired=1 every wave): pretr = 1, 1.75, 2.3125 (decay = 1 - 1/4 = 0.75).
        // e = Σ_t pretr_i(t)·fired_j(t) = 1 + 1.75 + 2.3125 = 5.0625 for every synapse.
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].len(), 1); // one edge on layer 0
        assert_eq!(e[1].len(), 0); // top layer: no outgoing edge
        for &v in &e[0][0] {
            assert!((v - 5.0625).abs() < 1e-4, "e = {v}");
        }
    }

    #[test]
    fn temporal_eligibility_beta_changes_result_and_is_deterministic() {
        let net = tiny_net();
        let rec = dense_records(16, 2, 4);
        let entries = vec![vec![Edge { level: 1, count: 1, radius: 0 }], vec![]];
        let base = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let adapt = EligParams { elig_beta: 0.4, ..base };
        let e0a = temporal_eligibility(&net, &entries, &rec, &base);
        let e0b = temporal_eligibility(&net, &entries, &rec, &base);
        let ea = temporal_eligibility(&net, &entries, &rec, &adapt);
        assert_eq!(e0a, e0b, "eligibility must be deterministic");
        assert!(ea[0][0] != e0a[0][0], "β>0 (ALIF εᵃ) must change the eligibility");
    }

    #[test]
    fn multilayer_dfa_step_raises_weights_on_negative_signal() {
        let mut net = tiny_net();
        let ls = 16;
        let rec = dense_records(ls, 2, 3);
        let entries = vec![vec![Edge { level: 1, count: 1, radius: 0 }], vec![]];
        // signal into the top layer (tz = 1) is negative → weights should rise (fire more).
        let signal = vec![vec![0.0f32; ls], vec![-1.0f32; ls]];
        let p = EligParams { rec_tau: 4.0, elig_beta: 0.0, elig_psi_width: PSI_WIDTH, use_bump: false, adapt_decay: 6 };
        let before: f32 = net.with_layer(0, |lz| lz.out_shadow.iter().sum());
        multilayer_dfa_step(&mut net, &entries, &rec, &signal, 0.02, &p);
        let after: f32 = net.with_layer(0, |lz| lz.out_shadow.iter().sum());
        // Δ per synapse = -lr·signal·e = -0.02·(-1)·5.0625 > 0
        assert!(after > before + 1.0, "negative signal + positive eligibility must raise layer-0 weights: {before} -> {after}");
    }
}
