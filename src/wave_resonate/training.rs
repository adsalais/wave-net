//! `training` — HYPR online-eligibility training for the BRF engine: the double-Gaussian surrogate, the
//! eligibility knobs, and the bit-exact `dense_eligibility` oracle. The per-wave accrual and the shadow
//! update live on `Network` (they need the layer stack + per-wave fired sets).

/// Reference double-Gaussian surrogate `∂z/∂x` at `v = x − ϑ_c − q` (StepDoubleGaussianGrad):
/// `ψ(v) = γ·[(1+p)·N(v;0,σ₁) − 2p·N(v;0,σ₂)]`, `N(v;μ,σ)=exp(−(v−μ)²/2σ²)/(σ√2π)`.
#[inline]
pub fn surrogate(v: f32) -> f32 {
    const P: f32 = 0.15;
    const S1: f32 = 0.5;
    const S2: f32 = 3.0;
    const G: f32 = 0.5;
    let inv_sqrt_2pi = 1.0f32 / (2.0 * std::f32::consts::PI).sqrt();
    let n = |mu: f32, sigma: f32| (inv_sqrt_2pi / sigma) * (-((v - mu) * (v - mu)) / (2.0 * sigma * sigma)).exp();
    G * ((1.0 + P) * n(0.0, S1) - 2.0 * P * n(0.0, S2))
}

/// HYPR eligibility knobs. `dt` mirrors `Config::dt` (the eligibility recursion uses the same δ). `eps_cut`
/// zeroes a trace slot once `|ε^x|,|ε^y|` fall below it (bounds the trace + keeps the dense oracle exact).
/// `train_omega_b` gates the per-neuron ω/b′ updates (Phase 2b; false here).
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub dt: f32,
    pub eps_cut: f32,
    pub train_omega_b: bool,
}

impl Default for EligParams {
    fn default() -> Self {
        EligParams { dt: 0.05, eps_cut: 1.0 / 1024.0, train_omega_b: false }
    }
}

/// One topology edge of a source layer, in built `LayerConfig` topology order (so `entries[z][e]` lines up
/// with the layer's `e`-th level). Mirrors the DFA credit wiring.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub level: i32,
    pub count: usize,
    pub radius: u32,
}

use crate::wave_resonate::config::Config;
use crate::wave_resonate::network::Network;
use std::sync::{Arc, Mutex};

/// Bit-exact reference eligibility. Builds a FRESH training net, drives it on `inputs`, and after EACH
/// wave captures the target snapshots (`b_eff, ψ, ω`) + firers, then advances the SAME 2-state ε
/// recursion (same coef/order/cutoff, same `elig += ψ·ε^x` guard) over ALL sources. Because the forward
/// is deterministic, the captured values equal the online run's, so this matches `Network`'s online
/// accrual bit-for-bit. Per-synapse updates are independent within a wave, so full-scan order (0..ls)
/// vs the online active-set order does not change any `widx`. Returns per-layer `elig` vectors.
pub fn dense_eligibility(cfg: &Config, inputs: &[Vec<u32>], p: &EligParams) -> Vec<Vec<f32>> {
    let size = cfg.size;
    let ls = (size as usize) * (size as usize);
    let l = cfg.layers.len();
    let dt = p.dt;
    let cut = p.eps_cut;

    let mut net = Network::new(cfg.clone());
    net.set_elig_params(*p);
    net.enable_training();

    let entries: Vec<Vec<Edge>> = net.entries().to_vec();
    let ts_by: Vec<usize> = (0..l).map(|z| net.with_layer(z, |lz| lz.total_slots)).collect();
    let mut elig: Vec<Vec<f32>> = (0..l).map(|z| vec![0f32; ls * ts_by[z]]).collect();
    let mut eps_x: Vec<Vec<f32>> = elig.iter().map(|e| vec![0f32; e.len()]).collect();
    let mut eps_y: Vec<Vec<f32>> = elig.iter().map(|e| vec![0f32; e.len()]).collect();
    let mut prev_fired = vec![vec![false; ls]; l];

    // capture each wave's firers per layer (listener overwrites its slot each wave)
    let cur: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 0..l {
        let c = cur.clone();
        net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] = f.to_vec()));
    }

    for inp in inputs {
        net.wave(inp);
        // this wave's target snapshots
        let mut b_snap = vec![vec![0f32; ls]; l];
        let mut psi_snap = vec![vec![0f32; ls]; l];
        let mut om_snap = vec![vec![0f32; ls]; l];
        for z in 0..l {
            net.with_layer(z, |lz| {
                om_snap[z].copy_from_slice(&lz.omega);
                if let Some(t) = lz.train.as_ref() {
                    b_snap[z].copy_from_slice(&t.b_eff);
                    psi_snap[z].copy_from_slice(&t.psi);
                }
            });
        }
        // full-scan accrual (all sources), byte-identical arithmetic to Network::accrue_eligibility
        for z in 0..l {
            let ts = ts_by[z];
            let ex_z = &mut eps_x[z];
            let ey_z = &mut eps_y[z];
            let el_z = &mut elig[z];
            let pf_z = &prev_fired[z];
            net.with_layer(z, |lz| {
                for i in 0..ls {
                    let inj = if pf_z[i] { dt } else { 0.0 };
                    for (e_idx, edge) in entries[z].iter().enumerate() {
                        let tz_i = z as i32 + edge.level;
                        if tz_i < 0 || tz_i as usize >= l {
                            continue;
                        }
                        let tz = tz_i as usize;
                        let (b_t, psi_t, om_t) = (&b_snap[tz], &psi_snap[tz], &om_snap[tz]);
                        let sbase = lz.slot_base(e_idx);
                        lz.for_wired(e_idx, i, |rank, cell| {
                            let j = lz.decode(e_idx, i as u32, cell, size) as usize;
                            let widx = i * ts + sbase + rank;
                            let ex = ex_z[widx];
                            let ey = ey_z[widx];
                            let coef = 1.0 + dt * b_t[j];
                            let mut nex = coef * ex - dt * om_t[j] * ey + inj;
                            let mut ney = dt * om_t[j] * ex + coef * ey;
                            if nex.abs() < cut {
                                nex = 0.0;
                            }
                            if ney.abs() < cut {
                                ney = 0.0;
                            }
                            ex_z[widx] = nex;
                            ey_z[widx] = ney;
                            if psi_t[j] != 0.0 && nex != 0.0 {
                                el_z[widx] += psi_t[j] * nex;
                            }
                        });
                    }
                }
            });
        }
        // roll prev_fired ← this wave's firers
        let fired_now = cur.lock().unwrap();
        for z in 0..l {
            for v in prev_fired[z].iter_mut() {
                *v = false;
            }
            for &i in &fired_now[z] {
                prev_fired[z][i as usize] = true;
            }
        }
    }
    net.clear_listeners();
    elig
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_peaks_at_zero_and_is_symmetric() {
        let z = surrogate(0.0);
        assert!(z > 0.0, "positive at threshold");
        assert!((surrogate(0.3) - surrogate(-0.3)).abs() < 1e-6, "symmetric");
        assert!(surrogate(0.0) > surrogate(0.5), "peaks near 0");
        assert!(surrogate(50.0).abs() < 1e-3, "≈0 far from threshold");
    }

    #[test]
    fn elig_params_default_is_weights_only() {
        let p = EligParams::default();
        assert!(!p.train_omega_b, "Phase 2a default: ω/b′ frozen");
        assert!(p.eps_cut > 0.0);
    }
}
