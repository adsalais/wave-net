//! `critical_init` — the default engine initialisation: a rate-free homeostatic weight-training that
//! drives each forward hop's branching ratio σ → 1 (the edge of chaos), layer-wise greedy bottom-up,
//! via the e-prop update primitive. Replaces the firing-rate calibration (now a `bench` fallback for
//! recurrent configs). **Feed-forward only.** Also exposes the σ diagnostic (`forward_avalanche`) and
//! the deterministic noise drive (`random_l0_input`) used by both the init and the diagnostic.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix, P_INPUT};

/// A deterministic per-wave input: injects each L0 local with probability `fraction_q16 / 2^16`.
pub fn random_l0_input(seed: u64, size: u32, fraction_q16: u32) -> impl Fn(usize) -> Vec<u32> {
    let ls = size * size;
    move |wave: usize| {
        let mut v = Vec::new();
        for local in 0..ls {
            let h = mix(key(seed, local, 0, wave as u32, P_INPUT));
            if ((h & 0xFFFF) as u32) < fraction_q16 {
                v.push(local);
            }
        }
        v
    }
}

/// σ diagnostic — per-hop **forward** damage-spreading footprint: inject a `burst` of extra L0 spikes
/// at wave `warmup` and, under deferred one-hop propagation, track the perturbation footprint as it
/// climbs (layer `z`'s front lands at wave `warmup + z`). Returns the mean per-layer footprint over
/// `n_perturb` sites; the per-hop branching ratio is `footprint[z]/footprint[z-1]` (≈1 critical, <1
/// the cue shrinks with depth, >1 super-critical). `footprint[0] ≈ burst` (the injected spikes). The
/// burst cancels in the ratio but the larger footprint cuts the noise.
pub fn forward_avalanche(net: &mut Network, drive_seed: u64, drive_frac_q16: u32, warmup: usize, n_perturb: usize, burst: usize) -> Vec<f64> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let size = net.size();
    let drive = random_l0_input(drive_seed, size, drive_frac_q16);
    let steps = warmup + l + 1;
    let run = |net: &mut Network, extra: &[u32]| -> Vec<Vec<Vec<u32>>> {
        let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
        for z in 0..l {
            let r = rec.clone();
            net.on_layer(z, Box::new(move |_w, fired: &[u32]| r.lock().unwrap()[z].push(fired.to_vec())));
        }
        net.reset_state();
        for w in 0..steps {
            let mut sites = drive(w);
            if w == warmup {
                for &e in extra {
                    if !sites.contains(&e) {
                        sites.push(e);
                    }
                }
            }
            net.wave(&sites);
        }
        net.clear_listeners();
        let out = rec.lock().unwrap().clone();
        out
    };
    let refr = run(net, &[]);
    let driven: HashSet<u32> = drive(warmup).into_iter().collect();
    let np = n_perturb.max(1);
    let b = burst.max(1);
    let mut footprint = vec![0f64; l];
    for k in 0..np {
        let mut extra: Vec<u32> = Vec::with_capacity(b);
        for j in 0..b {
            let mut p = (mix(key(drive_seed, k as u32, j as i32, 0, 0xC1)) % ls as u64) as u32;
            let mut guard = 0;
            while (driven.contains(&p) || extra.contains(&p)) && guard < ls {
                p = (p + 1) % ls as u32;
                guard += 1;
            }
            if !driven.contains(&p) && !extra.contains(&p) {
                extra.push(p);
            }
        }
        let pert = run(net, &extra);
        for z in 0..l {
            let wv = warmup + z;
            let a: HashSet<u32> = refr[z][wv].iter().copied().collect();
            let bset: HashSet<u32> = pert[z][wv].iter().copied().collect();
            footprint[z] += a.symmetric_difference(&bset).count() as f64 / np as f64;
        }
    }
    footprint
}

#[derive(Clone, Copy, Debug)]
pub struct CriticalInitParams {
    pub rounds: usize,    // max σ-gain rounds per layer edge
    pub lr: f32,          // per-synapse σ-error learning rate
    pub tol: f32,         // stop an edge when |σ_hop - 1| <= tol
    pub warmup: usize,    // avalanche / eligibility measurement warmup
    pub waves: usize,     // window for the source pre-trace
    pub n_perturb: usize, // perturbation sites averaged per σ measurement
    pub burst: usize,     // extra spikes injected per perturbation (bigger = less noisy σ)
}

impl Default for CriticalInitParams {
    fn default() -> CriticalInitParams {
        CriticalInitParams { rounds: 40, lr: 0.05, tol: 0.15, warmup: 32, waves: 96, n_perturb: 24, burst: 16 }
    }
}

impl Network {
    /// **Default init.** Rate-free criticality: drive each forward hop's branching ratio σ_hop → 1 by a
    /// per-synapse e-prop update with a uniform σ-error signal `(σ_hop − 1)` (rate is emergent, no
    /// set-point), layer-wise greedy bottom-up. σ_hop measured by `forward_avalanche`; the source's
    /// per-neuron `pre_i` heterogeneity thins/grows weights smoothly (and the f32 shadow crossing zero
    /// can flip signs to inhibition). Feed-forward (single up entry, index 0).
    pub fn critical_init(&mut self, drive_seed: u64, frac_q16: u32, params: &CriticalInitParams) {
        let l = self.layer_count();
        let ls = (self.size() * self.size()) as usize;
        let drive = random_l0_input(drive_seed, self.size(), frac_q16);
        for z in 1..l {
            let src = z - 1;
            for _ in 0..params.rounds {
                let fp = forward_avalanche(self, drive_seed, frac_q16, params.warmup, params.n_perturb, params.burst);
                let denom = fp[z - 1];
                let sigma = if denom > 0.0 { fp[z] / denom } else { 0.0 };
                if denom > 0.0 && (sigma - 1.0).abs() <= params.tol as f64 {
                    break;
                }
                let sig_err = (sigma - 1.0) as f32;
                let (pre, psi) = self.windowed_eligibility(params.warmup, params.waves, &drive);
                self.eprop_update(src, 0, &pre[src], &psi[z], &vec![sig_err; ls], params.lr, false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_input_hits_expected_fraction() {
        let input = random_l0_input(1, 8, 32768); // ~50%
        let total: usize = (0..200).map(|w| input(w).len()).sum();
        let frac = total as f64 / (200 * 64) as f64;
        assert!((frac - 0.5).abs() < 0.05, "fraction {frac} != ~0.5");
    }
}
