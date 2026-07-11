//! `critical_init` — a homeostatic weight-training **initialization** that replaces the brittle
//! threshold calibration. Instead of tuning per-layer baselines toward a firing-rate *proxy*, it
//! trains the stored **weights** with an e-prop update driven **only** by the per-neuron rate error
//! (the `rate_reg` learning signal, no task), on random noise, **layer-wise greedy bottom-up**.
//!
//! Why weights, not thresholds: a threshold only *gates* a fixed projection — it cannot manufacture
//! drive that never arrives, so calibration cannot revive a sub-critical (cue-dies-with-depth) stack.
//! Training weights raises the *gain*: a too-quiet neuron's incoming weights rise until it fires. This
//! is homeostatic synaptic scaling (the mechanism cortex is thought to use to self-organize toward
//! criticality). Greedy bottom-up is load-bearing: each edge `(z-1)->z` is trained only once its
//! source `z-1` is already live, so the target receives input and has non-zero eligibility (ψ > 0) to
//! learn from — otherwise a fully-dead layer gets no gradient. Feed-forward only (no readout / no
//! recurrence) for this first spike.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::bench::rsnn::target_of;
use crate::wave_net::calibrate::random_l0_input;
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix};

#[derive(Clone, Copy, Debug)]
pub struct CriticalInitParams {
    pub target_permille: u64, // desired per-neuron firing rate (e.g. 100 = 10%)
    pub lr: f32,              // e-prop learning rate for the homeostatic weight update
    pub rounds: usize,        // max update rounds per layer edge
    pub warmup: usize,        // waves discarded per measurement (let adaptation settle)
    pub waves: usize,         // waves the eligibility/rate window integrates over
    pub tol_permille: u64,    // stop an edge when |rate - target| <= tol
    pub use_psi: bool,        // gate the update by ψ (e-prop) or not (pure activity-driven scaling)
}

impl Default for CriticalInitParams {
    fn default() -> CriticalInitParams {
        CriticalInitParams { target_permille: 100, lr: 0.02, rounds: 80, warmup: 32, waves: 96, tol_permille: 15, use_psi: true }
    }
}

/// Windowed per-neuron eligibility for every layer: the pre-trace (spike count) and ψ (near-threshold
/// count) accumulated over `waves` *after* a `warmup` transient, read via the difference of the
/// engine's running `elig_pre` / `elig_post` accumulators (so the boots-hot transient is excluded).
fn windowed_eligibility(
    net: &mut Network,
    warmup: usize,
    waves: usize,
    input: &impl Fn(usize) -> Vec<u32>,
) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let l = net.layer_count();
    net.reset_state();
    for w in 0..warmup {
        net.wave(&input(w));
    }
    let pre0: Vec<Vec<i32>> = (0..l).map(|z| net.with_layer(z, |x| x.elig_pre.clone())).collect();
    let psi0: Vec<Vec<i32>> = (0..l).map(|z| net.with_layer(z, |x| x.elig_post.clone())).collect();
    for w in 0..waves {
        net.wave(&input(warmup + w));
    }
    let pre: Vec<Vec<i32>> =
        (0..l).map(|z| net.with_layer(z, |x| x.elig_pre.iter().zip(&pre0[z]).map(|(a, b)| a - b).collect())).collect();
    let psi: Vec<Vec<i32>> =
        (0..l).map(|z| net.with_layer(z, |x| x.elig_post.iter().zip(&psi0[z]).map(|(a, b)| a - b).collect())).collect();
    (pre, psi)
}

/// Train the net to a live ~`target` regime by homeostatic weight scaling, layer-wise bottom-up.
/// `seed` must be the engine's construction seed (needed to recover each synapse's target). Assumes a
/// uniform single-entry feed-forward topology (level +1). Mutates the stored weights in place.
pub fn rate_reg_init(net: &mut Network, seed: u64, params: &CriticalInitParams, input: &impl Fn(usize) -> Vec<u32>) {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let size = net.size();
    let r_target = params.target_permille as f32 / 1000.0;
    let waves_f = params.waves as f32;
    // Forward fan-out (count) and radius, read from a computational layer's topology.
    let (up, up_radius) = net.with_layer(1, |lz| {
        let e = &lz.topology[0];
        (e.count as usize, e.radius)
    });

    // Edge (z-1) -> z: adjust the SOURCE layer's out-weights so target layer z fires near target.
    for z in 1..l {
        for _round in 0..params.rounds {
            let (pre, psi) = windowed_eligibility(net, params.warmup, params.waves, input);
            let rate_z = pre[z].iter().sum::<i32>() as f32 / (params.waves * ls) as f32;
            if ((rate_z - r_target).abs() * 1000.0) as u64 <= params.tol_permille {
                break;
            }
            // Per-target-neuron learning signal: the rate error (rate_reg with no task term).
            let l_sig: Vec<f32> = (0..ls).map(|j| pre[z][j] as f32 / waves_f - r_target).collect();
            let src = z - 1;
            net.with_layer_mut(src, |lz| {
                for i in 0..ls {
                    let pre_i = pre[src][i] as f32;
                    if pre_i == 0.0 {
                        continue;
                    }
                    let sg = (src * ls + i) as u32;
                    for kk in 0..up {
                        let j = target_of(seed, sg, i as u32, 1, kk as u32, up_radius, size) as usize;
                        // too-quiet target (l_sig<0) -> weights rise (fires more); mirrors rsnn rate_reg.
                        let pf = if params.use_psi { psi[z][j] as f32 } else { 1.0 };
                        lz.out_shadow[i * up + kk] += -params.lr * l_sig[j] * pre_i * pf;
                    }
                }
                for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                    *wq = s.round().clamp(-127.0, 127.0) as i8;
                }
            });
        }
    }
}

/// Per-hop **forward** damage-spreading avalanche footprint: inject one extra L0 spike at wave
/// `warmup` and, under deferred one-hop propagation, track its footprint as it climbs — layer `z`'s
/// front lands at wave `warmup + z`. Returns the mean per-layer footprint (index `z` = # neurons in
/// layer `z` that fire differently because of the one spike), averaged over `n_perturb` sites. The
/// per-hop branching ratio is `footprint[z+1] / footprint[z]`: ≈1 critical, <1 the cue shrinks with
/// depth (sub-critical), >1 grows. Unlike the whole-network `sigma_probe`, this isolates each hop
/// (no cross-layer accumulation), so it's the right σ read for a feed-forward stack. Bench-side.
pub fn forward_avalanche(net: &mut Network, drive_seed: u64, drive_frac_q16: u32, warmup: usize, n_perturb: usize, burst: usize) -> Vec<f64> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let size = net.size();
    let drive = random_l0_input(drive_seed, size, drive_frac_q16);
    let steps = warmup + l + 1;
    // Inject `burst` extra L0 spikes at `warmup` (bigger avalanche = less noisy σ ratio; burst cancels).
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
        // `b` distinct extra sites for this trial, none already driven at `warmup`.
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
            let wv = warmup + z; // wave the avalanche front reaches layer z (z=0: the injected spikes)
            let a: HashSet<u32> = refr[z][wv].iter().copied().collect();
            let bset: HashSet<u32> = pert[z][wv].iter().copied().collect();
            footprint[z] += a.symmetric_difference(&bset).count() as f64 / np as f64;
        }
    }
    footprint // footprint[0] ≈ burst; σ_hop(z-1→z) = footprint[z]/footprint[z-1]
}

#[derive(Clone, Copy, Debug)]
pub struct SigmaInitParams {
    pub rounds: usize,   // max gain-scaling rounds per layer edge
    pub alpha: f32,      // damped step: g = (1/σ)^alpha (0<alpha<=1)
    pub tol: f32,        // stop an edge when |σ_hop - 1| <= tol
    pub warmup: usize,   // avalanche measurement warmup
    pub n_perturb: usize, // perturbation sites averaged per σ measurement
    pub burst: usize,    // extra spikes injected per perturbation (bigger = less noisy σ)
}

impl Default for SigmaInitParams {
    fn default() -> SigmaInitParams {
        SigmaInitParams { rounds: 30, alpha: 0.5, tol: 0.15, warmup: 32, n_perturb: 24, burst: 16 }
    }
}

/// Multiply layer `z`'s out-weights (via the f32 shadow, then requantize) by gain `g`. Rescales the
/// random projection uniformly — changes the effective gain (hence σ) while preserving its structure.
fn scale_layer(net: &mut Network, z: usize, g: f32) {
    net.with_layer_mut(z, |lz| {
        for s in lz.out_shadow.iter_mut() {
            *s *= g;
        }
        for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
            *wq = s.round().clamp(-127.0, 127.0) as i8;
        }
    });
}

#[derive(Clone, Copy, Debug)]
pub struct SigmaEpropParams {
    pub rounds: usize,
    pub lr: f32,          // per-synapse σ-error learning rate
    pub tol: f32,         // stop an edge when |σ_hop - 1| <= tol
    pub warmup: usize,
    pub waves: usize,     // window for the source pre-trace
    pub n_perturb: usize, // sites averaged per σ measurement
    pub burst: usize,     // extra spikes injected per perturbation (bigger = less noisy σ)
}

impl Default for SigmaEpropParams {
    fn default() -> SigmaEpropParams {
        SigmaEpropParams { rounds: 40, lr: 0.05, tol: 0.15, warmup: 32, waves: 96, n_perturb: 24, burst: 16 }
    }
}

/// **Rate-free criticality init with sign-flipping.** Same σ_hop→1 objective as `sigma_gain_init`, but
/// a **per-synapse** update instead of a uniform scale: `out_shadow[i,kk] += -lr·(σ_hop-1)·pre_i`,
/// driven by the source's activity `pre_i` (no ψ, so dead targets still revive). The f32 latent shadow
/// crossing zero flips `+1 → 0 → -1` — so a super-critical hop is tamed by turning the most-active
/// sources **inhibitory** (and the per-source spread in `pre_i` gives the heterogeneity a uniform
/// scalar lacked at the int8 quantization floor). σ>1 → weights down/negative, σ<1 (or dead) → up.
/// Layer-wise greedy bottom-up; int8 quantizer (BitNet ternary is a separate path).
pub fn sigma_eprop_init(net: &mut Network, drive_seed: u64, drive_frac_q16: u32, params: &SigmaEpropParams) {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let size = net.size();
    let drive = random_l0_input(drive_seed, size, drive_frac_q16);
    for z in 1..l {
        let src = z - 1;
        let up = net.with_layer(src, |lz| lz.topology[0].count as usize);
        for _ in 0..params.rounds {
            let fp = forward_avalanche(net, drive_seed, drive_frac_q16, params.warmup, params.n_perturb, params.burst);
            let denom = fp[z - 1];
            let sigma = if denom > 0.0 { fp[z] / denom } else { 0.0 };
            if denom > 0.0 && (sigma - 1.0).abs() <= params.tol as f64 {
                break;
            }
            let sig_err = (sigma - 1.0) as f32; // >0 super-critical (shrink/flip), <0 sub-critical/dead (grow)
            let (pre, _psi) = windowed_eligibility(net, params.warmup, params.waves, &drive);
            let lr = params.lr;
            net.with_layer_mut(src, |lz| {
                for i in 0..ls {
                    let pre_i = pre[src][i] as f32;
                    if pre_i == 0.0 {
                        continue;
                    }
                    let delta = -lr * sig_err * pre_i;
                    for kk in 0..up {
                        lz.out_shadow[i * up + kk] += delta;
                    }
                }
                for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                    *wq = s.round().clamp(-127.0, 127.0) as i8;
                }
            });
        }
    }
}

/// **Rate-free criticality init.** Drive each forward hop's branching ratio σ_hop → 1 by scaling the
/// source layer's weight *gain* (no firing-rate target — the rate is whatever criticality produces),
/// layer-wise greedy bottom-up. σ_hop is measured by `forward_avalanche` (per-hop damage spreading);
/// σ>1 → shrink the gain, σ<1 (or a dead target) → grow it, until |σ-1|≤tol, then freeze and move up.
pub fn sigma_gain_init(net: &mut Network, drive_seed: u64, drive_frac_q16: u32, params: &SigmaInitParams) {
    let l = net.layer_count();
    for z in 1..l {
        for _ in 0..params.rounds {
            let fp = forward_avalanche(net, drive_seed, drive_frac_q16, params.warmup, params.n_perturb, params.burst);
            let denom = fp[z - 1];
            let sigma = if denom > 0.0 { fp[z] / denom } else { 0.0 };
            if denom > 0.0 && (sigma - 1.0).abs() <= params.tol as f64 {
                break;
            }
            // σ<1 or a dead target → grow the gain; σ>1 → shrink it. Damped by `alpha` for stability.
            let alpha = params.alpha as f64;
            let g = if sigma <= 0.0 { 1.0 + alpha } else { (1.0 / sigma).powf(alpha) };
            scale_layer(net, z - 1, g as f32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::readout::NearestCentroid;
    use crate::bench::regime::{as_f64, effective_dim};
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::TopologyLevel;

    /// Top-computational-layer spike-count feature over a read window, given a per-wave input `cue`.
    /// Warm up (present − read waves, uncounted), then count the top layer's spikes over `read` waves.
    fn top_features(net: &mut Network, cue: impl Fn(usize) -> Vec<u32>, present: usize, read: usize) -> Vec<u32> {
        let l = net.layer_count();
        let ls = (net.size() * net.size()) as usize;
        let top = l - 1;
        net.reset_state();
        let warm = present.saturating_sub(read);
        for w in 0..warm {
            net.wave(&cue(w));
        }
        let counts = Arc::new(Mutex::new(vec![0u32; ls]));
        {
            let c = counts.clone();
            net.on_layer(top, Box::new(move |_w, fired: &[u32]| {
                let mut g = c.lock().unwrap();
                for &loc in fired {
                    g[loc as usize] += 1;
                }
            }));
        }
        for w in 0..read {
            net.wave(&cue(warm + w));
        }
        net.clear_listeners();
        std::mem::take(&mut *counts.lock().unwrap())
    }

    /// Class-`class` noisy cue: template = fixed ~`base` subset of L0 for that class, flipped per (trial,wave).
    fn pattern_cue(class_seed: u64, ls: usize, class: usize, trial: usize, base: u32, noise: u32) -> impl Fn(usize) -> Vec<u32> {
        move |w| {
            (0..ls as u32)
                .filter(|&local| {
                    let tmpl = (mix(key(class_seed, local, class as i32, 0, 0xC0E)) & 0xFFFF) < base as u64;
                    let nz = (mix(key(class_seed, local, class as i32, (trial * 131 + w + 1) as u32, 0xC0F)) & 0xFFFF) < noise as u64;
                    tmpl ^ nz
                })
                .collect()
        }
    }

    /// Spatial-XOR cue: region A (first half of L0) active iff `f1`, region B iff `f2`; label = f1^f2.
    /// Linear-inseparable ({00,11} vs {01,10}), so top-layer NC accuracy > chance requires the reservoir
    /// to mix the two regions nonlinearly — a stronger test than the linear pattern task.
    fn xor_cue(class_seed: u64, ls: usize, f1: bool, f2: bool, trial: usize, base: u32, noise: u32) -> impl Fn(usize) -> Vec<u32> {
        let half = (ls / 2) as u32;
        move |w| {
            (0..ls as u32)
                .filter(|&local| {
                    let region_a = local < half;
                    let active = if region_a { f1 } else { f2 };
                    let tmpl = active && (mix(key(class_seed, local, region_a as i32, 0, 0xD0E)) & 0xFFFF) < base as u64;
                    let nz = (mix(key(class_seed, local, 2, (trial * 131 + w + 1) as u32, 0xD0F)) & 0xFFFF) < noise as u64;
                    tmpl ^ nz
                })
                .collect()
        }
    }

    /// Held-out nearest-centroid accuracy (permille) + effective-dim from labelled features.
    fn score(feats: &[Vec<u32>], labels: &[usize], k: usize) -> (u64, f64) {
        let trials = feats.len();
        let half = trials / 2;
        let nc = NearestCentroid::fit(&feats[..half], &labels[..half], k);
        let test = (trials - half).max(1);
        let correct = (half..trials).filter(|&i| nc.predict(&feats[i]) == labels[i]).count();
        ((correct * 1000 / test) as u64, effective_dim(&as_f64(feats)))
    }

    /// K-class linear pattern classification on the top layer.
    fn quality_pattern(net: &mut Network, class_seed: u64, k: usize, trials: usize, present: usize, read: usize, base: u32, noise: u32) -> (u64, f64) {
        let ls = (net.size() * net.size()) as usize;
        let mut feats = Vec::with_capacity(trials);
        let mut labels = Vec::with_capacity(trials);
        for t in 0..trials {
            let class = t % k;
            feats.push(top_features(net, pattern_cue(class_seed, ls, class, t, base, noise), present, read));
            labels.push(class);
        }
        score(&feats, &labels, k)
    }

    /// 2-class nonlinear spatial-XOR on the top layer.
    fn quality_xor(net: &mut Network, class_seed: u64, trials: usize, present: usize, read: usize, base: u32, noise: u32) -> (u64, f64) {
        let ls = (net.size() * net.size()) as usize;
        let mut feats = Vec::with_capacity(trials);
        let mut labels = Vec::with_capacity(trials);
        for t in 0..trials {
            let (f1, f2) = ((t & 1) == 1, (t & 2) == 2);
            feats.push(top_features(net, xor_cue(class_seed, ls, f1, f2, t, base, noise), present, read));
            labels.push((f1 ^ f2) as usize);
        }
        score(&feats, &labels, 2)
    }

    const SEED: u64 = 0xC0FFEE_1234_5678;

    /// The 32×32 × 5 uniform feed-forward config, parameterized by forward fan-out `up_count`, spatial
    /// `up_radius`, and `adapt_bump` (0 = LIF, >0 = ALIF).
    fn ff_config(up_count: u32, up_radius: u32, adapt_bump: i16) -> Config {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay: 6,
        };
        Config { seed: SEED, size: 32, layers: vec![layer; 5] }
    }

    fn pct(r: &[f64]) -> Vec<f64> {
        r.iter().map(|x| (x * 1000.0).round() / 10.0).collect()
    }

    /// Sanity (fast): on a small starved stack, rate-init must lift the top layer's firing well above
    /// the untrained ±1 net (i.e. it revives depth, which is the whole point).
    #[test]
    fn rate_init_revives_a_starved_stack() {
        let cfg = {
            let layer = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 2, count: 6 }], // low fan-out -> starves
                leak: (3, 5),
                cooldown_base: 2,
                inhibitor_ratio: 0,
                threshold_jitter: 16,
                baseline_init: 6,
                adapt_bump: 5,
                adapt_decay: 6,
            };
            Config { seed: SEED, size: 8, layers: vec![layer; 4] }
        };
        let input = random_l0_input(SEED, 8, 20000);
        let top = cfg.layers.len() - 1;

        let mut untrained = Network::new(cfg.clone());
        let before = untrained.measure_layer_rates(16, 64, &input)[top];

        let mut net = Network::new(cfg);
        let params = CriticalInitParams { rounds: 40, ..CriticalInitParams::default() };
        rate_reg_init(&mut net, SEED, &params, &input);
        let after = net.measure_layer_rates(16, 64, &input)[top];

        assert!(after > before + 0.02, "rate-init should revive the top layer: {before:.3} -> {after:.3}");
    }

    /// Experiment (run manually): does rate-init reach criticality (σ≈1), and how does **density**
    /// (fan-out `up_count`, spatial `up_radius`) + **ALIF** affect it? For each config: rate-init, then
    /// measure per-layer rate and the per-hop forward-avalanche footprint. σ_hop = footprint[k+1]/[k]:
    /// ≈1 critical, <1 the cue dies with depth, >1 super-critical. Hypothesis under test: ALIF holds
    /// the *rate* at target while high density lets *branching* (σ) run >1 — i.e. adaptation masks a
    /// density-driven super-criticality that the rate alone can't see.
    ///   cargo test --release rate_init_criticality_vs_density -- --ignored --nocapture
    #[test]
    #[ignore]
    fn rate_init_criticality_vs_density() {
        let init = CriticalInitParams::default();
        let hops = |f: &[f64]| -> Vec<f64> {
            (1..f.len() - 1).map(|z| if f[z] > 0.0 { (f[z + 1] / f[z] * 100.0).round() / 100.0 } else { 0.0 }).collect()
        };
        let fp = |f: &[f64]| f[1..].iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>();
        let mut run = |up_count: u32, up_radius: u32, adapt_bump: i16| {
            let input = random_l0_input(SEED, 32, 20000);
            let mut net = Network::new(ff_config(up_count, up_radius, adapt_bump));
            rate_reg_init(&mut net, SEED, &init, &input);
            let r = net.measure_layer_rates(32, 128, &input);
            let f = forward_avalanche(&mut net, SEED ^ 0xABCD, 20000, 32, 16, 16);
            let alif = if adapt_bump > 0 { "ALIF" } else { "LIF " };
            println!("uc={up_count:<2} r={up_radius} {alif}: rates={:?} footprint{:?} σ_hop{:?}", pct(&r), fp(&f), hops(&f));
        };
        println!("== density (up_count) sweep, radius 3, ALIF ==");
        for uc in [8u32, 16, 24, 32, 48] {
            run(uc, 3, 5);
        }
        println!("== radius sweep, up_count 16, ALIF ==");
        for rad in [2u32, 4] {
            run(16, rad, 5);
        }
        println!("== ALIF vs LIF (adapt_bump 0), radius 3 ==");
        for uc in [16u32, 32] {
            run(uc, 3, 5);
            run(uc, 3, 0);
        }
    }

    /// Experiment (run manually): the **rate-free σ-gain init** across the density sweep. Targets
    /// σ_hop≈1 by scaling gain (no rate set-point); the rate is emergent. Expect σ_hop≈1 at every
    /// density, with the rate self-selecting (high density → lower rate, low density → higher).
    ///   cargo test --release sigma_gain_init_vs_density -- --ignored --nocapture
    #[test]
    #[ignore]
    fn sigma_gain_init_vs_density() {
        let params = SigmaInitParams::default();
        let hops = |f: &[f64]| -> Vec<f64> {
            (1..f.len()).map(|z| if f[z - 1] > 0.0 { (f[z] / f[z - 1] * 100.0).round() / 100.0 } else { 0.0 }).collect()
        };
        let fp = |f: &[f64]| f.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>();
        for up_count in [8u32, 16, 24, 32, 48] {
            let mut net = Network::new(ff_config(up_count, 3, 5));
            sigma_gain_init(&mut net, SEED ^ 0xABCD, 20000, &params);
            let input = random_l0_input(SEED, 32, 20000);
            let r = net.measure_layer_rates(32, 128, &input);
            let f = forward_avalanche(&mut net, SEED ^ 0xABCD, 20000, 32, 32, 16);
            println!("uc={up_count:<2}: rates(emergent)={:?} footprint(L0..){:?} σ_hop{:?}", pct(&r), fp(&f), hops(&f));
        }
    }

    /// Experiment (run manually): the **sign-flipping** σ-eprop init across density. Same σ_hop≈1,
    /// rate-free objective, but a per-synapse update that can turn sources inhibitory — so it should
    /// tame the high-density super-criticality where the uniform-scaling controller oscillated. Also
    /// prints the fraction of negative (inhibitory) weights per layer, to see the brake being built.
    ///   cargo test --release sigma_eprop_init_vs_density -- --ignored --nocapture
    #[test]
    #[ignore]
    fn sigma_eprop_init_vs_density() {
        let params = SigmaEpropParams::default();
        let hops = |f: &[f64]| -> Vec<f64> {
            (1..f.len()).map(|z| if f[z - 1] > 0.0 { (f[z] / f[z - 1] * 100.0).round() / 100.0 } else { 0.0 }).collect()
        };
        for up_count in [8u32, 16, 24, 32, 48] {
            let mut net = Network::new(ff_config(up_count, 3, 5));
            sigma_eprop_init(&mut net, SEED ^ 0xABCD, 20000, &params);
            let input = random_l0_input(SEED, 32, 20000);
            let r = net.measure_layer_rates(32, 128, &input);
            let f = forward_avalanche(&mut net, SEED ^ 0xABCD, 20000, 32, 32, 16);
            let neg: Vec<f64> = (0..net.layer_count() - 1)
                .map(|z| {
                    let (n, tot) = net.with_layer(z, |l| (l.out_weights.iter().filter(|&&w| w < 0).count(), l.out_weights.len()));
                    (n as f64 / tot as f64 * 100.0).round() / 10.0 * 10.0
                })
                .collect();
            println!("uc={up_count:<2}: rates={:?} σ_hop{:?} neg%(L0..)={:?}", pct(&r), hops(&f), neg);
        }
    }

    /// Experiment (run manually): the FLAT-RATE init retested with the fixes — ψ dropped (pure
    /// activity-driven scaling, the stability/revival fix from the σ-eprop work) and σ read with the
    /// clean burst estimate. Question: does hitting a fixed 10% rate now still leave σ super-critical
    /// at high density, or does the improved rule change that?
    ///   cargo test --release rate_init_fixed_vs_density -- --ignored --nocapture
    #[test]
    #[ignore]
    fn rate_init_fixed_vs_density() {
        let hops = |f: &[f64]| -> Vec<f64> {
            (1..f.len()).map(|z| if f[z - 1] > 0.0 { (f[z] / f[z - 1] * 100.0).round() / 100.0 } else { 0.0 }).collect()
        };
        for use_psi in [true, false] {
            println!("== flat-rate init, use_psi={use_psi} (clean burst σ) ==");
            let params = CriticalInitParams { use_psi, ..CriticalInitParams::default() };
            for up_count in [8u32, 16, 24, 32, 48] {
                let mut net = Network::new(ff_config(up_count, 3, 5));
                let input = random_l0_input(SEED, 32, 20000);
                rate_reg_init(&mut net, SEED, &params, &input);
                let r = net.measure_layer_rates(32, 128, &input);
                let f = forward_avalanche(&mut net, SEED ^ 0xABCD, 20000, 32, 32, 16);
                println!("uc={up_count:<2}: rates={:?} σ_hop{:?}", pct(&r), hops(&f));
            }
        }
    }

    /// Experiment (run manually): **computational** effect of the two inits. For each density, init
    /// the FF net two ways (flat-rate with ψ; rate-free σ-eprop), then measure the intrinsic quality of
    /// the top-layer representation — held-out nearest-centroid accuracy (separability) + effective-dim
    /// on a 4-class noisy-cue task. Answers: does σ≈1 (rate-decaying) actually compute better than a
    /// flat rate (super-critical off the critical density), or not, under ALIF?
    ///   cargo test --release computation_vs_density -- --ignored --nocapture
    #[test]
    #[ignore]
    fn computation_vs_density() {
        let rate_params = CriticalInitParams::default(); // with ψ (the good flat-rate init)
        let sigma_params = SigmaEpropParams::default();
        let class_seed = SEED ^ 0x5EED;
        let (k, trials, present, read, base, noise) = (4usize, 200usize, 16usize, 8usize, 13107u32, 3277u32);
        println!("(chance = {}‰)", 1000 / k);
        for up_count in [8u32, 16, 24, 32, 48] {
            let input = random_l0_input(SEED, 32, 20000);
            let mut a = Network::new(ff_config(up_count, 3, 5));
            rate_reg_init(&mut a, SEED, &rate_params, &input);
            let (acc_a, dim_a) = quality_pattern(&mut a, class_seed, k, trials, present, read, base, noise);

            let mut b = Network::new(ff_config(up_count, 3, 5));
            sigma_eprop_init(&mut b, SEED ^ 0xABCD, 20000, &sigma_params);
            let (acc_b, dim_b) = quality_pattern(&mut b, class_seed, k, trials, present, read, base, noise);

            println!("uc={up_count:<2}: flat-rate acc={acc_a}‰ dim={dim_a:.1}  |  σ-eprop acc={acc_b}‰ dim={dim_b:.1}");
        }
    }

    /// Experiment (run manually): multi-seed × two-task confirmation of the computational verdict.
    /// Averages held-out accuracy over several reservoir/task seeds, on both the linear PATTERN task
    /// (4-class, chance 250‰) and the nonlinear spatial-XOR task (2-class, chance 500‰). Firms up
    /// whether σ-eprop really beats the flat-rate init (and checks the odd uc16 dip).
    ///   cargo test --release computation_multiseed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn computation_multiseed() {
        let seeds = [0xC0FFEE_1234_5678u64, 0x00A1_1CE5, 0x00B0_B0B0, 0xDEAD_BEEF, 0x1234_ABCD];
        let (trials, present, read, base, noise) = (200usize, 16usize, 8usize, 13107u32, 3277u32);
        let mk = |seed: u64, up: u32| {
            let layer = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 3, count: up }],
                leak: (3, 5),
                cooldown_base: 2,
                inhibitor_ratio: 0,
                threshold_jitter: 32,
                baseline_init: 6,
                adapt_bump: 5,
                adapt_decay: 6,
            };
            Config { seed, size: 32, layers: vec![layer; 5] }
        };
        println!("(pattern chance 250‰, xor chance 500‰; mean over {} seeds)", seeds.len());
        for up in [16u32, 24, 32] {
            let (mut fp, mut sp, mut fx, mut sx) = (0u64, 0u64, 0u64, 0u64);
            for &s in &seeds {
                let cs = s ^ 0x5EED;
                let input = random_l0_input(s, 32, 20000);
                let mut a = Network::new(mk(s, up));
                rate_reg_init(&mut a, s, &CriticalInitParams::default(), &input);
                fp += quality_pattern(&mut a, cs, 4, trials, present, read, base, noise).0;
                fx += quality_xor(&mut a, cs, trials, present, read, base, noise).0;

                let mut b = Network::new(mk(s, up));
                sigma_eprop_init(&mut b, s, 20000, &SigmaEpropParams::default());
                sp += quality_pattern(&mut b, cs, 4, trials, present, read, base, noise).0;
                sx += quality_xor(&mut b, cs, trials, present, read, base, noise).0;
            }
            let n = seeds.len() as u64;
            println!("uc={up}: PATTERN flat={}‰ σ={}‰  |  XOR flat={}‰ σ={}‰", fp / n, sp / n, fx / n, sx / n);
        }
    }
}
