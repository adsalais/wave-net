//! Reservoir-regime diagnostic — measures properties of the calibrated-but-untrained reservoir to find
//! which predict learnability (V1 & V2b) and how topology couples to the other knobs. Bench-side, f64.

use crate::bench::eprop::{calibrated_reservoir, pick_class, EpropConfig};
use crate::bench::readout::NearestCentroid;
use crate::bench::store_recall::{cue_realization, probe_pattern};
use crate::wave_net::network::Network;
use std::sync::{Arc, Mutex};

/// One trial (reset → present cue → delay → probe); returns per-computational-layer spike counts. Each
/// site in `flip` is toggled in every present wave (empty = unperturbed; the perturbation set for σ).
pub fn reservoir_states(
    net: &mut Network,
    cfg: &EpropConfig,
    class: usize,
    trial: usize,
    flip: &[u32],
) -> Vec<Vec<u32>> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![vec![0u32; ls]; l]));
    for z in 1..l {
        let c = counts.clone();
        net.on_layer(
            z,
            Box::new(move |_w: usize, fired: &[u32]| {
                let mut g = c.lock().unwrap();
                for &loc in fired {
                    g[z][loc as usize] += 1;
                }
            }),
        );
    }
    net.reset_state();
    for w in 0..cfg.present_waves {
        let mut sites = cue_realization(cfg.seed, cfg.size, class, trial, w, cfg.base_q16, cfg.keep_q16, cfg.noise_q16);
        for &s in flip {
            match sites.iter().position(|&x| x == s) {
                Some(pos) => {
                    sites.remove(pos);
                }
                None => sites.push(s),
            }
        }
        net.wave(&sites);
    }
    for _ in 0..cfg.delay {
        net.wave(&[]);
    }
    let probe = probe_pattern(cfg.seed, cfg.size, cfg.probe_q16);
    for _ in 0..cfg.read_waves {
        net.wave(&probe);
    }
    net.clear_listeners();
    let g = counts.lock().unwrap();
    (1..l).map(|z| g[z].clone()).collect()
}

/// The readout-accessible state: the **top** computational layer's spike counts. (The full reservoir's
/// lower layers directly carry the cue, so any linear readout separates them trivially — that saturates
/// and predicts nothing; learning reads only the top layer, so that is what the metrics must measure.)
pub fn top_state(layered: &[Vec<u32>]) -> Vec<u32> {
    layered.last().cloned().unwrap_or_default()
}

/// Collect `trials` top-layer reservoir states with their class labels (no training).
pub fn collect_states(cfg: &EpropConfig, trials: usize) -> (Vec<Vec<u32>>, Vec<usize>) {
    let mut net = calibrated_reservoir(cfg);
    let mut states = Vec::with_capacity(trials);
    let mut labels = Vec::with_capacity(trials);
    for t in 0..trials {
        let class = pick_class(cfg.seed, t, cfg.k);
        states.push(top_state(&reservoir_states(&mut net, cfg, class, t, &[])));
        labels.push(class);
    }
    (states, labels)
}

/// Held-out NearestCentroid accuracy (permille) on the reservoir states — the intrinsic separability.
pub fn separation_ceiling(cfg: &EpropConfig, trials: usize) -> u64 {
    let (states, labels) = collect_states(cfg, trials);
    let half = trials / 2;
    if half == 0 || half == trials {
        return 500;
    }
    let nc = NearestCentroid::fit(&states[..half], &labels[..half], cfg.k);
    let correct = (half..trials).filter(|&i| nc.predict(&states[i]) == labels[i]).count();
    (correct as u64 * 1000) / (trials - half) as u64
}

/// Fisher discriminant ratio S_B / S_W (trace form): between-class over within-class scatter.
pub fn fisher_ratio(states: &[Vec<u32>], labels: &[usize], k: usize) -> f64 {
    let n = states.len();
    let d = states[0].len();
    let mut mu = vec![0f64; d];
    for x in states {
        for (j, &v) in x.iter().enumerate() {
            mu[j] += v as f64;
        }
    }
    for m in &mut mu {
        *m /= n as f64;
    }
    let mut cmu = vec![vec![0f64; d]; k];
    let mut cn = vec![0f64; k];
    for (x, &c) in states.iter().zip(labels) {
        cn[c] += 1.0;
        for (j, &v) in x.iter().enumerate() {
            cmu[c][j] += v as f64;
        }
    }
    for c in 0..k {
        if cn[c] > 0.0 {
            for m in &mut cmu[c] {
                *m /= cn[c];
            }
        }
    }
    let mut sb = 0f64;
    for c in 0..k {
        let d2: f64 = (0..d).map(|j| (cmu[c][j] - mu[j]).powi(2)).sum();
        sb += cn[c] * d2;
    }
    let mut sw = 0f64;
    for (x, &c) in states.iter().zip(labels) {
        for j in 0..d {
            sw += (x[j] as f64 - cmu[c][j]).powi(2);
        }
    }
    // Degenerate top layer (constant/silent states): S_W = 0. With no between-class signal either
    // (S_B = 0) that is *no* discriminability → 0; a rare S_B > 0 with S_W = 0 is perfect separation.
    if sw <= 0.0 {
        return if sb > 0.0 { f64::INFINITY } else { 0.0 };
    }
    sb / sw
}

/// Participation ratio (tr C)² / tr(C²) of the state covariance — the effective dimensionality.
pub fn effective_dim(states: &[Vec<f64>]) -> f64 {
    let n = states.len();
    if n == 0 {
        return 0.0;
    }
    let d = states[0].len();
    let mut mu = vec![0f64; d];
    for x in states {
        for j in 0..d {
            mu[j] += x[j];
        }
    }
    for m in &mut mu {
        *m /= n as f64;
    }
    let mut c = vec![vec![0f64; d]; d];
    for x in states {
        for a in 0..d {
            let xa = x[a] - mu[a];
            for b in 0..d {
                c[a][b] += xa * (x[b] - mu[b]);
            }
        }
    }
    let tr: f64 = (0..d).map(|a| c[a][a] / n as f64).sum();
    let tr_sq: f64 = c.iter().flatten().map(|&v| (v / n as f64).powi(2)).sum();
    if tr_sq <= 0.0 { 0.0 } else { tr * tr / tr_sq }
}

/// Cast integer states to f64 rows.
pub fn as_f64(states: &[Vec<u32>]) -> Vec<Vec<f64>> {
    states.iter().map(|x| x.iter().map(|&v| v as f64).collect()).collect()
}

/// Legenstein–Maass power: effective rank across distinct inputs (both classes) minus effective rank
/// across noisy copies of one input (one class, different noise realizations). PR is the soft rank.
pub fn kernel_minus_gen_rank(cfg: &EpropConfig) -> f64 {
    let m = 64usize;
    let mut net = calibrated_reservoir(cfg);
    let kernel: Vec<Vec<u32>> = (0..m)
        .map(|t| top_state(&reservoir_states(&mut net, cfg, pick_class(cfg.seed, t, cfg.k), t, &[])))
        .collect();
    let noisy: Vec<Vec<u32>> = (0..m).map(|t| top_state(&reservoir_states(&mut net, cfg, 0, t, &[]))).collect();
    effective_dim(&as_f64(&kernel)) - effective_dim(&as_f64(&noisy))
}

/// σ / edge-of-chaos: flip a small localized set of L0 sites, measure per-layer Hamming divergence
/// (neurons whose count differs) between base and perturbed states, and return the geometric growth of
/// divergence up the stack. (A single-site flip is too weak to register in these small integer nets.)
pub fn perturbation_spread(cfg: &EpropConfig) -> f64 {
    use crate::wave_net::synapse::{key, mix};
    let mut net = calibrated_reservoir(cfg);
    let ls = cfg.size * cfg.size;
    let nflip = (ls / 16).max(4);
    let mut sites: Vec<u32> = (0..nflip).map(|i| (mix(key(cfg.seed, i, 0, 0, 71)) % ls as u64) as u32).collect();
    sites.sort_unstable();
    sites.dedup();
    let base = reservoir_states(&mut net, cfg, 0, 0, &[]);
    let pert = reservoir_states(&mut net, cfg, 0, 0, &sites);
    let div: Vec<f64> = base
        .iter()
        .zip(&pert)
        .map(|(b, p)| b.iter().zip(p).filter(|(x, y)| x != y).count() as f64)
        .collect();
    let mut log_sum = 0f64;
    let mut n = 0u32;
    for z in 1..div.len() {
        if div[z - 1] > 0.0 && div[z] > 0.0 {
            log_sum += (div[z] / div[z - 1]).ln();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (log_sum / n as f64).exp() }
}

/// Mean firing fraction per computational layer over `trials` trials.
pub fn layer_gain(cfg: &EpropConfig, trials: usize) -> Vec<f64> {
    let mut net = calibrated_reservoir(cfg);
    let ls = (cfg.size * cfg.size) as usize;
    let waves = (cfg.present_waves + cfg.delay + cfg.read_waves).max(1);
    let mut sum: Vec<f64> = Vec::new();
    for t in 0..trials {
        let layered = reservoir_states(&mut net, cfg, pick_class(cfg.seed, t, cfg.k), t, &[]);
        if sum.is_empty() {
            sum = vec![0.0; layered.len()];
        }
        for (z, layer) in layered.iter().enumerate() {
            sum[z] += layer.iter().sum::<u32>() as f64 / (ls * waves) as f64;
        }
    }
    sum.iter().map(|s| s / trials as f64).collect()
}

/// Degeneracy: (dead fraction, saturated fraction, sampled mean |pairwise correlation|). `waves` is the
/// per-trial spike ceiling used to flag saturation.
pub fn degeneracy(states: &[Vec<u32>], waves: u32) -> (f64, f64, f64) {
    let n = states.len();
    let d = states[0].len();
    let mut dead = 0usize;
    let mut sat = 0usize;
    for j in 0..d {
        let total: u64 = states.iter().map(|x| x[j] as u64).sum();
        let mx = states.iter().map(|x| x[j]).max().unwrap_or(0);
        if total == 0 {
            dead += 1;
        }
        if mx >= waves {
            sat += 1;
        }
    }
    // synchrony: mean |Pearson| over a deterministic sample of neuron-index pairs (stride 7)
    let mut sync_sum = 0f64;
    let mut pairs = 0u32;
    let mut a = 0usize;
    while a + 7 < d && pairs < 200 {
        let b = a + 7;
        let ma: f64 = states.iter().map(|x| x[a] as f64).sum::<f64>() / n as f64;
        let mb: f64 = states.iter().map(|x| x[b] as f64).sum::<f64>() / n as f64;
        let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
        for x in states {
            let da = x[a] as f64 - ma;
            let db = x[b] as f64 - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        if va > 0.0 && vb > 0.0 {
            sync_sum += (cov / (va.sqrt() * vb.sqrt())).abs();
            pairs += 1;
        }
        a += 1;
    }
    let sync = if pairs == 0 { 0.0 } else { sync_sum / pairs as f64 };
    (dead as f64 / d as f64, sat as f64 / d as f64, sync)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> EpropConfig {
        let mut cfg = EpropConfig::demo();
        cfg.calib.warmup = 8;
        cfg.calib.waves = 24;
        cfg.calib.max_steps = 12;
        cfg.calib.refine_passes = 2;
        cfg
    }

    fn dead_cfg() -> EpropConfig {
        let mut cfg = small();
        cfg.up_count = 8; // known dead regime from the sweep
        cfg
    }

    #[test]
    fn collect_states_shape_and_determinism() {
        let cfg = small();
        let (s1, y1) = collect_states(&cfg, 20);
        let (s2, y2) = collect_states(&cfg, 20);
        assert_eq!(s1.len(), 20);
        assert_eq!(s1[0].len(), (cfg.size * cfg.size) as usize); // top layer only
        assert_eq!((s1, y1), (s2, y2), "collection must be deterministic");
    }

    #[test]
    fn separation_ceiling_discriminates_working_from_dead() {
        let work = separation_ceiling(&small(), 200);
        let dead = separation_ceiling(&dead_cfg(), 200);
        eprintln!("ceiling work {work} dead {dead}");
        assert!(work > 600, "working reservoir separates classes: {work}");
        assert!(work > dead + 80, "working {work} > dead {dead}");
    }

    #[test]
    fn fisher_ratio_discriminates_working_from_dead() {
        let (sw, yw) = collect_states(&small(), 200);
        let (sd, yd) = collect_states(&dead_cfg(), 200);
        let fw = fisher_ratio(&sw, &yw, 2);
        let fd = fisher_ratio(&sd, &yd, 2);
        eprintln!("fisher work {fw:.4} dead {fd:.4}");
        assert!(fw > fd, "working Fisher {fw} > dead {fd}");
    }

    #[test]
    fn effective_dim_matches_known_participation_ratio() {
        // rank-1 (all rows a scalar multiple of one direction) → PR ≈ 1
        let rank1: Vec<Vec<f64>> = (1..=10).map(|a| vec![a as f64, 2.0 * a as f64, 3.0 * a as f64]).collect();
        let pr1 = effective_dim(&rank1);
        assert!((pr1 - 1.0).abs() < 0.05, "rank-1 PR ~ 1, got {pr1}");
        // isotropic 3-D (independent ±unit per axis, equal variance, zero covariance) → PR ≈ 3
        let iso: Vec<Vec<f64>> = (0..3)
            .flat_map(|d| {
                let mut p = vec![0.0; 3];
                p[d] = 1.0;
                let mut n = vec![0.0; 3];
                n[d] = -1.0;
                [p, n]
            })
            .collect();
        let pr3 = effective_dim(&iso);
        assert!((pr3 - 3.0).abs() < 0.05, "isotropic PR ~ 3, got {pr3}");
    }

    #[test]
    fn kernel_minus_gen_rank_is_finite_and_deterministic() {
        let a = kernel_minus_gen_rank(&small());
        let b = kernel_minus_gen_rank(&small());
        assert!(a.is_finite());
        assert_eq!(a, b, "deterministic");
    }

    #[test]
    fn perturbation_spread_orders_regimes() {
        // a starved (sparse) reservoir should spread a perturbation no more than the denser baseline
        let sparse = perturbation_spread(&dead_cfg());
        let dense = perturbation_spread(&small());
        eprintln!("sigma sparse {sparse:.3} dense {dense:.3}");
        assert!(sparse.is_finite() && dense.is_finite());
        assert!(dense >= sparse, "denser reservoir spreads at least as much: dense {dense} vs sparse {sparse}");
    }

    #[test]
    fn degeneracy_flags_dead_and_saturated() {
        // 4 features: #0 dead (all 0), #3 saturated (== waves every trial); waves = 5
        let states: Vec<Vec<u32>> = (0..10).map(|i| vec![0, (i % 3) as u32, (i % 2) as u32, 5]).collect();
        let (dead, sat, _sync) = degeneracy(&states, 5);
        assert!((dead - 0.25).abs() < 1e-9, "one of four dead: {dead}");
        assert!((sat - 0.25).abs() < 1e-9, "one of four saturated: {sat}");
    }

    fn learned(mut cfg: EpropConfig, readout: bool, broadcast: bool, lr: f64) -> u64 {
        use crate::bench::eprop::train;
        cfg.readout = readout;
        cfg.broadcast = broadcast;
        if broadcast {
            cfg.softmax_temp = 10.0;
        }
        cfg.trials = 1500;
        cfg.block = 250;
        let c = train(&cfg, lr).accuracy_permille;
        let h = c.len() / 2;
        c[h..].iter().sum::<u64>() / (c.len() - h).max(1) as u64
    }

    #[test]
    #[ignore]
    fn _regime_vs_learnability() {
        let base = small();
        let cases: Vec<(&str, EpropConfig)> = vec![
            ("baseline", base.clone()),
            ("up_count=8", { let mut c = base.clone(); c.up_count = 8; c }),
            ("up_count=24", { let mut c = base.clone(); c.up_count = 24; c }),
            ("up_radius=2", { let mut c = base.clone(); c.up_radius = 2; c }),
            ("layers=2", { let mut c = base.clone(); c.layers = 2; c }),
            ("layers=4", { let mut c = base.clone(); c.layers = 4; c }),
        ];
        let waves = (base.present_waves + base.delay + base.read_waves) as u32;
        eprintln!(
            "{:<12}{:>6}{:>8}{:>7}{:>7}{:>7}{:>6}{:>6}{:>6}",
            "cfg", "ceil", "fisher", "edim", "k-g", "sigma", "dead", "V1", "V2b"
        );
        for (name, cfg) in &cases {
            let ceil = separation_ceiling(cfg, 200);
            let (s, y) = collect_states(cfg, 200);
            let fish = fisher_ratio(&s, &y, cfg.k);
            let edim = effective_dim(&as_f64(&s));
            let kg = kernel_minus_gen_rank(cfg);
            let sig = perturbation_spread(cfg);
            let (dead, _sat, _sync) = degeneracy(&s, waves);
            let v1 = learned(cfg.clone(), false, false, 0.3);
            let v2b = learned(cfg.clone(), true, true, 0.5);
            eprintln!(
                "{name:<12}{ceil:>6}{fish:>8.2}{edim:>7.1}{kg:>7.1}{sig:>7.2}{dead:>6.2}{v1:>6}{v2b:>6}"
            );
        }
    }

    #[test]
    #[ignore]
    fn _topology_interaction_grid() {
        let base = small();
        eprintln!("separation ceiling — rows up_count, cols adapt_bump");
        eprint!("{:>10}", "cnt\\bump");
        for b in [5i16, 10, 20, 40] {
            eprint!("{b:>6}");
        }
        eprintln!();
        for cnt in [8u32, 12, 16, 24] {
            eprint!("{cnt:>10}");
            for b in [5i16, 10, 20, 40] {
                let mut c = base.clone();
                c.up_count = cnt;
                c.adapt_bump = b;
                eprint!("{:>6}", separation_ceiling(&c, 160));
            }
            eprintln!();
        }
    }

}
