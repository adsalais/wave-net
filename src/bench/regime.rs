//! Reservoir-regime diagnostic — measures properties of the calibrated-but-untrained reservoir to find
//! which predict learnability (V1 & V2b) and how topology couples to the other knobs. Bench-side, f64.

use crate::bench::eprop::{calibrated_reservoir, pick_class, EpropConfig};
use crate::bench::readout::NearestCentroid;
use crate::bench::store_recall::{cue_realization, probe_pattern};
use crate::wave_net::network::Network;
use std::sync::{Arc, Mutex};

/// One trial (reset → present cue → delay → probe); returns per-computational-layer spike counts. If
/// `flip` is set, that L0 site is toggled in every present wave (the perturbation for σ).
pub fn reservoir_states(
    net: &mut Network,
    cfg: &EpropConfig,
    class: usize,
    trial: usize,
    flip: Option<u32>,
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
        if let Some(s) = flip {
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
        states.push(top_state(&reservoir_states(&mut net, cfg, class, t, None)));
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
}
