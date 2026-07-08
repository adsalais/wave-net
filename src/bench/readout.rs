//! `readout` — per-neuron spike-count features over a multi-wave window, and an integer
//! nearest-centroid classifier. No floats: centroids are integer means, distances are i64.

use std::sync::{Arc, Mutex};

use crate::bench::linalg::{xt_x, xt_y, Lu};
use crate::wave_net::network::Network;

/// Ridge-regression linear readout. Factors `(XᵀX + λI)` once from the training design matrix
/// (which must already include a bias column); each target column is solved by back-substitution.
pub struct RidgeReadout {
    lu: Lu,
}

impl RidgeReadout {
    pub fn fit(x_train: &[Vec<f64>], lambda: f64) -> RidgeReadout {
        let mut a = xt_x(x_train);
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += lambda;
        }
        RidgeReadout { lu: Lu::factor(a) }
    }

    /// Weight vector reconstructing one target column `y_train` from `x_train`.
    pub fn weights(&self, x_train: &[Vec<f64>], y_train: &[f64]) -> Vec<f64> {
        self.lu.solve(&xt_y(x_train, y_train))
    }

    /// Prediction `X · w`.
    pub fn predict(x: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
        x.iter().map(|row| row.iter().zip(w).map(|(a, b)| a * b).sum()).collect()
    }
}

/// Run `waves` waves feeding `input(w)` each wave, returning per-neuron spike counts over the
/// computational layers `1..L` concatenated (layer 0, the transducer, excluded). Installs counting
/// listeners, runs, then clears listeners. Does not reset state — the caller sets up the run.
pub fn record_response(net: &mut Network, waves: usize, input: impl Fn(usize) -> Vec<u32>) -> Vec<u32> {
    let l = net.layer_count();
    let ls = (net.size() * net.size()) as usize;
    let counts = Arc::new(Mutex::new(vec![0u32; l.saturating_sub(1) * ls]));
    for z in 1..l {
        let c = counts.clone();
        net.on_layer(
            z,
            Box::new(move |_w: usize, fired: &[u32]| {
                let mut g = c.lock().unwrap();
                let base = (z - 1) * ls;
                for &local in fired {
                    g[base + local as usize] += 1;
                }
            }),
        );
    }
    for w in 0..waves {
        net.wave(&input(w));
    }
    net.clear_listeners();
    std::mem::take(&mut *counts.lock().unwrap())
}

/// Integer nearest-centroid classifier over fixed-length `u32` feature vectors.
pub struct NearestCentroid {
    centroids: Vec<Vec<i64>>, // one centroid (integer mean) per class
}

impl NearestCentroid {
    /// Fit `k` class centroids (integer means) from labelled feature vectors (labels in `0..k`).
    pub fn fit(features: &[Vec<u32>], labels: &[usize], k: usize) -> NearestCentroid {
        let dim = features.first().map(|f| f.len()).unwrap_or(0);
        let mut sums = vec![vec![0i64; dim]; k];
        let mut counts = vec![0i64; k];
        for (f, &lab) in features.iter().zip(labels) {
            counts[lab] += 1;
            for (acc, &v) in sums[lab].iter_mut().zip(f) {
                *acc += v as i64;
            }
        }
        let centroids = sums
            .iter()
            .zip(&counts)
            .map(|(sum, &c)| {
                let denom = c.max(1);
                sum.iter().map(|&s| s / denom).collect()
            })
            .collect();
        NearestCentroid { centroids }
    }

    /// Index of the class whose centroid is nearest in squared L2 distance (i64).
    pub fn predict(&self, feature: &[u32]) -> usize {
        let mut best = 0usize;
        let mut best_dist = i64::MAX;
        for (c, centroid) in self.centroids.iter().enumerate() {
            let mut dist = 0i64;
            for (&f, &m) in feature.iter().zip(centroid) {
                let d = f as i64 - m;
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    fn two_layer_low() -> Config {
        // L0 -> L1 straight up; L1 baseline low so L0 injection makes it fire.
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
            leak: (3, 5),
            cooldown_base: 1,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 2,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let l1 = LayerConfig { topology: vec![], ..l0.clone() };
        Config { seed: 1, size: 4, layers: vec![l0, l1] }
    }

    #[test]
    fn record_response_counts_spikes() {
        let mut net = Network::new(two_layer_low());
        let ls = 16;
        // silent run -> all zero, correct length ((L-1)*ls = 1*16)
        net.reset_state();
        let silent = record_response(&mut net, 4, |_w| Vec::new());
        assert_eq!(silent.len(), ls);
        assert!(silent.iter().all(|&c| c == 0), "silent run must record no spikes");
        // drive all L0 every wave -> L1 should fire, so some counts are non-zero
        net.reset_state();
        let all_l0: Vec<u32> = (0..ls as u32).collect();
        let driven = record_response(&mut net, 6, move |_w| all_l0.clone());
        assert_eq!(driven.len(), ls);
        assert!(driven.iter().any(|&c| c > 0), "driven run must record L1 spikes");
    }

    use crate::wave_net::synapse::mix;

    fn synth_design(n: usize, d: usize) -> Vec<Vec<f64>> {
        // Deterministic rows in [-1,1) with a trailing bias column of 1.0.
        (0..n)
            .map(|i| {
                let mut row: Vec<f64> = (0..d - 1)
                    .map(|j| {
                        let h = mix(((i as u64) << 20) ^ ((j as u64) << 3) ^ 0x9E37_79B9);
                        ((h & 0xFFFF) as f64 / 65536.0) * 2.0 - 1.0
                    })
                    .collect();
                row.push(1.0);
                row
            })
            .collect()
    }

    #[test]
    fn ridge_recovers_planted_linear_map() {
        let (n, d) = (60usize, 4usize); // 3 features + bias
        let x = synth_design(n, d);
        let w_true = [1.5, -2.0, 0.5, 0.25];
        let y: Vec<f64> = x.iter().map(|r| r.iter().zip(&w_true).map(|(a, b)| a * b).sum()).collect();
        let ridge = RidgeReadout::fit(&x, 1e-6);
        let w = ridge.weights(&x, &y);
        for (got, want) in w.iter().zip(&w_true) {
            assert!((got - want).abs() < 1e-2, "weight {got} != {want}");
        }
        let pred = RidgeReadout::predict(&x, &w);
        let max_err = pred.iter().zip(&y).map(|(p, t)| (p - t).abs()).fold(0.0, f64::max);
        assert!(max_err < 1e-2, "prediction error {max_err} too large");
    }

    #[test]
    fn nearest_centroid_separates_clusters() {
        // Class 0 clusters near (10,0), class 1 near (0,10).
        let features = vec![
            vec![10u32, 0], vec![9, 1], vec![11, 0],
            vec![0, 10], vec![1, 9], vec![0, 11],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let clf = NearestCentroid::fit(&features, &labels, 2);
        assert_eq!(clf.predict(&[10, 1]), 0);
        assert_eq!(clf.predict(&[1, 10]), 1);
    }
}
