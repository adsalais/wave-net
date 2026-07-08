//! `readout` — per-neuron spike-count features over a multi-wave window, and an integer
//! nearest-centroid classifier. No floats: centroids are integer means, distances are i64.

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
