//! `eprop` — the official e-prop learning rule on the live engine: a generic per-layer weight-update
//! primitive from the stored eligibility, plus (Task 4) a feed-forward training driver. Learning
//! *signals* (task error, DFA, rate, σ) are computed by the caller (`bench` for tasks, `critical_init`
//! for the σ target) and passed in — the engine owns only the update mechanism.

use crate::wave_net::network::Network;
use crate::wave_net::synapse::target_of;

impl Network {
    /// Apply one e-prop weight update to the `entry_idx`-th topology entry of layer `source_z`, using
    /// caller-supplied `pre` (source pre-trace), `psi` (target ψ) and per-target `signal`:
    /// `out_shadow[i, slot] += -lr · signal[j] · pre_i · (psi_j if use_psi else 1)`, then requantise.
    /// `target_of` recovers each synapse's target `j` (no re-scatter). Generic over the topology entry,
    /// so feed-forward (the up entry) and — later — side-car edges both reuse it. No-op if the entry's
    /// target layer is out of range.
    pub fn eprop_update(&mut self, source_z: usize, entry_idx: usize, pre: &[i32], psi: &[i32], signal: &[f32], lr: f32, use_psi: bool) {
        let seed = self.seed_val();
        let size = self.size();
        let l = self.layer_count();
        let ls = (size as usize) * (size as usize);
        let (level, radius, count, slot_base, total_slots) = self.with_layer(source_z, |lz| {
            let e = &lz.topology[entry_idx];
            let slot_base: usize = lz.topology[..entry_idx].iter().map(|t| t.count as usize).sum();
            (e.level, e.radius, e.count as usize, slot_base, lz.total_slots)
        });
        let tz = source_z as i32 + level;
        if tz < 0 || tz as usize >= l {
            return;
        }
        let base = source_z * ls;
        self.with_layer_mut(source_z, |lz| {
            for i in 0..ls {
                let pre_i = pre[i] as f32;
                if pre_i == 0.0 {
                    continue;
                }
                let sg = (base + i) as u32;
                for kk in 0..count {
                    let j = target_of(seed, sg, i as u32, level, kk as u32, radius, size) as usize;
                    let pf = if use_psi { psi[j] as f32 } else { 1.0 };
                    lz.out_shadow[i * total_slots + slot_base + kk] += -lr * signal[j] * pre_i * pf;
                }
            }
            for (wq, s) in lz.out_weights.iter_mut().zip(&lz.out_shadow) {
                *wq = s.round().clamp(-127.0, 127.0) as i8;
            }
        });
    }

    /// Windowed per-neuron eligibility for every layer: the pre-trace + ψ accumulated over `waves`
    /// *after* a `warmup` transient, via the difference of the running `elig_pre`/`elig_post`
    /// accumulators (so the boots-hot transient is excluded).
    pub fn windowed_eligibility(&mut self, warmup: usize, waves: usize, input: &impl Fn(usize) -> Vec<u32>) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
        let l = self.layer_count();
        self.reset_state();
        for w in 0..warmup {
            self.wave(&input(w));
        }
        let pre0: Vec<Vec<i32>> = (0..l).map(|z| self.with_layer(z, |x| x.elig_pre.clone())).collect();
        let psi0: Vec<Vec<i32>> = (0..l).map(|z| self.with_layer(z, |x| x.elig_post.clone())).collect();
        for w in 0..waves {
            self.wave(&input(warmup + w));
        }
        let pre = (0..l).map(|z| self.with_layer(z, |x| x.elig_pre.iter().zip(&pre0[z]).map(|(a, b)| a - b).collect())).collect();
        let psi = (0..l).map(|z| self.with_layer(z, |x| x.elig_post.iter().zip(&psi0[z]).map(|(a, b)| a - b).collect())).collect();
        (pre, psi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::TopologyLevel;

    #[test]
    fn eprop_update_applies_expected_delta() {
        // radius-0, count-1 up entry: target of source local `i` is local `i` in the layer above.
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
        let mut net = Network::new(Config { seed: 1, size: 4, layers: vec![lc; 2] });
        let ls = 16;
        let pre = vec![2i32; ls];
        let psi = vec![3i32; ls];
        let signal = vec![0.5f32; ls];
        let before = net.with_layer(0, |l| l.out_shadow[0]);
        net.eprop_update(0, 0, &pre, &psi, &signal, 0.1, true);
        let after = net.with_layer(0, |l| l.out_shadow[0]);
        // Δ = -lr·signal[0]·pre[0]·psi[0] = -0.1·0.5·2·3 = -0.3
        assert!((after - before + 0.3).abs() < 1e-4, "{before} -> {after}");
    }
}
