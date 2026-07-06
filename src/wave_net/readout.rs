//! Online linear readout trained by Recursive Least Squares (RLS) — the incremental form of the
//! ridge regression used offline. It reads a feature vector (e.g. the top layer's per-bit spike
//! counts plus a bias) and learns to predict a target online, giving a training loop a live
//! performance signal instead of a batch refit. Float math, downstream of the integer engine
//! (like the offline ridge); it does not affect engine determinism.

/// An online least-squares readout. `w` are the weights; `p` is the running inverse correlation
/// matrix `(XᵀX + λI)⁻¹` that RLS maintains so each step matches the batch ridge solution so far.
pub struct OnlineReadout {
    w: Vec<f64>,
    p: Vec<Vec<f64>>,
}

impl OnlineReadout {
    /// `dim` is the feature dimension (include a bias entry in the features for an intercept).
    /// `lambda` is the ridge regularization: `P` is initialised to `(1/lambda) I`.
    pub fn new(dim: usize, lambda: f64) -> OnlineReadout {
        let mut p = vec![vec![0.0f64; dim]; dim];
        let inv = 1.0 / lambda;
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = inv;
        }
        OnlineReadout { w: vec![0.0; dim], p }
    }

    /// Predicted scalar output `w · x`.
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.w.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    /// RLS step toward target `y` given features `x` (no forgetting factor). Standard recursion:
    /// `k = Px / (1 + xᵀPx)`, `w += k·(y - wᵀx)`, `P -= k (Px)ᵀ`, keeping `P` symmetric.
    pub fn update(&mut self, x: &[f64], y: f64) {
        let d = self.w.len();
        // px = P x
        let mut px = vec![0.0f64; d];
        for i in 0..d {
            let mut s = 0.0;
            for j in 0..d {
                s += self.p[i][j] * x[j];
            }
            px[i] = s;
        }
        let denom = 1.0 + x.iter().zip(&px).map(|(a, b)| a * b).sum::<f64>();
        let err = y - self.predict(x);
        // w += (px / denom) * err
        for i in 0..d {
            self.w[i] += px[i] * err / denom;
        }
        // P -= (px pxᵀ) / denom
        for i in 0..d {
            for j in 0..d {
                self.p[i][j] -= px[i] * px[j] / denom;
            }
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_reservoir::hash::mix;

    #[test]
    fn rls_recovers_a_linear_map() {
        // Noiseless y = 2·x0 + 3·x1 + 0.5; RLS must converge to predict it near-exactly.
        let mut ro = OnlineReadout::new(3, 1e-6);
        for t in 0..300u64 {
            let x0 = (mix(t) % 1000) as f64 / 1000.0;
            let x1 = (mix(t ^ 0xABC) % 1000) as f64 / 1000.0;
            let x = [x0, x1, 1.0];
            ro.update(&x, 2.0 * x0 + 3.0 * x1 + 0.5);
        }
        let x = [0.3, 0.7, 1.0];
        let pred = ro.predict(&x);
        assert!((pred - (2.0 * 0.3 + 3.0 * 0.7 + 0.5)).abs() < 1e-3, "RLS did not converge: pred {pred}");
    }

    #[test]
    fn online_top_layer_readout_learns_xor() {
        // Online RLS readout over the TOP LAYER'S per-bit spike counts only. Train online over
        // the train split, evaluate on the held-out split. Must beat the ~0.5 control, i.e. the
        // top layer already supports temporal-XOR to some degree (the untrained baseline that
        // reservoir training is meant to lift).
        use crate::wave_net::calibrate::{calibrate, CalibrateParams};
        use crate::wave_net::stream::{fair_bit, BipolarInput};
        use crate::wave_reservoir::config::IntConfig;
        use crate::wave_reservoir::index::Dims;
        use crate::wave_reservoir::pipeline::LayerNet;
        use std::sync::{Arc, Mutex};

        const WPB: usize = 8;
        const WASHOUT: usize = 30;
        const TRAIN: usize = 400;
        const TEST: usize = 200;
        const TAU: usize = 1;
        let total = WASHOUT + TRAIN + TEST;

        let mut cfg = IntConfig::demo();
        let dims = Dims::new(cfg.w, cfg.h, cfg.l);
        let ls = (cfg.w * cfg.h) as usize;
        let top = cfg.l as usize - 1;
        let input = BipolarInput::scatter_bottom(&dims, 0x0B17_5EED, 24, 4);

        // calibrate on the task's calibration stream (distinct seed from the eval bits)
        {
            let inp = input.clone();
            calibrate(&mut cfg, &CalibrateParams::default(), &move |w, buf| {
                inp.drive_into(buf, fair_bit(0x5EED_C0DE ^ 0xCA1B, (w / WPB) as u64))
            });
        }

        // stream the eval bits, collect the top layer's per-bit spike-count vector
        let bhist: Vec<u8> = (0..total).map(|t| fair_bit(0x5EED_C0DE, t as u64)).collect();
        let feats = Arc::new(Mutex::new(vec![vec![0.0f64; ls]; total]));
        let mut net = LayerNet::new(cfg.clone());
        net.reset_state();
        {
            let f = feats.clone();
            net.on_layer(top, Box::new(move |wave, fired| {
                let bit = wave / WPB;
                let mut ff = f.lock().unwrap();
                for &loc in fired {
                    ff[bit][loc as usize] += 1.0;
                }
            }));
        }
        let bh = bhist.clone();
        net.run_stream(total * WPB, 1, move |w, buf| input.drive_into(buf, bh[w / WPB]));
        let features = std::mem::take(&mut *feats.lock().unwrap());

        // online-train the readout over the train split, evaluate on the test split
        let mut ro = OnlineReadout::new(ls + 1, 1.0);
        let target = |t: usize| (bhist[t] ^ bhist[t - TAU]) as f64;
        for t in WASHOUT..(WASHOUT + TRAIN) {
            let mut x = features[t].clone();
            x.push(1.0);
            ro.update(&x, target(t));
        }
        let mut correct = 0;
        for t in (WASHOUT + TRAIN)..total {
            let mut x = features[t].clone();
            x.push(1.0);
            if ((ro.predict(&x) >= 0.5) as u8) as f64 == target(t) {
                correct += 1;
            }
        }
        let acc = correct as f32 / TEST as f32;
        eprintln!("online top-layer XOR accuracy = {acc:.3} (control ~0.5, offline all-layer ~0.9)");
        // The top layer barely supports XOR linearly (~0.62): it encodes the individual bits but
        // not their interaction. That gap to the all-layer ~0.9 is the headroom reservoir training
        // (Spec 3) is meant to recover. Guard only that the readout clearly beats chance.
        assert!(acc > 0.58, "online top-layer readout should beat control on XOR: {acc:.3}");
    }
}
