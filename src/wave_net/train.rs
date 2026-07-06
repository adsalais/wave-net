//! Gradient-free per-neuron **field** training by node perturbation (stochastic hill-climbing):
//! perturb the field, keep the change if the reward improves. The reward is arbitrary — in
//! practice the online top-layer readout accuracy (see `examples/field_training.rs`), a mock in
//! tests. Works directly on the non-differentiable integer engine: no gradients needed.

use crate::wave_reservoir::hash::mix;

/// Add a per-neuron field into a (zeroed) drive buffer, saturating: `buf[i] += field[i]`.
/// This is how the field "rides on the drive" without touching the engine.
pub fn add_field(buf: &mut [i16], field: &[i16]) {
    for (b, &f) in buf.iter_mut().zip(field) {
        *b = b.saturating_add(f);
    }
}

/// Node-perturbation knobs.
#[derive(Clone, Copy, Debug)]
pub struct PerturbParams {
    pub iters: usize,
    /// Percent of parameters kicked each iteration.
    pub density_pct: u64,
    /// Kick magnitude (added or subtracted).
    pub step: i16,
    /// Bound on `|param|`.
    pub clamp: i16,
    pub seed: u64,
}

/// Result of a hill-climb: the best parameter vector, its reward, and the best-so-far reward
/// after each iteration (length `iters + 1`).
pub struct Outcome {
    pub params: Vec<i16>,
    pub reward: f64,
    pub history: Vec<f64>,
}

/// Deterministically perturb `field` into `out`: each parameter is independently kicked by
/// `±step` with probability `density_pct%`, then clamped.
fn perturb(field: &[i16], cfg: &PerturbParams, iter: usize, out: &mut Vec<i16>) {
    out.clear();
    out.extend_from_slice(field);
    for (i, v) in out.iter_mut().enumerate() {
        let h = mix(cfg.seed
            ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (iter as u64).wrapping_mul(0xD1B5_4A32_9E37_79B9));
        if h % 100 < cfg.density_pct {
            let delta = if (h >> 63) & 1 == 0 { cfg.step } else { -cfg.step };
            *v = (*v + delta).clamp(-cfg.clamp, cfg.clamp);
        }
    }
}

/// Stochastic hill-climb: each iteration perturb and keep the trial iff its reward improves.
pub fn hill_climb(init: Vec<i16>, cfg: &PerturbParams, reward: impl Fn(&[i16]) -> f64) -> Outcome {
    let mut field = init;
    let mut best = reward(&field);
    let mut history = Vec::with_capacity(cfg.iters + 1);
    history.push(best);
    let mut trial = Vec::with_capacity(field.len());
    for it in 0..cfg.iters {
        perturb(&field, cfg, it, &mut trial);
        let r = reward(&trial);
        if r > best {
            best = r;
            field.clone_from(&trial);
        }
        history.push(best);
    }
    Outcome { params: field, reward: best, history }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_field_accumulates_into_buffer() {
        let mut buf = vec![10i16, -5, 0];
        add_field(&mut buf, &[1, 2, 3]);
        assert_eq!(buf, vec![11, -3, 3]);
    }

    #[test]
    fn hill_climb_improves_on_a_quadratic() {
        // Reward = -||f - target||². Keep-if-better must drive the distance down and never let
        // the running best regress.
        let target: Vec<i16> = (0..24).map(|i| ((i as i16 % 7) - 3) * 5).collect();
        let dist = |f: &[i16]| -> f64 {
            f.iter().zip(&target).map(|(&a, &b)| { let d = (a - b) as f64; d * d }).sum()
        };
        let init = vec![0i16; target.len()];
        let d0 = dist(&init);
        let cfg = PerturbParams { iters: 2000, density_pct: 40, step: 1, clamp: 60, seed: 0x1234 };
        let out = hill_climb(init, &cfg, |f| -dist(f));
        let df = dist(&out.params);
        assert!(out.history.windows(2).all(|w| w[1] >= w[0]), "best reward must be non-decreasing");
        assert!(df < 0.25 * d0, "hill-climb should cut distance to target: {d0} -> {df}");
    }
}
