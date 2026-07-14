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
