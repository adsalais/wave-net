//! `training` — online, activity-scaled multi-layer-DFA training for `wave_driven`: membrane e-prop
//! eligibility with spike-ψ, accrued on the frontier during `wave()`. Types here; the accrual and
//! shadow-update live on `Network` (they need the layer stack + per-wave fired sets). The offline
//! `dense_eligibility` oracle (this file) is the bit-exact reference for the online accrual.

/// Eligibility knobs (membrane-only, spike-ψ). `rec_tau` sets the presynaptic-trace decay
/// (`decay = 1 − 1/rec_tau`); `epsilon` is the hard trace cutoff (activity-scaling + exact oracle).
#[derive(Clone, Copy, Debug)]
pub struct EligParams {
    pub rec_tau: f32,
    pub epsilon: f32,
}

impl Default for EligParams {
    fn default() -> Self {
        EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 }
    }
}

/// One topology edge of a source layer, in built `LayerConfig` topology order (so `entries[z][e]`
/// lines up with the layer's `e`-th level). Mirrors the DFA credit wiring.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub level: i32,
    pub count: usize,
    pub radius: u32,
}
