//! `neurons` — a `Layer`'s per-neuron state, its delivery inbox/outbox pair, and its
//! per-layer parameters. The `threshold` field is the ALIF **baseline**: it inits low
//! (`baseline_init + jitter`, clamped to [1, i16::MAX]) and is tuned by calibration. Each
//! neuron also carries `adapt`, a slow variable bumped on fire and decayed each wave; the
//! effective firing threshold is `threshold + (adapt >> ADAPT_SHIFT)`.
//!
//! `adapt` is stored in **fixed point** (i32, scaled by `2^ADAPT_SHIFT`) so its geometric decay
//! `adapt -= adapt >> adapt_decay` stays exponential with time constant ≈ `2^adapt_decay` waves
//! *independent of magnitude* — the fixed-point scale pushes the integer right-shift dead zone below
//! `1 / 2^ADAPT_SHIFT` of a threshold unit, so adaptation always relaxes (no ratchet / lock-out).
//! This holds only while `adapt_decay <= ADAPT_SHIFT` (beyond that the dead zone returns at the real
//! scale); `Config::validate` enforces it.

use crate::wave_net::config::LayerConfig;
use crate::wave_net::synapse::{key, map_range, mix, Synapse, TopologyLevel, P_TARGET, P_THRESHOLD};

/// Fixed-point scale for `adapt`: it holds the effective threshold contribution × `2^ADAPT_SHIFT`.
/// Bounded by the i32 overflow limit on the bump-add (`2·ADAPT_MAX = i16::MAX << (SHIFT+1)` must fit
/// i32, i.e. `SHIFT <= 14`); 12 keeps ~8× margin and allows `adapt_decay` up to 12 (τ ≈ 4096 waves).
pub const ADAPT_SHIFT: u32 = 12;
/// Ceiling for `adapt`, so the effective contribution never exceeds `i16::MAX` (overflow guard).
pub const ADAPT_MAX: i32 = (i16::MAX as i32) << ADAPT_SHIFT;

/// Weight quantizer for the shadow→weight requantize step. `Int8`: per-weight round/clamp to [-127,127]
/// (the default). `Ternary`: BitNet-style {−1,0,+1} with a **per-row** (per source neuron) absmean γ that
/// sets which weights prune to 0; delivered magnitude stays ±1 (pure ternary, no delivery scale).
/// `TernaryScaled`: same ternary sign/zero, but delivery is **±g** with `g = round(γ)` a per-row integer
/// gain (≥1) — BitNet b1.58-style per-row scale, restoring weight magnitude / loop-gain control.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WeightQuant {
    Int8,
    Ternary,
    TernaryScaled,
}

pub struct Layer {
    // wave-mutable hot state
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub adapt: Vec<i32>,      // ALIF adaptation (Q12 fixed point): rest 0, >= 0; bumped on fire, decayed each wave
    pub inbox: Vec<Synapse>,  // drained THIS wave (filled last wave). The next-wave deliveries accumulate
                              // in the Network's scratch buffer and are swapped in here at wave end.

    // tunable params (calibration/training will rewrite these between phases)
    pub threshold: Vec<i16>, // ALIF baseline; effective threshold is threshold + (adapt >> ADAPT_SHIFT)

    // fixed structure
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<TopologyLevel>,
    pub adapt_bump: i16,   // added to adapt on each fire (0 = plain LIF)
    pub adapt_decay: u8,   // right-shift decay of adapt per wave
    pub readout: bool,     // non-spiking drain-only output layer: integrates input, never fires
    pub total_slots: usize,   // Σ topology counts — the stride for out_weights[local·total_slots + slot]
    pub out_weights: Vec<i8>, // stored plastic weight per (source local, slot); addresses stay procedural
    pub out_shadow: Vec<f32>, // higher-precision training accumulator, quantised into out_weights
    pub weight_quant: WeightQuant, // shadow→weight quantizer (default Int8)
    pub ternary_threshold: f32,    // ternary prune threshold t: |shadow|/γ < t → 0 (default 0.5 = round)
    pub elig_pre: Vec<i32>,   // e-prop presynaptic trace: this neuron's spike count this trial
    pub elig_post: Vec<i32>,  // e-prop postsynaptic pseudo-derivative accumulated this trial
    pub decide_potential: Vec<i16>, // potential at the decide step (pre fire-reset/leak); per-wave snapshot
    pub decide_eff: Vec<i32>, // effective threshold `baseline + (adapt >> ADAPT_SHIFT)` at the decide step
                              // (captured BEFORE the fire-bump mutates adapt); per-wave snapshot, pairs with decide_potential
}

impl Layer {
    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;
        let mut threshold = vec![0i16; ls];
        for (local, th) in threshold.iter_mut().enumerate() {
            let global = (base + local) as u32;
            let h = mix(key(seed, global, 0, 0, P_THRESHOLD));
            let jitter = map_range(h as u32, cfg.threshold_jitter as u32) as i32; // [0, jitter)
            *th = (cfg.baseline_init as i32 + jitter).clamp(1, i16::MAX as i32) as i16;
        }
        // Stored weights: init to the old procedural sign (magnitude 1) so the net is behaviour-identical;
        // training later moves them into the full int8 range. Addresses stay hash-generated in generate_into.
        let total_slots: usize = cfg.topology.iter().map(|e| e.count as usize).sum();
        let mut out_weights = vec![0i8; ls * total_slots];
        for local in 0..ls {
            let source_global = (base + local) as u32;
            let mut slot = 0usize;
            for entry in &cfg.topology {
                for k in 0..entry.count {
                    let h = mix(key(seed, source_global, entry.level, k, P_TARGET));
                    out_weights[local * total_slots + slot] =
                        if ((h & 0xFFFF) as u32) < cfg.inhibitor_ratio { -1 } else { 1 };
                    slot += 1;
                }
            }
        }
        let out_shadow: Vec<f32> = out_weights.iter().map(|&w| w as f32).collect();
        Layer {
            potential: vec![0; ls],
            cooldown: vec![0; ls],
            adapt: vec![0; ls],
            inbox: Vec::new(),
            threshold,
            leak: cfg.leak,
            cooldown_base: cfg.cooldown_base,
            topology: cfg.topology.clone(),
            adapt_bump: cfg.adapt_bump,
            adapt_decay: cfg.adapt_decay,
            readout: false,
            total_slots,
            out_weights,
            out_shadow,
            weight_quant: WeightQuant::Int8,
            ternary_threshold: 0.5,
            elig_pre: vec![0; ls],
            elig_post: vec![0; ls],
            decide_potential: vec![0; ls],
            decide_eff: vec![0; ls],
        }
    }

    pub fn max_threshold(&self) -> i16 {
        self.threshold.iter().copied().max().unwrap_or(0)
    }

    /// Subtract `delta` from every threshold (delta>0 lowers), clamped to [1, i16::MAX].
    /// Uniform shift, so per-neuron jitter is preserved.
    pub fn shift_threshold(&mut self, delta: i32) {
        for t in self.threshold.iter_mut() {
            *t = ((*t as i32) - delta).clamp(1, i16::MAX as i32) as i16;
        }
    }

    /// Requantise source neuron `i`'s row (`out_{weights,shadow}[i*total_slots .. +total_slots]`) from the
    /// shadow, per `weight_quant`. Int8: per-weight round/clamp. Ternary: per-row absmean γ sets zeros
    /// (|shadow| < 0.5γ → 0), delivery ±1. No-op for a no-outgoing layer (`total_slots == 0`).
    pub fn requantize_row(&mut self, i: usize) {
        let ts = self.total_slots;
        if ts == 0 {
            return;
        }
        let base = i * ts;
        match self.weight_quant {
            WeightQuant::Int8 => {
                for s in 0..ts {
                    self.out_weights[base + s] = self.out_shadow[base + s].round().clamp(-127.0, 127.0) as i8;
                }
            }
            WeightQuant::Ternary => {
                let mut sum = 0.0f32;
                for s in 0..ts {
                    sum += self.out_shadow[base + s].abs();
                }
                let gamma = sum / ts as f32;
                let t = self.ternary_threshold; // |shadow|/γ < t → 0 (t=0.5 == round-to-nearest)
                for s in 0..ts {
                    let x = if gamma <= 0.0 { 0.0 } else { self.out_shadow[base + s] / gamma };
                    self.out_weights[base + s] = if x.abs() < t { 0 } else if x > 0.0 { 1 } else { -1 };
                }
            }
            WeightQuant::TernaryScaled => {
                let mut sum = 0.0f32;
                for s in 0..ts {
                    sum += self.out_shadow[base + s].abs();
                }
                let gamma = sum / ts as f32;
                // per-row integer gain g = round(γ), floored at 1 so a row never dies; delivery ±g or 0
                let g = if gamma <= 0.0 { 1 } else { (gamma.round() as i32).clamp(1, 127) };
                let t = self.ternary_threshold;
                for s in 0..ts {
                    let x = if gamma <= 0.0 { 0.0 } else { self.out_shadow[base + s] / gamma };
                    let sign = if x.abs() < t { 0 } else if x > 0.0 { 1 } else { -1 };
                    self.out_weights[base + s] = (g * sign).clamp(-127, 127) as i8;
                }
            }
        }
    }

    /// One measure-informed tuning step toward `target` (fractions in 0..1). Returns whether it
    /// adjusted. Geometric step `max_threshold >> step_shift`; lower when too cold, raise when hot,
    /// no-op inside the tolerance band.
    pub fn calibrate_step(&mut self, rate: f64, target: f64, tol: f64, step_shift: u32) -> bool {
        if (rate - target).abs() <= tol {
            return false;
        }
        let step = ((self.max_threshold() as i32) >> step_shift).max(1);
        let delta = if rate < target { step } else { -step };
        self.shift_threshold(delta);
        true
    }

    pub fn thresholds(&self) -> &[i16] {
        &self.threshold
    }

    pub fn set_thresholds(&mut self, t: Vec<i16>) {
        assert_eq!(t.len(), self.threshold.len(), "threshold length mismatch");
        self.threshold = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::LayerConfig;
    use crate::wave_net::synapse::TopologyLevel;

    fn lc(jitter: u16) -> LayerConfig {
        LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: jitter,
            baseline_init: i16::MAX,
            adapt_bump: 0,
            adapt_decay: 5,
        }
    }

    fn lc_baseline(jitter: u16, baseline: i16) -> LayerConfig {
        LayerConfig { baseline_init: baseline, ..lc(jitter) }
    }

    #[test]
    fn thresholds_near_baseline_within_jitter() {
        let l = Layer::new(&lc_baseline(128, 12), 1, 0, 8);
        for &t in &l.threshold {
            assert!((12..12 + 128).contains(&t), "threshold {t} out of [12, 140) band");
        }
    }

    #[test]
    fn new_zeroes_adaptation() {
        let l = Layer::new(&lc_baseline(128, 12), 1, 0, 8);
        assert_eq!(l.adapt.len(), 64);
        assert!(l.adapt.iter().all(|&a| a == 0));
    }

    #[test]
    fn new_sizes_and_zeroes() {
        let l = Layer::new(&lc(128), 1, 0, 8);
        assert_eq!(l.potential.len(), 64);
        assert_eq!(l.cooldown.len(), 64);
        assert_eq!(l.threshold.len(), 64);
        assert!(l.potential.iter().all(|&p| p == 0));
        assert!(l.inbox.is_empty());
    }

    #[test]
    fn thresholds_near_i16_max_within_jitter() {
        let l = Layer::new(&lc(128), 1, 0, 8);
        for &t in &l.threshold {
            assert!(t > i16::MAX - 128 && t <= i16::MAX, "threshold {t} out of band");
        }
        assert_eq!(l.max_threshold(), *l.threshold.iter().max().unwrap());
    }

    #[test]
    fn thresholds_deterministic() {
        let a = Layer::new(&lc(128), 7, 2, 8);
        let b = Layer::new(&lc(128), 7, 2, 8);
        assert_eq!(a.threshold, b.threshold);
    }

    #[test]
    fn shift_threshold_clamps_and_preserves_jitter() {
        let mut l = Layer::new(&lc(128), 1, 0, 8);
        let before = l.thresholds().to_vec();
        l.shift_threshold(1000);
        for (a, b) in before.iter().zip(l.thresholds()) {
            assert_eq!(*b as i32, (*a as i32 - 1000).max(1));
        }
        l.shift_threshold(i16::MAX as i32); // drive well past the floor
        assert!(l.thresholds().iter().all(|&t| t == 1));
        l.shift_threshold(-(i16::MAX as i32)); // raise past the cap
        assert!(l.thresholds().iter().all(|&t| t == i16::MAX));
    }

    #[test]
    fn calibrate_step_lowers_cold_raises_hot_holds_in_band() {
        let mut l = Layer::new(&lc(0), 1, 0, 8); // jitter 0 -> all i16::MAX
        let m0 = l.max_threshold();
        assert!(l.calibrate_step(0.0, 0.1, 0.02, 2)); // cold -> lower
        assert!(l.max_threshold() < m0);
        let m1 = l.max_threshold();
        assert!(!l.calibrate_step(0.1, 0.1, 0.02, 2)); // in band -> no change
        assert_eq!(l.max_threshold(), m1);
        assert!(l.calibrate_step(0.5, 0.1, 0.02, 2)); // hot -> raise
        assert!(l.max_threshold() > m1);
    }

    #[test]
    fn thresholds_round_trip() {
        let mut l = Layer::new(&lc(128), 1, 0, 8);
        let snap = l.thresholds().to_vec();
        l.shift_threshold(500);
        l.set_thresholds(snap.clone());
        assert_eq!(l.thresholds(), snap.as_slice());
    }

    #[test]
    fn requantize_row_int8_and_ternary() {
        let cfg = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 0, count: 4 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 6,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let mut layer = Layer::new(&cfg, 1, 0, 2); // size 2 → ls 4, total_slots 4
        assert_eq!(layer.total_slots, 4);
        assert_eq!(layer.weight_quant, WeightQuant::Int8);
        // Int8: per-weight round/clamp
        layer.out_shadow[0..4].copy_from_slice(&[3.7, -50.0, 0.4, 200.0]);
        layer.requantize_row(0);
        assert_eq!(&layer.out_weights[0..4], &[4i8, -50, 0, 127]);
        // Ternary: γ = mean|shadow| = (2+2+0.1+0.1)/4 = 1.05; 2/1.05→1, 0.1/1.05→0
        layer.weight_quant = WeightQuant::Ternary;
        layer.out_shadow[0..4].copy_from_slice(&[2.0, 2.0, 0.1, 0.1]);
        layer.requantize_row(0);
        assert_eq!(&layer.out_weights[0..4], &[1i8, 1, 0, 0]);
        // TernaryScaled: γ = (10+10+0.1+0.1)/4 = 5.05 → g = round = 5; signs [1,1,0,0] → [5,5,0,0]
        layer.weight_quant = WeightQuant::TernaryScaled;
        layer.out_shadow[0..4].copy_from_slice(&[10.0, 10.0, 0.1, 0.1]);
        layer.requantize_row(0);
        assert_eq!(&layer.out_weights[0..4], &[5i8, 5, 0, 0]);
        // Prune threshold controls zeros: [1,1,0.7,0.7] γ=0.85; x=[1.18,1.18,0.82,0.82].
        layer.weight_quant = WeightQuant::Ternary;
        layer.out_shadow[0..4].copy_from_slice(&[1.0, 1.0, 0.7, 0.7]);
        layer.ternary_threshold = 0.5; // round: nothing prunes
        layer.requantize_row(0);
        assert_eq!(&layer.out_weights[0..4], &[1i8, 1, 1, 1]);
        layer.ternary_threshold = 0.9; // the 0.82 pair now < 0.9 → 0
        layer.requantize_row(0);
        assert_eq!(&layer.out_weights[0..4], &[1i8, 1, 0, 0]);
    }
}
