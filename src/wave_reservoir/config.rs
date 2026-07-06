#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefractoryMode {
    CarryOver,
    Drop,
}

#[derive(Clone, Copy, Debug)]
pub struct IntLevel {
    pub level: i32,
    pub radius: u32,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct IntLayer {
    pub p_inh_q16: u32,
    pub topology: Vec<IntLevel>,
    pub leak_a: u8,
    pub leak_b: u8,
    pub threshold_base: i32,
    pub spread_log2: u8,
    pub refractory: u8,
}

#[derive(Clone, Debug)]
pub struct IntConfig {
    pub seed: u64,
    pub w: u32,
    pub h: u32,
    pub l: u32,
    pub saturation: i16,
    pub waves: usize,
    pub refractory_mode: RefractoryMode,
    pub layers: Vec<IntLayer>,
}

/// Heterogeneity `spread_log2` for a threshold base: spread ≈ base/8 (±12.5%),
/// as a power of two so the threshold jitter uses a mask, not a divide.
pub fn spread_log2_for(threshold_base: i32) -> u8 {
    (31 - (threshold_base.max(1) as u32).leading_zeros()).saturating_sub(3) as u8
}

/// Neuron state (`potential`, `threshold`, `delivery`) is stored as `i16` to halve the
/// per-neuron memory that dominates a large reservoir. `saturation` is capped here so the
/// clamped potential — plus the pre-clamp transient (drive + fan-in deliveries folded in
/// before the end-of-sweep clamp) — stays inside `i16`, leaving ~4× headroom under
/// `i16::MAX`. The bound is a property of local fan-in, not `N`, so it never scales up.
pub const MAX_SATURATION: i16 = 1 << 13;

impl IntConfig {
    pub fn n_total(&self) -> usize {
        (self.w as usize) * (self.h as usize) * (self.l as usize)
    }

    pub fn demo() -> IntConfig {
        let l = 6;
        // Binary weights {-1,+1}; threshold init is a plain constant (~4 coincident
        // spikes to fire), since there is no magnitude to scale it to. Calibration adjusts.
        let tb = 4;
        let layer = IntLayer {
            p_inh_q16: 9830, // round(0.15 * 65536)
            topology: vec![
                IntLevel { level: 1, radius: 2, count: 6 },
                IntLevel { level: 2, radius: 1, count: 2 },
                IntLevel { level: 0, radius: 1, count: 2 },
                IntLevel { level: -1, radius: 0, count: 1 },
            ],
            leak_a: 3,
            leak_b: 5,
            threshold_base: tb, // initial; calibration adjusts
            spread_log2: spread_log2_for(tb),
            refractory: 2,
        };
        IntConfig {
            seed: 0x1234_5678_9ABC_DEF0, // identical to wave_reservoir::demo for shared wiring
            w: 16,
            h: 16,
            l,
            saturation: (tb << 6) as i16,
            waves: 4,
            refractory_mode: RefractoryMode::CarryOver,
            layers: vec![layer; l as usize],
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.w.is_power_of_two() {
            return Err(format!("w must be a power of two, got {}", self.w));
        }
        if !self.h.is_power_of_two() {
            return Err(format!("h must be a power of two, got {}", self.h));
        }
        if self.l == 0 {
            return Err("l must be >= 1".into());
        }
        if self.layers.len() != self.l as usize {
            return Err(format!("layers.len()={} must equal l={}", self.layers.len(), self.l));
        }
        if self.waves == 0 {
            return Err("waves must be >= 1".into());
        }
        if self.saturation < 1 || self.saturation > MAX_SATURATION {
            return Err(format!(
                "saturation must be in 1..={MAX_SATURATION} (state is stored as i16), got {}",
                self.saturation
            ));
        }
        for (z, lc) in self.layers.iter().enumerate() {
            if lc.refractory == 0 {
                return Err(format!("layer {z}: refractory must be >= 1"));
            }
            if lc.leak_a == 0 || lc.leak_b == 0 {
                return Err(format!("layer {z}: leak shifts must be >= 1"));
            }
            if lc.threshold_base < 1 {
                return Err(format!("layer {z}: threshold_base must be >= 1"));
            }
            // The stored threshold is `(threshold_base + jitter) as i16`; keep its peak
            // (`base + 2^spread_log2`) inside i16 so it never truncates.
            let off = 1i64 << lc.spread_log2.min(30);
            if lc.threshold_base as i64 + off > i16::MAX as i64 {
                return Err(format!(
                    "layer {z}: threshold_base {} + spread must fit i16 (<= {})",
                    lc.threshold_base,
                    i16::MAX
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_is_valid() {
        let cfg = IntConfig::demo();
        assert!(cfg.validate().is_ok(), "{:?}", cfg.validate());
        assert_eq!(cfg.n_total(), (cfg.w * cfg.h * cfg.l) as usize);
    }

    #[test]
    fn validate_rejects_zero_refractory() {
        let mut cfg = IntConfig::demo();
        cfg.layers[0].refractory = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_nonpositive_saturation() {
        let mut cfg = IntConfig::demo();
        cfg.saturation = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_oversized_saturation() {
        let mut cfg = IntConfig::demo();
        cfg.saturation = MAX_SATURATION + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_threshold_exceeding_i16() {
        let mut cfg = IntConfig::demo();
        cfg.layers[0].threshold_base = 40_000;
        assert!(cfg.validate().is_err());
    }
}
