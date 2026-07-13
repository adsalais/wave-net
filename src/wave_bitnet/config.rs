//! Construction input for the engine: a shared square `size`, a seed, and one `LayerConfig` per layer.
//! Thresholds are computed per neuron in `Layer::new`. `validate` enforces `count <= (2r+1)²` because
//! the engine stores topology as a per-cell occupancy bitset (fan-in is capped at the neighborhood size).

use crate::wave_bitnet::synapse::TopologyLevel;

pub const THRESHOLD_JITTER_DEFAULT: u16 = 128;

#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub topology: Vec<TopologyLevel>,
    pub leak: (u8, u8),        // right-shift amounts a, b in `p -= (p>>a) + (p>>b)`
    pub cooldown_base: u8,     // refractory reload on fire
    pub inhibitor_ratio: u32,  // Q16: inhibitory iff (hash & 0xFFFF) < inhibitor_ratio
    pub threshold_jitter: u16, // baseline = baseline_init + rand(0..threshold_jitter)
    pub baseline_init: i16,    // construction center for the baseline threshold (low, not i16::MAX)
    pub adapt_bump: i16,       // added to `adapt` on each fire (β); 0 = plain LIF dynamics
    pub adapt_decay: u8,       // right-shift decay of `adapt` per wave: adapt -= adapt >> adapt_decay (>= 1)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub size: u32, // square side; power of two
    pub layers: Vec<LayerConfig>,
}

impl Config {
    pub fn layer_size(&self) -> usize {
        (self.size as usize) * (self.size as usize)
    }

    pub fn n_total(&self) -> usize {
        self.layer_size() * self.layers.len()
    }

    /// A small, valid, deterministic network for tests and bring-up.
    pub fn demo() -> Config {
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 2, radius: 1, count: 2 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
                TopologyLevel { level: -1, radius: 0, count: 1 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 9830, // ~0.15 * 65536
            threshold_jitter: THRESHOLD_JITTER_DEFAULT,
            baseline_init: 12,
            adapt_bump: 5,
            adapt_decay: 5,
        };
        Config { seed: 0x1234_5678_9ABC_DEF0, size: 16, layers: vec![layer; 6] }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.size < 1 || !self.size.is_power_of_two() {
            return Err(format!("size must be a power of two >= 1, got {}", self.size));
        }
        if self.layers.is_empty() {
            return Err("layers must not be empty".into());
        }
        for (z, lc) in self.layers.iter().enumerate() {
            if lc.leak.0 == 0 || lc.leak.1 == 0 {
                return Err(format!("layer {z}: leak shifts must be >= 1"));
            }
            if lc.cooldown_base == 0 {
                return Err(format!("layer {z}: cooldown_base must be >= 1"));
            }
            if lc.adapt_decay == 0 || lc.adapt_decay as u32 > crate::wave_bitnet::neurons::ADAPT_SHIFT {
                return Err(format!(
                    "layer {z}: adapt_decay must be in 1..={} (ADAPT_SHIFT; larger reintroduces the fixed-point dead zone)",
                    crate::wave_bitnet::neurons::ADAPT_SHIFT
                ));
            }
            for t in &lc.topology {
                let n = crate::wave_bitnet::synapse::neigh_size(t.radius);
                if t.count as usize > n {
                    return Err(format!(
                        "layer {z}: topology count {} exceeds neighborhood size {} for radius {} \
                         (a per-cell occupancy bitset caps fan-in at (2r+1)^2)",
                        t.count, n, t.radius
                    ));
                }
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
        assert!(Config::demo().validate().is_ok());
    }

    #[test]
    fn rejects_non_power_of_two_size() {
        let mut c = Config::demo();
        c.size = 12;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_empty_layers() {
        let mut c = Config::demo();
        c.layers.clear();
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_adapt_decay() {
        let mut c = Config::demo();
        c.layers[0].adapt_decay = 0;
        assert!(c.validate().is_err());
    }

    fn one_level(radius: u32, count: u32) -> Config {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius, count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump: 5,
            adapt_decay: 6,
        };
        let top = LayerConfig { topology: vec![], ..lc.clone() };
        Config { seed: 1, size: 8, layers: vec![lc, top] }
    }

    #[test]
    fn validate_accepts_fan_in_within_neighborhood() {
        // r2 -> N=25; count 16 <= 25 ok
        assert!(one_level(2, 16).validate().is_ok());
    }

    #[test]
    fn validate_rejects_fan_in_over_neighborhood() {
        // r2 -> N=25; count 30 > 25 -> Err
        let e = one_level(2, 30).validate().unwrap_err();
        assert!(e.contains("count") && e.contains("neighborhood"), "descriptive error: {e}");
    }
}
