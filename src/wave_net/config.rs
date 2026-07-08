//! Construction input for the engine: a shared square `size`, a seed, and one
//! `LayerConfig` per layer. Thresholds are computed per neuron in `Layer::new`.

use crate::wave_net::synapse::TopologyLevel;

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
            adapt_bump: 16,
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
            if lc.adapt_decay == 0 {
                return Err(format!("layer {z}: adapt_decay must be >= 1"));
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

    #[test]
    fn rejects_zero_leak_and_cooldown() {
        let mut c = Config::demo();
        c.layers[0].leak = (0, 5);
        assert!(c.validate().is_err());
        let mut c = Config::demo();
        c.layers[0].cooldown_base = 0;
        assert!(c.validate().is_err());
    }
}
