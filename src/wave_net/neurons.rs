//! `neurons` — a `Layer`'s per-neuron state, its delivery inbox/outbox pair, and its
//! per-layer parameters. Thresholds start near `i16::MAX` (silent) with a small hash jitter.

use crate::wave_net::config::LayerConfig;
use crate::wave_net::synapse::{key, map_range, mix, Synapse, TopologyLevel, P_THRESHOLD};

pub struct Layer {
    // wave-mutable hot state
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub inbox: Vec<Synapse>,  // drained THIS wave (filled last wave)
    pub outbox: Vec<Synapse>, // filled for NEXT wave; swapped with inbox at wave end

    // tunable params (calibration/training will rewrite these between phases)
    pub threshold: Vec<i16>,
    pub saturation: i16,

    // fixed structure
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<TopologyLevel>,
    pub inhibitor_ratio: u32,
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
            *th = (i16::MAX as i32 - jitter) as i16;
        }
        Layer {
            potential: vec![0; ls],
            cooldown: vec![0; ls],
            inbox: Vec::new(),
            outbox: Vec::new(),
            threshold,
            saturation: cfg.saturation,
            leak: cfg.leak,
            cooldown_base: cfg.cooldown_base,
            topology: cfg.topology.clone(),
            inhibitor_ratio: cfg.inhibitor_ratio,
        }
    }

    pub fn max_threshold(&self) -> i16 {
        self.threshold.iter().copied().max().unwrap_or(0)
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
            saturation: i16::MAX,
        }
    }

    #[test]
    fn new_sizes_and_zeroes() {
        let l = Layer::new(&lc(128), 1, 0, 8);
        assert_eq!(l.potential.len(), 64);
        assert_eq!(l.cooldown.len(), 64);
        assert_eq!(l.threshold.len(), 64);
        assert!(l.potential.iter().all(|&p| p == 0));
        assert!(l.inbox.is_empty() && l.outbox.is_empty());
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
}
