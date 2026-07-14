//! Construction input for the BRF engine: a shared square `size`, a seed, global BRF constants
//! (`dt, gamma, theta_c`), and one `LayerConfig` per layer. Mirrors wave_driven::config but drops all
//! LIF/adaptation fields (leak/cooldown/adapt_*) and adds the resonator init ranges + divergence-boundary
//! validation (`δ·ω ≤ 1`).

use crate::wave_resonate::synapse::{neigh_size, TopologyLevel};

#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub topology: Vec<TopologyLevel>,
    pub inhibitor_ratio: u32,      // Q16: inhibitory iff (hash & 0xFFFF) < inhibitor_ratio
    pub omega_init: (f32, f32),    // per-neuron ω ~ U[lo, hi]
    pub b_offset_init: (f32, f32), // per-neuron b' ~ U[lo, hi], b' >= 0
    pub tau_out: f32,              // readout leaky-integrator time constant (used only if readout)
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub size: u32,    // power of two
    pub dt: f32,      // δ integration step (global)
    pub gamma: f32,   // refractory decay γ (global)
    pub theta_c: f32, // base threshold ϑ_c (global)
    pub layers: Vec<LayerConfig>,
}

impl Config {
    pub fn layer_size(&self) -> usize {
        (self.size as usize) * (self.size as usize)
    }
    pub fn n_total(&self) -> usize {
        self.layer_size() * self.layers.len()
    }

    /// A small, valid, deterministic network for tests and bring-up. δ·ω_hi = 0.05·10 = 0.5 ≤ 1.
    pub fn demo() -> Config {
        let mk = |topology: Vec<TopologyLevel>| LayerConfig {
            topology,
            inhibitor_ratio: 9830,
            omega_init: (5.0, 10.0),
            b_offset_init: (0.1, 1.0),
            tau_out: 20.0,
        };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 6 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 6 }]),
            mk(vec![]),
        ];
        Config { seed: 0x1234_5678_9ABC_DEF0, size: 16, dt: 0.05, gamma: 0.9, theta_c: 1.0, layers }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.size < 1 || !self.size.is_power_of_two() {
            return Err(format!("size must be a power of two >= 1, got {}", self.size));
        }
        if self.layers.is_empty() {
            return Err("layers must not be empty".into());
        }
        if !(self.dt > 0.0) {
            return Err(format!("dt must be > 0, got {}", self.dt));
        }
        if !(self.gamma >= 0.0 && self.gamma <= 1.0) {
            return Err(format!("gamma must be in [0,1], got {}", self.gamma));
        }
        if !(self.theta_c > 0.0) {
            return Err(format!("theta_c must be > 0, got {}", self.theta_c));
        }
        for (z, lc) in self.layers.iter().enumerate() {
            let (olo, ohi) = lc.omega_init;
            if !(olo > 0.0 && ohi >= olo) {
                return Err(format!("layer {z}: omega_init must satisfy 0 < lo <= hi, got {olo}..{ohi}"));
            }
            if self.dt * ohi > 1.0 {
                return Err(format!(
                    "layer {z}: divergence boundary requires δ·ω ≤ 1 (dt*omega); dt={} · omega_hi={} = {} > 1",
                    self.dt, ohi, self.dt * ohi
                ));
            }
            let (blo, bhi) = lc.b_offset_init;
            if !(blo >= 0.0 && bhi >= blo) {
                return Err(format!("layer {z}: b_offset_init must satisfy 0 <= lo <= hi, got {blo}..{bhi}"));
            }
            if !(lc.tau_out > 0.0) {
                return Err(format!("layer {z}: tau_out must be > 0, got {}", lc.tau_out));
            }
            for t in &lc.topology {
                let n = neigh_size(t.radius);
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
    use crate::wave_resonate::synapse::TopologyLevel;

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
    fn rejects_divergence_boundary_violation() {
        // δ·ω must be ≤ 1; push omega_init.hi over 1/dt
        let mut c = Config::demo();
        c.dt = 0.1;
        c.layers[0].omega_init = (5.0, 20.0); // 0.1*20 = 2 > 1
        let e = c.validate().unwrap_err();
        assert!(e.contains("divergence") || e.contains("δ·ω") || e.contains("dt*omega"), "descriptive: {e}");
    }

    #[test]
    fn rejects_fan_in_over_neighborhood() {
        let mut c = Config::demo();
        c.layers[0].topology = vec![TopologyLevel { level: 1, radius: 2, count: 30 }]; // 30 > 25
        let e = c.validate().unwrap_err();
        assert!(e.contains("count") && e.contains("neighborhood"), "descriptive: {e}");
    }

    #[test]
    fn rejects_bad_omega_or_gamma() {
        let mut c = Config::demo();
        c.gamma = 1.5;
        assert!(c.validate().is_err());
        let mut c2 = Config::demo();
        c2.layers[0].omega_init = (10.0, 5.0); // lo > hi
        assert!(c2.validate().is_err());
    }
}
