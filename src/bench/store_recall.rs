//! `store_recall` — the Tier-0 delayed-match task and the ALIF-vs-LIF memory-horizon experiment.

use crate::bench::readout::record_response;
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix};

const P_CUE: u64 = 7; // base cue membership per class
const P_TRIAL: u64 = 11; // per-trial keep of base sites
const P_NOISE: u64 = 13; // per-trial noise additions
const P_PROBE: u64 = 17; // fixed probe pattern

/// True iff the low 16 bits of the mixed key fall under `thresh_q16` (Q16 probability).
fn selected(seed: u64, site: u32, class_slot: i32, wave_slot: u32, purpose: u64, thresh_q16: u32) -> bool {
    ((mix(key(seed, site, class_slot, wave_slot, purpose)) & 0xFFFF) as u32) < thresh_q16
}

/// L0 injection set for one wave of one trial: the class's base sites (kept with prob `keep_q16`)
/// plus a few noise sites (non-base, added with prob `noise_q16`). Base membership per class is
/// fixed by `base_q16`; per-trial variability comes from `(trial, wave)` folded into the slot.
fn cue_realization(
    seed: u64,
    size: u32,
    class: usize,
    trial: usize,
    wave: usize,
    base_q16: u32,
    keep_q16: u32,
    noise_q16: u32,
) -> Vec<u32> {
    let ls = size * size;
    let slot = (trial as u32).wrapping_mul(1009).wrapping_add(wave as u32);
    let mut v = Vec::new();
    for s in 0..ls {
        let base = selected(seed, s, class as i32, 0, P_CUE, base_q16);
        let hit = if base {
            selected(seed, s, class as i32, slot, P_TRIAL, keep_q16)
        } else {
            selected(seed, s, class as i32, slot, P_NOISE, noise_q16)
        };
        if hit {
            v.push(s);
        }
    }
    v
}

/// Fixed probe pattern (same for every cue and trial): L0 sites selected at `density_q16`.
fn probe_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
    let ls = size * size;
    (0..ls).filter(|&s| selected(seed, s, 0, 0, P_PROBE, density_q16)).collect()
}

/// Encoding + timing knobs for one store-recall trial.
#[derive(Clone, Debug)]
pub struct TaskParams {
    pub seed: u64,
    pub size: u32,
    pub present_waves: usize,
    pub read_waves: usize,
    pub base_q16: u32,  // base cue density per class
    pub keep_q16: u32,  // prob a base site is injected on a given trial/wave
    pub noise_q16: u32, // prob a non-base site is injected (noise)
    pub probe_q16: u32, // probe density
}

/// One trial: reset, present the noisy cue for `present_waves`, stay silent for `delay` waves, then
/// inject the fixed probe for `read_waves` and return the per-neuron spike-count feature vector.
pub fn run_trial(net: &mut Network, tp: &TaskParams, class: usize, trial: usize, delay: usize) -> Vec<u32> {
    net.reset_state();
    for w in 0..tp.present_waves {
        let sites = cue_realization(tp.seed, tp.size, class, trial, w, tp.base_q16, tp.keep_q16, tp.noise_q16);
        net.wave(&sites);
    }
    for _ in 0..delay {
        net.wave(&[]);
    }
    let probe = probe_pattern(tp.seed, tp.size, tp.probe_q16);
    record_response(net, tp.read_waves, move |_w| probe.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    fn small_net(adapt_bump: i16) -> Network {
        let layer = LayerConfig {
            topology: vec![
                TopologyLevel { level: 1, radius: 2, count: 6 },
                TopologyLevel { level: 0, radius: 1, count: 2 },
                TopologyLevel { level: -1, radius: 1, count: 2 },
            ],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay: 5,
        };
        Network::new(Config { seed: 7, size: 8, layers: vec![layer; 4] })
    }

    fn task_params() -> TaskParams {
        TaskParams {
            seed: 7,
            size: 8,
            present_waves: 6,
            read_waves: 6,
            base_q16: 20000,
            keep_q16: 60000,
            noise_q16: 2000,
            probe_q16: 20000,
        }
    }

    #[test]
    fn cue_encoding_is_deterministic_and_distinct() {
        let (seed, size) = (42u64, 8u32);
        let a = cue_realization(seed, size, 0, 3, 1, 20000, 60000, 2000);
        let b = cue_realization(seed, size, 0, 3, 1, 20000, 60000, 2000);
        assert_eq!(a, b, "same args must reproduce the same injection set");
        let other = cue_realization(seed, size, 1, 3, 1, 20000, 60000, 2000);
        assert_ne!(a, other, "different classes must differ");
        // probe is fixed and reproducible
        let p1 = probe_pattern(seed, size, 20000);
        let p2 = probe_pattern(seed, size, 20000);
        assert_eq!(p1, p2);
        assert!(!p1.is_empty(), "probe should select some sites at ~30% density");
    }

    #[test]
    fn run_trial_shape_and_determinism() {
        let mut net = small_net(16);
        let tp = task_params();
        let f1 = run_trial(&mut net, &tp, 0, 0, 8);
        let f2 = run_trial(&mut net, &tp, 0, 0, 8);
        // length = (L-1)*size*size = 3*64 = 192
        assert_eq!(f1.len(), 3 * 64);
        assert_eq!(f1, f2, "trial must be deterministic (reset each time)");
        // a probe response should produce some spikes for an ALIF net at a short delay
        assert!(f1.iter().any(|&c| c > 0), "probe should elicit a response");
    }
}
