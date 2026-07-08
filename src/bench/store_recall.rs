//! `store_recall` — the Tier-0 delayed-match task and the ALIF-vs-LIF memory-horizon experiment.

use crate::bench::readout::{record_response, NearestCentroid};
use crate::wave_net::calibrate::{random_l0_input, CalibrateParams};
use crate::wave_net::config::{Config, LayerConfig};
use crate::wave_net::network::Network;
use crate::wave_net::synapse::{key, mix, TopologyLevel};

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
pub(crate) fn cue_realization(
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
pub(crate) fn probe_pattern(seed: u64, size: u32, density_q16: u32) -> Vec<u32> {
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

/// Full configuration for the store-recall memory-horizon experiment.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub seed: u64,
    pub size: u32,
    pub layers: usize,
    pub k: usize,           // number of cue classes (chance = 1/k)
    pub baseline_init: i16,
    pub adapt_bump: i16,    // ALIF value; LIF variant passes 0 to memory_horizon
    pub adapt_decay: u8,
    pub trials_per_class: usize,
    pub delays: Vec<usize>, // swept, ascending
    pub task: TaskParams,
    pub calib: CalibrateParams,
    pub calib_fraction_q16: u32,
}

impl BenchConfig {
    /// Small, fast, deterministic config tuned for the inline test. adapt_decay 6 -> tau ~64 waves,
    /// well past the leak horizon (~15-20 waves for leak (3,5)); the longest delay sits between them.
    pub fn demo() -> BenchConfig {
        let seed = 0xB0A7_57ED;
        let size = 8;
        BenchConfig {
            seed,
            size,
            layers: 4,
            k: 4,
            baseline_init: 6,
            adapt_bump: 20,
            adapt_decay: 6,
            trials_per_class: 10,
            delays: vec![0, 8, 24],
            task: TaskParams {
                seed,
                size,
                present_waves: 6,
                read_waves: 6,
                base_q16: 18000,
                keep_q16: 60000,
                noise_q16: 1500,
                probe_q16: 20000,
            },
            calib: CalibrateParams {
                warmup: 16,
                waves: 48,
                max_steps: 24,
                refine_passes: 3,
                ..CalibrateParams::default()
            },
            calib_fraction_q16: 20000,
        }
    }

    fn to_engine_config(&self, adapt_bump: i16) -> Config {
        let layer = LayerConfig {
            // Feed-forward for the store-recall diagnostic: recurrence (level 0 / -1) would let the
            // reservoir self-sustain activity through the delay and carry the cue on its own,
            // confounding the ALIF-vs-LIF contrast. Feed-forward goes quiet during the delay, so
            // only the slow (silent) adaptation state can survive it.
            topology: vec![TopologyLevel { level: 1, radius: 2, count: 6 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: self.baseline_init,
            adapt_bump,
            adapt_decay: self.adapt_decay,
        };
        Config { seed: self.seed, size: self.size, layers: vec![layer; self.layers] }
    }
}

/// Accuracy (permille) at each swept delay, for one variant.
#[derive(Clone, Debug)]
pub struct HorizonCurve {
    pub delays: Vec<usize>,
    pub accuracy_permille: Vec<u64>,
}

/// Run the store-recall sweep for one variant. `adapt_bump` selects the variant (0 = plain LIF).
/// The net is built + calibrated once, then trials reuse the calibrated baselines (reset per trial).
pub fn memory_horizon(cfg: &BenchConfig, adapt_bump: i16) -> HorizonCurve {
    let mut net = Network::new(cfg.to_engine_config(adapt_bump));
    let input = random_l0_input(cfg.seed ^ 0xCA11B, cfg.size, cfg.calib_fraction_q16);
    net.calibrate(&cfg.calib, &input);

    let mut accuracy_permille = Vec::with_capacity(cfg.delays.len());
    for &delay in &cfg.delays {
        let mut feats: Vec<Vec<u32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for t in 0..cfg.trials_per_class {
            for c in 0..cfg.k {
                feats.push(run_trial(&mut net, &cfg.task, c, t, delay));
                labels.push(c);
            }
        }
        // Deterministic split: even trial index -> train, odd -> test (balanced across classes).
        let (mut tr_f, mut tr_l, mut te_f, mut te_l): (Vec<Vec<u32>>, Vec<usize>, Vec<Vec<u32>>, Vec<usize>) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, (f, l)) in feats.into_iter().zip(labels).enumerate() {
            if (i / cfg.k) % 2 == 0 {
                tr_f.push(f);
                tr_l.push(l);
            } else {
                te_f.push(f);
                te_l.push(l);
            }
        }
        let clf = NearestCentroid::fit(&tr_f, &tr_l, cfg.k);
        let mut correct = 0usize;
        for (f, &l) in te_f.iter().zip(&te_l) {
            if clf.predict(f) == l {
                correct += 1;
            }
        }
        let acc = if te_f.is_empty() { 0 } else { (correct as u64 * 1000) / te_f.len() as u64 };
        accuracy_permille.push(acc);
    }
    HorizonCurve { delays: cfg.delays.clone(), accuracy_permille }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn memory_horizon_is_deterministic() {
        let cfg = BenchConfig::demo();
        let a = memory_horizon(&cfg, cfg.adapt_bump);
        let b = memory_horizon(&cfg, cfg.adapt_bump);
        assert_eq!(a.delays, b.delays);
        assert_eq!(a.accuracy_permille, b.accuracy_permille);
    }

    #[test]
    fn store_recall_alif_beats_lif_at_long_delay() {
        let cfg = BenchConfig::demo();
        let alif = memory_horizon(&cfg, cfg.adapt_bump);
        let lif = memory_horizon(&cfg, 0);
        eprintln!("delays {:?}", alif.delays);
        eprintln!("ALIF   {:?}", alif.accuracy_permille);
        eprintln!("LIF    {:?}", lif.accuracy_permille);
        let chance = 1000 / cfg.k as u64;
        let last = cfg.delays.len() - 1;
        // (1) encodable: both decode well above chance at the shortest delay
        assert!(alif.accuracy_permille[0] > 650, "ALIF should decode at short delay");
        assert!(lif.accuracy_permille[0] > 650, "LIF should decode at short delay");
        // (2) ALIF holds, LIF forgets, at the longest delay
        assert!(
            alif.accuracy_permille[last] > lif.accuracy_permille[last] + 100,
            "ALIF should beat LIF at long delay (ALIF {} vs LIF {})",
            alif.accuracy_permille[last],
            lif.accuracy_permille[last]
        );
        assert!(alif.accuracy_permille[last] > chance + 80, "ALIF should stay above chance at long delay");
    }
}
