//! `network` — owns the layer stack, drives each wave, and routes each layer's
//! generated synapses into the target layers' inboxes for the next wave.

use std::sync::{Arc, Mutex};

use crate::wave_net::config::Config;
use crate::wave_net::neurons::{Layer, ADAPT_SHIFT};
use crate::wave_net::synapse::Synapse;
use crate::wave_net::wave::process_layer;

pub struct Network {
    seed: u64,
    size: u32,
    layers: Vec<Layer>,
    wave_id: usize,
    scratch: Scratch,
    /// Whether each wave accrues the e-prop eligibility (per-neuron `decide_potential`/`decide_eff`
    /// snapshots and `elig_pre`/`elig_post` traces). Training needs it (default `true`); a pure
    /// forward pass (inference, throughput) turns it off to skip that per-neuron write traffic.
    record_eligibility: bool,
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

/// Reusable per-wave scratch owned by the `Network` — cleared/overwritten each wave, never
/// reallocated on the hot path. `deliveries[z]` accumulates the synapses bound for layer `z` on the
/// *next* wave (the old per-layer outbox, hoisted here so generation can scatter directly by
/// absolute target layer — no intermediate per-level grouping, no second copy). It is disjoint from
/// every layer's `inbox`, so pending deliveries are never overwritten; at wave end each layer's
/// (now drained) `inbox` is swapped with its `deliveries` buffer.
struct Scratch {
    acc: Vec<i32>,
    fired: Vec<u32>,
    deliveries: Vec<Vec<Synapse>>,
}

impl Network {
    pub fn new(config: Config) -> Network {
        Network::build(config, false)
    }

    /// Like `new`, but flags the **last** layer as a non-spiking drain-only readout (output sink):
    /// it integrates its input into potential and never fires. Mirrors L0's input-transducer role.
    pub fn new_with_readout(config: Config) -> Network {
        Network::build(config, true)
    }

    fn build(config: Config, readout_last: bool) -> Network {
        config.validate().expect("invalid config");
        let size = config.size;
        let l = config.layers.len();
        let ls = (size as usize) * (size as usize);
        let mut layers = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let mut layer = Layer::new(lc, config.seed, z as u32, size);
            if z == 0 {
                // L0 is the input transducer: forced to fire only on injection (baseline i16::MAX)
                // and to never adapt (adapt_bump 0). This keeps input encoding decoupled from
                // adaptation and guarantees injected spikes always fire (effective threshold stays
                // == baseline <= i16::MAX). ALIF dynamics apply to the computational layers 1..L.
                layer.threshold.iter_mut().for_each(|t| *t = i16::MAX);
                layer.adapt_bump = 0;
            }
            if readout_last && z == l - 1 {
                layer.readout = true;
            }
            layers.push(layer);
        }
        Network {
            seed: config.seed,
            size,
            layers,
            wave_id: 0,
            scratch: Scratch {
                acc: vec![0i32; ls],
                fired: Vec::new(),
                deliveries: (0..l).map(|_| Vec::new()).collect(),
            },
            record_eligibility: true,
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&mut self, input: &[u32]) {
        let w = self.wave_id;
        self.wave_id += 1;
        let l = self.layers.len();
        let seed = self.seed;
        let size = self.size;
        let record_elig = self.record_eligibility;
        // Disjoint mutable borrows: each layer is processed while generation scatters into the
        // separate `deliveries` buffer, so a source layer's level-0 self-connections and its own
        // in-flight state never alias.
        let Self { layers, scratch, listeners, .. } = self;
        let Scratch { acc, fired, deliveries } = scratch;

        for z in 0..l {
            let inp: &[u32] = if z == 0 { input } else { &[] };
            process_layer(&mut layers[z], z as u32, seed, size, inp, acc, deliveries, fired, record_elig);
            if let Some(listener) = &listeners[z] {
                listener(w, fired);
            }
        }

        // Swap each layer's now-drained `inbox` with its accumulated `deliveries` (the next inbox).
        // `deliveries` is disjoint from every `inbox`, so nothing pending was overwritten during the
        // wave; after the swap `deliveries[i]` holds the emptied inbox, ready to reaccumulate.
        for i in 0..l {
            std::mem::swap(&mut layers[i].inbox, &mut deliveries[i]);
        }
    }

    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>) {
        self.listeners[layer] = Some(listener);
    }

    pub fn clear_listeners(&mut self) {
        for l in self.listeners.iter_mut() {
            *l = None;
        }
    }

    /// Toggle e-prop eligibility recording (default on). Turn it off for a pure forward pass
    /// (inference / throughput) to skip the per-neuron `decide_potential`/`decide_eff` snapshots and
    /// `elig_pre`/`elig_post` traces; turn it back on before a training pass that reads them.
    pub fn set_record_eligibility(&mut self, on: bool) {
        self.record_eligibility = on;
    }

    pub fn reset_state(&mut self) {
        for g in self.layers.iter_mut() {
            g.potential.iter_mut().for_each(|p| *p = 0);
            g.cooldown.iter_mut().for_each(|c| *c = 0);
            g.adapt.iter_mut().for_each(|a| *a = 0);
            g.elig_pre.iter_mut().for_each(|e| *e = 0);
            g.elig_post.iter_mut().for_each(|e| *e = 0);
            g.decide_potential.iter_mut().for_each(|p| *p = 0);
            g.inbox.clear();
        }
        for d in self.scratch.deliveries.iter_mut() {
            d.clear();
        }
        self.wave_id = 0;
    }

    pub fn potential(&self, layer: usize, local: usize) -> i16 {
        self.layers[layer].potential[local]
    }

    /// Raw Q12 fixed-point adaptation state (effective threshold contribution is `>> ADAPT_SHIFT`).
    pub fn adaptation(&self, layer: usize, local: usize) -> i32 {
        self.layers[layer].adapt[local]
    }

    /// Force neuron `local` in layer `z` to fire on the *next* `wave()` — sets its potential to `i16::MAX`
    /// and clears its cooldown, so the decide step fires it (the effective threshold of the low-baseline
    /// computational layers is well under `i16::MAX`). Also zeroes its adaptation so the fire is guaranteed
    /// even if the neuron was heavily adapted. For criticality perturbation probes; no effect unless called.
    pub fn force_spike(&mut self, z: usize, local: usize) {
        let l = &mut self.layers[z];
        l.potential[local] = i16::MAX;
        l.cooldown[local] = 0;
        l.adapt[local] = 0;
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// The construction seed (learning rules need it to recover synapse targets via `target_of`).
    pub(crate) fn seed_val(&self) -> u64 {
        self.seed
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn n_total(&self) -> usize {
        self.layers.len() * (self.size as usize) * (self.size as usize)
    }

    /// Mutable access to one layer (how calibration/training reach `Layer` methods and state).
    pub(crate) fn with_layer_mut<R>(&mut self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R {
        f(&mut self.layers[z])
    }

    /// Read-only access to one layer (introspection that must not require `&mut`).
    pub(crate) fn with_layer<R>(&self, z: usize, f: impl FnOnce(&Layer) -> R) -> R {
        f(&self.layers[z])
    }

    /// A copy of a layer's per-neuron thresholds (introspection / determinism tests).
    pub fn layer_thresholds(&self, z: usize) -> Vec<i16> {
        self.with_layer(z, |l| l.thresholds().to_vec())
    }

    /// Per-neuron membrane potential captured at the last decide step (pre fire-reset/leak).
    pub fn layer_decide_potential(&self, z: usize) -> Vec<i16> {
        self.layers[z].decide_potential.clone()
    }

    /// Per-neuron effective ALIF firing threshold `baseline + (adapt >> ADAPT_SHIFT)` from the CURRENT
    /// adaptation state. Read-only; no dynamics change. (For a decide-time-aligned value that pairs with
    /// `layer_decide_potential`, use `layer_decide_effective_threshold` — the current adapt has already
    /// been fire-bumped and decayed by the time a wave returns.)
    pub fn layer_effective_threshold(&self, z: usize) -> Vec<i32> {
        let l = &self.layers[z];
        l.threshold
            .iter()
            .zip(l.adapt.iter())
            .map(|(&t, &a)| t as i32 + (a >> ADAPT_SHIFT))
            .collect()
    }

    /// Per-neuron effective firing threshold captured at the last decide step (before that wave's fire-bump
    /// mutated adapt) — the value actually compared against `layer_decide_potential`. This is the correct
    /// reference for a pseudo-derivative ψ = f(decide_potential − decide_eff).
    pub fn layer_decide_effective_threshold(&self, z: usize) -> Vec<i32> {
        self.layers[z].decide_eff.clone()
    }

    /// Reset, run `warmup` waves (discarded), then `waves` counted; per-layer firing rate =
    /// spikes / (layer_size * waves). Saves and restores the caller's listeners around the run.
    pub(crate) fn measure_layer_rates(
        &mut self,
        warmup: usize,
        waves: usize,
        input: &impl Fn(usize) -> Vec<u32>,
    ) -> Vec<f64> {
        let l = self.layers.len();
        // Move the caller's listeners aside (boxed Fn is not Clone), install counters.
        let saved = std::mem::replace(&mut self.listeners, (0..l).map(|_| None).collect());
        let counts = Arc::new(Mutex::new(vec![0u64; l]));
        for z in 0..l {
            let c = counts.clone();
            self.listeners[z] = Some(Box::new(move |_w: usize, fired: &[u32]| {
                c.lock().unwrap()[z] += fired.len() as u64;
            }));
        }
        self.reset_state();
        for w in 0..warmup {
            self.wave(&input(w));
        }
        counts.lock().unwrap().iter_mut().for_each(|c| *c = 0); // discard warmup
        for w in 0..waves {
            self.wave(&input(warmup + w));
        }
        self.listeners = saved; // restore caller's listeners; counters dropped
        let counts = std::mem::take(&mut *counts.lock().unwrap());
        let ls = (self.size as u64) * (self.size as u64);
        let denom = (ls * waves as u64) as f64;
        counts.iter().map(|&s| s as f64 / denom).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::synapse::TopologyLevel;
    use std::sync::{Arc, Mutex};

    #[test]
    fn effective_threshold_is_baseline_plus_adapt_shifted() {
        let mut net = Network::new(two_layer());
        // baseline with zero adaptation == the threshold itself
        let base = net.layer_thresholds(1);
        let eff0 = net.layer_effective_threshold(1);
        assert_eq!(eff0, base.iter().map(|&t| t as i32).collect::<Vec<_>>());
        // inject adaptation on neuron 0 of layer 1 and check the >> ADAPT_SHIFT contribution
        net.with_layer_mut(1, |l| l.adapt[0] = 5 << ADAPT_SHIFT);
        let eff1 = net.layer_effective_threshold(1);
        assert_eq!(eff1[0], base[0] as i32 + 5);
        assert_eq!(&eff1[1..], &eff0[1..]);
    }

    #[test]
    fn force_spike_fires_the_neuron() {
        let mut net = Network::new(two_layer());
        let fired = Arc::new(Mutex::new(Vec::new()));
        {
            let f = fired.clone();
            net.on_layer(1, Box::new(move |_w, fd: &[u32]| f.lock().unwrap().extend_from_slice(fd)));
        }
        net.force_spike(1, 5);
        net.wave(&[]); // no input — only the forced neuron should fire in L1
        assert_eq!(fired.lock().unwrap().as_slice(), &[5], "force_spike fires exactly the targeted neuron");
    }

    // two 4x4 layers, L0 -> L1 straight up (level+1, radius 0), all excitatory
    fn two_layer() -> Config {
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 0, count: 1 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: i16::MAX,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let l1 = LayerConfig { topology: vec![], ..l0.clone() };
        Config { seed: 99, size: 4, layers: vec![l0, l1] }
    }

    // 2 layers, L0 -> L1 (level+1, radius 1, 4 targets), low baseline + strong adaptation on L1.
    fn alif_two_layer() -> Config {
        let l0 = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
            leak: (3, 5),
            cooldown_base: 1,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 2,
            adapt_bump: 200,
            adapt_decay: 4,
        };
        let l1 = LayerConfig { topology: vec![], ..l0.clone() };
        Config { seed: 5, size: 4, layers: vec![l0, l1] }
    }

    #[test]
    fn adaptation_accessor_and_reset() {
        let mut net = Network::new(alif_two_layer());
        let all_l0 = (0..16u32).collect::<Vec<u32>>();
        for _ in 0..5 {
            net.wave(&all_l0); // injection drives L0 (transducer, no adapt) -> L1 fires and adapts
        }
        let any_nonzero = (0..16).any(|i| net.adaptation(1, i) > 0);
        assert!(any_nonzero, "L1 adaptation should be >0 after firing");
        assert!((0..16).all(|i| net.adaptation(0, i) == 0), "L0 transducer must never adapt");
        net.reset_state();
        for z in 0..net.layer_count() {
            for i in 0..16 {
                assert_eq!(net.adaptation(z, i), 0, "reset must zero adaptation");
            }
        }
    }

    #[test]
    fn determinism_includes_adaptation() {
        let inputs: [&[u32]; 4] = [&[0, 1, 2, 3], &[4, 5], &[], &[6, 7, 8]];
        let run = || {
            let mut net = Network::new(Config::demo());
            for _ in 0..6 {
                for inp in inputs {
                    net.wave(inp);
                }
            }
            (0..net.layer_count())
                .flat_map(|z| (0..(net.size() * net.size()) as usize).map(move |i| (z, i)))
                .map(|(z, i)| net.adaptation(z, i))
                .collect::<Vec<i32>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn adaptation_self_limits_rate() {
        // Same constant drive into L1, with vs. without adaptation. Adaptation must reduce total
        // L1 firing over the window — the self-limiting effect, robustly measured as a total count.
        fn total_l1_spikes(adapt_bump: i16) -> usize {
            let l0 = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
                leak: (3, 5),
                cooldown_base: 1,
                inhibitor_ratio: 0,
                threshold_jitter: 0,
                baseline_init: 2,
                adapt_bump,
                adapt_decay: 4,
            };
            let l1 = LayerConfig { topology: vec![], ..l0.clone() };
            let mut net = Network::new(Config { seed: 5, size: 4, layers: vec![l0, l1] });
            let count = Arc::new(Mutex::new(0usize));
            {
                let c = count.clone();
                net.on_layer(1, Box::new(move |_w, fired: &[u32]| *c.lock().unwrap() += fired.len()));
            }
            let all_l0 = (0..16u32).collect::<Vec<u32>>();
            for _ in 0..60 {
                net.wave(&all_l0); // constant maximal drive into L1
            }
            let n = *count.lock().unwrap();
            n
        }
        let without = total_l1_spikes(0);
        let with = total_l1_spikes(30);
        assert!(without > 0, "L1 should fire under constant drive with no adaptation");
        assert!(with < without, "adaptation should suppress total L1 firing: {with} vs {without}");
    }

    #[test]
    fn new_builds_expected_size() {
        assert_eq!(Network::new(two_layer()).n_total(), 32);
    }

    #[test]
    fn injection_fires_exactly_l0_targets() {
        let fired = Arc::new(Mutex::new(Vec::new()));
        let mut net = Network::new(two_layer());
        {
            let f = fired.clone();
            net.on_layer(0, Box::new(move |_w, locals| *f.lock().unwrap() = locals.to_vec()));
        }
        net.wave(&[0, 5]);
        assert_eq!(*fired.lock().unwrap(), vec![0, 5]);
    }

    #[test]
    fn deferred_delivery_is_one_hop() {
        // With L1 baseline 1, a single +1 delivery makes L1 fire — on the wave AFTER L0 fires, not
        // the same wave. (Verified via firing, not a residual potential, which the leak floor would
        // erase.) two_layer sets L1 near i16::MAX; drop it to 1 so the delivery is decisive.
        let mut cfg = two_layer();
        cfg.layers[1].baseline_init = 1;
        let mut net = Network::new(cfg);
        let fired_waves = Arc::new(Mutex::new(Vec::<usize>::new()));
        {
            let f = fired_waves.clone();
            net.on_layer(1, Box::new(move |w, fired: &[u32]| {
                if !fired.is_empty() {
                    f.lock().unwrap().push(w);
                }
            }));
        }
        net.wave(&[0]); // wave 0: L0 neuron 0 fires; delivery queued for L1
        net.wave(&[]); // wave 1: L1 drains the +1 and fires
        assert_eq!(*fired_waves.lock().unwrap(), vec![1], "L1 fires only on the wave after L0's spike");
    }

    #[test]
    fn deterministic_across_runs() {
        let inputs: [&[u32]; 3] = [&[0, 1, 2], &[], &[3]];
        let run = || {
            let mut net = Network::new(Config::demo());
            for inp in inputs {
                net.wave(inp);
            }
            (0..net.layer_count())
                .flat_map(|z| (0..(net.size() * net.size()) as usize).map(move |i| (z, i)))
                .map(|(z, i)| net.potential(z, i))
                .collect::<Vec<i16>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn reset_state_zeros_everything() {
        let mut net = Network::new(Config::demo());
        for _ in 0..5 {
            net.wave(&[0, 1, 2, 3]);
        }
        net.reset_state();
        for z in 0..net.layer_count() {
            for i in 0..(net.size() * net.size()) as usize {
                assert_eq!(net.potential(z, i), 0);
            }
        }
    }

    #[test]
    fn layer_thresholds_reads_layer() {
        let net = Network::new(two_layer()); // jitter 0 -> all i16::MAX
        let t = net.layer_thresholds(1);
        assert_eq!(t.len(), 16); // size 4 -> 16
        assert!(t.iter().all(|&x| x == i16::MAX));
    }

    #[test]
    fn measure_rates_reflects_l0_injection() {
        // inject 4 of the 16 L0 locals (size 4) every wave -> rates[0] = 0.25; L1 silent (near-max)
        let mut net = Network::new(two_layer());
        let input = |_w: usize| (0..4u32).collect::<Vec<u32>>();
        let rates = net.measure_layer_rates(4, 32, &input);
        assert!((rates[0] - 0.25).abs() < 0.02, "L0 rate {} != ~0.25", rates[0]);
        assert!(rates[1] < 0.01, "L1 should be silent, got {}", rates[1]);
    }

    #[test]
    fn measure_preserves_listeners() {
        let mut net = Network::new(two_layer());
        let hits = Arc::new(Mutex::new(0usize));
        {
            let h = hits.clone();
            net.on_layer(0, Box::new(move |_w, _f| *h.lock().unwrap() += 1));
        }
        let input = |_w: usize| vec![0u32];
        net.measure_layer_rates(2, 8, &input);
        *hits.lock().unwrap() = 0; // reset, then one wave must still hit the user listener
        net.wave(&[0]);
        assert_eq!(*hits.lock().unwrap(), 1, "user listener must survive measurement");
    }
}
