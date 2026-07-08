//! `network` — owns the layer stack, drives each wave, and routes each layer's
//! generated synapses into the target layers' inboxes for the next wave.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::wave_net::config::Config;
use crate::wave_net::neurons::Layer;
use crate::wave_net::synapse::SynapseGroup;
use crate::wave_net::wave::process_layer;

pub struct Network {
    seed: u64,
    size: u32,
    layers: Vec<Mutex<Layer>>,
    wave_id: AtomicUsize,
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

impl Network {
    pub fn new(config: Config) -> Network {
        config.validate().expect("invalid config");
        let size = config.size;
        let l = config.layers.len();
        let mut layers = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let layer = Layer::new(lc, config.seed, z as u32, size);
            layers.push(Mutex::new(layer));
        }
        Network {
            seed: config.seed,
            size,
            layers,
            wave_id: AtomicUsize::new(0),
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&self, input: &[u32]) {
        let w = self.wave_id.fetch_add(1, Ordering::Relaxed);
        let l = self.layers.len();
        let ls = (self.size as usize) * (self.size as usize);
        let mut acc = vec![0i32; ls];
        let mut fired: Vec<u32> = Vec::new();

        for z in 0..l {
            let mut out: Vec<SynapseGroup>;
            {
                let mut g = self.layers[z].lock().unwrap();
                out = g
                    .topology
                    .iter()
                    .map(|e| SynapseGroup { level: e.level, synapses: Vec::new() })
                    .collect();
                let inp: &[u32] = if z == 0 { input } else { &[] };
                process_layer(&mut g, z as u32, self.seed, self.size, inp, &mut acc, &mut out, &mut fired);
            }
            // route: Network resolves absolute target layers and feeds their outboxes
            for grp in out.iter() {
                let tl = z as i32 + grp.level;
                if tl >= 0 && (tl as usize) < l {
                    self.layers[tl as usize].lock().unwrap().outbox.extend(grp.synapses.iter().copied());
                }
            }
            if let Some(listener) = &self.listeners[z] {
                listener(w, &fired);
            }
        }

        // swap inbox <- outbox so this wave's deliveries drain next wave
        for layer in self.layers.iter() {
            let mut guard = layer.lock().unwrap();
            let g = &mut *guard; // deref once so inbox/outbox borrow disjointly
            std::mem::swap(&mut g.inbox, &mut g.outbox);
            g.outbox.clear();
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

    pub fn reset_state(&self) {
        for layer in self.layers.iter() {
            let mut g = layer.lock().unwrap();
            g.potential.iter_mut().for_each(|p| *p = 0);
            g.cooldown.iter_mut().for_each(|c| *c = 0);
            g.adapt.iter_mut().for_each(|a| *a = 0);
            g.inbox.clear();
            g.outbox.clear();
        }
        self.wave_id.store(0, Ordering::Relaxed);
    }

    pub fn potential(&self, layer: usize, local: usize) -> i16 {
        self.layers[layer].lock().unwrap().potential[local]
    }

    pub fn adaptation(&self, layer: usize, local: usize) -> i16 {
        self.layers[layer].lock().unwrap().adapt[local]
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn n_total(&self) -> usize {
        self.layers.len() * (self.size as usize) * (self.size as usize)
    }

    /// Locked mutable access to one layer (how calibration reaches Layer methods).
    pub(crate) fn with_layer_mut<R>(&self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R {
        let mut g = self.layers[z].lock().unwrap();
        f(&mut g)
    }

    /// A copy of a layer's per-neuron thresholds (introspection / determinism tests).
    pub fn layer_thresholds(&self, z: usize) -> Vec<i16> {
        self.with_layer_mut(z, |l| l.thresholds().to_vec())
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
    use crate::wave_net::synapse::{local_of, TopologyLevel};
    use std::sync::{Arc, Mutex};

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
        let net = Network::new(alif_two_layer());
        let all_l0 = (0..16u32).collect::<Vec<u32>>();
        for _ in 0..3 {
            net.wave(&all_l0); // injection forces L0 to fire -> bumps L0 adapt
        }
        let any_nonzero = (0..16).any(|i| net.adaptation(0, i) > 0);
        assert!(any_nonzero, "L0 adaptation should be >0 after repeated firing");
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
            let net = Network::new(Config::demo());
            for _ in 0..6 {
                for inp in inputs {
                    net.wave(inp);
                }
            }
            (0..net.layer_count())
                .flat_map(|z| (0..(net.size() * net.size()) as usize).map(move |i| (z, i)))
                .map(|(z, i)| net.adaptation(z, i))
                .collect::<Vec<i16>>()
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
        let net = Network::new(two_layer());
        net.wave(&[0]); // L0 neuron 0 fires; delivery queued for L1
        assert_eq!(net.potential(1, local_of(0, 0, 4) as usize), 0, "not delivered same wave");
        net.wave(&[]); // L1 drains: +1 arrives
        assert_eq!(net.potential(1, local_of(0, 0, 4) as usize), 1, "delivered next wave");
    }

    #[test]
    fn deterministic_across_runs() {
        let inputs: [&[u32]; 3] = [&[0, 1, 2], &[], &[3]];
        let run = || {
            let net = Network::new(Config::demo());
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
        let net = Network::new(Config::demo());
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
