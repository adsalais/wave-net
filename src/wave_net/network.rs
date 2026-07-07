//! `network` — owns the layer stack, drives each wave, and routes each layer's
//! generated synapses into the target layers' inboxes for the next wave.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

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
            assert!(
                layer.saturation >= layer.max_threshold(),
                "layer {z}: saturation {} must be >= max threshold {}",
                layer.saturation,
                layer.max_threshold()
            );
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
            g.inbox.clear();
            g.outbox.clear();
        }
        self.wave_id.store(0, Ordering::Relaxed);
    }

    pub fn potential(&self, layer: usize, local: usize) -> i16 {
        self.layers[layer].lock().unwrap().potential[local]
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
            saturation: i16::MAX,
        };
        let l1 = LayerConfig { topology: vec![], ..l0.clone() };
        Config { seed: 99, size: 4, layers: vec![l0, l1] }
    }

    #[test]
    fn new_asserts_invariants() {
        assert_eq!(Network::new(two_layer()).n_total(), 32);
    }

    #[test]
    #[should_panic(expected = "saturation")]
    fn new_rejects_saturation_below_threshold() {
        let mut c = two_layer();
        c.layers[0].threshold_jitter = 0; // threshold == i16::MAX
        c.layers[0].saturation = 100; // < max threshold -> panic
        Network::new(c);
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
}
