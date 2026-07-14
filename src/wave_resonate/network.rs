//! `network` — owns the BRF layer stack and drives each wave: process every layer (dense membrane +
//! firer-gated delivery), route generated deliveries one hop, swap into each layer's `pending` at wave
//! end (deferred propagation). L0 is the transducer; the last layer is either compute or a readout.

use crate::wave_resonate::config::Config;
use crate::wave_resonate::neurons::Layer;
use crate::wave_resonate::wave::process_layer;

pub struct Network {
    size: u32,
    layers: Vec<Layer>,
    wave_id: u32,
    fired: Vec<u32>,
    deliv: Vec<Vec<i32>>, // per layer: NEXT wave's incoming accumulator
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

impl Network {
    pub fn new(config: Config) -> Network {
        Network::build(config, false)
    }
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
            let mut layer = Layer::new(lc, config.dt, config.gamma, config.theta_c, config.seed, z as u32, size);
            if z == 0 {
                layer.transducer = true;
            }
            if readout_last && z == l - 1 {
                layer.readout = true;
            }
            layers.push(layer);
        }
        Network {
            size,
            layers,
            wave_id: 0,
            fired: Vec::new(),
            deliv: (0..l).map(|_| vec![0i32; ls]).collect(),
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&mut self, input: &[u32]) {
        let w = self.wave_id;
        self.wave_id = self.wave_id.wrapping_add(1);
        let l = self.layers.len();
        let size = self.size;
        let Self { layers, deliv, fired, listeners, .. } = self;
        for z in 0..l {
            let inp: &[u32] = if z == 0 { input } else { &[] };
            process_layer(&mut layers[z], z as u32, size, inp, deliv, fired);
            if let Some(cb) = &listeners[z] {
                cb(w as usize, fired);
            }
        }
        // deferred one hop: this wave's deliveries become next wave's pending
        for z in 0..l {
            std::mem::swap(&mut layers[z].pending, &mut deliv[z]);
        }
    }

    pub fn reset_state(&mut self) {
        for g in self.layers.iter_mut() {
            g.x.iter_mut().for_each(|v| *v = 0.0);
            g.y.iter_mut().for_each(|v| *v = 0.0);
            g.q.iter_mut().for_each(|v| *v = 0.0);
            g.pending.iter_mut().for_each(|p| *p = 0);
        }
        for d in self.deliv.iter_mut() {
            d.iter_mut().for_each(|x| *x = 0);
        }
        self.wave_id = 0;
    }

    pub fn size(&self) -> u32 {
        self.size
    }
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    pub fn with_layer<R>(&self, z: usize, f: impl FnOnce(&Layer) -> R) -> R {
        f(&self.layers[z])
    }
    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>) {
        self.listeners[layer] = Some(listener);
    }
    pub fn clear_listeners(&mut self) {
        for l in self.listeners.iter_mut() {
            *l = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;

    fn three_layer(size: u32) -> Config {
        let mk = |topology: Vec<TopologyLevel>| LayerConfig { topology, inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0 };
        Config {
            seed: 9,
            size,
            dt: 0.05,
            gamma: 0.9,
            theta_c: 1.0,
            layers: vec![
                mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
                mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
                mk(vec![]),
            ],
        }
    }

    #[test]
    fn l0_is_transducer_last_is_compute_by_default() {
        let net = Network::new(three_layer(8));
        net.with_layer(0, |l| assert!(l.transducer && !l.readout));
        net.with_layer(2, |l| assert!(!l.transducer && !l.readout));
    }

    #[test]
    fn new_with_readout_flags_last_layer() {
        let net = Network::new_with_readout(three_layer(8));
        net.with_layer(2, |l| assert!(l.readout && !l.transducer));
    }

    #[test]
    fn wave_is_deterministic() {
        let mut a = Network::new(three_layer(8));
        let mut b = Network::new(three_layer(8));
        let inputs: [&[u32]; 6] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[], &[1]];
        for inp in inputs {
            a.wave(inp);
            b.wave(inp);
        }
        a.with_layer(1, |la| {
            b.with_layer(1, |lb| {
                assert_eq!(la.x, lb.x);
                assert_eq!(la.y, lb.y);
                assert_eq!(la.q, lb.q);
            })
        });
    }

    #[test]
    fn activity_propagates_up_the_stack() {
        // Drive L0 for many waves; a middle compute layer should develop nonzero membrane state
        // (signal climbed the deferred one-hop stack).
        let mut net = Network::new(three_layer(16));
        for w in 0..60 {
            net.wave(if w % 2 == 0 { &[0, 1, 2, 16, 17, 18, 32, 33] } else { &[] });
        }
        let any = net.with_layer(1, |l| l.x.iter().any(|&v| v != 0.0) || l.y.iter().any(|&v| v != 0.0));
        assert!(any, "layer 1 developed oscillator activity from L0 drive");
    }

    #[test]
    fn readout_never_fires_but_integrates() {
        // 2-layer net: L0 transducer delivers directly into the readout (one hop), so input is
        // guaranteed regardless of how sub-critical a deeper untrained stack would be. Tests the readout
        // MECHANISM at the network level (never fires; accumulates); deep-stack propagation is a Phase-3
        // question, not an inference invariant.
        let mk = |topology: Vec<TopologyLevel>| LayerConfig { topology, inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0 };
        let cfg = Config {
            seed: 9,
            size: 8,
            dt: 0.05,
            gamma: 0.9,
            theta_c: 1.0,
            layers: vec![mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]), mk(vec![])],
        };
        let mut net = Network::new_with_readout(cfg);
        let fired_top = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let ft = fired_top.clone();
        net.on_layer(1, Box::new(move |_w, f| *ft.lock().unwrap() += f.len()));
        for w in 0..40 {
            net.wave(if w % 2 == 0 { &[0, 1, 2, 8, 9, 10] } else { &[] });
        }
        assert_eq!(*fired_top.lock().unwrap(), 0, "readout never fires");
        let any = net.with_layer(1, |l| l.x.iter().any(|&v| v != 0.0));
        assert!(any, "readout integrated some signal");
    }

    #[test]
    fn reset_state_clears_membrane() {
        let mut net = Network::new(three_layer(8));
        for _ in 0..10 {
            net.wave(&[0, 1, 2]);
        }
        net.reset_state();
        net.with_layer(1, |l| {
            assert!(l.x.iter().all(|&v| v == 0.0) && l.y.iter().all(|&v| v == 0.0) && l.q.iter().all(|&v| v == 0.0));
            assert!(l.pending.iter().all(|&p| p == 0));
        });
    }
}
