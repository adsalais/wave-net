//! `network` — owns the layer stack and drives each wave. Sparse mode processes only per-layer
//! frontiers and rebuilds them; dense mode processes all neurons (the equivalence oracle). Deliveries
//! are deferred one hop: generated into `deliv`, swapped into each layer's `pending` at wave end.

use crate::wave_driven::config::Config;
use crate::wave_driven::frontier::Frontier;
use crate::wave_driven::neurons::Layer;
use crate::wave_driven::wave::{process_layer, Work};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Sparse,
    Dense,
}

pub struct Network {
    size: u32,
    layers: Vec<Layer>,
    wave_id: u32,
    mode: Mode,
    fired: Vec<u32>,
    deliv: Vec<Vec<i32>>,         // per layer: NEXT wave's incoming accumulator
    frontier: Vec<Frontier>,      // per layer: current worklist (sparse only)
    frontier_next: Vec<Frontier>, // per layer: worklist being built (sparse only)
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

impl Network {
    pub fn new(config: Config) -> Network {
        Network::build(config, false, Mode::Sparse)
    }
    pub fn new_with_readout(config: Config) -> Network {
        Network::build(config, true, Mode::Sparse)
    }
    /// Dense oracle build (processes all neurons every wave; no readout). For equivalence testing.
    pub fn new_dense(config: Config) -> Network {
        Network::build(config, false, Mode::Dense)
    }

    fn build(config: Config, readout_last: bool, mode: Mode) -> Network {
        config.validate().expect("invalid config");
        let size = config.size;
        let l = config.layers.len();
        let ls = (size as usize) * (size as usize);
        let mut layers = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let mut layer = Layer::new(lc, config.seed, z as u32, size);
            if z == 0 {
                // L0 transducer: fires only on injection (baseline i16::MAX), never adapts.
                layer.threshold.iter_mut().for_each(|t| *t = i16::MAX);
                layer.adapt_bump = 0;
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
            mode,
            fired: Vec::new(),
            deliv: (0..l).map(|_| vec![0i32; ls]).collect(),
            frontier: (0..l).map(|_| Frontier::new(ls)).collect(),
            frontier_next: (0..l).map(|_| Frontier::new(ls)).collect(),
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&mut self, input: &[u32]) {
        let w = self.wave_id;
        self.wave_id = self.wave_id.wrapping_add(1);
        let l = self.layers.len();
        let size = self.size;
        match self.mode {
            Mode::Dense => {
                let Self { layers, deliv, fired, listeners, .. } = self;
                for z in 0..l {
                    let inp: &[u32] = if z == 0 { input } else { &[] };
                    process_layer(&mut layers[z], z as u32, size, inp, w, Work::Dense, deliv, fired);
                    if let Some(cb) = &listeners[z] {
                        cb(w as usize, fired);
                    }
                }
                for z in 0..l {
                    std::mem::swap(&mut layers[z].pending, &mut deliv[z]);
                }
            }
            Mode::Sparse => {
                // seed L0's current frontier with the injection sites so they are visited this wave
                for &a in input {
                    self.frontier[0].push(a);
                }
                let Self { layers, deliv, fired, frontier, frontier_next, listeners, .. } = self;
                for z in 0..l {
                    let inp: &[u32] = if z == 0 { input } else { &[] };
                    let cur = &frontier[z].list;
                    process_layer(&mut layers[z], z as u32, size, inp, w, Work::Sparse { cur, frontier_next }, deliv, fired);
                    if let Some(cb) = &listeners[z] {
                        cb(w as usize, fired);
                    }
                }
                // deferred one hop: this wave's deliveries become next wave's pending
                for z in 0..l {
                    std::mem::swap(&mut layers[z].pending, &mut deliv[z]);
                }
                // install the freshly built worklists as current; empty the consumed ones for reuse
                std::mem::swap(frontier, frontier_next);
                for f in frontier_next.iter_mut() {
                    f.clear();
                }
            }
        }
    }

    pub fn reset_state(&mut self) {
        for g in self.layers.iter_mut() {
            g.potential.iter_mut().for_each(|p| *p = 0);
            g.cooldown.iter_mut().for_each(|c| *c = 0);
            g.adapt_ref.iter_mut().for_each(|a| *a = 0);
            g.fire_wave.iter_mut().for_each(|f| *f = 0);
            g.pending.iter_mut().for_each(|p| *p = 0);
        }
        for d in self.deliv.iter_mut() {
            d.iter_mut().for_each(|x| *x = 0);
        }
        for f in self.frontier.iter_mut() {
            f.clear();
        }
        for f in self.frontier_next.iter_mut() {
            f.clear();
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
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::synapse::TopologyLevel;

    fn two_layer(size: u32) -> Config {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Config { seed: 9, size, layers: vec![up, top] }
    }

    #[test]
    fn l0_is_forced_transducer() {
        let net = Network::new(two_layer(8));
        net.with_layer(0, |l| {
            assert!(l.threshold.iter().all(|&t| t == i16::MAX), "L0 threshold forced to i16::MAX");
            assert_eq!(l.adapt_bump, 0, "L0 does not adapt");
        });
    }

    #[test]
    fn wave_is_deterministic() {
        let mut a = Network::new(two_layer(8));
        let mut b = Network::new(two_layer(8));
        let inputs: [&[u32]; 6] = [&[0, 1, 2], &[0, 1, 2], &[], &[5, 9], &[], &[1]];
        for inp in inputs {
            a.wave(inp);
            b.wave(inp);
        }
        a.with_layer(1, |la| {
            b.with_layer(1, |lb| {
                assert_eq!(la.potential, lb.potential);
                assert_eq!(la.adapt_ref, lb.adapt_ref);
                assert_eq!(la.fire_wave, lb.fire_wave);
            })
        });
    }

    #[test]
    fn readout_integrates_without_firing() {
        // Last layer is a drain-only readout: it accumulates potential and never fires.
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        let cfg = Config { seed: 4, size: 8, layers: vec![up.clone(), LayerConfig { topology: vec![], ..up }] };
        let mut net = Network::new_with_readout(cfg);
        let fired_top = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let ft = fired_top.clone();
        net.on_layer(1, Box::new(move |_w, fired| *ft.lock().unwrap() += fired.len()));
        for _ in 0..12 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        assert_eq!(*fired_top.lock().unwrap(), 0, "readout never fires");
        let any_pot = net.with_layer(1, |l| l.potential.iter().any(|&p| p != 0));
        assert!(any_pot, "readout integrated some potential");
    }
}
