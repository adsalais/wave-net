//! `network` — owns the layer stack, drives each wave, routes generated synapses into target inboxes.
//! Mirrors `wave_net::network`, minus the runtime seed (targets are materialized in each `Layer`'s
//! occupancy bitset at construction, so no hashing happens at wave time). Adds the shadow-based
//! `eprop_update_synaptic` whose targets are decoded from the occupancy.

use crate::wave_bitnet::config::Config;
use crate::wave_bitnet::neurons::Layer;
use crate::wave_bitnet::wave::process_layer;

pub struct Network {
    size: u32,
    layers: Vec<Layer>,
    wave_id: usize,
    scratch: Scratch,
    record_eligibility: bool,
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

/// Reusable per-wave scratch: `deliveries[z]` accumulates synapses bound for layer `z` next wave; it is
/// disjoint from every `inbox`, so each layer's drained `inbox` is swapped with its `deliveries` at wave end.
struct Scratch {
    fired: Vec<u32>,
    deliv: Vec<Vec<i32>>, // per layer: dense per-target incoming accumulator for the NEXT wave
}

impl Network {
    pub fn new(config: Config) -> Network {
        Network::build(config, false)
    }

    /// Like `new`, but flags the **last** layer as a non-spiking drain-only readout (output sink).
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
                // L0 is the input transducer: forced to fire only on injection (baseline i16::MAX) and to
                // never adapt (adapt_bump 0). Giving L0 adaptation would let it swallow later injections.
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
            scratch: Scratch {
                fired: Vec::new(),
                deliv: (0..l).map(|_| vec![0i32; ls]).collect(),
            },
            record_eligibility: true,
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn wave(&mut self, input: &[u32]) {
        let w = self.wave_id;
        self.wave_id += 1;
        let l = self.layers.len();
        let size = self.size;
        let record_elig = self.record_eligibility;
        let Self { layers, scratch, listeners, .. } = self;
        let Scratch { fired, deliv } = scratch;
        for z in 0..l {
            let inp: &[u32] = if z == 0 { input } else { &[] };
            process_layer(&mut layers[z], z as u32, size, inp, deliv, fired, record_elig);
            if let Some(listener) = &listeners[z] {
                listener(w, fired);
            }
        }
        // Swap each layer's (now drained-and-cleared) `pending` with the accumulator just scattered
        // into for it — so next wave folds this wave's deliveries, and `deliv` returns to all-zeros.
        for i in 0..l {
            std::mem::swap(&mut layers[i].pending, &mut deliv[i]);
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
    /// (inference / throughput) to skip the per-neuron decide/elig snapshots.
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
            g.pending.iter_mut().for_each(|p| *p = 0);
        }
        for d in self.scratch.deliv.iter_mut() {
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

    /// Mutable access to one layer.
    pub(crate) fn with_layer_mut<R>(&mut self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R {
        f(&mut self.layers[z])
    }

    /// Read-only access to the layer stack (used by persistence).
    pub(crate) fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Assemble a `Network` from already-built layers (used by `load_model`). Fresh runtime:
    /// `wave_id = 0`, zeroed delivery scratch, eligibility recording on, no listeners.
    pub(crate) fn from_layers(size: u32, layers: Vec<Layer>) -> Network {
        let l = layers.len();
        let ls = (size as usize) * (size as usize);
        Network {
            size,
            layers,
            wave_id: 0,
            scratch: Scratch { fired: Vec::new(), deliv: (0..l).map(|_| vec![0i32; ls]).collect() },
            record_eligibility: true,
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    /// Read-only access to one layer (introspection).
    pub fn with_layer<R>(&self, z: usize, f: impl FnOnce(&Layer) -> R) -> R {
        f(&self.layers[z])
    }

    /// Per-neuron membrane potential captured at the last decide step (pre fire-reset/leak).
    pub fn layer_decide_potential(&self, z: usize) -> Vec<i16> {
        self.layers[z].decide_potential.clone()
    }

    /// Per-neuron effective firing threshold captured at the last decide step (the value compared
    /// against `layer_decide_potential`).
    pub fn layer_decide_effective_threshold(&self, z: usize) -> Vec<i32> {
        self.layers[z].decide_eff.clone()
    }

    /// Apply one e-prop update to layer `source_z`'s `level_idx` topology entry from a per-synapse
    /// eligibility `elig` (indexed `[i*count + r]`, r = wired-synapse rank) and per-target `signal`:
    /// `shadow[i*total_slots + slot_base + r] += -lr·signal[target]·elig[i*count+r]`, then repack each
    /// touched row. Targets are decoded from the occupancy bitset (no hash). No-op if the target layer
    /// is off-stack or into L0 (`tz ∉ [1, L−1]`).
    pub fn eprop_update_synaptic(&mut self, source_z: usize, level_idx: usize, elig: &[f32], signal: &[f32], lr: f32) {
        let size = self.size();
        let ls = (size as usize) * (size as usize);
        let l = self.layer_count();
        self.with_layer_mut(source_z, |lz| {
            let entry = lz.topology[level_idx].clone();
            let tz = source_z as i32 + entry.level;
            if tz < 1 || tz as usize >= l {
                return;
            }
            let count = entry.count as usize;
            let sbase = lz.slot_bases[level_idx];
            let ts = lz.total_slots;
            // (rank, target) for one neuron, word-scanned + decoded once, then applied to the shadow.
            let mut wired: Vec<(usize, usize)> = Vec::with_capacity(count);
            for i in 0..ls {
                wired.clear();
                lz.for_wired(level_idx, i, |r, c| wired.push((r, lz.decode(level_idx, i as u32, c, size) as usize)));
                let mut touched = false;
                for &(r, target) in &wired {
                    let e = elig[i * count + r];
                    if e != 0.0 {
                        touched = true;
                        lz.shadow[i * ts + sbase + r] += -lr * signal[target] * e;
                    }
                }
                if touched {
                    lz.repack_row(i);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::{Config, LayerConfig};
    use crate::wave_bitnet::synapse::TopologyLevel;

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
        for _ in 0..5 {
            a.wave(&[0, 1, 2]);
            b.wave(&[0, 1, 2]);
        }
        a.with_layer(1, |la| {
            b.with_layer(1, |lb| {
                assert_eq!(la.potential, lb.potential);
                assert_eq!(la.shadow, lb.shadow);
            })
        });
    }

    #[test]
    fn update_with_negative_signal_raises_pruned_synapse() {
        let mut net = Network::new(two_layer(8));
        let ls = 64usize;
        // neuron 0, level 0: zero its whole row shadow then repack -> all pruned
        net.with_layer_mut(0, |l| {
            let ts = l.total_slots;
            for s in 0..ts {
                l.shadow[0 * ts + s] = 0.0;
            }
            l.repack_row(0);
            assert_eq!(l.weight_at(0), 0, "row starts fully pruned");
        });
        let count = 8usize;
        let mut elig = vec![0f32; ls * count];
        elig[0 * count + 0] = 1.0;
        let signal = vec![-1.0f32; ls];
        net.eprop_update_synaptic(0, 0, &elig, &signal, 0.02);
        net.with_layer(0, |l| {
            assert!(l.shadow[0] > 0.0, "shadow raised by -lr·(-1)·1 > 0: {}", l.shadow[0]);
        });
    }
}
