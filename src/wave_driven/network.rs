//! `network` — owns the layer stack and drives each wave. Sparse mode processes only per-layer
//! frontiers and rebuilds them; dense mode processes all neurons (the equivalence oracle). Deliveries
//! are deferred one hop: generated into `deliv`, swapped into each layer's `pending` at wave end.

use crate::wave_driven::config::Config;
use crate::wave_driven::frontier::Frontier;
use crate::wave_driven::neurons::Layer;
use crate::wave_driven::synapse::{local_of, wrap, xy_of};
use crate::wave_driven::training::EligParams;
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
    fired_by_layer: Vec<Vec<u32>>, // this wave's fired ids per layer (captured during wave, training only)
    fired_bitset: Vec<Vec<u64>>,  // per layer: "did neuron fire this wave" (ceil(ls/64) words)
    pretr_active: Vec<Frontier>,  // per layer: sources with a live presynaptic trace
    dirty_rows: Vec<Frontier>,    // per layer: source neurons whose elig row got accrual
    elig_params: EligParams,
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
            fired_by_layer: (0..l).map(|_| Vec::new()).collect(),
            fired_bitset: (0..l).map(|_| vec![0u64; (ls + 63) / 64]).collect(),
            pretr_active: (0..l).map(|_| Frontier::new(ls)).collect(),
            dirty_rows: (0..l).map(|_| Frontier::new(ls)).collect(),
            elig_params: EligParams::default(),
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
                let training = self.is_training();
                let Self { layers, deliv, fired, frontier, frontier_next, fired_by_layer, listeners, .. } = self;
                for z in 0..l {
                    let inp: &[u32] = if z == 0 { input } else { &[] };
                    let cur = &frontier[z].list;
                    process_layer(&mut layers[z], z as u32, size, inp, w, Work::Sparse { cur, frontier_next }, deliv, fired);
                    if training {
                        fired_by_layer[z].clear();
                        fired_by_layer[z].extend_from_slice(fired);
                    }
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
        if self.mode == Mode::Sparse && self.is_training() {
            self.accrue_eligibility();
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
        self.reset_eligibility();
    }

    pub fn enable_training(&mut self) {
        for l in self.layers.iter_mut() {
            l.enable_training();
        }
    }

    pub fn disable_training(&mut self) {
        for l in self.layers.iter_mut() {
            l.disable_training();
        }
    }

    pub fn is_training(&self) -> bool {
        self.layers.first().map(|l| l.train.is_some()).unwrap_or(false)
    }

    pub fn set_elig_params(&mut self, p: EligParams) {
        self.elig_params = p;
    }

    /// Per-neuron spike count accumulated since the last reset (for rate_reg). Requires training.
    pub fn layer_spike_count(&self, z: usize) -> &[u32] {
        &self.layers[z].train.as_ref().expect("layer_spike_count requires training enabled").spike_count
    }

    /// Accrue membrane spike-ψ eligibility for this wave (source-driven scan). Called after the wave's
    /// layer step, when training. `e_ij += pretr_i` for every synapse whose target fired this wave.
    fn accrue_eligibility(&mut self) {
        let size = self.size;
        let l = self.layers.len();
        let decay = 1.0 - 1.0 / self.elig_params.rec_tau.max(1.0);
        let eps = self.elig_params.epsilon;
        let Self { layers, fired_by_layer, fired_bitset, pretr_active, dirty_rows, .. } = self;

        // 1. fired bitset + spike_count
        for z in 0..l {
            for &j in &fired_by_layer[z] {
                fired_bitset[z][(j >> 6) as usize] |= 1u64 << (j & 63);
            }
            if let Some(t) = layers[z].train.as_mut() {
                for &j in &fired_by_layer[z] {
                    t.spike_count[j as usize] += 1;
                }
            }
        }

        // 2. pretr update: decay -> eps-drop -> bump firers (canonical order; matches the dense oracle)
        for z in 0..l {
            let Some(t) = layers[z].train.as_mut() else { continue };
            let pretr = &mut t.pretr;
            let old: Vec<u32> = std::mem::take(&mut pretr_active[z].list);
            for &i in &old {
                pretr_active[z].mark[(i >> 6) as usize] &= !(1u64 << (i & 63));
            }
            for &i in &old {
                let iu = i as usize;
                pretr[iu] *= decay;
                if pretr[iu] < eps {
                    pretr[iu] = 0.0;
                } else {
                    pretr_active[z].push(i);
                }
            }
            for &j in &fired_by_layer[z] {
                pretr[j as usize] += 1.0;
                pretr_active[z].push(j);
            }
        }

        // 3. accrue: for each source with a live trace, scan its fan-out, add pretr where target fired
        for z in 0..l {
            if layers[z].train.is_none() {
                continue;
            }
            let ts = layers[z].total_slots;
            let Layer { topology, slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
            let tr = train.as_mut().unwrap();
            for &iu in &pretr_active[z].list {
                let i = iu as usize;
                let pr = tr.pretr[i];
                if pr == 0.0 {
                    continue;
                }
                let (sx, sy) = xy_of(iu, size);
                for (e_idx, entry) in topology.iter().enumerate() {
                    let tz_i = z as i32 + entry.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let sbase = slot_bases[e_idx];
                    let wpn = occ_wpn[e_idx];
                    let words = &occ[e_idx][i * wpn..i * wpn + wpn];
                    let fb = &fired_bitset[tz];
                    let mut rank = 0usize;
                    for (wi, &w0) in words.iter().enumerate() {
                        let mut word = w0;
                        let cbase = wi * 64;
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            let cell = cbase + bit;
                            let (dx, dy) = offsets[e_idx][cell];
                            let j = local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size);
                            if fb[(j >> 6) as usize] & (1u64 << (j & 63)) != 0 {
                                tr.elig[i * ts + sbase + rank] += pr;
                                dirty_rows[z].push(iu);
                            }
                            rank += 1;
                            word &= word - 1;
                        }
                    }
                }
            }
        }

        // 4. clear this wave's fired bitset for reuse next wave
        for z in 0..l {
            for &j in &fired_by_layer[z] {
                fired_bitset[z][(j >> 6) as usize] &= !(1u64 << (j & 63));
            }
        }
    }

    /// Clear all per-trial training accumulators (elig over dirty rows, pretr over the active set,
    /// spike_count densely) and the per-wave work-sets. Called by `reset_state`.
    pub fn reset_eligibility(&mut self) {
        let l = self.layers.len();
        let Self { layers, pretr_active, dirty_rows, fired_by_layer, fired_bitset, .. } = self;
        for z in 0..l {
            let ts = layers[z].total_slots;
            if let Some(t) = layers[z].train.as_mut() {
                for &i in &dirty_rows[z].list {
                    let base = i as usize * ts;
                    for s in 0..ts {
                        t.elig[base + s] = 0.0;
                    }
                }
                for &i in &pretr_active[z].list {
                    t.pretr[i as usize] = 0.0;
                }
                t.spike_count.iter_mut().for_each(|c| *c = 0);
            }
            dirty_rows[z].clear();
            pretr_active[z].clear();
            for &j in &fired_by_layer[z] {
                fired_bitset[z][(j >> 6) as usize] &= !(1u64 << (j & 63));
            }
            fired_by_layer[z].clear();
        }
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

    #[cfg(test)]
    pub(crate) fn with_layer_mut_test<R>(&mut self, z: usize, f: impl FnOnce(&mut Layer) -> R) -> R {
        f(&mut self.layers[z])
    }

    #[cfg(test)]
    pub(crate) fn seed_worksets_test(&mut self, z: usize, i: u32) {
        self.dirty_rows[z].push(i);
        self.pretr_active[z].push(i);
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

    #[test]
    fn training_toggles_and_reports() {
        let mut net = Network::new(two_layer(8));
        assert!(!net.is_training());
        net.enable_training();
        assert!(net.is_training());
        net.with_layer(0, |l| assert_eq!(l.train.as_ref().unwrap().shadow.len(), l.synapse_count()));
        net.disable_training();
        assert!(!net.is_training());
    }

    #[test]
    fn accrual_marks_eligibility_and_is_deterministic() {
        let cfg = {
            let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 3, adapt_bump: 0, adapt_decay: 6 };
            let top = LayerConfig { topology: vec![], ..up.clone() };
            Config { seed: 21, size: 8, layers: vec![up, top] }
        };
        let mut a = Network::new(cfg.clone());
        let mut b = Network::new(cfg);
        a.enable_training();
        b.enable_training();
        for _ in 0..12 {
            a.wave(&[0, 1, 2, 8, 9, 10]);
            b.wave(&[0, 1, 2, 8, 9, 10]);
        }
        a.with_layer(0, |la| {
            b.with_layer(0, |lb| {
                assert_eq!(la.train.as_ref().unwrap().elig, lb.train.as_ref().unwrap().elig, "deterministic elig");
            })
        });
        let any = a.with_layer(0, |l| l.train.as_ref().unwrap().elig.iter().any(|&e| e > 0.0));
        assert!(any, "some L0->L1 eligibility accrued once L1 neurons fire");
    }

    #[test]
    fn reset_eligibility_clears_accumulators() {
        let mut net = Network::new(two_layer(8));
        net.enable_training();
        net.with_layer_mut_test(0, |l| {
            let t = l.train.as_mut().unwrap();
            t.elig[0] = 5.0;
            t.pretr[0] = 2.0;
            t.spike_count[0] = 7;
        });
        net.seed_worksets_test(0, 0); // register neuron 0 as dirty + pretr-active
        net.reset_eligibility();
        net.with_layer(0, |l| {
            let t = l.train.as_ref().unwrap();
            assert_eq!(t.elig[0], 0.0);
            assert_eq!(t.pretr[0], 0.0);
            assert!(t.spike_count.iter().all(|&c| c == 0));
        });
    }
}
