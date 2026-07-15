//! `network` — owns the layer stack and drives each wave. Sparse mode processes only per-layer
//! frontiers and rebuilds them; dense mode processes all neurons (the equivalence oracle). Deliveries
//! are deferred one hop: generated into `deliv`, swapped into each layer's `pending` at wave end.

use crate::wave_driven::config::Config;
use crate::wave_driven::frontier::Frontier;
use crate::wave_driven::neurons::Layer;
use crate::wave_driven::synapse::{local_of, wrap, xy_of};
use crate::wave_driven::training::{Edge, EligParams};
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
    elig_active: Vec<Frontier>,   // per layer: sources fired since reset (εᵃ scan set, β≠0)
    elig_params: EligParams,
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

impl Network {
    pub fn new(config: Config) -> Network {
        Network::build(config, Mode::Sparse)
    }
    /// Dense oracle build (processes all neurons every wave). For equivalence testing.
    pub fn new_dense(config: Config) -> Network {
        Network::build(config, Mode::Dense)
    }

    fn build(config: Config, mode: Mode) -> Network {
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
            elig_active: (0..l).map(|_| Frontier::new(ls)).collect(),
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
        let alloc = self.elig_params.elig_beta != 0.0;
        for l in self.layers.iter_mut() {
            l.enable_training(alloc);
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
        let beta = self.elig_params.elig_beta;
        let eps_a_cut = self.elig_params.epsilon_a;
        let use_ea = beta != 0.0;
        // ρ per layer = 1 − 2^(−adapt_decay); εᵃ decays at the TARGET layer's ρ.
        let rho: Vec<f32> = self.layers.iter().map(|lz| 1.0 - 2f32.powi(-(lz.adapt_decay as i32))).collect();
        let Self { layers, fired_by_layer, fired_bitset, pretr_active, dirty_rows, elig_active, .. } = self;

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
        if use_ea {
            for z in 0..l {
                for &j in &fired_by_layer[z] {
                    elig_active[z].push(j);
                }
            }
        }

        // 3. accrual
        if !use_ea {
            // Phase 2a: membrane, over pretr_active
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
        } else {
            // εᵃ: over elig_active, with the adaptation-eligibility recursion (spike-ψ). Prune the scan
            // set: a source with no live presynaptic trace AND an all-zero εᵃ row contributes exactly 0
            // on every future wave until it fires again (which re-adds it) — so drop it here instead of
            // letting elig_active grow monotonically per trial. Bit-exact vs the dense oracle.
            for z in 0..l {
                if layers[z].train.is_none() {
                    continue;
                }
                let ts = layers[z].total_slots;
                // Scan in place, compacting survivors toward the front (write pointer `keep`); the inner
                // loop stays byte-identical to the un-pruned version — survival is decided per row after.
                let mut scan = std::mem::take(&mut elig_active[z].list);
                let Layer { topology, slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
                let tr = train.as_mut().unwrap();
                let mut keep = 0usize;
                for r in 0..scan.len() {
                    let iu = scan[r];
                    let i = iu as usize;
                    let pr = tr.pretr[i]; // 0 if the presynaptic trace already decayed (silent-source coupling)
                    let (sx, sy) = xy_of(iu, size);
                    for (e_idx, entry) in topology.iter().enumerate() {
                        let tz_i = z as i32 + entry.level;
                        if tz_i < 0 || tz_i as usize >= l {
                            continue;
                        }
                        let tz = tz_i as usize;
                        let r_tz = rho[tz];
                        let sbase = slot_bases[e_idx];
                        let wpn = occ_wpn[e_idx];
                        // SAFETY: e_idx < topology.len() == occ.len() == offsets.len(); i < ls and
                        // occ[e_idx].len() == ls*wpn, so [i*wpn, i*wpn+wpn) is in bounds; offsets[e_idx]
                        // is this entry's neighborhood LUT (len == cell count). Mirrors the sanctioned
                        // unsafe in wave_bitnet::process_layer (same word-scan invariants).
                        let words = unsafe { occ.get_unchecked(e_idx).get_unchecked(i * wpn..i * wpn + wpn) };
                        let lut = unsafe { offsets.get_unchecked(e_idx) };
                        let fb = &fired_bitset[tz];
                        let mut rank = 0usize;
                        for (wi, &w0) in words.iter().enumerate() {
                            let mut word = w0;
                            let cbase = wi * 64;
                            while word != 0 {
                                let bit = word.trailing_zeros() as usize;
                                let cell = cbase + bit;
                                // SAFETY: `cell` is a SET occupancy bit => a sampled cell < lut.len()
                                // (padding bits are never set). widx = i*ts + sbase + rank, rank < count,
                                // sbase+count <= ts, i < ls => widx < ls*ts == eps_a.len() == elig.len().
                                // j = local_of(wrap,..) < ls => j>>6 < fb.len() (ceil(ls/64) words).
                                let (dx, dy) = unsafe { *lut.get_unchecked(cell) };
                                let j = local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size);
                                let widx = i * ts + sbase + rank;
                                let ea = unsafe { *tr.eps_a.get_unchecked(widx) };
                                let fired = unsafe { *fb.get_unchecked((j >> 6) as usize) } & (1u64 << (j & 63)) != 0;
                                let new_ea = if fired {
                                    unsafe { *tr.elig.get_unchecked_mut(widx) += pr - beta * ea };
                                    dirty_rows[z].push(iu);
                                    pr + (r_tz - beta) * ea
                                } else {
                                    r_tz * ea
                                };
                                unsafe { *tr.eps_a.get_unchecked_mut(widx) = if new_ea.abs() < eps_a_cut { 0.0 } else { new_ea } };
                                rank += 1;
                                word &= word - 1;
                            }
                        }
                    }
                    // Keep the source only while it can still contribute: a live presynaptic trace (can
                    // inject on a future target-fire), or any still-live εᵃ slot. The row's εᵃ slots are
                    // contiguous and out-of-range entries stay 0, so scanning the whole row is exact; the
                    // `pr != 0` short-circuit means the hot path (live trace) pays nothing for the check.
                    let survive = pr != 0.0 || tr.eps_a[i * ts..i * ts + ts].iter().any(|&e| e != 0.0);
                    if survive {
                        scan[keep] = iu;
                        keep += 1;
                    } else {
                        elig_active[z].mark[(iu >> 6) as usize] &= !(1u64 << (iu & 63));
                    }
                }
                scan.truncate(keep);
                elig_active[z].list = scan;
            }
        }

        // 4. clear this wave's fired bitset for reuse next wave
        for z in 0..l {
            for &j in &fired_by_layer[z] {
                fired_bitset[z][(j >> 6) as usize] &= !(1u64 << (j & 63));
            }
        }
    }

    /// Apply one multi-layer-DFA update from the accumulated eligibility: for each trainable edge
    /// (`tz = z + level ∈ [1, L)`), `shadow[i,edge,r] += −lr·signal[tz][j]·elig[i,edge,r]` over the
    /// dirty rows, then repack each touched row. Targets decoded from the occupancy (inlined).
    pub fn dfa_update(&mut self, entries: &[Vec<Edge>], signal: &[Vec<f32>], lr: f32) {
        let size = self.size;
        let l = self.layers.len();
        let Self { layers, dirty_rows, .. } = self;
        for z in 0..l {
            if layers[z].train.is_none() {
                continue;
            }
            for ri in 0..dirty_rows[z].list.len() {
                let iu = dirty_rows[z].list[ri];
                let i = iu as usize;
                let mut touched = false;
                {
                    let ts = layers[z].total_slots;
                    let Layer { slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
                    let tr = train.as_mut().unwrap();
                    let (sx, sy) = xy_of(iu, size);
                    for (e_idx, edge) in entries[z].iter().enumerate() {
                        let tz_i = z as i32 + edge.level;
                        if tz_i < 1 || tz_i as usize >= l {
                            continue;
                        }
                        let tz = tz_i as usize;
                        let sbase = slot_bases[e_idx];
                        let wpn = occ_wpn[e_idx];
                        let words = &occ[e_idx][i * wpn..i * wpn + wpn];
                        let mut rank = 0usize;
                        for (wi, &w0) in words.iter().enumerate() {
                            let mut word = w0;
                            let cbase = wi * 64;
                            while word != 0 {
                                let bit = word.trailing_zeros() as usize;
                                let cell = cbase + bit;
                                let (dx, dy) = offsets[e_idx][cell];
                                let j = local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize;
                                let widx = i * ts + sbase + rank;
                                let e = tr.elig[widx];
                                if e != 0.0 {
                                    tr.shadow[widx] += -lr * signal[tz][j] * e;
                                    touched = true;
                                }
                                rank += 1;
                                word &= word - 1;
                            }
                        }
                    }
                }
                if touched {
                    layers[z].repack_row(i);
                }
            }
        }
    }

    /// Clear all per-trial training accumulators (elig over dirty rows, pretr over the active set,
    /// spike_count densely) and the per-wave work-sets. Called by `reset_state`.
    pub fn reset_eligibility(&mut self) {
        let l = self.layers.len();
        let Self { layers, pretr_active, dirty_rows, elig_active, fired_by_layer, fired_bitset, .. } = self;
        for z in 0..l {
            let ts = layers[z].total_slots;
            if let Some(t) = layers[z].train.as_mut() {
                for &i in &dirty_rows[z].list {
                    let base = i as usize * ts;
                    for s in 0..ts {
                        t.elig[base + s] = 0.0;
                    }
                }
                if !t.eps_a.is_empty() {
                    for &i in &elig_active[z].list {
                        let base = i as usize * ts;
                        for s in 0..ts {
                            t.eps_a[base + s] = 0.0;
                        }
                    }
                }
                for &i in &pretr_active[z].list {
                    t.pretr[i as usize] = 0.0;
                }
                t.spike_count.iter_mut().for_each(|c| *c = 0);
            }
            dirty_rows[z].clear();
            pretr_active[z].clear();
            elig_active[z].clear();
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

    /// Size of the εᵃ scan set (sources whose row is still accrued each wave). Exposed for the pruning
    /// tests: a fully-decayed set should shrink to 0, not grow monotonically with cumulative activity.
    #[cfg(test)]
    pub(crate) fn elig_active_len(&self, z: usize) -> usize {
        self.elig_active[z].len()
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
    fn dfa_update_with_negative_signal_raises_eligible_synapse() {
        let cfg = {
            let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 3, adapt_bump: 0, adapt_decay: 6 };
            let top = LayerConfig { topology: vec![], ..up.clone() };
            Config { seed: 5, size: 8, layers: vec![up, top] }
        };
        let entries = vec![vec![Edge { level: 1, count: 8, radius: 2 }], vec![]];
        let mut net = Network::new(cfg);
        net.enable_training();
        // zero L0 row 0's shadow then repack -> fully pruned
        net.with_layer_mut_test(0, |l| {
            let ts = l.total_slots;
            for s in 0..ts {
                l.train.as_mut().unwrap().shadow[s] = 0.0;
            }
            l.repack_row(0);
        });
        net.reset_state();
        for _ in 0..12 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        let ls = 64usize;
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
        let before: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow[0..l.total_slots].iter().sum());
        net.dfa_update(&entries, &signal, 0.05);
        let after: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow[0..l.total_slots].iter().sum());
        assert!(after > before, "negative target signal + positive eligibility raises L0 row-0 shadow: {before}->{after}");
    }

    #[test]
    fn eps_a_accrual_changes_elig_and_is_deterministic() {
        // Same net/input trained at β=0 vs β=0.4 must produce DIFFERENT elig (εᵃ has an effect),
        // and two β=0.4 builds must match (determinism).
        let cfg = {
            let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: 0, radius: 1, count: 3 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 3, adapt_bump: 5, adapt_decay: 6 };
            let top = LayerConfig { topology: vec![], ..up.clone() };
            Config { seed: 21, size: 8, layers: vec![up, top] }
        };
        let run = |beta: f32| {
            let mut net = Network::new(cfg.clone());
            net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0, elig_beta: beta, epsilon_a: 1.0 / 1024.0 });
            net.enable_training();
            for _ in 0..16 {
                net.wave(&[0, 1, 2, 8, 9, 10]);
            }
            net.with_layer(0, |l| l.train.as_ref().unwrap().elig.clone())
        };
        let b0 = run(0.0);
        let b4a = run(0.4);
        let b4b = run(0.4);
        assert_eq!(b4a, b4b, "β=0.4 deterministic");
        assert_ne!(b0, b4a, "εᵃ (β=0.4) changes the eligibility vs β=0");
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
