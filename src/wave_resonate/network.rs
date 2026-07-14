//! `network` — owns the BRF layer stack and drives each wave: process every layer (dense membrane +
//! firer-gated delivery), route generated deliveries one hop, swap into each layer's `pending` at wave
//! end (deferred propagation). L0 is the transducer; the last layer is either compute or a readout. When
//! training, after each wave it accrues the online HYPR eligibility (per-synapse 2-state ε^x/ε^y).

use crate::wave_resonate::config::Config;
use crate::wave_resonate::neurons::Layer;
use crate::wave_resonate::synapse::{local_of, wrap, xy_of};
use crate::wave_resonate::training::{Edge, EligParams};
use crate::wave_resonate::wave::process_layer;

pub struct Network {
    size: u32,
    layers: Vec<Layer>,
    wave_id: u32,
    fired: Vec<u32>,
    deliv: Vec<Vec<i32>>, // per layer: NEXT wave's incoming accumulator
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
    // --- training state (buffers always allocated; used only while training) ---
    elig_params: EligParams,
    entries: Vec<Vec<Edge>>,           // per layer: topology edges (source-layer view)
    fired_by_layer: Vec<Vec<u32>>,     // this wave's firers per layer (captured during the sweep)
    prev_fired_bitset: Vec<Vec<u64>>,  // PREVIOUS wave's firers per layer (source injection z_i^{t−1})
    cur_fired_bitset: Vec<Vec<u64>>,   // THIS wave's firers per layer (keeps a just-fired source alive one wave)
    elig_active: Vec<Vec<u32>>,        // per layer: sources with a live ε trace (accrual scan set)
    elig_mark: Vec<Vec<u64>>,          // dedup bitset for elig_active
    dirty_rows: Vec<Vec<u32>>,         // per layer: sources whose elig row got accrual (drives dfa_update)
    dirty_mark: Vec<Vec<u64>>,         // dedup bitset for dirty_rows
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
        let words = (ls + 63) / 64;
        let mut layers = Vec::with_capacity(l);
        let mut entries = Vec::with_capacity(l);
        for (z, lc) in config.layers.iter().enumerate() {
            let mut layer = Layer::new(lc, config.dt, config.gamma, config.theta_c, config.seed, z as u32, size);
            if z == 0 {
                layer.transducer = true;
            }
            if readout_last && z == l - 1 {
                layer.readout = true;
            }
            entries.push(lc.topology.iter().map(|t| Edge { level: t.level, count: t.count as usize, radius: t.radius }).collect());
            layers.push(layer);
        }
        Network {
            size,
            layers,
            wave_id: 0,
            fired: Vec::new(),
            deliv: (0..l).map(|_| vec![0i32; ls]).collect(),
            listeners: (0..l).map(|_| None).collect(),
            elig_params: EligParams { dt: config.dt, ..EligParams::default() },
            entries,
            fired_by_layer: (0..l).map(|_| Vec::new()).collect(),
            prev_fired_bitset: (0..l).map(|_| vec![0u64; words]).collect(),
            cur_fired_bitset: (0..l).map(|_| vec![0u64; words]).collect(),
            elig_active: (0..l).map(|_| Vec::new()).collect(),
            elig_mark: (0..l).map(|_| vec![0u64; words]).collect(),
            dirty_rows: (0..l).map(|_| Vec::new()).collect(),
            dirty_mark: (0..l).map(|_| vec![0u64; words]).collect(),
        }
    }

    pub fn wave(&mut self, input: &[u32]) {
        let w = self.wave_id;
        self.wave_id = self.wave_id.wrapping_add(1);
        let l = self.layers.len();
        let size = self.size;
        let training = self.is_training();
        {
            let Self { layers, deliv, fired, listeners, fired_by_layer, .. } = self;
            for z in 0..l {
                let inp: &[u32] = if z == 0 { input } else { &[] };
                process_layer(&mut layers[z], z as u32, size, inp, deliv, fired);
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
        }
        if training {
            self.accrue_eligibility();
        }
    }

    /// Accrue the online HYPR eligibility for this wave. Source-driven scan over each layer's active set;
    /// per synapse i→j advance the 2-state ε recursion (TARGET j's b_eff/ψ/ω, SOURCE i's PREVIOUS-wave
    /// spike), accumulate `elig += ψ_j·ε^x`. Canonical order — the dense oracle mirrors it exactly.
    fn accrue_eligibility(&mut self) {
        let l = self.layers.len();
        let size = self.size;
        let dt = self.elig_params.dt;
        let cut = self.elig_params.eps_cut;
        // per-layer read-only TARGET snapshots (b_eff, ψ from train; ω from the layer) — decouples the
        // mutable per-source-layer borrow from the immutable per-target-layer reads.
        let b_snap: Vec<Vec<f32>> = self.layers.iter().map(|lz| lz.train.as_ref().map(|t| t.b_eff.clone()).unwrap_or_default()).collect();
        let psi_snap: Vec<Vec<f32>> = self.layers.iter().map(|lz| lz.train.as_ref().map(|t| t.psi.clone()).unwrap_or_default()).collect();
        let om_snap: Vec<Vec<f32>> = self.layers.iter().map(|lz| lz.omega.clone()).collect();
        let Self { layers, fired_by_layer, prev_fired_bitset, cur_fired_bitset, elig_active, elig_mark, dirty_rows, dirty_mark, .. } = self;

        // 1. add this wave's firers to each layer's active set (dedup) + build cur_fired_bitset
        for z in 0..l {
            for w in cur_fired_bitset[z].iter_mut() {
                *w = 0;
            }
            for &i in &fired_by_layer[z] {
                let w = (i >> 6) as usize;
                let bit = 1u64 << (i & 63);
                cur_fired_bitset[z][w] |= bit;
                if elig_mark[z][w] & bit == 0 {
                    elig_mark[z][w] |= bit;
                    elig_active[z].push(i);
                }
            }
        }

        // 2. scan each source layer's active set, compacting survivors toward the front
        for z in 0..l {
            if layers[z].train.is_none() {
                continue;
            }
            let ts = layers[z].total_slots;
            let mut scan = std::mem::take(&mut elig_active[z]);
            let mut keep = 0usize;
            for r in 0..scan.len() {
                let iu = scan[r];
                let i = iu as usize;
                let src_fired_prev = prev_fired_bitset[z][(iu >> 6) as usize] & (1u64 << (iu & 63)) != 0;
                let inj = if src_fired_prev { dt } else { 0.0 };
                let (sx, sy) = xy_of(iu, size);
                let mut any_live = false;
                let Layer { topology, slot_bases, occ_wpn, occ, offsets, train, .. } = &mut layers[z];
                let tr = train.as_mut().unwrap();
                for (e_idx, entry) in topology.iter().enumerate() {
                    let tz_i = z as i32 + entry.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let (b_t, psi_t, om_t) = (&b_snap[tz], &psi_snap[tz], &om_snap[tz]);
                    let sbase = slot_bases[e_idx];
                    let wpn = occ_wpn[e_idx];
                    let words = &occ[e_idx][i * wpn..i * wpn + wpn];
                    let lut = &offsets[e_idx];
                    let mut rank = 0usize;
                    for (wi, &w0) in words.iter().enumerate() {
                        let mut word = w0;
                        let cbase = wi * 64;
                        while word != 0 {
                            let bit = word.trailing_zeros() as usize;
                            let cell = cbase + bit;
                            let (dx, dy) = lut[cell];
                            let j = local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize;
                            let widx = i * ts + sbase + rank;
                            let ex = tr.eps_x[widx];
                            let ey = tr.eps_y[widx];
                            let coef = 1.0 + dt * b_t[j];
                            let mut nex = coef * ex - dt * om_t[j] * ey + inj;
                            let mut ney = dt * om_t[j] * ex + coef * ey;
                            if nex.abs() < cut {
                                nex = 0.0;
                            }
                            if ney.abs() < cut {
                                ney = 0.0;
                            }
                            tr.eps_x[widx] = nex;
                            tr.eps_y[widx] = ney;
                            if psi_t[j] != 0.0 && nex != 0.0 {
                                tr.elig[widx] += psi_t[j] * nex;
                                let dw = (iu >> 6) as usize;
                                let db = 1u64 << (iu & 63);
                                if dirty_mark[z][dw] & db == 0 {
                                    dirty_mark[z][dw] |= db;
                                    dirty_rows[z].push(iu);
                                }
                            }
                            if nex != 0.0 || ney != 0.0 {
                                any_live = true;
                            }
                            rank += 1;
                            word &= word - 1;
                        }
                    }
                }
                // Keep a source while it can still contribute: a live ε trace, OR it fired THIS wave (its
                // injection δ·z_i lands NEXT wave — dropping it now would lose that injection, the bug the
                // dense oracle exposed). A source that fired last wave is already covered: `inj` was applied
                // this wave, so its ε is now live (`any_live`).
                let fired_now = cur_fired_bitset[z][(iu >> 6) as usize] & (1u64 << (iu & 63)) != 0;
                if any_live || fired_now {
                    scan[keep] = iu;
                    keep += 1;
                } else {
                    elig_mark[z][(iu >> 6) as usize] &= !(1u64 << (iu & 63));
                }
            }
            scan.truncate(keep);
            elig_active[z] = scan;
        }

        // 3. roll prev_fired_bitset ← this wave's firers (source injection z_i^{t−1} for next wave)
        std::mem::swap(prev_fired_bitset, cur_fired_bitset);
    }

    /// Apply one multi-layer-DFA update from the accumulated eligibility: for each trainable edge
    /// (`tz = z + level ∈ [1, L)`), `shadow[i,edge,r] += −lr·signal[tz][j]·elig[i,edge,r]` over the dirty
    /// rows, then repack each touched row. Targets decoded from the occupancy (inlined).
    pub fn dfa_update(&mut self, entries: &[Vec<Edge>], signal: &[Vec<f32>], lr: f32) {
        let size = self.size;
        let l = self.layers.len();
        let Self { layers, dirty_rows, .. } = self;
        for z in 0..l {
            if layers[z].train.is_none() {
                continue;
            }
            for ri in 0..dirty_rows[z].len() {
                let iu = dirty_rows[z][ri];
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

    /// Clear all per-trial training accumulators (elig over dirty rows, ε traces over the active set,
    /// spike_count densely) and the per-wave work-sets. Called by `reset_state`.
    pub fn reset_eligibility(&mut self) {
        let l = self.layers.len();
        let Self { layers, elig_active, elig_mark, dirty_rows, dirty_mark, prev_fired_bitset, cur_fired_bitset, fired_by_layer, .. } = self;
        for z in 0..l {
            let ts = layers[z].total_slots;
            if let Some(t) = layers[z].train.as_mut() {
                for &i in &dirty_rows[z] {
                    let base = i as usize * ts;
                    for s in 0..ts {
                        t.elig[base + s] = 0.0;
                    }
                }
                for &i in &elig_active[z] {
                    let base = i as usize * ts;
                    for s in 0..ts {
                        t.eps_x[base + s] = 0.0;
                        t.eps_y[base + s] = 0.0;
                    }
                }
                t.spike_count.iter_mut().for_each(|c| *c = 0);
                t.g_om_x.iter_mut().for_each(|v| *v = 0.0);
                t.g_om_y.iter_mut().for_each(|v| *v = 0.0);
                t.g_bo_x.iter_mut().for_each(|v| *v = 0.0);
                t.g_bo_y.iter_mut().for_each(|v| *v = 0.0);
                t.om_grad.iter_mut().for_each(|v| *v = 0.0);
                t.bo_grad.iter_mut().for_each(|v| *v = 0.0);
            }
            elig_active[z].clear();
            for w in elig_mark[z].iter_mut() {
                *w = 0;
            }
            dirty_rows[z].clear();
            for w in dirty_mark[z].iter_mut() {
                *w = 0;
            }
            for w in prev_fired_bitset[z].iter_mut() {
                *w = 0;
            }
            for w in cur_fired_bitset[z].iter_mut() {
                *w = 0;
            }
            fired_by_layer[z].clear();
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
        self.reset_eligibility();
    }

    pub fn enable_training(&mut self) {
        let tob = self.elig_params.train_omega_b;
        for l in self.layers.iter_mut() {
            l.enable_training();
            l.train_omega_b = tob;
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
        for l in self.layers.iter_mut() {
            l.train_omega_b = p.train_omega_b;
        }
    }

    /// Apply one DFA update to the per-neuron BRF params: for each compute layer `z` (train_omega_b, not
    /// transducer/readout), `ω[j] += −lr·signal[z][j]·om_grad[j]` (clamped to keep `δ·ω ≤ 1`), and
    /// `b′[j] += −lr·signal[z][j]·bo_grad[j]` (clamped `≥ 0`). No-op when `train_omega_b` is off (grads 0).
    pub fn omega_b_update(&mut self, signal: &[Vec<f32>], lr: f32) {
        let l = self.layers.len();
        for z in 0..l {
            let Layer { omega, b_off, train, transducer, readout, train_omega_b, dt, .. } = &mut self.layers[z];
            if !*train_omega_b || *transducer || *readout {
                continue;
            }
            let Some(t) = train.as_ref() else { continue };
            let om_hi = 0.99 / *dt;
            for j in 0..omega.len() {
                let s = signal[z][j];
                omega[j] = (omega[j] - lr * s * t.om_grad[j]).clamp(0.5, om_hi);
                b_off[j] = (b_off[j] - lr * s * t.bo_grad[j]).max(0.0);
            }
        }
    }
    /// The per-layer topology edges (source-layer view), built at construction. Convenience for callers
    /// that drive `dfa_update` (the DFA credit wiring lines up index-for-index with these).
    pub fn entries(&self) -> &[Vec<Edge>] {
        &self.entries
    }
    /// Per-neuron spike count accumulated since the last reset (for rate_reg / liveness). Requires training.
    pub fn layer_spike_count(&self, z: usize) -> &[u32] {
        &self.layers[z].train.as_ref().expect("layer_spike_count requires training enabled").spike_count
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
        let mut net = Network::new(three_layer(16));
        for w in 0..60 {
            net.wave(if w % 2 == 0 { &[0, 1, 2, 16, 17, 18, 32, 33] } else { &[] });
        }
        let any = net.with_layer(1, |l| l.x.iter().any(|&v| v != 0.0) || l.y.iter().any(|&v| v != 0.0));
        assert!(any, "layer 1 developed oscillator activity from L0 drive");
    }

    #[test]
    fn readout_never_fires_but_integrates() {
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

#[cfg(test)]
mod training_tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;

    fn two_layer(size: u32) -> Config {
        let up = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 2, count: 8 }], inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0 };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        Config { seed: 9, size, dt: 0.05, gamma: 0.9, theta_c: 1.0, layers: vec![up, top] }
    }

    #[test]
    fn training_toggles() {
        let mut net = Network::new(two_layer(8));
        assert!(!net.is_training());
        net.enable_training();
        assert!(net.is_training());
        net.with_layer(0, |l| assert_eq!(l.train.as_ref().unwrap().shadow.len(), l.synapse_count()));
        net.disable_training();
        assert!(!net.is_training());
    }

    #[test]
    fn accrual_is_deterministic_and_nonzero() {
        let cfg = two_layer(8);
        let run = || {
            let mut net = Network::new(cfg.clone());
            net.enable_training();
            for _ in 0..40 {
                net.wave(&[0, 1, 2, 8, 9, 10]);
            }
            net.with_layer(0, |l| l.train.as_ref().unwrap().elig.clone())
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "deterministic elig");
        assert!(a.iter().any(|&e| e != 0.0), "some L0→L1 eligibility accrued once L1 fires");
    }

    #[test]
    fn dfa_update_with_negative_signal_moves_eligible_shadow() {
        let cfg = two_layer(8);
        let entries = vec![vec![Edge { level: 1, count: 8, radius: 2 }], vec![]];
        let mut net = Network::new(cfg);
        net.enable_training();
        for _ in 0..40 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        let ls = 64usize;
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
        let before: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow.iter().sum());
        net.dfa_update(&entries, &signal, 0.05);
        let after: f32 = net.with_layer(0, |l| l.train.as_ref().unwrap().shadow.iter().sum());
        assert!(after != before, "a negative signal × accrued eligibility moves the shadow: {before}->{after}");
    }

    #[test]
    fn omega_b_frozen_when_disabled() {
        let mut net = Network::new(two_layer(8));
        net.enable_training(); // train_omega_b default false
        let before = net.with_layer(1, |l| (l.omega.clone(), l.b_off.clone()));
        for _ in 0..30 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        let ls = 64usize;
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
        net.omega_b_update(&signal, 0.5);
        let after = net.with_layer(1, |l| (l.omega.clone(), l.b_off.clone()));
        assert_eq!(before, after, "ω/b′ frozen when train_omega_b=false");
    }

    #[test]
    fn omega_b_train_moves_params_within_clamp() {
        let mut net = Network::new(two_layer(8));
        net.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: true });
        net.enable_training();
        let before = net.with_layer(1, |l| l.omega.clone());
        for _ in 0..30 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        let ls = 64usize;
        let signal = vec![vec![0f32; ls], vec![-1.0f32; ls]];
        net.omega_b_update(&signal, 5.0);
        net.with_layer(1, |l| {
            assert!(l.omega != before, "ω moves when trained");
            assert!(l.omega.iter().all(|&w| w >= 0.5 && w <= 0.99 / 0.05), "ω stays within δω≤1 clamp");
            assert!(l.b_off.iter().all(|&b| b >= 0.0), "b′ stays ≥ 0");
        });
    }

    #[test]
    fn reset_clears_param_eligibility() {
        let mut net = Network::new(two_layer(8));
        net.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: true });
        net.enable_training();
        for _ in 0..20 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        net.reset_state();
        net.with_layer(1, |l| {
            let t = l.train.as_ref().unwrap();
            assert!(t.om_grad.iter().all(|&v| v == 0.0) && t.g_om_x.iter().all(|&v| v == 0.0));
        });
    }

    #[test]
    fn reset_eligibility_clears_accumulators() {
        let mut net = Network::new(two_layer(8));
        net.enable_training();
        for _ in 0..20 {
            net.wave(&[0, 1, 2, 8, 9, 10]);
        }
        net.reset_state();
        net.with_layer(0, |l| {
            let t = l.train.as_ref().unwrap();
            assert!(t.elig.iter().all(|&e| e == 0.0));
            assert!(t.eps_x.iter().all(|&e| e == 0.0) && t.eps_y.iter().all(|&e| e == 0.0));
            assert!(t.spike_count.iter().all(|&c| c == 0));
        });
    }
}
