//! Wavefront-pipelined int engine. Per-layer *mutable* state lives in `Vec<Mutex<Layer>>`;
//! read-only per-layer config lives beside it in `Vec<LayerCfg>` (a mutex should only guard
//! what mutates). A worker carries one wave up the stack holding a contiguous band of layer
//! guards, and consecutive waves stagger so their footprints never overlap.
//!
//! A wave is a moving window: leak+decay runs on the **leading edge** (`fwd` layers ahead of
//! the decide, so a layer is leaked before any delivery reaches it — the sequential leak-all-
//! then-fold order); decide+apply is the middle; clamp finalizes layers as they
//! **trail out** of the window. The band is the whole window `[s-back, s+fwd]`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::wave_reservoir::hash::{key, mix, P_THRESHOLD};
use crate::wave_reservoir::index::Dims;
use crate::wave_reservoir::config::{IntConfig, IntLevel, RefractoryMode};
use crate::wave_reservoir::wiring::{scatter_layered, LayeredSynapse};

/// One layer's mutable state — the only thing the `Mutex` guards.
struct Layer {
    potential: Vec<i16>,
    cooldown: Vec<u8>,
}

/// One layer's read-only config + precomputed thresholds. Read lock-free via `&self`.
struct LayerCfg {
    topology: Vec<IntLevel>,
    leak_a: u8,
    leak_b: u8,
    refractory: u8,
    p_inh_q16: u32,
    threshold: Vec<i16>,
}

pub struct LayerNet {
    layers: Vec<Mutex<Layer>>,
    cfgs: Vec<LayerCfg>,
    seed: u64,
    dims: Dims,
    l: u32,
    ls: usize,
    sat: i16,
    drop: bool,
    /// global forward reach (max `max_level` over layers, ≥ 0) — the leak-front lead.
    fwd: usize,
    /// per source layer: the contiguous guard band `[lo, hi]` = `[s-back, s+fwd]` clamped.
    band: Vec<(usize, usize)>,
    /// layers to finalize (clamp) after processing each source layer.
    clamp_after: Vec<Vec<usize>>,
    /// optional per-layer spike listener, emitted (in wave order) at that layer's decide.
    #[allow(clippy::type_complexity)]
    listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}

/// `(min_level, max_level)` of a layer's topology; `(0, 0)` for an empty topology.
fn level_range(topology: &[IntLevel]) -> (i32, i32) {
    let min = topology.iter().map(|e| e.level).min().unwrap_or(0);
    let max = topology.iter().map(|e| e.level).max().unwrap_or(0);
    (min, max)
}

impl LayerNet {
    pub fn new(cfg: IntConfig) -> LayerNet {
        assert!(cfg.validate().is_ok(), "invalid config: {:?}", cfg.validate());
        let dims = Dims::new(cfg.w, cfg.h, cfg.l);
        let l = cfg.l as usize;
        let ls = (cfg.w * cfg.h) as usize;
        let last = l as i32 - 1;

        // Global reaches: the band and the leak-front use the max over all layers, so the
        // leading edge is always inside the locked band regardless of per-layer topology.
        let fwd = (0..l).map(|s| level_range(&cfg.layers[s].topology).1.max(0)).max().unwrap_or(0);
        let back = (0..l).map(|s| (-level_range(&cfg.layers[s].topology).0).max(0)).max().unwrap_or(0);
        let fwd = fwd as usize;
        let back = back as usize;
        let band: Vec<(usize, usize)> = (0..l)
            .map(|s| {
                let lo = (s as i32 - back as i32).clamp(0, last) as usize;
                let hi = (s + fwd).min(l - 1);
                (lo, hi)
            })
            .collect();

        // last_source[j] = highest source layer that writes j's potential this wave (its own
        // processing, init = j, or a delivery reaching it); j is finalized once that source is done.
        let mut last_source: Vec<usize> = (0..l).collect();
        for s in 0..l {
            let (min_lvl, max_lvl) = level_range(&cfg.layers[s].topology);
            let dlo = (s as i32 + min_lvl).clamp(0, last) as usize;
            let dhi = (s as i32 + max_lvl).clamp(0, last) as usize;
            for j in dlo..=dhi {
                last_source[j] = last_source[j].max(s);
            }
        }
        let mut clamp_after: Vec<Vec<usize>> = vec![Vec::new(); l];
        for (j, &s) in last_source.iter().enumerate() {
            clamp_after[s].push(j);
        }

        let mut cfgs = Vec::with_capacity(l);
        let mut layers = Vec::with_capacity(l);
        for z in 0..l {
            let lc = &cfg.layers[z];
            let mut threshold = vec![0i16; ls];
            for (local, th) in threshold.iter_mut().enumerate() {
                let global = (z * ls + local) as u32;
                let ht = mix(key(cfg.seed, global, 0, 0, P_THRESHOLD));
                let mask = (1u64 << (lc.spread_log2 as u64 + 1)) - 1;
                let off = 1i32 << lc.spread_log2;
                *th = (lc.threshold_base + ((ht & mask) as i32 - off)) as i16;
            }
            cfgs.push(LayerCfg {
                topology: lc.topology.clone(),
                leak_a: lc.leak_a,
                leak_b: lc.leak_b,
                refractory: lc.refractory,
                p_inh_q16: lc.p_inh_q16,
                threshold,
            });
            layers.push(Mutex::new(Layer { potential: vec![0; ls], cooldown: vec![0u8; ls] }));
        }

        LayerNet {
            layers,
            cfgs,
            seed: cfg.seed,
            dims,
            l: cfg.l,
            ls,
            sat: cfg.saturation as i16,
            drop: cfg.refractory_mode == RefractoryMode::Drop,
            fwd,
            band,
            clamp_after,
            listeners: (0..l).map(|_| None).collect(),
        }
    }

    pub fn n_total(&self) -> usize {
        self.l as usize * self.ls
    }

    /// Subscribe to a layer's spikes: `listener(wave_id, &fired_locals)` is called (in wave
    /// order) each time a wave decides `layer`. Register before `run`/`run_stream`.
    pub fn on_layer(&mut self, layer: usize, listener: Box<dyn Fn(usize, &[u32]) + Send + Sync>) {
        self.listeners[layer] = Some(listener);
    }

    pub fn clear_listeners(&mut self) {
        for l in &mut self.listeners {
            *l = None;
        }
    }

    /// Zero all potential and cooldown (for reruns between trials).
    pub fn reset_state(&self) {
        for layer in &self.layers {
            let mut g = layer.lock().unwrap();
            g.potential.iter_mut().for_each(|p| *p = 0);
            g.cooldown.iter_mut().for_each(|c| *c = 0);
        }
    }

    /// cooldown decay + native-i16 saturating leak + drive, for one layer (the leading edge).
    fn leak_decay(&self, layer: &mut Layer, idx: usize, drive: &[i16]) {
        for c in &mut layer.cooldown {
            *c = c.saturating_sub(1);
        }
        let (la, lb) = (self.cfgs[idx].leak_a, self.cfgs[idx].leak_b);
        let base = idx * self.ls;
        for local in 0..self.ls {
            let p = layer.potential[local];
            layer.potential[local] = (p - (p >> la) - (p >> lb)).saturating_add(drive[base + local]);
        }
    }

    /// clamp to `±saturation` and apply `Drop` zeroing (finalize a trailing layer).
    fn finalize(&self, layer: &mut Layer) {
        for local in 0..self.ls {
            let mut p = layer.potential[local].clamp(-self.sat, self.sat);
            if self.drop && layer.cooldown[local] > 0 {
                p = 0;
            }
            layer.potential[local] = p;
        }
    }

    /// Process one source layer within an already-held band (`guards[i]` == layer `lo+i`):
    /// leak the leading edge, decide+apply layer `s`, finalize its trailing layers. Shared by
    /// the sequential `wave` and the threaded `run`.
    fn process_source(
        &self,
        s: usize,
        wave_id: usize,
        lo: usize,
        guards: &mut [MutexGuard<'_, Layer>],
        leaked_upto: &mut usize,
        drive: &[i16],
        buf: &mut Vec<LayeredSynapse>,
        firers: &mut Vec<u32>,
    ) {
        // 1. leak+decay the leading edge before any delivery reaches those layers
        let lead = (s + self.fwd).min(self.l as usize - 1);
        while *leaked_upto <= lead {
            self.leak_decay(&mut guards[*leaked_upto - lo], *leaked_upto, drive);
            *leaked_upto += 1;
        }

        // 2. decide layer s on its (leaked + already-delivered) snapshot; collect firers locally
        firers.clear();
        {
            let layer = &mut *guards[s - lo];
            let th = &self.cfgs[s].threshold;
            let refractory = self.cfgs[s].refractory;
            for local in 0..self.ls {
                if layer.cooldown[local] == 0 && layer.potential[local] >= th[local] {
                    layer.potential[local] = 0;
                    layer.cooldown[local] = refractory;
                    firers.push(local as u32);
                }
            }
        }

        // emit this layer's spikes to a subscriber (lazy: nothing assembled if unsubscribed)
        if let Some(listener) = &self.listeners[s] {
            listener(wave_id, firers);
        }

        // 3. apply: config is lock-free (self.cfgs) and potential goes through fast slices
        {
            let mut pots: Vec<&mut [i16]> =
                guards.iter_mut().map(|g| g.potential.as_mut_slice()).collect();
            let topo = &self.cfgs[s].topology;
            let p_inh = self.cfgs[s].p_inh_q16;
            for &local in firers.iter() {
                let src = (s * self.ls + local as usize) as u32;
                scatter_layered(src, self.seed, topo, p_inh, &self.dims, buf);
                for syn in buf.iter() {
                    let d = if syn.inhibitory { -1 } else { 1 };
                    pots[syn.target_layer as usize - lo][syn.local as usize] += d;
                }
            }
        }

        // 4. finalize layers whose last writer is s
        for &j in &self.clamp_after[s] {
            self.finalize(&mut guards[j - lo]);
        }
    }

    /// Sequential wavefront: one wave, layers bottom-to-top. Listeners receive `wave_id = 0`
    /// on every call — use `run_stream` when the wave index matters.
    pub fn wave(&self, drive: &[i16]) {
        assert_eq!(drive.len(), self.n_total(), "drive length {} != n_total() {}", drive.len(), self.n_total());
        let l = self.l as usize;
        let mut leaked_upto = 0usize;
        let mut buf = Vec::new();
        let mut firers = Vec::new();
        for s in 0..l {
            let (lo, hi) = self.band[s];
            let mut guards: Vec<_> = (lo..=hi).map(|i| self.layers[i].lock().unwrap()).collect();
            self.process_source(s, 0, lo, &mut guards, &mut leaked_upto, drive, &mut buf, &mut firers);
        }
    }

    /// Threaded wavefront: pipeline `waves` waves across `threads` workers. Each worker carries
    /// one wave up the stack, holding a contiguous band of guards (hand-over-hand, acquired in
    /// increasing layer order → deadlock-free); the locks self-regulate the stagger so waves
    /// never overlap. Bit-identical to `waves` sequential `wave` calls.
    /// Streaming variant: `drive_fn(wave_id, &mut buf)` supplies each wave's drive on demand.
    /// `buf` arrives zeroed with length `n_total()` — write only the entries you need, and keep
    /// the length unchanged (worker buffers are reused across waves; the engine re-zeroes them
    /// so no wave's drive can leak into another). `run` is the constant-drive wrapper.
    pub fn run_stream(
        &self,
        waves: usize,
        threads: usize,
        drive_fn: impl Fn(usize, &mut Vec<i16>) + Sync,
    ) {
        assert!(threads >= 1, "threads must be >= 1");
        let l = self.l as usize;
        let n = self.n_total();
        let next_wave = AtomicUsize::new(0);
        // Waves must ENTER the pipeline (take the bottom band) in wave order: otherwise a
        // lower-numbered wave delayed right after `fetch_add` could be overtaken by a higher
        // one racing for layer 0 → wrong processing order → non-deterministic. `entry`
        // serializes that first acquisition; the band locks preserve the order from there on.
        let entry = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let next_wave = &next_wave;
                let entry = &entry;
                let drive_fn = &drive_fn;
                scope.spawn(move || {
                    let mut buf = Vec::new();
                    let mut firers = Vec::new();
                    let mut drive_buf = vec![0i16; n];
                    loop {
                        let w = next_wave.fetch_add(1, Ordering::Relaxed);
                        if w >= waves {
                            break;
                        }
                        drive_buf.fill(0);
                        drive_fn(w, &mut drive_buf);
                        assert_eq!(drive_buf.len(), n, "drive_fn must keep the buffer at n_total() = {n} elements");
                        // take the whole bottom band before wave w+1 is allowed to enter
                        while entry.load(Ordering::Acquire) != w {
                            std::hint::spin_loop();
                        }
                        let (lo0, hi0) = self.band[0];
                        let mut guards: Vec<MutexGuard<'_, Layer>> = Vec::new();
                        for layer in lo0..=hi0 {
                            guards.push(self.layers[layer].lock().unwrap());
                        }
                        entry.store(w + 1, Ordering::Release); // wave w+1 may now enter
                        let mut leaked_upto = 0usize;
                        let mut next_acquire = hi0 + 1;
                        let mut held_lo = lo0;
                        for s in 0..l {
                            let (lo, hi) = self.band[s];
                            // extend the band up to hi (acquire leading, increasing order)…
                            while next_acquire <= hi {
                                guards.push(self.layers[next_acquire].lock().unwrap());
                                next_acquire += 1;
                            }
                            // …then release trailing layers below lo
                            while held_lo < lo {
                                drop(guards.remove(0));
                                held_lo += 1;
                            }
                            self.process_source(
                                s, w, lo, &mut guards, &mut leaked_upto, &drive_buf, &mut buf, &mut firers,
                            );
                        }
                        // guards drop here → the whole band is released before the next wave
                    }
                });
            }
        });
    }

    pub fn run(&self, drive: &[i16], waves: usize, threads: usize) {
        assert_eq!(drive.len(), self.n_total(), "drive length {} != n_total() {}", drive.len(), self.n_total());
        self.run_stream(waves, threads, |_, buf| buf.copy_from_slice(drive));
    }

    /// Read a neuron's potential by global index (`idx = layer*ls + local`).
    pub fn potential_global(&self, idx: usize) -> i16 {
        let (layer, local) = (idx / self.ls, idx % self.ls);
        self.layers[layer].lock().unwrap().potential[local]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_reservoir::config::MAX_SATURATION;
    use std::sync::{Arc, Mutex};

    #[test]
    fn construct_and_band_offsets() {
        let cfg = IntConfig::demo(); // topology levels {-1,0,1,2}, l=6
        let net = LayerNet::new(cfg);
        assert_eq!(net.n_total(), 16 * 16 * 6);
        let mut seen = vec![0u32; net.l as usize];
        for list in &net.clamp_after {
            for &j in list {
                seen[j] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "each layer finalized exactly once");
        assert!(net.clamp_after[(net.l - 1) as usize].contains(&((net.l - 1) as usize)));
    }

    #[test]
    fn threaded_matches_sequential_all_thread_counts() {
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let seq = LayerNet::new(cfg.clone());
        for _ in 0..20 {
            seq.wave(&drive);
        }
        let golden: Vec<i16> = (0..n).map(|i| seq.potential_global(i)).collect();

        for t in [1usize, 2, 4, 8, 16] {
            let net = LayerNet::new(cfg.clone());
            net.run(&drive, 20, t);
            for i in 0..n {
                assert_eq!(net.potential_global(i), golden[i], "mismatch at neuron {i}, {t} threads");
            }
        }
    }

    #[test]
    fn reset_state_zeros_everything() {
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let net = LayerNet::new(cfg);
        net.run(&drive, 10, 1);
        net.reset_state();
        for i in 0..n {
            assert_eq!(net.potential_global(i), 0, "potential at {i} after reset");
        }
    }

    #[test]
    fn threaded_deterministic_under_stress() {
        // A deep config widens the wavefront (more waves in flight), so the pipeline-entry
        // race — a lower wave overtaken at the start — surfaces here without entry ordering.
        // Looped to sample many interleavings against the sequential golden.
        let mut cfg = IntConfig::demo();
        cfg.w = 8;
        cfg.h = 8;
        cfg.l = 32;
        cfg.layers = vec![cfg.layers[0].clone(); 32];
        let n = cfg.n_total();
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let seq = LayerNet::new(cfg.clone());
        for _ in 0..40 {
            seq.wave(&drive);
        }
        let golden: Vec<i16> = (0..n).map(|i| seq.potential_global(i)).collect();
        for _ in 0..20 {
            let net = LayerNet::new(cfg.clone());
            net.run(&drive, 40, 8);
            for i in 0..n {
                assert_eq!(net.potential_global(i), golden[i], "deep pipeline diverged (entry race?)");
            }
        }
    }

    #[test]
    fn run_stream_matches_per_wave_drives() {
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let drives: Vec<Vec<i16>> = (0..10)
            .map(|w| {
                let mut d = vec![0i16; n];
                for j in 0..(cfg.w * cfg.h) as usize {
                    d[j] = ((w as i16) * 7 + 3) % 40;
                }
                d
            })
            .collect();
        let seq = LayerNet::new(cfg.clone());
        for d in &drives {
            seq.wave(d);
        }
        let golden: Vec<i16> = (0..n).map(|i| seq.potential_global(i)).collect();
        for t in [1usize, 4, 8] {
            let net = LayerNet::new(cfg.clone());
            net.run_stream(drives.len(), t, |w, buf| {
                buf.clear();
                buf.extend_from_slice(&drives[w]);
            });
            for i in 0..n {
                assert_eq!(net.potential_global(i), golden[i], "mismatch at {i}, {t} threads");
            }
        }
    }

    #[test]
    fn drive_buffer_arrives_zeroed_each_wave() {
        // Contract: `drive_fn` receives a zeroed, n_total()-length buffer every wave, so a
        // sparse writer needs no manual clear — and worker buffer reuse can never leak one
        // wave's drive into another (which would differ across thread counts).
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let ls = (cfg.w * cfg.h) as usize;
        let seq = LayerNet::new(cfg.clone());
        for w in 0..12 {
            let mut d = vec![0i16; n];
            d[w % ls] = 50; // one wave-dependent site, all else zero
            seq.wave(&d);
        }
        let golden: Vec<i16> = (0..n).map(|i| seq.potential_global(i)).collect();
        for t in [1usize, 4] {
            let net = LayerNet::new(cfg.clone());
            net.run_stream(12, t, |w, buf| {
                buf[w % ls] = 50; // sparse write, relies on the zeroed buffer
            });
            for i in 0..n {
                assert_eq!(net.potential_global(i), golden[i], "stale drive at {i}, {t} threads");
            }
        }
    }

    #[test]
    #[should_panic(expected = "drive length")]
    fn wave_rejects_wrong_drive_length() {
        let net = LayerNet::new(IntConfig::demo());
        net.wave(&[0i16; 3]);
    }

    #[test]
    #[should_panic(expected = "threads must be >= 1")]
    fn run_stream_rejects_zero_threads() {
        let cfg = IntConfig::demo();
        let drive = vec![0i16; cfg.n_total()];
        let net = LayerNet::new(cfg);
        net.run(&drive, 1, 0);
    }

    #[test]
    #[should_panic]
    fn run_stream_rejects_drive_fn_resizing_buffer() {
        let net = LayerNet::new(IntConfig::demo());
        net.run_stream(2, 1, |_, buf| buf.push(1)); // len n+1 violates the contract
    }

    #[test]
    fn listener_stream_deterministic_across_threads() {
        // The per-layer event stream is serialized + wave-ordered by the layer lock, so a
        // layer-0 subscriber sees the identical (wave, count) sequence at any thread count.
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let record = |threads: usize| {
            let rec = Arc::new(Mutex::new(Vec::new()));
            let mut net = LayerNet::new(cfg.clone());
            {
                let r = rec.clone();
                net.on_layer(0, Box::new(move |wave, fired| r.lock().unwrap().push((wave, fired.len()))));
            }
            net.run_stream(20, threads, |_, buf| {
                buf.clear();
                buf.extend_from_slice(&drive);
            });
            std::mem::take(&mut *rec.lock().unwrap())
        };
        let a = record(1);
        let b = record(4);
        assert_eq!(a.len(), 20, "one event per wave (emitted even on zero-firing waves)");
        assert!(a.iter().enumerate().all(|(i, &(w, _))| w == i), "events in wave order");
        assert_eq!(a, b, "listener stream identical (order + counts) across thread counts");
    }

    fn single_neuron_cfg(refractory: u8) -> IntConfig {
        let mut cfg = IntConfig::demo();
        cfg.w = 1;
        cfg.h = 1;
        cfg.l = 1;
        let mut layer = cfg.layers[0].clone();
        layer.p_inh_q16 = 0;
        layer.topology = vec![]; // no synapses; isolate the fire test
        layer.threshold_base = 100;
        layer.spread_log2 = 0; // threshold in [99,100]
        layer.leak_a = 3;
        layer.leak_b = 5;
        layer.refractory = refractory;
        cfg.layers = vec![layer];
        cfg.saturation = MAX_SATURATION;
        cfg
    }

    #[test]
    fn refractory_blocks_refiring() {
        // A single neuron driven every wave fires only when its cooldown reaches zero.
        let fired_waves = Arc::new(Mutex::new(Vec::new()));
        let mut net = LayerNet::new(single_neuron_cfg(3));
        {
            let fw = fired_waves.clone();
            net.on_layer(
                0,
                Box::new(move |wave, fired| {
                    if !fired.is_empty() {
                        fw.lock().unwrap().push(wave);
                    }
                }),
            );
        }
        net.run_stream(7, 1, |_, buf| {
            buf.clear();
            buf.push(200);
        });
        assert_eq!(*fired_waves.lock().unwrap(), vec![0, 3, 6]);
    }

    #[test]
    fn leak_decays_potential() {
        // No firing (threshold above any transient), no drive after wave 0: potential shrinks.
        let mut cfg = single_neuron_cfg(2);
        cfg.layers[0].threshold_base = 20_000; // never fires; still inside i16
        cfg.layers[0].spread_log2 = 0;
        let net = LayerNet::new(cfg);
        net.wave(&[4_000i16]); // load potential (below saturation, so no clamp)
        let p0 = net.potential_global(0);
        net.wave(&[0i16]);
        let p1 = net.potential_global(0);
        // leak (3,5): p1 = p0 - (p0>>3) - (p0>>5) ~ 0.844*p0
        assert!(p1 < p0 && p1 > p0 / 2, "leak should shrink but not zero: {p0}->{p1}");
    }

    #[test]
    fn forward_delivery_reaches_next_layer() {
        // 1x1x2: L0's single level-1 synapse delivers into L1 (which integrates, never fires).
        let mut cfg = IntConfig::demo();
        cfg.w = 1;
        cfg.h = 1;
        cfg.l = 2;
        let mut layer = cfg.layers[0].clone();
        layer.p_inh_q16 = 0; // excitatory (+1)
        layer.topology = vec![IntLevel { level: 1, radius: 0, count: 1 }];
        layer.threshold_base = 1;
        layer.spread_log2 = 0;
        cfg.layers = vec![layer; 2];
        cfg.layers[1].threshold_base = 100; // L1 integrates, does not fire
        cfg.saturation = MAX_SATURATION;
        let n = cfg.n_total();
        let net = LayerNet::new(cfg);
        let mut drive = vec![0i16; n];
        drive[0] = 5; // L0 over threshold
        net.wave(&drive);
        assert!(net.potential_global(1) > 0, "forward delivery must reach L1: {}", net.potential_global(1));
    }

    #[test]
    fn top_layer_trajectory_golden() {
        // Exact demo trajectory anchor: the top layer's per-wave spike counts and index
        // checksum must reproduce bit-for-bit — any change to wiring or dynamics trips this.
        let cfg = IntConfig::demo();
        let n = cfg.n_total();
        let top = cfg.l as usize - 1;
        let waves = 4; // the golden anchors exactly 4 waves
        let mut drive = vec![0i16; n];
        for j in 0..(cfg.w * cfg.h) as usize {
            drive[j] = 50;
        }
        let traj = Arc::new(Mutex::new(Vec::new()));
        let mut net = LayerNet::new(cfg);
        {
            let t = traj.clone();
            net.on_layer(top, Box::new(move |_wave, fired| t.lock().unwrap().push(fired.to_vec())));
        }
        net.run_stream(waves, 1, |_, buf| {
            buf.clear();
            buf.extend_from_slice(&drive);
        });
        let traj = std::mem::take(&mut *traj.lock().unwrap());
        let counts: Vec<usize> = traj.iter().map(|w| w.len()).collect();
        let checksum: u64 = traj.iter().flatten().map(|&x| x as u64).sum();
        assert_eq!(counts, vec![132, 75, 162, 80], "top-layer spike counts per wave");
        assert_eq!(checksum, 57437, "trajectory checksum");
    }
}
