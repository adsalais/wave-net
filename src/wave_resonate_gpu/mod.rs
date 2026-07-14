//! `wave_resonate_gpu` — a de-risking spike: GPU port of the HYPR eligibility accrual
//! (`Network::accrue_eligibility`), validated vs the CPU `dense_eligibility` oracle. The std-only parts
//! here (flat edge-list layout, the `cpu_accrue` reference, the `GpuBackend` trait + `CpuBackend`) compile
//! in the default build; the CUDA/wgpu backends are feature-gated. See
//! docs/superpowers/specs/2026-07-14-wave-resonate-gpu-eligibility-spike-design.md.

use crate::wave_resonate::config::Config;
use crate::wave_resonate::network::Network;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Flat per-synapse edge list derived once from a `Network`. Synapse `e` corresponds to the CPU's
/// `widx = i*ts + sbase + rank` within layer `z`, offset by `syn_base[z]`. `tgt_g[e]`/`src_g[e]` are the
/// GLOBAL neuron ids (`layer*ls + local`) of the synapse's target and source.
pub struct Layout {
    pub n: usize,             // total synapses across the stack
    pub ls: usize,            // size*size
    pub l: usize,             // layer count
    pub syn_base: Vec<usize>, // len l+1; synapse-index base per layer (prefix sum of ls*ts)
    pub tgt_g: Vec<u32>,
    pub src_g: Vec<u32>,
}

/// Build the flat edge list: walk every source's wired cells once (setup-time `for_wired` + `decode`,
/// folding the target-layer offset `z+level` into a global id). Every slot of every synapse-bearing layer
/// is a wired synapse (Σ count_e == total_slots), so no sentinels are needed for the FF spike config.
pub fn build_layout(net: &Network) -> Layout {
    let size = net.size();
    let ls = (size as usize) * (size as usize);
    let l = net.layer_count();
    let ts_by: Vec<usize> = (0..l).map(|z| net.with_layer(z, |lz| lz.total_slots)).collect();
    let mut syn_base = vec![0usize; l + 1];
    for z in 0..l {
        syn_base[z + 1] = syn_base[z] + ls * ts_by[z];
    }
    let n = syn_base[l];
    let mut tgt_g = vec![0u32; n];
    let mut src_g = vec![0u32; n];
    for z in 0..l {
        let ts = ts_by[z];
        if ts == 0 {
            continue;
        }
        let base_z = syn_base[z];
        net.with_layer(z, |lz| {
            for i in 0..ls {
                for (e_idx, edge) in lz.topology.iter().enumerate() {
                    let tz_i = z as i32 + edge.level;
                    if tz_i < 0 || tz_i as usize >= l {
                        continue;
                    }
                    let tz = tz_i as usize;
                    let sbase = lz.slot_bases[e_idx];
                    lz.for_wired(e_idx, i, |rank, cell| {
                        let j = lz.decode(e_idx, i as u32, cell, size) as usize;
                        let e = base_z + i * ts + sbase + rank;
                        tgt_g[e] = (tz * ls + j) as u32;
                        src_g[e] = (z * ls + i) as u32;
                    });
                }
            }
        });
    }
    Layout { n, ls, l, syn_base, tgt_g, src_g }
}

/// Per-wave forward outputs, packed by GLOBAL neuron id `g = z*ls + local` (length `L*ls` each).
/// `prev_fired_g` is the PREVIOUS wave's firers (the source injection `z_i^{t-1}`); `omega_g` is constant
/// while `train_omega_b=false` but captured per wave for generality.
pub struct Captured {
    pub b_eff_g: Vec<f32>,
    pub omega_g: Vec<f32>,
    pub psi_g: Vec<f32>,
    pub prev_fired_g: Vec<u32>,
}

/// Drive a fresh training net on `inputs`, capturing each wave's (b_eff, psi, omega, prev_fired) into the
/// global-id layout. Mirrors what `dense_eligibility` consumes, so `cpu_accrue`/the GPU kernels fed this
/// reproduce the oracle. Returns the `Layout` (same net) alongside the per-wave captures.
pub fn capture_inputs(cfg: &Config, inputs: &[Vec<u32>]) -> (Layout, Vec<Captured>) {
    use crate::wave_resonate::training::EligParams;
    let size = cfg.size;
    let ls = (size as usize) * (size as usize);
    let l = cfg.layers.len();
    let mut net = Network::new(cfg.clone());
    net.set_elig_params(EligParams { dt: cfg.dt, eps_cut: 1.0 / 1024.0, train_omega_b: false });
    net.enable_training();
    let layout = build_layout(&net);

    // capture each wave's firers per layer via listeners (listener overwrites its slot each wave)
    let cur: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
    for z in 0..l {
        let c = cur.clone();
        net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] = f.to_vec()));
    }

    let mut prev_fired_g = vec![0u32; l * ls]; // wave 0: no previous firers
    let mut seq = Vec::with_capacity(inputs.len());
    for inp in inputs {
        net.wave(inp);
        let mut b_eff_g = vec![0f32; l * ls];
        let mut omega_g = vec![0f32; l * ls];
        let mut psi_g = vec![0f32; l * ls];
        for z in 0..l {
            net.with_layer(z, |lz| {
                omega_g[z * ls..(z + 1) * ls].copy_from_slice(&lz.omega);
                if let Some(t) = lz.train.as_ref() {
                    b_eff_g[z * ls..(z + 1) * ls].copy_from_slice(&t.b_eff);
                    psi_g[z * ls..(z + 1) * ls].copy_from_slice(&t.psi);
                }
            });
        }
        seq.push(Captured { b_eff_g, omega_g, psi_g, prev_fired_g: prev_fired_g.clone() });
        // roll prev_fired ← THIS wave's firers for the next wave
        prev_fired_g.iter_mut().for_each(|v| *v = 0);
        let fired_now = cur.lock().unwrap();
        for z in 0..l {
            for &i in &fired_now[z] {
                prev_fired_g[z * ls + i as usize] = 1;
            }
        }
    }
    net.clear_listeners();
    (layout, seq)
}

/// The flat-edge-list eligibility reference — the EXACT per-synapse recursion the CUDA/WGSL kernels mirror.
/// Per synapse `e`: advance the 2-state ε from the TARGET's (b_eff, ω, ψ) and the SOURCE's prev-spike,
/// apply `eps_cut`, accumulate `elig += ψ·εˣ`. Per-synapse independent → order-free (matches the oracle).
pub fn cpu_accrue(layout: &Layout, seq: &[Captured], dt: f32, cut: f32) -> Vec<f32> {
    let n = layout.n;
    let mut eps_x = vec![0f32; n];
    let mut eps_y = vec![0f32; n];
    let mut elig = vec![0f32; n];
    for cap in seq {
        for e in 0..n {
            let j = layout.tgt_g[e] as usize;
            let i = layout.src_g[e] as usize;
            let inj = if cap.prev_fired_g[i] != 0 { dt } else { 0.0 };
            let b = cap.b_eff_g[j];
            let om = cap.omega_g[j];
            let psi = cap.psi_g[j];
            let ex = eps_x[e];
            let ey = eps_y[e];
            let coef = 1.0 + dt * b;
            let mut nex = coef * ex - dt * om * ey + inj;
            let mut ney = dt * om * ex + coef * ey;
            if nex.abs() < cut {
                nex = 0.0;
            }
            if ney.abs() < cut {
                ney = 0.0;
            }
            eps_x[e] = nex;
            eps_y[e] = ney;
            if psi != 0.0 && nex != 0.0 {
                elig[e] += psi * nex;
            }
        }
    }
    elig
}

/// A backend advances the per-synapse eligibility one wave at a time and can read back `elig`. `step`
/// must be synchronous (complete before returning) so `run_backend` measures true wall time.
pub trait GpuBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self
    where
        Self: Sized;
    fn reset(&mut self);
    fn step(&mut self, cap: &Captured);
    fn download_elig(&self) -> Vec<f32>;
}

/// Reference backend: the same per-synapse recursion as `cpu_accrue`, but one wave per `step` with
/// resident `eps_x/eps_y/elig` — the shape every GPU backend implements. Also the harness baseline.
pub struct CpuBackend {
    tgt_g: Vec<u32>,
    src_g: Vec<u32>,
    eps_x: Vec<f32>,
    eps_y: Vec<f32>,
    elig: Vec<f32>,
    dt: f32,
    cut: f32,
}

impl GpuBackend for CpuBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self {
        CpuBackend {
            tgt_g: layout.tgt_g.clone(),
            src_g: layout.src_g.clone(),
            eps_x: vec![0.0; layout.n],
            eps_y: vec![0.0; layout.n],
            elig: vec![0.0; layout.n],
            dt,
            cut,
        }
    }
    fn reset(&mut self) {
        self.eps_x.iter_mut().for_each(|v| *v = 0.0);
        self.eps_y.iter_mut().for_each(|v| *v = 0.0);
        self.elig.iter_mut().for_each(|v| *v = 0.0);
    }
    fn step(&mut self, cap: &Captured) {
        let (dt, cut) = (self.dt, self.cut);
        for e in 0..self.elig.len() {
            let j = self.tgt_g[e] as usize;
            let i = self.src_g[e] as usize;
            let inj = if cap.prev_fired_g[i] != 0 { dt } else { 0.0 };
            let (b, om, psi) = (cap.b_eff_g[j], cap.omega_g[j], cap.psi_g[j]);
            let (ex, ey) = (self.eps_x[e], self.eps_y[e]);
            let coef = 1.0 + dt * b;
            let mut nex = coef * ex - dt * om * ey + inj;
            let mut ney = dt * om * ex + coef * ey;
            if nex.abs() < cut {
                nex = 0.0;
            }
            if ney.abs() < cut {
                ney = 0.0;
            }
            self.eps_x[e] = nex;
            self.eps_y[e] = ney;
            if psi != 0.0 && nex != 0.0 {
                self.elig[e] += psi * nex;
            }
        }
    }
    fn download_elig(&self) -> Vec<f32> {
        self.elig.clone()
    }
}

/// Run a full capture sequence through a backend, returning (final elig, wall-clock over the wave loop).
/// Construction is excluded from timing; the loop is the per-wave `step` (upload + launch for GPU).
pub fn run_backend<B: GpuBackend>(layout: &Layout, seq: &[Captured], dt: f32, cut: f32) -> (Vec<f32>, Duration) {
    let mut b = B::new(layout, dt, cut);
    let t = Instant::now();
    for cap in seq {
        b.step(cap);
    }
    let dur = t.elapsed();
    (b.download_elig(), dur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;

    // FF config with the TOP layer empty-topology (0 source synapses — read directly), so every
    // allocated eligibility slot is an in-range wired synapse (no inert out-of-range slots to special-case).
    fn ff_cfg(size: u32, layers: usize) -> Config {
        let up = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
            inhibitor_ratio: 0,
            omega_init: (5.0, 10.0),
            b_offset_init: (0.0, 0.2),
            tau_out: 20.0,
        };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        let layers_vec: Vec<LayerConfig> =
            (0..layers).map(|z| if z == layers - 1 { top.clone() } else { up.clone() }).collect();
        Config { seed: 7, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: layers_vec }
    }

    #[test]
    fn layout_indices_line_up_with_widx_and_decode() {
        let size = 8u32;
        let ls = (size * size) as usize;
        let net = Network::new(ff_cfg(size, 3));
        let layout = build_layout(&net);
        // top layer (empty topology) contributes 0 synapses; layers 0,1 contribute ls*32 each.
        assert_eq!(layout.syn_base[3], layout.n);
        assert_eq!(layout.n, 2 * ls * 32);
        // spot-check: for layer 0, source i, edge 0, the flat entry matches decode() and the global ids.
        net.with_layer(0, |lz| {
            let ts = lz.total_slots;
            for &i in &[0usize, 5, ls - 1] {
                lz.for_wired(0, i, |rank, cell| {
                    let e = layout.syn_base[0] + i * ts + lz.slot_bases[0] + rank;
                    let j = lz.decode(0, i as u32, cell, size) as usize;
                    assert_eq!(layout.tgt_g[e] as usize, ls + j, "target in layer 1");
                    assert_eq!(layout.src_g[e] as usize, i, "source in layer 0");
                });
            }
        });
    }

    #[test]
    fn cpu_accrue_matches_dense_eligibility_oracle() {
        use crate::wave_resonate::training::{dense_eligibility, EligParams};
        let size = 8u32;
        let ls = (size * size) as usize;
        let cfg = ff_cfg(size, 3);
        // a representative drive sequence (some cue waves, some silent)
        let inputs: Vec<Vec<u32>> =
            (0..40).map(|w| if w % 3 == 0 { vec![0u32, 1, 2, 8, 9, 10] } else { vec![] }).collect();
        let p = EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b: false };

        let oracle = dense_eligibility(&cfg, &inputs, &p); // Vec<Vec<f32>> per layer, indexed by widx
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        let flat = cpu_accrue(&layout, &seq, p.dt, p.eps_cut);

        let l = cfg.layers.len();
        let mut max_abs = 0f32;
        for z in 0..l {
            for widx in 0..oracle[z].len() {
                let e = layout.syn_base[z] + widx;
                max_abs = max_abs.max((flat[e] - oracle[z][widx]).abs());
            }
        }
        assert!(max_abs < 1e-6, "cpu_accrue matches oracle, max_abs={max_abs}");
        assert!(flat.iter().any(|&e| e != 0.0), "some eligibility accrued");
    }

    #[test]
    fn cpu_backend_stepwise_equals_batch() {
        let size = 8u32;
        let cfg = ff_cfg(size, 3);
        let inputs: Vec<Vec<u32>> =
            (0..30).map(|w| if w % 3 == 0 { vec![0u32, 1, 2] } else { vec![] }).collect();
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        let batch = cpu_accrue(&layout, &seq, 0.05, 1e-6);
        let (stepwise, _dur) = run_backend::<CpuBackend>(&layout, &seq, 0.05, 1e-6);
        assert_eq!(batch, stepwise, "stepwise CpuBackend == batch cpu_accrue");
    }
}
