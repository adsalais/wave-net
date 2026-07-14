//! CUDA backend for the eligibility kernel (cudarc 0.19 + NVRTC). `eps_x/eps_y/elig/tgt_g/src_g` stay
//! resident on the device; each `step` uploads only the per-neuron arrays (b_eff/omega/psi/prev_fired)
//! and launches. Semantics mirror `super::cpu_accrue` (validated <1e-5 vs the CPU oracle in tests).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::{Captured, GpuBackend, Layout};

pub struct CudaBackend {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    n: u32,
    dt: f32,
    cut: f32,
    // resident device buffers
    tgt_g: CudaSlice<u32>,
    src_g: CudaSlice<u32>,
    eps_x: CudaSlice<f32>,
    eps_y: CudaSlice<f32>,
    elig: CudaSlice<f32>,
    b_eff: CudaSlice<f32>,
    omega: CudaSlice<f32>,
    psi: CudaSlice<f32>,
    prev_fired: CudaSlice<u32>,
    ls_l: usize, // L*ls (length of the per-neuron arrays)
}

impl GpuBackend for CudaBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self {
        let ctx = CudaContext::new(0).expect("cuda context 0");
        let stream = ctx.default_stream();
        let ptx = compile_ptx(include_str!("elig.cu")).expect("nvrtc compile elig.cu");
        let module = ctx.load_module(ptx).expect("load module");
        let func = module.load_function("accrue").expect("load accrue");
        let n = layout.n;
        let ls_l = layout.l * layout.ls; // per-neuron array length = L*ls (global-id domain)
        let tgt_g = stream.clone_htod(&layout.tgt_g).unwrap();
        let src_g = stream.clone_htod(&layout.src_g).unwrap();
        let eps_x = stream.alloc_zeros::<f32>(n).unwrap();
        let eps_y = stream.alloc_zeros::<f32>(n).unwrap();
        let elig = stream.alloc_zeros::<f32>(n).unwrap();
        let b_eff = stream.alloc_zeros::<f32>(ls_l).unwrap();
        let omega = stream.alloc_zeros::<f32>(ls_l).unwrap();
        let psi = stream.alloc_zeros::<f32>(ls_l).unwrap();
        let prev_fired = stream.alloc_zeros::<u32>(ls_l).unwrap();
        stream.synchronize().unwrap();
        CudaBackend { stream, func, n: n as u32, dt, cut, tgt_g, src_g, eps_x, eps_y, elig, b_eff, omega, psi, prev_fired, ls_l }
    }

    fn reset(&mut self) {
        let zeros_n = vec![0f32; self.n as usize];
        self.stream.memcpy_htod(&zeros_n, &mut self.eps_x).unwrap();
        self.stream.memcpy_htod(&zeros_n, &mut self.eps_y).unwrap();
        self.stream.memcpy_htod(&zeros_n, &mut self.elig).unwrap();
        self.stream.synchronize().unwrap();
    }

    fn step(&mut self, cap: &Captured) {
        debug_assert_eq!(cap.b_eff_g.len(), self.ls_l);
        self.stream.memcpy_htod(&cap.b_eff_g, &mut self.b_eff).unwrap();
        self.stream.memcpy_htod(&cap.omega_g, &mut self.omega).unwrap();
        self.stream.memcpy_htod(&cap.psi_g, &mut self.psi).unwrap();
        self.stream.memcpy_htod(&cap.prev_fired_g, &mut self.prev_fired).unwrap();
        let cfg = LaunchConfig::for_num_elems(self.n);
        let mut lb = self.stream.launch_builder(&self.func);
        lb.arg(&self.tgt_g).arg(&self.src_g);
        lb.arg(&self.b_eff).arg(&self.omega).arg(&self.psi).arg(&self.prev_fired);
        lb.arg(&mut self.eps_x).arg(&mut self.eps_y).arg(&mut self.elig);
        lb.arg(&self.n).arg(&self.dt).arg(&self.cut);
        unsafe { lb.launch(cfg).unwrap() };
        self.stream.synchronize().unwrap(); // synchronous step for honest timing
    }

    fn download_elig(&self) -> Vec<f32> {
        self.stream.clone_dtoh(&self.elig).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;
    use crate::wave_resonate_gpu::{allclose, capture_inputs, cpu_accrue, run_backend};

    // FF config with the TOP layer empty-topology (matches src/wave_resonate_gpu/mod.rs tests).
    fn ff_cfg(size: u32, layers: usize) -> Config {
        let up = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
            inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.0, 0.2), tau_out: 20.0,
        };
        let top = LayerConfig { topology: vec![], ..up.clone() };
        let v: Vec<LayerConfig> = (0..layers).map(|z| if z == layers - 1 { top.clone() } else { up.clone() }).collect();
        Config { seed: 7, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers: v }
    }

    #[test]
    fn cuda_matches_oracle_within_tolerance() {
        let cfg = ff_cfg(16, 4);
        let inputs: Vec<Vec<u32>> =
            (0..40).map(|w| if w % 3 == 0 { vec![0u32, 1, 2, 16, 17] } else { vec![] }).collect();
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        let cpu = cpu_accrue(&layout, &seq, 0.05, 1e-6);
        let (gpu, _dur) = run_backend::<CudaBackend>(&layout, &seq, 0.05, 1e-6);
        // allclose(atol=1e-5, rtol=1e-3): absolute agreement is the meaningful bar (GPU matches to the f32
        // FMA-reordering floor ~1e-7); rtol only loosens it for large values. See mod::allclose.
        let (ok, max_abs, max_rel) = allclose(&gpu, &cpu, 1e-5, 1e-3);
        assert!(ok, "cuda vs oracle allclose(1e-5,1e-3) failed: max_abs={max_abs} max_rel={max_rel}");
    }
}
