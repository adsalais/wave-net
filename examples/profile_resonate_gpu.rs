//! GPU eligibility spike throughput: CPU vs CUDA vs wgpu waves/s across size {32,64,128}, plus max error
//! vs the CPU reference (itself validated <1e-6 vs the dense_eligibility oracle in the unit tests). The
//! per-wave loop uploads only the O(L·ls) neuron arrays; eps/elig stay resident on device.
//!
//! Run: `cargo run --release --features cuda --example profile_resonate_gpu`
//!  or: `cargo run --release --features "cuda wgpu" --example profile_resonate_gpu`
//! (No features → CPU baseline only.)

use wave_net::wave_resonate::config::{Config, LayerConfig};
use wave_net::wave_resonate::synapse::{random_l0_input, TopologyLevel};
use wave_net::wave_resonate_gpu::{capture_inputs, run_backend, CpuBackend};

const SEED: u64 = 0xC0FFEE_1234_5678;
const WAVES: usize = 256;

// FF config with the TOP layer empty-topology (0 source synapses — read directly), matching the spike/tests.
fn ff_cfg(size: u32) -> Config {
    let up = LayerConfig {
        topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }],
        inhibitor_ratio: 0,
        omega_init: (5.0, 10.0),
        b_offset_init: (0.0, 0.2),
        tau_out: 20.0,
    };
    let top = LayerConfig { topology: vec![], ..up.clone() };
    let layers: Vec<LayerConfig> = (0..5).map(|z| if z == 4 { top.clone() } else { up.clone() }).collect();
    Config { seed: SEED, size, dt: 0.05, gamma: 0.9, theta_c: 0.1, layers }
}

fn waves_per_s(dur: std::time::Duration) -> f64 {
    WAVES as f64 / dur.as_secs_f64()
}

fn main() {
    println!("wave_resonate GPU eligibility spike — {WAVES} waves, per-synapse accrual\n");
    for &size in &[32u32, 64, 128] {
        let cfg = ff_cfg(size);
        let input = random_l0_input(SEED, size, 8000); // ~12% L0 drive
        let inputs: Vec<Vec<u32>> = (0..WAVES).map(&input).collect();
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        println!("size {size}: {} synapses", layout.n);

        let (cpu, cdur) = run_backend::<CpuBackend>(&layout, &seq, 0.05, 1e-6);
        println!("    {:<6} {:>11.0} waves/s   (reference)", "cpu", waves_per_s(cdur));

        #[cfg(feature = "cuda")]
        {
            let (elig, dur) = run_backend::<wave_net::wave_resonate_gpu::CudaBackend>(&layout, &seq, 0.05, 1e-6);
            let (_ok, ma, mr) = wave_net::wave_resonate_gpu::allclose(&elig, &cpu, 1e-5, 1e-3);
            println!("    {:<6} {:>11.0} waves/s   ({:.1}x)   max_abs={ma:.2e} max_rel={mr:.2e}", "cuda", waves_per_s(dur), waves_per_s(dur) / waves_per_s(cdur));
        }
        #[cfg(feature = "wgpu")]
        {
            let (elig, dur) = run_backend::<wave_net::wave_resonate_gpu::WgpuBackend>(&layout, &seq, 0.05, 1e-6);
            let (_ok, ma, mr) = wave_net::wave_resonate_gpu::allclose(&elig, &cpu, 1e-5, 1e-3);
            println!("    {:<6} {:>11.0} waves/s   ({:.1}x)   max_abs={ma:.2e} max_rel={mr:.2e}", "wgpu", waves_per_s(dur), waves_per_s(dur) / waves_per_s(cdur));
        }
        let _ = &cpu;
    }
}
