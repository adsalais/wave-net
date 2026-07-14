//! WebGPU backend for the eligibility kernel (wgpu 29 + WGSL). Storage buffers for tgt/src/eps/elig stay
//! resident; each `step` writes the 4 per-neuron arrays and dispatches ceil(n/256) workgroups. Semantics
//! mirror `super::cpu_accrue` (validated <1e-5 vs the CPU oracle in tests). Portable (Vulkan/Metal/DX/GL).

use wgpu::util::DeviceExt;

use super::{Captured, GpuBackend, Layout};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    dt: f32,
    cut: f32,
    _pad: u32,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    b_eff: wgpu::Buffer,
    omega: wgpu::Buffer,
    psi: wgpu::Buffer,
    prev_fired: wgpu::Buffer,
    eps_x: wgpu::Buffer,
    eps_y: wgpu::Buffer,
    elig: wgpu::Buffer,
    readback: wgpu::Buffer,
    n: u32,
    elig_bytes: u64,
}

fn storage_init<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T], rw: bool) -> wgpu::Buffer {
    let extra = if rw { wgpu::BufferUsages::COPY_SRC } else { wgpu::BufferUsages::empty() };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
    })
}

impl GpuBackend for WgpuBackend {
    fn new(layout: &Layout, dt: f32, cut: f32) -> Self {
        pollster::block_on(async move {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .expect("no wgpu adapter");
            // The kernel binds 9 storage buffers; the WebGPU baseline caps a compute stage at 8, so request
            // the adapter's real (higher) limits. A browser-portable version would pack the index arrays
            // (tgt_g+src_g) into one buffer to fit the 8-storage-buffer baseline — deferred (native spike).
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() })
                .await
                .expect("request_device");

            let n = layout.n;
            let ls_l = layout.l * layout.ls;
            let tgt_g = storage_init(&device, &layout.tgt_g, false);
            let src_g = storage_init(&device, &layout.src_g, false);
            let b_eff = storage_init(&device, &vec![0f32; ls_l], false);
            let omega = storage_init(&device, &vec![0f32; ls_l], false);
            let psi = storage_init(&device, &vec![0f32; ls_l], false);
            let prev_fired = storage_init(&device, &vec![0u32; ls_l], false);
            let eps_x = storage_init(&device, &vec![0f32; n], true);
            let eps_y = storage_init(&device, &vec![0f32; n], true);
            let elig = storage_init(&device, &vec![0f32; n], true);
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params { n: n as u32, dt, cut, _pad: 0 }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(include_str!("elig.wgsl").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("accrue"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tgt_g.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: src_g.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: b_eff.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: omega.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: psi.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: prev_fired.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: eps_x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: eps_y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: elig.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: params.as_entire_binding() },
                ],
            });
            let elig_bytes = (n * std::mem::size_of::<f32>()) as u64;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: elig_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            WgpuBackend { device, queue, pipeline, bind_group, b_eff, omega, psi, prev_fired, eps_x, eps_y, elig, readback, n: n as u32, elig_bytes }
        })
    }

    fn reset(&mut self) {
        let zeros = vec![0f32; self.n as usize];
        self.queue.write_buffer(&self.eps_x, 0, bytemuck::cast_slice(&zeros));
        self.queue.write_buffer(&self.eps_y, 0, bytemuck::cast_slice(&zeros));
        self.queue.write_buffer(&self.elig, 0, bytemuck::cast_slice(&zeros));
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    fn step(&mut self, cap: &Captured) {
        self.queue.write_buffer(&self.b_eff, 0, bytemuck::cast_slice(&cap.b_eff_g));
        self.queue.write_buffer(&self.omega, 0, bytemuck::cast_slice(&cap.omega_g));
        self.queue.write_buffer(&self.psi, 0, bytemuck::cast_slice(&cap.psi_g));
        self.queue.write_buffer(&self.prev_fired, 0, bytemuck::cast_slice(&cap.prev_fired_g));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((self.n + 255) / 256, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap(); // synchronous step for honest timing
    }

    fn download_elig(&self) -> Vec<f32> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.elig, 0, &self.readback, 0, self.elig_bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.readback.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::synapse::TopologyLevel;
    use crate::wave_resonate_gpu::{allclose, capture_inputs, cpu_accrue, run_backend};

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
    fn wgpu_matches_oracle_within_tolerance() {
        let cfg = ff_cfg(16, 4);
        let inputs: Vec<Vec<u32>> =
            (0..40).map(|w| if w % 3 == 0 { vec![0u32, 1, 2, 16, 17] } else { vec![] }).collect();
        let (layout, seq) = capture_inputs(&cfg, &inputs);
        let cpu = cpu_accrue(&layout, &seq, 0.05, 1e-6);
        let (gpu, _dur) = run_backend::<WgpuBackend>(&layout, &seq, 0.05, 1e-6);
        let (ok, max_abs, max_rel) = allclose(&gpu, &cpu, 1e-5, 1e-3);
        assert!(ok, "wgpu vs oracle allclose(1e-5,1e-3) failed: max_abs={max_abs} max_rel={max_rel}");
    }
}
