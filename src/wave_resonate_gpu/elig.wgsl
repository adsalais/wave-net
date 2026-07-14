// One thread per synapse: advance the 2-state HYPR eligibility and accumulate elig. Mirrors cpu_accrue.
struct Params { n: u32, dt: f32, cut: f32, pad: u32, };

@group(0) @binding(0) var<storage, read>       tgt_g: array<u32>;
@group(0) @binding(1) var<storage, read>       src_g: array<u32>;
@group(0) @binding(2) var<storage, read>       b_eff_g: array<f32>;
@group(0) @binding(3) var<storage, read>       omega_g: array<f32>;
@group(0) @binding(4) var<storage, read>       psi_g: array<f32>;
@group(0) @binding(5) var<storage, read>       prev_fired_g: array<u32>;
@group(0) @binding(6) var<storage, read_write> eps_x: array<f32>;
@group(0) @binding(7) var<storage, read_write> eps_y: array<f32>;
@group(0) @binding(8) var<storage, read_write> elig: array<f32>;
@group(0) @binding(9) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn accrue(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= params.n) { return; }
    let j = tgt_g[e];
    let i = src_g[e];
    var inj = 0.0;
    if (prev_fired_g[i] != 0u) { inj = params.dt; }
    let b = b_eff_g[j];
    let om = omega_g[j];
    let psi = psi_g[j];
    let ex = eps_x[e];
    let ey = eps_y[e];
    let coef = 1.0 + params.dt * b;
    var nex = coef * ex - params.dt * om * ey + inj;
    var ney = params.dt * om * ex + coef * ey;
    if (abs(nex) < params.cut) { nex = 0.0; }
    if (abs(ney) < params.cut) { ney = 0.0; }
    eps_x[e] = nex;
    eps_y[e] = ney;
    if (psi != 0.0 && nex != 0.0) { elig[e] = elig[e] + psi * nex; }
}
