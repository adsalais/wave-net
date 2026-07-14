//! `wave` — one BRF layer's per-wave step: drain the integer delivery accumulator into the input current,
//! (L0) act as a pass-through transducer or (last) leaky-integrator readout, else run the complex
//! resonator (dense over all neurons) and collect firers, then scatter each firer's ternary weights into
//! the target layers' accumulators (`generate`, firer-gated — the sparse, dominant cost).

use crate::wave_resonate::neurons::{pw, Layer, WCODE};
use crate::wave_resonate::synapse::{local_of, wrap, xy_of};
use crate::wave_resonate::training::surrogate;

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    size: u32,
    input: &[u32],
    deliv: &mut [Vec<i32>],
    fired: &mut Vec<u32>,
) {
    let ls = (size as usize) * (size as usize);
    fired.clear();

    // --- transducer (L0): fire exactly the injected sites; no membrane; clear any pending ---
    if layer.transducer {
        for p in layer.pending.iter_mut() {
            *p = 0;
        }
        for &a in input {
            fired.push(a);
        }
        generate(layer, layer_index, size, deliv, fired);
        return;
    }

    // --- readout: leaky-integrate the drained current into x; never fire, never deliver ---
    if layer.readout {
        let k = layer.kappa;
        for i in 0..ls {
            let cur = layer.pending[i] as f32;
            layer.x[i] = k * layer.x[i] + cur;
            layer.pending[i] = 0;
        }
        return;
    }

    // --- compute: dense BRF oscillator update + decide (+ training capture) ---
    let (dt, gamma, theta_c) = (layer.dt, layer.gamma, layer.theta_c);
    let training = layer.train.is_some();
    for i in 0..ls {
        let cur = layer.pending[i] as f32;
        layer.pending[i] = 0;
        let (x, y, q, omega, b_off) = (layer.x[i], layer.y[i], layer.q[i], layer.omega[i], layer.b_off[i]);
        let b = pw(omega, dt) - b_off.abs() - q;
        let nx = x + dt * (b * x - omega * y + cur);
        let ny = y + dt * (omega * x + b * y);
        let spike = nx - theta_c - q > 0.0;
        layer.x[i] = nx;
        layer.y[i] = ny;
        layer.q[i] = gamma * q + if spike { 1.0 } else { 0.0 };
        if spike {
            fired.push(i as u32);
        }
        if training {
            // capture the target-neuron values the HYPR accrual needs this wave: b_j^t and ψ_j^t (at the
            // pre-update q, matching the forward threshold), plus the running spike count.
            let t = layer.train.as_mut().unwrap();
            t.b_eff[i] = b;
            t.psi[i] = surrogate(nx - theta_c - q);
            if spike {
                t.spike_count[i] += 1;
            }
        }
    }

    generate(layer, layer_index, size, deliv, fired);
}

/// Firer-gated ternary delivery: each firer word-scans its occupancy bitset, decodes each wired cell to
/// its target local index, and scatter-adds the packed ±1/0 weight into `deliv[target_layer][target]`.
fn generate(layer: &Layer, layer_index: u32, size: u32, deliv: &mut [Vec<i32>], fired: &[u32]) {
    let layer_count = deliv.len() as i32;
    for &local in fired.iter() {
        let li = local as usize;
        let (sx, sy) = xy_of(local, size);
        for (lvl, entry) in layer.topology.iter().enumerate() {
            let tl = layer_index as i32 + entry.level;
            if tl < 0 || tl >= layer_count {
                continue;
            }
            let tz = tl as usize;
            let wpn = layer.occ_wpn[lvl];
            let words = &layer.occ[lvl][li * wpn..li * wpn + wpn];
            let wbase = li * layer.total_slots + layer.slot_bases[lvl];
            let lut = &layer.offsets[lvl];
            let flat = &layer.off_flat[lvl];
            let r = entry.radius;
            let hi = size.saturating_sub(r);
            let interior = sx >= r && sx < hi && sy >= r && sy < hi;
            let li_i = li as i32;
            let mut rank = 0usize;
            for (wi, &w0) in words.iter().enumerate() {
                let mut word = w0;
                let cbase = wi * 64;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    let cell = cbase + bit;
                    let widx = wbase + rank;
                    let wt = WCODE[((layer.codes[widx >> 5] >> ((widx & 31) * 2)) & 0b11) as usize] as i32;
                    let target = if interior {
                        (li_i + flat[cell]) as usize
                    } else {
                        let (dx, dy) = lut[cell];
                        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize
                    };
                    deliv[tz][target] += wt;
                    rank += 1;
                    word &= word - 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::LayerConfig;
    use crate::wave_resonate::neurons::Layer;
    use crate::wave_resonate::synapse::TopologyLevel;

    fn compute_layer(size: u32, omega: f32, b_off: f32, dt: f32, gamma: f32, theta_c: f32) -> Layer {
        let cfg = LayerConfig { topology: vec![], inhibitor_ratio: 0, omega_init: (omega, omega), b_offset_init: (b_off, b_off), tau_out: 20.0 };
        Layer::new(&cfg, dt, gamma, theta_c, 5, 0, size)
    }

    // hand-rolled reference BRF neuron (plain f32 loop) — the fidelity oracle
    fn ref_brf(dt: f32, gamma: f32, theta_c: f32, omega: f32, b_off: f32, i_seq: &[f32]) -> Vec<(f32, f32, f32, u8)> {
        let (mut x, mut y, mut q) = (0f32, 0f32, 0f32);
        let mut out = Vec::new();
        for &i in i_seq {
            let p = (-1.0 + (1.0 - (dt * omega) * (dt * omega)).sqrt()) / dt;
            let b = p - b_off.abs() - q;
            let nx = x + dt * (b * x - omega * y + i);
            let ny = y + dt * (omega * x + b * y);
            let z = if nx - theta_c - q > 0.0 { 1u8 } else { 0u8 };
            let nq = gamma * q + z as f32;
            x = nx;
            y = ny;
            q = nq;
            out.push((x, y, q, z));
        }
        out
    }

    #[test]
    fn single_neuron_matches_reference_bit_exact() {
        let (dt, gamma, theta_c, omega, b_off) = (0.05f32, 0.9f32, 1.0f32, 10.0f32, 0.3f32);
        let i_seq: Vec<f32> = vec![3.0, 3.0, 3.0, 0.0, 0.0, 5.0, 0.0, -2.0, 0.0, 0.0, 4.0, 4.0];
        let want = ref_brf(dt, gamma, theta_c, omega, b_off, &i_seq);
        let mut l = compute_layer(1, omega, b_off, dt, gamma, theta_c);
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
        let mut fired = Vec::new();
        for (t, &i) in i_seq.iter().enumerate() {
            l.pending[0] = i as i32;
            process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
            let (wx, wy, wq, wz) = want[t];
            assert_eq!(l.x[0], wx, "x @ t={t}");
            assert_eq!(l.y[0], wy, "y @ t={t}");
            assert_eq!(l.q[0], wq, "q @ t={t}");
            assert_eq!(fired.contains(&0), wz == 1, "spike @ t={t}");
        }
    }

    #[test]
    fn divergence_free_stays_bounded_under_strong_drive() {
        let mut l = compute_layer(1, 10.0, 0.3, 0.05, 0.9, 1.0);
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
        let mut fired = Vec::new();
        for _ in 0..2000 {
            l.pending[0] = 50; // sustained strong drive
            process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
            assert!(l.x[0].abs() < 1e4 && l.y[0].abs() < 1e4, "BRF stays bounded: x={} y={}", l.x[0], l.y[0]);
        }
    }

    #[test]
    fn vanilla_rf_reference_diverges_documenting_control() {
        // Documenting control: a fixed positive-b RF (no p(ω) boundary) grows unbounded, whereas the BRF
        // reference stays bounded. Pure hand-rolled ref (no engine change) — motivates p(ω).
        let (dt, omega) = (0.05f32, 10.0f32);
        let (mut x, mut y) = (0.1f32, 0.0f32);
        for _ in 0..3000 {
            let b = 0.5f32;
            let nx = x + dt * (b * x - omega * y);
            let ny = y + dt * (omega * x + b * y);
            x = nx;
            y = ny;
        }
        // Diverges: |x|,|y| blow past f32 range (→ inf, and inf−inf → NaN). Either way it is NOT bounded
        // like the BRF neuron, which is the point. (A `> 1e6` check would miss the NaN endpoint.)
        assert!(!(x.abs() < 1e6 && y.abs() < 1e6), "fixed +b RF should not stay bounded, got x={x} y={y}");
    }

    #[test]
    fn resonance_prefers_matched_frequency() {
        // A neuron with ω has discrete period P ≈ 2π/(δω) waves. Drive the SAME number of impulses, spaced
        // in-phase (stride = P, constructive) vs anti-phase (stride = P/2, destructive); the resonant
        // (in-phase) drive reaches a larger peak |x|. Matched impulse count isolates resonance from energy.
        let (dt, omega) = (0.05f32, 10.0f32);
        let period = (2.0 * std::f32::consts::PI / (dt * omega)).round() as usize; // ≈ 13
        let n_impulses = 8usize;
        let run = |stride: usize| -> f32 {
            let mut l = compute_layer(1, omega, 0.05, dt, 0.9, 1.0);
            let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
            let mut fired = Vec::new();
            let mut peak = 0f32;
            let total = stride * n_impulses + period;
            for t in 0..total {
                l.pending[0] = if t % stride == 0 && t / stride < n_impulses { 6 } else { 0 };
                process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
                peak = peak.max(l.x[0].abs());
            }
            peak
        };
        let resonant = run(period);
        let off = run((period / 2).max(1));
        assert!(resonant > off, "resonant peak {resonant} should exceed off-frequency peak {off}");
    }

    #[test]
    fn transducer_fires_iff_injected_and_does_not_oscillate() {
        let mut l = compute_layer(4, 10.0, 0.3, 0.05, 0.9, 1.0);
        l.transducer = true;
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 16]; 1];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 4, &[2, 5, 9], &mut deliv, &mut fired);
        let mut f = fired.clone();
        f.sort_unstable();
        assert_eq!(f, vec![2, 5, 9], "transducer fires exactly the injected sites");
        assert!(l.x.iter().all(|&v| v == 0.0) && l.y.iter().all(|&v| v == 0.0), "no oscillation");
    }

    #[test]
    fn readout_integrates_and_never_fires() {
        let mut l = compute_layer(4, 10.0, 0.3, 0.05, 0.9, 1.0);
        l.readout = true;
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 16]; 1];
        let mut fired = Vec::new();
        l.pending[0] = 10;
        process_layer(&mut l, 0, 4, &[], &mut deliv, &mut fired);
        assert!(fired.is_empty(), "readout never fires");
        assert!(l.x[0] > 0.0, "readout integrated input into x (leaky accumulator)");
        let x1 = l.x[0];
        l.pending[0] = 0;
        process_layer(&mut l, 0, 4, &[], &mut deliv, &mut fired);
        assert!(l.x[0] < x1 && l.x[0] > 0.0, "with no input the accumulator leaks toward 0: {} < {}", l.x[0], x1);
    }

    #[test]
    fn training_capture_fills_b_eff_and_psi() {
        use crate::wave_resonate::training::surrogate;
        let dt = 0.05f32;
        let mut l = compute_layer(1, 10.0, 0.3, dt, 0.9, 1.0);
        l.enable_training();
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; 1]; 1];
        let mut fired = Vec::new();
        l.pending[0] = 40; // drive toward threshold
        process_layer(&mut l, 0, 1, &[], &mut deliv, &mut fired);
        let t = l.train.as_ref().unwrap();
        // b_eff = pw(omega) - |b_off| - q_old (q_old was 0 on the first wave)
        let expect_b = crate::wave_resonate::neurons::pw(l.omega[0], dt) - l.b_off[0].abs();
        assert!((t.b_eff[0] - expect_b).abs() < 1e-6, "b_eff captured");
        assert_eq!(t.psi[0], surrogate(l.x[0] - l.theta_c - 0.0), "psi captured at (x - θ_c - q_old)");
    }

    #[test]
    fn firer_scatters_decoded_weights_into_target_accumulator() {
        let size = 4u32;
        let ls = (size * size) as usize;
        let cfg = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }], inhibitor_ratio: 0, omega_init: (10.0, 10.0), b_offset_init: (0.3, 0.3), tau_out: 20.0 };
        let mut l = Layer::new(&cfg, 0.05, 0.9, 1.0, 5, 0, size);
        // force neuron 0 to fire: huge drive, zero q
        l.pending[0] = 1000;
        // expected: sum decoded nonzero weights per target for neuron 0
        let base = l.slot_base(0);
        let mut expect = vec![0i32; ls];
        l.for_wired(0, 0, |r, cell| {
            let wt = l.weight_at(base + r);
            if wt != 0 {
                expect[l.decode(0, 0, cell, size) as usize] += wt as i32;
            }
        });
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; ls]; 2];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, size, &[], &mut deliv, &mut fired);
        assert!(fired.contains(&0), "neuron 0 fires under strong drive");
        assert_eq!(deliv[1], expect, "scatter-adds decoded ±1 weights into layer 1's accumulator");
    }
}
