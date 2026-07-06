//! OFAT parameter study: impact of the integer wave engine's knobs on temporal-XOR
//! computation, with binary {-1,+1} weights. Each sweep changes one field of the demo
//! and recalibrates on the actual bit stream, so effects aren't confounded with
//! firing-rate shifts.

use wave_net::wave_reservoir::hash::mix;
use wave_net::wave_reservoir::index::Dims;
use wave_net::wave_reservoir::config::{IntConfig, IntLevel, RefractoryMode};
use wave_net::wave_reservoir::pipeline::LayerNet;
use wave_net::wave_net::calibrate::{calibrate, CalibrateParams};
use wave_net::wave_net::linalg::ridge_fit;
use wave_net::wave_net::stream::{fair_bit, BipolarInput};
use std::sync::{Arc, Mutex};

const READOUT_DIM: usize = 128;
const PER_CHANNEL: usize = 24;
const INPUT_SEED: u64 = 0x0B17_5EED;
const TASK_SEED: u64 = 0x5EED_C0DE;
const WASHOUT: usize = 30;
const TRAIN: usize = 400;
const TEST: usize = 200;
const TAU: usize = 1;
const LAMBDA: f64 = 1.0;
const CONTROL_LAMBDA: f64 = 1e3;
const FIXED_WPB: usize = 8;
const TARGET_PERMILLE: u64 = 120; // ~12% firing, near the inverted-U peak
const CAL_BITS: usize = 40;
const CAL_PASSES: usize = 10;
const INPUT_LEVEL: i16 = 4; // fixed input level; calibration moves the threshold, not this
// The demo is shallow (depth 6), so the wavefront can't pipeline it — run LayerNet
// single-threaded here; the deep-net parallelism is exercised by the wavefront bench.
const THREADS: usize = 1;
const NSEEDS: usize = 3;
const SEEDS: [u64; NSEEDS] = [
    0x1234_5678_9ABC_DEF0,
    0x2222_2222_2222_2222,
    0x3333_3333_3333_3333,
];

// --- integer XOR harness (ridge readout + accuracy); linalg lifted to wave_net::linalg ---
fn sample_neurons(n: usize, dim: usize, seed: u64) -> Vec<usize> {
    use std::collections::BTreeSet;
    let mut set = BTreeSet::new();
    let mut k = 0u64;
    while set.len() < dim.min(n) {
        let h = mix(seed ^ k.wrapping_mul(0x9E37_79B9));
        set.insert((h % n as u64) as usize);
        k += 1;
    }
    set.into_iter().collect()
}

fn stream_xor(
    cfg: &IntConfig,
    waves_per_bit: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = cfg.n_total();
    let ls = (cfg.w * cfg.h) as usize;
    let dims = Dims::new(cfg.w, cfg.h, cfg.l);
    let input = BipolarInput::scatter_bottom(&dims, INPUT_SEED, PER_CHANNEL, INPUT_LEVEL);
    let sample = sample_neurons(n, READOUT_DIM, seed ^ 0x5A11);
    let total = WASHOUT + TRAIN + TEST;
    let total_waves = total * waves_per_bit;
    let bhist: Vec<u8> = (0..total).map(|t| fair_bit(seed, t as u64)).collect();

    // For each layer, map local index -> feature column (None if that neuron isn't sampled).
    let mut col_of: Vec<Vec<Option<usize>>> = (0..cfg.l as usize).map(|_| vec![None; ls]).collect();
    let mut layer_sampled = vec![false; cfg.l as usize];
    for (j, &nrn) in sample.iter().enumerate() {
        col_of[nrn / ls][nrn % ls] = Some(j);
        layer_sampled[nrn / ls] = true;
    }

    // Per-bit feature rows; the last column is the bias, filled after the run.
    let features = Arc::new(Mutex::new(vec![vec![0.0f64; sample.len() + 1]; total]));
    let mut net = LayerNet::new(cfg.clone());
    net.reset_state();
    for (layer, map) in col_of.into_iter().enumerate() {
        if !layer_sampled[layer] {
            continue;
        }
        let feats = features.clone();
        net.on_layer(
            layer,
            Box::new(move |wave, fired| {
                let bit = wave / waves_per_bit;
                let mut f = feats.lock().unwrap();
                for &local in fired {
                    if let Some(j) = map[local as usize] {
                        f[bit][j] += 1.0;
                    }
                }
            }),
        );
    }

    // Stream the bits: wave w carries bit w / waves_per_bit.
    net.run_stream(total_waves, THREADS, |w, buf| {
        input.drive_into(buf, bhist[w / waves_per_bit]);
    });

    let features = std::mem::take(&mut *features.lock().unwrap());
    let (mut feats, mut tgts, mut ctrl) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..total {
        let mut counts = features[t].clone();
        counts[sample.len()] = 1.0;
        if t >= WASHOUT {
            feats.push(counts);
            tgts.push((bhist[t] ^ bhist[t - TAU]) as f64);
            ctrl.push(vec![bhist[t] as f64, bhist[t - TAU] as f64, 1.0]);
        }
    }
    (feats, tgts, ctrl)
}

fn accuracy(w: &[f64], x: &[Vec<f64>], y: &[f64]) -> f32 {
    let mut correct = 0;
    for (xi, &yi) in x.iter().zip(y) {
        let pred: f64 = xi.iter().zip(w).map(|(a, b)| a * b).sum();
        if ((pred >= 0.5) as u8) as f64 == yi {
            correct += 1;
        }
    }
    correct as f32 / y.len() as f32
}

fn xor_eval(cfg: &IntConfig, waves_per_bit: usize, seed: u64) -> (f32, f32) {
    let (feats, tgts, ctrl) = stream_xor(cfg, waves_per_bit, seed);
    let (trf, tef) = feats.split_at(TRAIN);
    let (trt, tet) = tgts.split_at(TRAIN);
    let (trc, tec) = ctrl.split_at(TRAIN);
    let wr = ridge_fit(trf, trt, LAMBDA);
    let res = accuracy(&wr, tef, tet);
    let wc = ridge_fit(trc, trt, CONTROL_LAMBDA);
    let ctl = accuracy(&wc, tec, tet);
    (res, ctl)
}

// --- config-variant builders ---
fn build_at(target: u64, seed: u64, mutate: &impl Fn(&mut IntConfig)) -> IntConfig {
    let mut cfg = IntConfig::demo();
    cfg.seed = seed;
    mutate(&mut cfg);
    // Calibrate on the same bipolar bit stream the task uses (its own seed, distinct from the
    // eval bits) — identical to the routine this used to carry inline, now in wave_net::calibrate.
    let dims = Dims::new(cfg.w, cfg.h, cfg.l);
    let input = BipolarInput::scatter_bottom(&dims, INPUT_SEED, PER_CHANNEL, INPUT_LEVEL);
    let params = CalibrateParams { target_permille: target, passes: CAL_PASSES, bits: CAL_BITS, wpb: FIXED_WPB };
    calibrate(&mut cfg, &params, &move |w: usize, buf: &mut Vec<i16>| {
        input.drive_into(buf, fair_bit(TASK_SEED ^ 0xCA1B, (w / FIXED_WPB) as u64))
    });
    cfg
}

/// Mean XOR accuracy at wpb=8 for a config calibrated to a given target firing rate.
fn avg_at(target: u64, mutate: &impl Fn(&mut IntConfig)) -> f32 {
    let mut r = 0.0f32;
    for &s in &SEEDS {
        let cfg = build_at(target, s, mutate);
        r += xor_eval(&cfg, FIXED_WPB, TASK_SEED).0;
    }
    r / NSEEDS as f32
}

fn build(seed: u64, mutate: &impl Fn(&mut IntConfig)) -> IntConfig {
    build_at(TARGET_PERMILLE, seed, mutate)
}

fn avg(wpb: usize, mutate: &impl Fn(&mut IntConfig)) -> (f32, f32) {
    let (mut r, mut c) = (0.0f32, 0.0f32);
    for &s in &SEEDS {
        let (rr, cc) = xor_eval(&build(s, mutate), wpb, TASK_SEED);
        r += rr;
        c += cc;
    }
    (r / NSEEDS as f32, c / NSEEDS as f32)
}

enum TopologyKind {
    Forward,
    Recurrent,
    Balanced,
}

fn topo_entries(k: &TopologyKind) -> Vec<IntLevel> {
    match k {
        TopologyKind::Forward => vec![
            IntLevel { level: 1, radius: 2, count: 6 },
            IntLevel { level: 2, radius: 1, count: 2 },
            IntLevel { level: 0, radius: 1, count: 2 },
            IntLevel { level: -1, radius: 0, count: 1 },
        ],
        TopologyKind::Recurrent => vec![
            IntLevel { level: 1, radius: 2, count: 3 },
            IntLevel { level: 0, radius: 1, count: 4 },
            IntLevel { level: -1, radius: 1, count: 3 },
        ],
        TopologyKind::Balanced => vec![
            IntLevel { level: 1, radius: 2, count: 4 },
            IntLevel { level: 2, radius: 1, count: 1 },
            IntLevel { level: 0, radius: 1, count: 3 },
            IntLevel { level: -1, radius: 0, count: 2 },
        ],
    }
}

fn set_topology(cfg: &mut IntConfig, entries: &[IntLevel]) {
    for l in cfg.layers.iter_mut() {
        l.topology = entries.to_vec();
    }
}

fn set_layers(cfg: &mut IntConfig, l: u32) {
    cfg.l = l;
    cfg.layers = vec![cfg.layers[0].clone(); l as usize];
}

fn main() {
    println!("Parameter impact on temporal-XOR (binary weights; control ~0.5; mean of 3 seeds)");
    println!("(waves_per_bit = {FIXED_WPB} for every sweep except [1])");

    println!("\n[1] waves_per_bit");
    println!("{:<16} {:>10} {:>10}", "waves_per_bit", "reservoir", "control");
    for wpb in [1usize, 2, 4, 8, 12, 16, 20, 24] {
        let (r, c) = avg(wpb, &|_cfg| {});
        println!("{:<16} {:>10.3} {:>10.3}", wpb, r, c);
    }

    println!("\n[2] refractory period");
    println!("{:<16} {:>10} {:>10}", "refractory", "reservoir", "control");
    for p in [1u8, 2, 3, 4] {
        let (r, c) = avg(FIXED_WPB, &|cfg| {
            for l in cfg.layers.iter_mut() {
                l.refractory = p;
            }
        });
        println!("{:<16} {:>10.3} {:>10.3}", p, r, c);
    }

    println!("\n[3] refractory mode (refractory=2)");
    println!("{:<16} {:>10} {:>10}", "mode", "reservoir", "control");
    for (name, mode) in [
        ("CarryOver", RefractoryMode::CarryOver),
        ("Drop", RefractoryMode::Drop),
    ] {
        let (r, c) = avg(FIXED_WPB, &|cfg| {
            cfg.refractory_mode = mode;
            for l in cfg.layers.iter_mut() {
                l.refractory = 2;
            }
        });
        println!("{:<16} {:>10.3} {:>10.3}", name, r, c);
    }

    println!("\n[4] leak (retention)");
    println!("{:<16} {:>10} {:>10}", "retention", "reservoir", "control");
    for (a, b) in [(3u8, 4u8), (3, 5), (3, 6), (4, 4), (4, 5), (4, 6), (5, 5)] {
        let (r, c) = avg(FIXED_WPB, &|cfg| {
            for l in cfg.layers.iter_mut() {
                l.leak_a = a;
                l.leak_b = b;
            }
        });
        let ret = 1.0 - (2f64.powi(-(a as i32)) + 2f64.powi(-(b as i32)));
        println!("{:<16.3} {:>10.3} {:>10.3}", ret, r, c);
    }

    println!("\n[5] topology");
    println!("{:<16} {:>10} {:>10}", "topology", "reservoir", "control");
    for (name, kind) in [
        ("Forward", TopologyKind::Forward),
        ("Recurrent", TopologyKind::Recurrent),
        ("Balanced", TopologyKind::Balanced),
    ] {
        let entries = topo_entries(&kind);
        let (r, c) = avg(FIXED_WPB, &|cfg| set_topology(cfg, &entries));
        println!("{:<16} {:>10.3} {:>10.3}", name, r, c);
    }

    println!("\n[6] layers");
    println!("{:<16} {:>10} {:>10}", "layers", "reservoir", "control");
    for l in [2u32, 4, 6, 8] {
        let (r, c) = avg(FIXED_WPB, &|cfg| set_layers(cfg, l));
        println!("{:<16} {:>10.3} {:>10.3}", l, r, c);
    }

    println!("\n[7] p_inh (inhibitory fraction)");
    println!("{:<16} {:>10} {:>10}", "p_inh", "reservoir", "control");
    for (pct, q16) in [(0.0, 0u32), (0.10, 6554), (0.15, 9830), (0.25, 16384), (0.40, 26214)] {
        let (r, c) = avg(FIXED_WPB, &|cfg| {
            for l in cfg.layers.iter_mut() {
                l.p_inh_q16 = q16;
            }
        });
        println!("{:<16.2} {:>10.3} {:>10.3}", pct, r, c);
    }

    // Capstone: the best single-knob winners are all defaults except p_inh=0, so the
    // "best config" is the demo with inhibition off. Its computation over waves:
    println!("\n[8] best config (p_inh=0) vs default — XOR over the wave progression");
    println!("{:<16} {:>10} {:>10} {:>10}", "waves_per_bit", "best", "default", "control");
    for wpb in [1usize, 2, 4, 6, 8, 12, 16, 20, 24] {
        let (rb, cb) = avg(wpb, &|cfg| {
            for l in cfg.layers.iter_mut() {
                l.p_inh_q16 = 0;
            }
        });
        let (rd, _cd) = avg(wpb, &|_cfg| {});
        println!("{:<16} {:>10.3} {:>10.3} {:>10.3}", wpb, rb, rd, cb);
    }

    // Adversarial check on the density fix: do the key verdicts (recurrent >= forward,
    // shallow vs deep) hold across the target firing rate, or are they an artifact of 12%?
    println!("\n[9] target-rate robustness — XOR @ wpb=8 (mean 3 seeds)");
    println!("{:<8} {:>7} {:>7} {:>7} {:>7}", "target", "Fwd", "Recur", "2lay", "6lay");
    for target in [80u64, 120, 160] {
        let fwd = avg_at(target, &|_c| {});
        let recur = avg_at(target, &|c| set_topology(c, &topo_entries(&TopologyKind::Recurrent)));
        let l2 = avg_at(target, &|c| set_layers(c, 2));
        let l6 = avg_at(target, &|c| set_layers(c, 6));
        println!(
            "{:<8} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            format!("{}%", target / 10),
            fwd,
            recur,
            l2,
            l6
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_computes_xor() {
        let cfg = build(SEEDS[0], &|_cfg| {});
        let (r, c) = xor_eval(&cfg, FIXED_WPB, TASK_SEED);
        assert!(r > c + 0.08, "binary-weight baseline must compute XOR at wpb={FIXED_WPB}: r={r} c={c}");
    }
}
