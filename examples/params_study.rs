//! OFAT parameter study: impact of the integer wave engine's knobs on temporal-XOR
//! computation, with binary {-1,+1} weights. Each sweep changes one field of the demo
//! and recalibrates on the actual bit stream, so effects aren't confounded with
//! firing-rate shifts.

use wave_net::wave_reservoir::hash::mix;
use wave_net::wave_reservoir::index::Dims;
use wave_net::wave_reservoir::input::InputMap;
use wave_net::wave_reservoir::config::{
    spread_log2_for, IntConfig, IntLevel, RefractoryMode, MAX_SATURATION,
};
use wave_net::wave_reservoir::pipeline::LayerNet;
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

// --- integer XOR harness (ridge readout + accuracy) ---
fn cholesky_solve(mut a: Vec<Vec<f64>>, b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            if i == j {
                a[i][i] = s.max(1e-12).sqrt();
            } else {
                a[i][j] = s / a[j][j];
            }
        }
    }
    let m = b[0].len();
    let mut x = vec![vec![0.0f64; m]; n];
    for col in 0..m {
        let mut z = vec![0.0f64; n];
        for i in 0..n {
            let mut s = b[i][col];
            for k in 0..i {
                s -= a[i][k] * z[k];
            }
            z[i] = s / a[i][i];
        }
        for i in (0..n).rev() {
            let mut s = z[i];
            for k in (i + 1)..n {
                s -= a[k][i] * x[k][col];
            }
            x[i][col] = s / a[i][i];
        }
    }
    x
}

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

fn fair_bit(seed: u64, t: u64) -> u8 {
    (mix(seed ^ t.wrapping_mul(0xD1B5_4A32)) & 1) as u8
}

/// Write this bit's bipolar bottom-layer drive into `buf` (arrives zeroed from `run_stream`).
fn bit_drive_into(buf: &mut [i16], sites: &[u32], b: u8, level: i16) {
    let v = if b == 1 { level } else { -level };
    for &s in sites {
        buf[s as usize] += v;
    }
}

fn stream_xor(
    cfg: &IntConfig,
    waves_per_bit: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = cfg.n_total();
    let ls = (cfg.w * cfg.h) as usize;
    let dims = Dims::new(cfg.w, cfg.h, cfg.l);
    let sites = InputMap::scatter_bottom(&dims, INPUT_SEED, 1, PER_CHANNEL).channels[0].clone();
    let sample = sample_neurons(n, READOUT_DIM, seed ^ 0x5A11);
    let level = INPUT_LEVEL;
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
        bit_drive_into(buf, &sites, bhist[w / waves_per_bit], level);
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

fn ridge_fit(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let d = x[0].len();
    let mut a = vec![vec![0.0f64; d]; d];
    let mut b = vec![vec![0.0f64; 1]; d];
    for (xi, &yi) in x.iter().zip(y) {
        for i in 0..d {
            b[i][0] += xi[i] * yi;
            for j in 0..d {
                a[i][j] += xi[i] * xi[j];
            }
        }
    }
    for i in 0..d {
        a[i][i] += lambda;
    }
    cholesky_solve(a, &b).into_iter().map(|r| r[0]).collect()
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

// --- stream-based calibration (matches the actual task's firing rate) ---
/// Per-layer (spikes, total) over a continuous bipolar bit stream — the real task
/// dynamics (reset once, never between bits), unlike the class-drive ensemble.
fn stream_layer_rates(cfg: &IntConfig) -> Vec<(u64, u64)> {
    let ls = (cfg.w * cfg.h) as u64;
    let dims = Dims::new(cfg.w, cfg.h, cfg.l);
    let sites = InputMap::scatter_bottom(&dims, INPUT_SEED, 1, PER_CHANNEL).channels[0].clone();
    let level = INPUT_LEVEL;
    let total_waves = CAL_BITS * FIXED_WPB;

    let spikes = Arc::new(Mutex::new(vec![0u64; cfg.l as usize]));
    let mut net = LayerNet::new(cfg.clone());
    net.reset_state();
    for z in 0..cfg.l as usize {
        let sp = spikes.clone();
        net.on_layer(
            z,
            Box::new(move |_wave, fired| {
                sp.lock().unwrap()[z] += fired.len() as u64;
            }),
        );
    }
    net.run_stream(total_waves, THREADS, |w, buf| {
        let b = fair_bit(TASK_SEED ^ 0xCA1B, (w / FIXED_WPB) as u64);
        bit_drive_into(buf, &sites, b, level);
    });

    let spikes = std::mem::take(&mut *spikes.lock().unwrap());
    let waves = total_waves as u64;
    (0..cfg.l as usize).map(|z| (spikes[z], ls * waves)).collect()
}

/// Tune each cascade layer's `threshold_base` (integer, shift step) so it fires at
/// `target_permille` on the actual bit-stream task — the density-matching fix.
fn calibrate_on_stream(cfg: &mut IntConfig, target_permille: u64) {
    for _ in 0..CAL_PASSES {
        let rates = stream_layer_rates(cfg);
        for z in 1..cfg.l as usize {
            let (spikes, total) = rates[z];
            let tb = cfg.layers[z].threshold_base;
            let step = (tb >> 2).max(1);
            if spikes * 1000 > total * target_permille {
                cfg.layers[z].threshold_base = tb + step;
            } else if spikes * 1000 < total * target_permille {
                cfg.layers[z].threshold_base = (tb - step).max(1);
            }
            let new_tb = cfg.layers[z].threshold_base;
            cfg.layers[z].spread_log2 = spread_log2_for(new_tb);
        }
        let max_t = cfg.layers.iter().map(|l| l.threshold_base).max().unwrap_or(1);
        cfg.saturation = max_t.saturating_mul(32).min(MAX_SATURATION as i32) as i16;
    }
}

// --- config-variant builders ---
fn build_at(target: u64, seed: u64, mutate: &impl Fn(&mut IntConfig)) -> IntConfig {
    let mut cfg = IntConfig::demo();
    cfg.seed = seed;
    mutate(&mut cfg);
    calibrate_on_stream(&mut cfg, target);
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
