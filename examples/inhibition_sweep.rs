//! Inhibition sweep: is the inhibitory synapse fraction `p_inh` helping or hurting, and does
//! the answer survive seed changes and bigger layers?
//!
//! Grid: p_inh ∈ {0, 2, 5, 10, 15}% × 3 seed pairs (task + input + perturbation) × layer sizes
//! {16×16, 32×32} (l = 6). Each cell runs field_training's honest protocol independently:
//! calibrate the substrate, train the online readout on TRAIN, drive node perturbation by VAL
//! accuracy, report held-out TEST. Input-site density is held constant across sizes (24 sites
//! per 256 bottom neurons). Synapse targets are drawn from different hash bits than the sign,
//! so within a seed the graph is identical across p_inh values — only the −1/+1 mix changes.
//!
//! Cells run in parallel (one thread each); rows print as they finish, then a mean-over-seeds
//! summary per (size, p_inh).
//!
//! Run: `cargo run --release --example inhibition_sweep [iters]` (default 60).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wave_net::wave_net::calibrate::{calibrate, CalibrateParams};
use wave_net::wave_net::config::IntConfig;
use wave_net::wave_net::hash::mix;
use wave_net::wave_net::index::Dims;
use wave_net::wave_net::pipeline::LayerNet;
use wave_net::wave_net::readout::OnlineReadout;
use wave_net::wave_net::stream::{fair_bit, BipolarInput};
use wave_net::wave_net::train::{add_field, hill_climb, PerturbParams};

const WPB: usize = 8;
const WASHOUT: usize = 30;
const TRAIN: usize = 400;
const VAL: usize = 200;
const TEST: usize = 200;
const TAU: usize = 1;

const SIZES: [u32; 2] = [32, 16]; // big cells first for load balance
const PCTS: [u32; 5] = [0, 2, 5, 10, 15];
const SEEDS: u64 = 3;

struct Row {
    size: u32,
    pct: u32,
    k: u64,
    base_val: f64,
    base_test: f64,
    trained_val: f64,
    trained_test: f64,
}

fn run_cell(size: u32, pct: u32, k: u64, iters: usize) -> Row {
    let total = WASHOUT + TRAIN + VAL + TEST;
    let val_lo = WASHOUT + TRAIN;
    let test_lo = val_lo + VAL;
    let task_seed = mix(0x5EED_C0DE ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let input_seed = mix(0x0B17_5EED ^ k.wrapping_mul(0xD1B5_4A32_9E37_79B9));
    let perturb_seed = mix(0x00F1_E1D0 ^ k.wrapping_mul(0xA24B_AED4_963E_E407));

    let mut cfg = IntConfig::demo();
    cfg.w = size;
    cfg.h = size;
    for lc in &mut cfg.layers {
        lc.p_inh_q16 = (pct * 65536 + 50) / 100;
    }
    let dims = Dims::new(cfg.w, cfg.h, cfg.l);
    let n = cfg.n_total();
    let ls = (cfg.w * cfg.h) as usize;
    let top = cfg.l as usize - 1;
    let top_base = top * ls;
    let sites = 24 * ls / 256; // constant input-site density across sizes
    let input = BipolarInput::scatter_bottom(&dims, input_seed, sites, 4);

    {
        let inp = input.clone();
        calibrate(&mut cfg, &CalibrateParams::default(), &move |w, buf| {
            inp.drive_into(buf, fair_bit(task_seed ^ 0xCA1B, (w / WPB) as u64))
        });
    }

    let bhist: Vec<u8> = (0..total).map(|t| fair_bit(task_seed, t as u64)).collect();
    let target = |t: usize| (bhist[t] ^ bhist[t - TAU]) as f64;

    let feats = Arc::new(Mutex::new(vec![vec![0.0f64; ls]; total]));
    let mut net = LayerNet::new(cfg.clone());
    {
        let f = feats.clone();
        net.on_layer(
            top,
            Box::new(move |wave, fired| {
                let bit = wave / WPB;
                let mut ff = f.lock().unwrap();
                for &loc in fired {
                    ff[bit][loc as usize] += 1.0;
                }
            }),
        );
    }

    let run_features = |top_field: &[i16]| -> Vec<Vec<f64>> {
        let mut full = vec![0i16; n];
        full[top_base..top_base + ls].copy_from_slice(top_field);
        {
            let mut ff = feats.lock().unwrap();
            for row in ff.iter_mut() {
                row.iter_mut().for_each(|v| *v = 0.0);
            }
        }
        net.reset_state();
        net.run_stream(total * WPB, 1, |w, buf| {
            input.drive_into(buf, bhist[w / WPB]);
            add_field(buf, &full);
        });
        let ff = feats.lock().unwrap();
        ff.iter()
            .map(|row| {
                let mut x = row.clone();
                x.push(1.0);
                x
            })
            .collect()
    };

    let acc_on = |features: &[Vec<f64>], lo: usize, hi: usize| -> f64 {
        let mut ro = OnlineReadout::new(ls + 1, 1.0);
        for t in WASHOUT..(WASHOUT + TRAIN) {
            ro.update(&features[t], target(t));
        }
        let mut correct = 0;
        for t in lo..hi {
            if ((ro.predict(&features[t]) >= 0.5) as u8) as f64 == target(t) {
                correct += 1;
            }
        }
        correct as f64 / (hi - lo) as f64
    };

    let reward = |top_field: &[i16]| -> f64 {
        let ft = run_features(top_field);
        acc_on(&ft, val_lo, val_lo + VAL)
    };

    let zeros = vec![0i16; ls];
    let base_ft = run_features(&zeros);
    let base_val = acc_on(&base_ft, val_lo, val_lo + VAL);
    let base_test = acc_on(&base_ft, test_lo, test_lo + TEST);

    let pp = PerturbParams { iters, density_pct: 15, step: 1, clamp: 8, seed: perturb_seed };
    let out = hill_climb(zeros, &pp, &reward);
    let best_ft = run_features(&out.params);
    let trained_test = acc_on(&best_ft, test_lo, test_lo + TEST);

    Row { size, pct, k, base_val, base_test, trained_val: out.reward, trained_test }
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let mut cells = Vec::new();
    for &size in &SIZES {
        for &pct in &PCTS {
            for k in 0..SEEDS {
                cells.push((size, pct, k));
            }
        }
    }
    println!(
        "Inhibition sweep: sizes {SIZES:?} × p_inh {PCTS:?}% × {SEEDS} seeds, {iters} iters — {} cells",
        cells.len()
    );

    let started = std::time::Instant::now();
    let next = AtomicUsize::new(0);
    let rows = Mutex::new(Vec::<Row>::new());
    let workers = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4).min(cells.len());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= cells.len() {
                    break;
                }
                let (size, pct, k) = cells[i];
                let r = run_cell(size, pct, k, iters);
                println!(
                    "{:>2}x{:<2} p_inh {:>2}% seed {}: base VAL {:.3} TEST {:.3} | trained VAL {:.3} TEST {:.3} | gain {:+.3}",
                    r.size, r.size, r.pct, r.k, r.base_val, r.base_test, r.trained_val,
                    r.trained_test, r.trained_test - r.base_test
                );
                rows.lock().unwrap().push(r);
            });
        }
    });
    let rows = rows.into_inner().unwrap();

    println!("\nmeans over {SEEDS} seeds ({:.0}s total):", started.elapsed().as_secs_f64());
    println!("size   p_inh   base TEST   trained TEST   gain     VAL-TEST gap");
    for &size in &SIZES {
        for &pct in &PCTS {
            let sel: Vec<&Row> = rows.iter().filter(|r| r.size == size && r.pct == pct).collect();
            let mean = |f: &dyn Fn(&Row) -> f64| sel.iter().map(|r| f(r)).sum::<f64>() / sel.len() as f64;
            println!(
                "{:>2}x{:<4} {:>2}%     {:.3}       {:.3}        {:+.3}      {:+.3}",
                size,
                size,
                pct,
                mean(&|r| r.base_test),
                mean(&|r| r.trained_test),
                mean(&|r| r.trained_test - r.base_test),
                mean(&|r| r.trained_val - r.trained_test),
            );
        }
    }
}
