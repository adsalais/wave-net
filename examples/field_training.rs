//! Spec 3, first experiment: train the per-neuron FIELD by node perturbation to raise the online
//! top-layer XOR accuracy above its ~0.62 baseline. The field rides on the drive; the engine is
//! untouched.
//!
//! Honest protocol (no test-set leakage): the online RLS readout trains on TRAIN; node
//! perturbation's reward is accuracy on a separate VAL split; the headline number is accuracy on
//! a never-selected TEST split. Reward(field) = best top-layer readout accuracy that field admits
//! on VAL, so each trial is scored fairly and the reported TEST number is a real generalization
//! estimate.
//!
//! Run: `cargo run --release --example field_training [iters]` (default 120).

use std::sync::{Arc, Mutex};
use wave_net::legacy_net::calibrate::{calibrate, CalibrateParams};
use wave_net::legacy_net::readout::OnlineReadout;
use wave_net::legacy_net::stream::{fair_bit, BipolarInput};
use wave_net::legacy_net::train::{add_field, hill_climb, PerturbParams};
use wave_net::legacy_net::config::IntConfig;
use wave_net::legacy_net::index::Dims;
use wave_net::legacy_net::pipeline::LayerNet;

const WPB: usize = 8;
const WASHOUT: usize = 30;
const TRAIN: usize = 400;
const VAL: usize = 200;
const TEST: usize = 200;
const TAU: usize = 1;
const TASK_SEED: u64 = 0x5EED_C0DE;
const INPUT_SEED: u64 = 0x0B17_5EED;

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(120);
    let total = WASHOUT + TRAIN + VAL + TEST;
    let val_lo = WASHOUT + TRAIN;
    let test_lo = WASHOUT + TRAIN + VAL;

    let mut cfg = IntConfig::demo();
    let dims = Dims::new(cfg.w, cfg.h, cfg.l);
    let n = cfg.n_total();
    let ls = (cfg.w * cfg.h) as usize;
    let top = cfg.l as usize - 1;
    let top_base = top * ls;
    let input = BipolarInput::scatter_bottom(&dims, INPUT_SEED, 24, 4);

    // calibrate the substrate once (fixed; only the field is trained)
    {
        let inp = input.clone();
        calibrate(&mut cfg, &CalibrateParams::default(), &move |w, buf| {
            inp.drive_into(buf, fair_bit(TASK_SEED ^ 0xCA1B, (w / WPB) as u64))
        });
    }

    let bhist: Vec<u8> = (0..total).map(|t| fair_bit(TASK_SEED, t as u64)).collect();
    let target = |t: usize| (bhist[t] ^ bhist[t - TAU]) as f64;

    // one net + top-layer feature buffer, reused per trial (reset each time)
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

    // Run the reservoir with a top-layer field and return the top layer's per-bit feature rows
    // (with bias appended). One reservoir run yields TRAIN/VAL/TEST features together.
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

    // Train the readout on TRAIN, return accuracy over [lo, hi).
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

    // Reward drives node perturbation: readout on TRAIN, accuracy on VAL (never TEST).
    let reward = |top_field: &[i16]| -> f64 {
        let ft = run_features(top_field);
        acc_on(&ft, val_lo, val_lo + VAL)
    };

    let zeros = vec![0i16; ls];
    let base_ft = run_features(&zeros);
    let base_val = acc_on(&base_ft, val_lo, val_lo + VAL);
    let base_test = acc_on(&base_ft, test_lo, test_lo + TEST);

    println!("Field training by node perturbation — top layer ({ls} params), {iters} iters");
    println!("split: TRAIN {TRAIN} / VAL {VAL} (reward) / TEST {TEST} (held out)");
    println!("baseline (field = 0):  VAL {base_val:.3}   TEST {base_test:.3}");

    let pp = PerturbParams { iters, density_pct: 15, step: 1, clamp: 8, seed: 0x00F1_E1D0 };
    let out = hill_climb(zeros, &pp, &reward);

    // Honest generalization number: TEST accuracy of the VAL-selected field.
    let best_ft = run_features(&out.params);
    let best_test = acc_on(&best_ft, test_lo, test_lo + TEST);
    let nonzero = out.params.iter().filter(|&&v| v != 0).count();

    println!("trained (field ≠ 0):   VAL {:.3}   TEST {best_test:.3}", out.reward);
    println!("  → TEST improvement over baseline: {:+.3}", best_test - base_test);
    println!("  field: {nonzero}/{ls} top neurons biased, range [{}, {}]",
        out.params.iter().min().unwrap(), out.params.iter().max().unwrap());
    let step = (out.history.len() / 12).max(1);
    print!("  VAL best-so-far:");
    for i in (0..out.history.len()).step_by(step) {
        print!(" {:.3}", out.history[i]);
    }
    println!();
}
