//! FF training harness for the `wave_driven` engine (test-only). Ports the `wave_bitnet_bench` shape
//! onto the event-driven engine's online eligibility: no per-wave pot/eff recording, per-neuron rate
//! read from the engine's `spike_count`, and `Network::dfa_update` applied from the accumulated
//! eligibility. Proves the activity-scaled trainer learns end-to-end (FF single-cue above chance).

#[cfg(test)]
mod tests {
    use crate::wave_driven::config::{Config, LayerConfig};
    use crate::wave_driven::network::Network;
    use crate::wave_driven::synapse::{key, mix, TopologyLevel};
    use crate::wave_driven::training::{Edge, EligParams};
    use std::sync::{Arc, Mutex};

    const CUE_P: u64 = 0xC0E;
    const P_DFA: u64 = 61;

    fn cue_sites(task_seed: u64, size: u32, class: usize) -> Vec<u32> {
        let ls = (size * size) as u32;
        (0..ls).filter(|&loc| mix(key(task_seed, loc, class as i32, 0, CUE_P)) & 3 == 0).collect()
    }

    fn softmax2(z0: f32, z1: f32) -> (f32, f32) {
        let m = z0.max(z1);
        let (e0, e1) = ((z0 - m).exp(), (z1 - m).exp());
        let s = (e0 + e1).max(1e-30);
        (e0 / s, e1 / s)
    }

    fn dfa_weight(seed: u64, neuron_global: u32, class: usize) -> f32 {
        if mix(key(seed, neuron_global, class as i32, 0, P_DFA)) & 1 == 1 { 1.0 } else { -1.0 }
    }

    /// Run one trial; returns (top-layer read-window spike counts `act`, total waves `ttot`).
    fn run_trial(net: &mut Network, size: u32, classes: &[usize], task_seed: u64, present: usize, delay: usize, read: usize) -> (Vec<f32>, usize) {
        let l = net.layer_count();
        let ls = (size * size) as usize;
        let top = l - 1;
        let top_spikes: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(Vec::new()));
        let ts = top_spikes.clone();
        net.on_layer(top, Box::new(move |_w, fired: &[u32]| ts.lock().unwrap().push(fired.to_vec())));
        net.reset_state();
        let mut ttot = 0usize;
        for (pos, &class) in classes.iter().enumerate() {
            if pos > 0 {
                for _ in 0..delay {
                    net.wave(&[]);
                    ttot += 1;
                }
            }
            for _ in 0..present {
                let sites = cue_sites(task_seed, size, class);
                net.wave(&sites);
                ttot += 1;
            }
        }
        let read_start = top_spikes.lock().unwrap().len();
        for _ in 0..read {
            net.wave(&[]);
            ttot += 1;
        }
        net.clear_listeners();
        let mut act = vec![0f32; ls];
        for wv in top_spikes.lock().unwrap().iter().skip(read_start) {
            for &loc in wv {
                act[loc as usize] += 1.0;
            }
        }
        (act, ttot)
    }

    struct TaskCfg {
        size: u32,
        present: usize,
        delay: usize,
        read: usize,
        holdout: usize,
        readout_lr: f32,
        hidden_lr: f32,
        rate_reg: f32,
        rate_target: f32,
    }

    /// Learning signal per computational layer/neuron: DFA feedback + rate_reg (rate from the engine).
    fn build_signal(net: &Network, w: &[Vec<f32>], err: &[f32], seed: u64, ttot: usize, cfg: &TaskCfg) -> Vec<Vec<f32>> {
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let denom = ttot.max(1) as f32;
        let mut signal = vec![vec![0f32; ls]; l];
        for tz in 1..l {
            let sc = net.layer_spike_count(tz);
            for j in 0..ls {
                let task_sig: f32 = (0..2)
                    .map(|c| {
                        let b = if tz == top { w[c][j] } else { dfa_weight(seed, (tz * ls + j) as u32, c) };
                        b * err[c]
                    })
                    .sum();
                let rate = sc[j] as f32 / denom;
                signal[tz][j] = task_sig + cfg.rate_reg * (rate - cfg.rate_target);
            }
        }
        signal
    }

    fn train_and_eval_best(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize), eval_every: usize, patience: usize, max_trials: usize) -> (u64, usize) {
        const EVAL_OFFSET: usize = 10_000_000;
        let ls = (cfg.size * cfg.size) as usize;
        let mut w = vec![vec![0f32; ls]; 2];
        let score = |w: &[Vec<f32>], a: &[f32]| -> (f32, f32) {
            (w[0].iter().zip(a).map(|(x, y)| x * y).sum(), w[1].iter().zip(a).map(|(x, y)| x * y).sum())
        };
        let (mut best, mut best_at, mut stale, mut t) = (0u64, 0usize, 0usize, 0usize);
        while t < max_trials {
            let stop = (t + eval_every).min(max_trials);
            while t < stop {
                let (classes, label) = task(task_seed, t);
                let (act, ttot) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                let (p0, p1) = softmax2(s0, s1);
                let err = [p0 - if label == 0 { 1.0 } else { 0.0 }, p1 - if label == 1 { 1.0 } else { 0.0 }];
                for c in 0..2 {
                    for j in 0..ls {
                        w[c][j] -= cfg.readout_lr * err[c] * act[j];
                    }
                }
                if cfg.hidden_lr != 0.0 {
                    let signal = build_signal(net, &w, &err, seed, ttot, cfg);
                    net.dfa_update(entries, &signal, cfg.hidden_lr);
                }
                t += 1;
            }
            let mut correct = 0usize;
            for i in 0..cfg.holdout {
                let (classes, label) = task(task_seed, EVAL_OFFSET + i);
                let (act, _) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                if ((s1 > s0) as usize) == label {
                    correct += 1;
                }
            }
            let acc = (correct as u64 * 1000) / cfg.holdout as u64;
            if acc > best {
                best = acc;
                best_at = t;
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }
        (best, best_at)
    }

    fn single_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let c = (mix(key(seed, t as u32, 0, 0, 71)) & 1) as usize;
        (vec![c], c)
    }

    fn make_ff(seed: u64, size: u32, layers: usize, up_count: u32, up_radius: u32, adapt_bump: i16, adapt_decay: u8) -> (Network, Vec<Vec<Edge>>) {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay,
        };
        let mut net = Network::new(Config { seed, size, layers: vec![lc; layers] });
        net.enable_training();
        net.set_elig_params(EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: up_count as usize, radius: up_radius }] })
            .collect();
        (net, entries)
    }

    fn ff_cfg() -> TaskCfg {
        TaskCfg { size: 16, present: 6, delay: 4, read: 6, holdout: 200, readout_lr: 0.02, hidden_lr: 0.004, rate_reg: 5.0, rate_target: 0.1 }
    }

    #[test]
    fn wave_driven_ff_trains_above_chance() {
        // 4-layer FF, size 16, generous fan-in; single-cue 2-class must beat chance. Integration proof
        // that online eligibility + shadow update + repack + readout trains end-to-end.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 5, 6);
        let cfg = ff_cfg();
        let (best, _at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 3, 1000);
        assert!(best > 600, "wave_driven FF should train above chance: {best}");
    }

    #[test]
    #[ignore] // smoke: run manually in --release
    fn wave_driven_ff_depth8_smoke() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        eprintln!("== wave_driven FF depth-8 pure ternary smoke (r4/c48, adapt=5, online elig) ==");
        let mut bests = Vec::new();
        for &s in &seeds {
            let (mut net, entries) = make_ff(s, 32, 8, 48, 4, 5, 6);
            let mut cfg = ff_cfg();
            cfg.size = 32;
            cfg.present = 8;
            cfg.read = 8;
            cfg.holdout = 300;
            let (best, at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, single_task, 300, 3, 3000);
            eprintln!("seed {s:#x}: best {best}@{at}");
            bests.push(best);
        }
        let worst = *bests.iter().min().unwrap();
        eprintln!("worst {worst} (target ~1000)");
        assert!(worst >= 900, "pure ternary FF depth-8 should hold ~1000 (worst {worst})");
    }

    #[test]
    #[ignore] // experiment: online vs offline-eligibility trainer throughput (run in --release)
    fn wave_driven_training_throughput() {
        use crate::wave_driven::training::dense_eligibility;
        use std::time::Instant;
        for &size in &[16u32, 32u32] {
            let seed = 0xC0FFEEu64;
            let (mut net, entries) = make_ff(seed, size, 4, 32, 3, 5, 6);
            let mut cfg = ff_cfg();
            cfg.size = size;
            let trials = 200usize;

            // online: run trials, accrual happens inside wave(); dfa_update reads engine elig
            let w = vec![vec![0f32; (size * size) as usize]; 2];
            let t0 = Instant::now();
            for t in 0..trials {
                let (classes, _label) = single_task(seed, t);
                let (_act, ttot) = run_trial(&mut net, size, &classes, seed, cfg.present, cfg.delay, cfg.read);
                let err = [0.1f32, -0.1f32];
                let signal = build_signal(&net, &w, &err, seed, ttot, &cfg);
                net.dfa_update(&entries, &signal, cfg.hidden_lr);
            }
            let online = t0.elapsed().as_secs_f64();

            // offline: record fired every wave, compute dense_eligibility per trial (the size-bound path)
            let (mut net2, entries2) = make_ff(seed, size, 4, 32, 3, 5, 6);
            let t1 = Instant::now();
            for t in 0..trials {
                let (classes, _label) = single_task(seed, t);
                let l = net2.layer_count();
                let rec: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
                for z in 0..l {
                    let r = rec.clone();
                    net2.on_layer(z, Box::new(move |_w, f: &[u32]| r.lock().unwrap()[z].push(f.to_vec())));
                }
                net2.reset_state();
                for _ in 0..(cfg.present + cfg.read) {
                    let sites = cue_sites(seed, size, classes[0]);
                    net2.wave(&sites);
                }
                net2.clear_listeners();
                let fired = rec.lock().unwrap().clone();
                let _e = dense_eligibility(&net2, &entries2, &fired, &EligParams { rec_tau: 6.0, epsilon: 1.0 / 1024.0 });
            }
            let offline = t1.elapsed().as_secs_f64();
            eprintln!("size {size}: online {:.1} trials/s, offline-eligibility {:.1} trials/s ({:.1}x)", trials as f64 / online, trials as f64 / offline, offline / online);
        }
    }
}
