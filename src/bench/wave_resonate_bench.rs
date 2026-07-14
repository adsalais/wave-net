//! FF training harness for the `wave_resonate` (BRF + HYPR) engine (test-only). Ports the
//! `wave_driven_bench` shape onto the resonate engine: BRF config, HYPR online eligibility, per-neuron
//! rate from `spike_count`, and `Network::dfa_update` from the accumulated eligibility. Proves the
//! BRF+HYPR trainer learns end-to-end (FF single-cue above chance).

#[cfg(test)]
mod tests {
    use crate::wave_resonate::config::{Config, LayerConfig};
    use crate::wave_resonate::network::Network;
    use crate::wave_resonate::synapse::{key, mix, TopologyLevel};
    use crate::wave_resonate::training::{Edge, EligParams};
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
        train_omega_b: bool,
        omega_b_lr: f32,
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
                if cfg.hidden_lr != 0.0 || cfg.train_omega_b {
                    let signal = build_signal(net, &w, &err, seed, ttot, cfg);
                    if cfg.hidden_lr != 0.0 {
                        net.dfa_update(entries, &signal, cfg.hidden_lr);
                    }
                    if cfg.train_omega_b {
                        net.omega_b_update(&signal, cfg.omega_b_lr);
                    }
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

    /// N-bit sequential parity: N deterministic cue bits, label = their XOR. (N=2 is temporal XOR.)
    fn task_parity(seed: u64, t: usize, n: usize) -> (Vec<usize>, usize) {
        let bits: Vec<usize> = (0..n).map(|i| (mix(key(seed, t as u32, 0, i as u32, 51)) & 1) as usize).collect();
        let label = bits.iter().fold(0usize, |a, &b| a ^ b);
        (bits, label)
    }

    /// `[a, distractor, b]` where the middle is a label-irrelevant cue (class 2); label = a XOR b.
    fn task_distractor(seed: u64, trial: usize) -> (Vec<usize>, usize) {
        let a = (mix(key(seed, trial as u32, 0, 0, 51)) & 1) as usize;
        let b = (mix(key(seed, trial as u32, 0, 0, 53)) & 1) as usize;
        (vec![a, 2, b], a ^ b)
    }

    /// `n_ops` set(class 0)/reset(class 1) ops; label = final state (set -> on 1, reset -> off 0).
    fn task_flipflop(seed: u64, trial: usize, n_ops: usize) -> (Vec<usize>, usize) {
        let ops: Vec<usize> = (0..n_ops).map(|i| (mix(key(seed, trial as u32, 0, i as u32, 57)) & 1) as usize).collect();
        let last = *ops.last().unwrap();
        (ops, if last == 0 { 1 } else { 0 })
    }

    #[test]
    fn task_labels_correct() {
        for trial in 0..25 {
            let (bits, label) = task_parity(42, trial, 4);
            assert_eq!(bits.len(), 4);
            assert!(bits.iter().all(|&b| b <= 1));
            assert_eq!(label, bits.iter().fold(0, |a, &b| a ^ b), "parity label is the XOR of the bits");

            let (classes, dlabel) = task_distractor(42, trial);
            assert_eq!(classes.len(), 3);
            assert_eq!(classes[1], 2, "middle cue is the class-2 distractor");
            assert!(classes[0] <= 1 && classes[2] <= 1);
            assert_eq!(dlabel, classes[0] ^ classes[2], "distractor label ignores the middle cue");

            let (ops, flabel) = task_flipflop(42, trial, 4);
            assert_eq!(ops.len(), 4);
            assert!(ops.iter().all(|&o| o <= 1));
            let last = *ops.last().unwrap();
            assert_eq!(flabel, if last == 0 { 1 } else { 0 }, "flip-flop label is the final state");
        }
    }

    fn make_ff(seed: u64, size: u32, layers: usize, up_count: u32, up_radius: u32, theta_c: f32, b_off: (f32, f32), train_omega_b: bool) -> (Network, Vec<Vec<Edge>>) {
        let lc = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
            inhibitor_ratio: 0,
            omega_init: (5.0, 10.0),
            b_offset_init: b_off,
            tau_out: 20.0,
        };
        let mut net = Network::new(Config { seed, size, dt: 0.05, gamma: 0.9, theta_c, layers: vec![lc; layers] });
        // eps_cut small: BRF ε traces are dt-scaled (~0.05·…), so a coarse cut would zero real gradient.
        net.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b });
        net.enable_training();
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: up_count as usize, radius: up_radius }] })
            .collect();
        (net, entries)
    }

    // hidden_lr is ~100× the integer engines' (0.004): BRF's HYPR eligibility is δ-scaled (~0.05·…), so
    // the shadow needs a proportionally larger step to move the ternary codes.
    // Backward-fed side-car (ported from wave_driven_bench::make_sidecar, BRF config):
    // L0→L1(+1); L1→L3(+2 skip); L2 self(0)+L2→L3(+1); L3→L2(−1 back)+L3→L4(+1); L4 read.
    fn make_sidecar(seed: u64, size: u32, uc: u32, ur: u32, n: u32, r: u32, theta_c: f32, b_off: (f32, f32), train_omega_b: bool) -> (Network, Vec<Vec<Edge>>) {
        let mk = |topology| LayerConfig { topology, inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: b_off, tau_out: 20.0 };
        let layers = vec![
            mk(vec![TopologyLevel { level: 1, radius: ur, count: uc }]),
            mk(vec![TopologyLevel { level: 2, radius: ur, count: uc }]),
            mk(vec![TopologyLevel { level: 0, radius: r, count: n }, TopologyLevel { level: 1, radius: r, count: n }]),
            mk(vec![TopologyLevel { level: -1, radius: r, count: n }, TopologyLevel { level: 1, radius: ur, count: uc }]),
            mk(vec![]),
        ];
        let mut net = Network::new(Config { seed, size, dt: 0.05, gamma: 0.9, theta_c, layers });
        net.set_elig_params(EligParams { dt: 0.05, eps_cut: 1e-6, train_omega_b });
        net.enable_training();
        let entries = vec![
            vec![Edge { level: 1, count: uc as usize, radius: ur }],
            vec![Edge { level: 2, count: uc as usize, radius: ur }],
            vec![Edge { level: 0, count: n as usize, radius: r }, Edge { level: 1, count: n as usize, radius: r }],
            vec![Edge { level: -1, count: n as usize, radius: r }, Edge { level: 1, count: uc as usize, radius: ur }],
            vec![],
        ];
        (net, entries)
    }

    #[test]
    #[ignore] // smoke (size 16): side-car builds + is live + trains above chance (--release --nocapture)
    fn wave_resonate_sidecar_smoke_size16() {
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_sidecar(seed, 16, 32, 3, 8, 4, 0.1, (0.0, 0.2), false);
        let (pct, sigma) = rate_profile(&mut net, 16, seed, 0, 16, 48);
        eprintln!("side-car size16 rate% {pct:?} σ≈{sigma:.2}");
        assert!(pct.iter().skip(1).any(|&r| r > 0.5), "side-car compute layers must be live: {pct:?}");
        let mut cfg = ff_cfg();
        cfg.size = 16;
        cfg.present = 6;
        cfg.delay = 8;
        cfg.read = 8;
        let task = |s: u64, t: usize| task_parity(s, t, 2);
        let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, task, 100, 4, 1500);
        eprintln!("side-car size16 temporal-XOR: best {best}@{at}");
        assert!(best > 600, "side-car should clear chance on temporal XOR: {best}");
    }

    fn ff_cfg() -> TaskCfg {
        TaskCfg { size: 16, present: 6, delay: 4, read: 6, holdout: 200, readout_lr: 0.02, hidden_lr: 2.0, rate_reg: 0.0, rate_target: 0.1, train_omega_b: false, omega_b_lr: 0.0 }
    }

    /// Per-layer firing rate (%/neuron/wave) over a cue-driven window + a coarse σ (mean
    /// consecutive-layer spike ratio over `1..l-1`) — the liveness + dynamics diagnostic.
    fn rate_profile(net: &mut Network, size: u32, task_seed: u64, class: usize, warmup: usize, waves: usize) -> (Vec<f64>, f64) {
        let l = net.layer_count();
        let counts = Arc::new(Mutex::new(vec![0u64; l]));
        for z in 0..l {
            let c = counts.clone();
            net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] += f.len() as u64));
        }
        net.reset_state();
        let sites = cue_sites(task_seed, size, class);
        for _ in 0..warmup {
            net.wave(&sites);
        }
        counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
        for _ in 0..waves {
            net.wave(&sites);
        }
        net.clear_listeners();
        let counts = std::mem::take(&mut *counts.lock().unwrap());
        let denom = ((size as u64) * (size as u64) * waves as u64) as f64;
        let pct: Vec<f64> = counts.iter().map(|&s| (s as f64 / denom * 1000.0).round() / 10.0).collect();
        let mut ratios = Vec::new();
        for z in 1..l.saturating_sub(1) {
            if counts[z] > 0 {
                ratios.push(counts[z + 1] as f64 / counts[z] as f64);
            }
        }
        let sigma = if ratios.is_empty() { 0.0 } else { ratios.iter().sum::<f64>() / ratios.len() as f64 };
        (pct, sigma)
    }

    #[test]
    #[ignore] // diagnostic: per-layer liveness across θ_c and b_off (a resonator barely responds to DC) (--release --nocapture)
    fn wave_resonate_liveness_probe() {
        let seed = 0xE9_0B_0A17u64;
        eprintln!("== BRF FF liveness (rate %/neuron/wave under sustained cue), 4 layers r3/c24 ==");
        for &b_off in &[(0.0f32, 0.2f32), (0.1, 1.0)] {
            for &theta_c in &[0.05f32, 0.1, 0.2, 0.5, 1.0] {
                let (mut net, _e) = make_ff(seed, 16, 4, 24, 3, theta_c, b_off, false);
                let (pct, _sigma) = rate_profile(&mut net, 16, seed, 0, 20, 40);
                eprintln!("  b_off {b_off:?} θ_c {theta_c:>4}: rate% {pct:?}");
            }
        }
    }

    #[test]
    #[ignore] // diagnostic: isolate where the class signal is (readout-only vs full) across depth (--release --nocapture)
    fn wave_resonate_ff_diagnose() {
        let seed = 0xE9_0B_0A17u64;
        // Where does the class signal live, and does HYPR credit recover it through depth? readout-only
        // reads the top layer with no hidden training; full also runs dfa_update. (BRF needs a ~100× larger
        // hidden_lr than the integer engines — its ε traces are δ-scaled; see ff_cfg.)
        eprintln!("== BRF FF diagnose (size 16, r3/c24, θ_c 0.1, b_off (0,0.2)) ==");
        for &layers in &[2usize, 3, 4] {
            let (mut n0, e0) = make_ff(seed, 16, layers, 24, 3, 0.1, (0.0, 0.2), false);
            let mut c0 = ff_cfg();
            c0.hidden_lr = 0.0;
            let (ro, _) = train_and_eval_best(&mut n0, &e0, seed, seed, &c0, single_task, 100, 4, 1500);
            let (mut n1, e1) = make_ff(seed, 16, layers, 24, 3, 0.1, (0.0, 0.2), false);
            let (full, at) = train_and_eval_best(&mut n1, &e1, seed, seed, &ff_cfg(), single_task, 100, 4, 1500);
            eprintln!("  layers {layers}: readout-only {ro} | full {full}@{at}");
        }
    }

    #[test]
    #[ignore] // smoke: run manually in --release (BRF f32 + HYPR eligibility is heavier than the integer engines)
    fn wave_resonate_ff_trains_above_chance() {
        // 4-layer BRF FF, size 16, generous fan-in; single-cue 2-class must beat chance. Integration proof
        // that BRF forward + HYPR online eligibility + shadow update + repack + readout trains end-to-end.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 24, 3, 0.1, (0.0, 0.2), false);
        let cfg = ff_cfg();
        let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 4, 1500);
        eprintln!("wave_resonate FF single-cue: best {best}@{at}");
        assert!(best > 600, "BRF+HYPR FF should train above chance: {best}");
    }

    struct Bench {
        name: &'static str,
        present: usize,
        delay: usize,
        read: usize,
        task: Box<dyn Fn(u64, usize) -> (Vec<usize>, usize)>,
    }

    fn temporal_benches() -> Vec<Bench> {
        vec![
            Bench { name: "temporal-XOR", present: 6, delay: 8, read: 8, task: Box::new(|s, t| task_parity(s, t, 2)) },
            Bench { name: "parity-4", present: 6, delay: 8, read: 8, task: Box::new(|s, t| task_parity(s, t, 4)) },
            Bench { name: "distractor-XOR", present: 6, delay: 20, read: 8, task: Box::new(|s, t| task_distractor(s, t)) },
            Bench { name: "flip-flop", present: 6, delay: 12, read: 8, task: Box::new(|s, t| task_flipflop(s, t, 4)) },
        ]
    }

    /// Core comparison: for each task, FF vs side-car × frozen vs trained ω/b′, worst+mean over seeds.
    /// `size`, `seeds`, `max_trials` parameterize the size-16 validation vs the size-32 study.
    fn run_temporal_study(size: u32, seeds: &[u64], max_trials: usize) {
        let (uc, ur, rec, rr) = (32u32, 3u32, 8u32, 4u32);
        eprintln!("== wave_resonate temporal study — size {size}, {} seeds, θ_c 0.1, FF/side-car × frozen/trained ω/b′ ==", seeds.len());
        eprintln!("   {:<15} | FF froz w/m | FF train w/m | side froz w/m | side train w/m | side σ", "task");
        for b in temporal_benches() {
            let mkcfg = |train_ob: bool| {
                let mut c = ff_cfg();
                c.size = size;
                c.present = b.present;
                c.delay = b.delay;
                c.read = b.read;
                c.holdout = 200;
                c.train_omega_b = train_ob;
                // omega_b_lr 1.0: the LR-sweep sweet spot — helps the hard tasks without destabilizing the
                // easy ones (lr 2.0 maxes distractor but breaks temporal-XOR).
                c.omega_b_lr = if train_ob { 1.0 } else { 0.0 };
                c
            };
            // returns (worst, mean)
            let run = |sidecar: bool, train_ob: bool| -> (u64, u64) {
                let bests: Vec<u64> = seeds
                    .iter()
                    .map(|&s| {
                        let (mut net, e) = if sidecar {
                            make_sidecar(s, size, uc, ur, rec, rr, 0.1, (0.0, 0.2), train_ob)
                        } else {
                            make_ff(s, size, 5, uc, ur, 0.1, (0.0, 0.2), train_ob)
                        };
                        train_and_eval_best(&mut net, &e, s, s, &mkcfg(train_ob), b.task.as_ref(), 300, 3, max_trials).0
                    })
                    .collect();
                (*bests.iter().min().unwrap(), bests.iter().sum::<u64>() / bests.len() as u64)
            };
            let ff_f = run(false, false);
            let ff_t = run(false, true);
            let sc_f = run(true, false);
            let sc_t = run(true, true);
            let sig = {
                let (mut net, _e) = make_sidecar(seeds[0], size, uc, ur, rec, rr, 0.1, (0.0, 0.2), false);
                rate_profile(&mut net, size, seeds[0], 0, 16, 64).1
            };
            eprintln!(
                "   {:<15} | {:>4}/{:<4} | {:>4}/{:<4} | {:>4}/{:<4} | {:>4}/{:<4} | {sig:.2}",
                b.name, ff_f.0, ff_f.1, ff_t.0, ff_t.1, sc_f.0, sc_f.1, sc_t.0, sc_t.1
            );
        }
    }

    #[test]
    #[ignore] // validation (size 16, 2 seeds): fast preliminary readout of the full matrix (--release --nocapture)
    fn wave_resonate_temporal_size16() {
        run_temporal_study(16, &[0xE9_0B_0A17u64, 0x1234_5678], 1500);
    }

    #[test]
    #[ignore] // the study (size 32, 3 seeds): the headline comparison — SLOW, run in background (--release --nocapture)
    fn wave_resonate_temporal_size32() {
        run_temporal_study(32, &[0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF], 2400);
    }

    #[test]
    #[ignore] // pick omega_b_lr: does a smaller LR keep the distractor gain without hurting XOR? (--release --nocapture)
    fn wave_resonate_omega_b_lr_sweep() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678];
        eprintln!("== ω/b′ LR sweep (size 16, FF, 2 seeds): temporal-XOR (frozen 1000) vs distractor-XOR (frozen ~515) ==");
        eprintln!("   {:<8} | XOR w/m | distractor w/m", "lr");
        let tasks: [(&str, usize, usize, usize, fn(u64, usize) -> (Vec<usize>, usize)); 2] =
            [("XOR", 8, 8, 2, |s, t| task_parity(s, t, 2)), ("distr", 20, 8, 3, |s, t| task_distractor(s, t))];
        for &lr in &[0.0f32, 0.25, 0.5, 1.0, 2.0] {
            let mut out = String::new();
            for (_name, delay, read, _n, task) in tasks.iter() {
                let bests: Vec<u64> = seeds
                    .iter()
                    .map(|&s| {
                        let train_ob = lr != 0.0;
                        let (mut net, e) = make_ff(s, 16, 5, 32, 3, 0.1, (0.0, 0.2), train_ob);
                        let mut c = ff_cfg();
                        c.size = 16;
                        c.present = 6;
                        c.delay = *delay;
                        c.read = *read;
                        c.holdout = 200;
                        c.train_omega_b = train_ob;
                        c.omega_b_lr = lr;
                        train_and_eval_best(&mut net, &e, s, s, &c, task, 300, 3, 1500).0
                    })
                    .collect();
                let (w, m) = (*bests.iter().min().unwrap(), bests.iter().sum::<u64>() / bests.len() as u64);
                out.push_str(&format!(" {w:>4}/{m:<4} |"));
            }
            eprintln!("   lr {lr:<5} |{out}");
        }
    }

    #[test]
    #[ignore] // smoke: FF trains with ω/b′ trainable; also prints frozen-vs-trained (--release --nocapture)
    fn wave_resonate_ff_trains_with_omega_b() {
        let seed = 0xE9_0B_0A17u64;
        // frozen baseline (ω/b′ fixed at init) for comparison
        let (mut nf, ef) = make_ff(seed, 16, 4, 24, 3, 0.1, (0.0, 0.2), false);
        let (frozen, fa) = train_and_eval_best(&mut nf, &ef, seed, seed, &ff_cfg(), single_task, 100, 4, 1500);
        // trainable ω/b′
        let (mut net, entries) = make_ff(seed, 16, 4, 24, 3, 0.1, (0.0, 0.2), true);
        let mut cfg = ff_cfg();
        cfg.train_omega_b = true;
        cfg.omega_b_lr = 2.0;
        let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 4, 1500);
        let (omin, omax) = net.with_layer(1, |l| (l.omega.iter().cloned().fold(f32::MAX, f32::min), l.omega.iter().cloned().fold(f32::MIN, f32::max)));
        eprintln!("wave_resonate FF: frozen {frozen}@{fa} | ω/b′-trained {best}@{at} (L1 ω range [{omin:.2},{omax:.2}])");
        assert!(best > 600, "BRF+HYPR FF with trainable ω/b′ should clear chance: {best}");
    }
}
