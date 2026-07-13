//! FF harness + smoke benchmark for the `wave_bitnet` engine (test-only). A minimal port of the
//! `multilayer_dfa` harness onto `wave_bitnet`: it proves the bitset engine trains end-to-end (FF
//! depth-8 pure ternary → ~1000, parity with wave_net) and reports the per-synapse memory footprint.

#[cfg(test)]
mod tests {
    use crate::wave_bitnet::config::{Config, LayerConfig};
    use crate::wave_bitnet::multilayer_dfa::{multilayer_dfa_step, Edge, EligParams, TrialRecords, PSI_WIDTH};
    use crate::wave_bitnet::network::Network;
    use crate::wave_bitnet::synapse::{key, mix, TopologyLevel};
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

    fn run_trial(net: &mut Network, size: u32, classes: &[usize], task_seed: u64, present: usize, delay: usize, read: usize) -> (Vec<f32>, TrialRecords) {
        let l = net.layer_count();
        let ls = (size * size) as usize;
        let top = l - 1;
        let spikes_acc: Arc<Mutex<Vec<Vec<Vec<u32>>>>> = Arc::new(Mutex::new(vec![Vec::new(); l]));
        for z in 0..l {
            let acc = spikes_acc.clone();
            net.on_layer(z, Box::new(move |_w, fired: &[u32]| acc.lock().unwrap()[z].push(fired.to_vec())));
        }
        net.reset_state();
        let mut pots: Vec<Vec<Vec<i16>>> = vec![Vec::new(); l];
        let mut effs: Vec<Vec<Vec<i32>>> = vec![Vec::new(); l];
        let snapshot = |net: &Network, pots: &mut Vec<Vec<Vec<i16>>>, effs: &mut Vec<Vec<Vec<i32>>>| {
            for z in 0..l {
                pots[z].push(net.layer_decide_potential(z));
                effs[z].push(net.layer_decide_effective_threshold(z));
            }
        };
        for (pos, &class) in classes.iter().enumerate() {
            if pos > 0 {
                for _ in 0..delay {
                    net.wave(&[]);
                    snapshot(net, &mut pots, &mut effs);
                }
            }
            for _ in 0..present {
                let sites = cue_sites(task_seed, size, class);
                net.wave(&sites);
                snapshot(net, &mut pots, &mut effs);
            }
        }
        let read_start = spikes_acc.lock().unwrap()[top].len();
        for _ in 0..read {
            net.wave(&[]);
            snapshot(net, &mut pots, &mut effs);
        }
        net.clear_listeners();
        let spikes = spikes_acc.lock().unwrap().clone();
        let mut act = vec![0f32; ls];
        for wv in spikes[top].iter().skip(read_start) {
            for &loc in wv {
                act[loc as usize] += 1.0;
            }
        }
        (act, TrialRecords { spikes, pots, effs })
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
        elig: EligParams,
    }

    fn build_signal(rec: &TrialRecords, w: &[Vec<f32>], err: &[f32], seed: u64, l: usize, ls: usize, top: usize, cfg: &TaskCfg) -> Vec<Vec<f32>> {
        let ttot = rec.spikes[top].len().max(1) as f32;
        let mut signal = vec![vec![0f32; ls]; l];
        for tz in 1..l {
            for j in 0..ls {
                let task_sig: f32 = (0..2)
                    .map(|c| {
                        let b = if tz == top { w[c][j] } else { dfa_weight(seed, (tz * ls + j) as u32, c) };
                        b * err[c]
                    })
                    .sum();
                let fired_j = rec.spikes[tz].iter().filter(|wv| wv.contains(&(j as u32))).count() as f32;
                let rate = fired_j / ttot;
                signal[tz][j] = task_sig + cfg.rate_reg * (rate - cfg.rate_target);
            }
        }
        signal
    }

    /// Train with periodic held-out eval; return `(best held-out permille, trials-at-best)`.
    fn train_and_eval_best(net: &mut Network, entries: &[Vec<Edge>], seed: u64, task_seed: u64, cfg: &TaskCfg, task: impl Fn(u64, usize) -> (Vec<usize>, usize), eval_every: usize, patience: usize, max_trials: usize) -> (u64, usize) {
        const EVAL_OFFSET: usize = 10_000_000;
        let l = net.layer_count();
        let ls = (cfg.size * cfg.size) as usize;
        let top = l - 1;
        let mut w = vec![vec![0f32; ls]; 2];
        let score = |w: &[Vec<f32>], a: &[f32]| -> (f32, f32) {
            (w[0].iter().zip(a).map(|(x, y)| x * y).sum(), w[1].iter().zip(a).map(|(x, y)| x * y).sum())
        };
        let (mut best, mut best_at, mut stale, mut t) = (0u64, 0usize, 0usize, 0usize);
        while t < max_trials {
            let stop = (t + eval_every).min(max_trials);
            while t < stop {
                let (classes, label) = task(task_seed, t);
                let (act, rec) = run_trial(net, cfg.size, &classes, task_seed, cfg.present, cfg.delay, cfg.read);
                let (s0, s1) = score(&w, &act);
                let (p0, p1) = softmax2(s0, s1);
                let err = [p0 - if label == 0 { 1.0 } else { 0.0 }, p1 - if label == 1 { 1.0 } else { 0.0 }];
                for c in 0..2 {
                    for j in 0..ls {
                        w[c][j] -= cfg.readout_lr * err[c] * act[j];
                    }
                }
                if cfg.hidden_lr != 0.0 {
                    let signal = build_signal(&rec, &w, &err, seed, l, ls, top, cfg);
                    multilayer_dfa_step(net, entries, &rec, &signal, cfg.hidden_lr, &cfg.elig);
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

    /// Single-cue 2-class separable task: present class c, label = c.
    fn single_task(seed: u64, t: usize) -> (Vec<usize>, usize) {
        let c = (mix(key(seed, t as u32, 0, 0, 71)) & 1) as usize;
        (vec![c], c)
    }

    fn ff_cfg(hidden_lr: f32, elig_beta: f32) -> TaskCfg {
        TaskCfg {
            size: 8,
            present: 6,
            delay: 4,
            read: 6,
            holdout: 200,
            readout_lr: 0.02,
            hidden_lr,
            rate_reg: 5.0,
            rate_target: 0.1,
            elig: EligParams { rec_tau: 6.0, elig_beta, elig_psi_width: PSI_WIDTH, use_bump: elig_beta != 0.0, adapt_decay: 6 },
        }
    }

    /// Fraction of pruned (nonzero == 0) weights over the computational layers `1..L`.
    fn weight_sparsity(net: &Network) -> f64 {
        let l = net.layer_count();
        let (mut zeros, mut total) = (0usize, 0usize);
        for z in 1..l {
            net.with_layer(z, |lz| {
                let n = lz.shadow.len(); // ls * total_slots
                total += n;
                zeros += (0..n).filter(|&s| lz.weight_at(s) == 0).count();
            });
        }
        if total == 0 { 0.0 } else { zeros as f64 / total as f64 }
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
        let net = Network::new(Config { seed, size, layers: vec![lc; layers] });
        let entries = (0..layers)
            .map(|z| if z == layers - 1 { vec![] } else { vec![Edge { level: 1, count: up_count as usize, radius: up_radius }] })
            .collect();
        (net, entries)
    }

    /// (training bytes/synapse, at-rest bytes/synapse) for the whole net. At rest the f32 shadow is dropped.
    fn bytes_per_synapse(net: &Network) -> (f64, f64) {
        let l = net.layer_count();
        let (mut syn, mut occ_bits) = (0usize, 0usize);
        for z in 0..l {
            net.with_layer(z, |lz| {
                let ls = lz.potential.len();
                syn += ls * lz.total_slots;
                for &wpn in &lz.occ_wpn {
                    occ_bits += ls * wpn * 64; // word-aligned occupancy storage
                }
            });
        }
        if syn == 0 {
            return (0.0, 0.0);
        }
        let weight_bits = 2.0 * syn as f64; // nonzero + sign
        let occ = occ_bits as f64;
        let shadow_bytes = 4.0 * syn as f64;
        let train = ((weight_bits + occ) / 8.0 + shadow_bytes) / syn as f64;
        let rest = ((weight_bits + occ) / 8.0) / syn as f64;
        (train, rest)
    }

    #[test]
    #[ignore] // diagnostic: find a threshold that keeps a uniform 5-layer FF alive (no critical_init)
    fn bitnet_rate_sweep() {
        use crate::wave_bitnet::synapse::random_l0_input;
        let (size, seed) = (32u32, 0xC0FFEE_1234_5678u64);
        for &bi in &[0i16, 1, 2, 3, 4, 6] {
            let lc = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 3, count: 32 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 32, baseline_init: bi, adapt_bump: 5, adapt_decay: 6 };
            let mut net = Network::new(Config { seed, size, layers: vec![lc; 5] });
            net.set_record_eligibility(false);
            let input = random_l0_input(seed, size, 20000);
            let l = net.layer_count();
            let counts = Arc::new(Mutex::new(vec![0u64; l]));
            for z in 0..l {
                let c = counts.clone();
                net.on_layer(z, Box::new(move |_w, f: &[u32]| c.lock().unwrap()[z] += f.len() as u64));
            }
            net.reset_state();
            for w in 0..32 { net.wave(&input(w)); }
            counts.lock().unwrap().iter_mut().for_each(|x| *x = 0);
            for w in 0..128 { net.wave(&input(32 + w)); }
            net.clear_listeners();
            let counts = std::mem::take(&mut *counts.lock().unwrap());
            let ls = (size as u64) * (size as u64);
            let pct: Vec<f64> = counts.iter().map(|&s| (s as f64 / (ls * 128) as f64 * 1000.0).round() / 10.0).collect();
            eprintln!("baseline_init {bi}: per-layer rate % {pct:?}");
        }
    }

    #[test]
    fn wave_bitnet_trains_above_chance() {
        // 4-layer FF, size 16, generous fan-in; 2-class separable task must beat chance. Integration proof
        // that the whole bitset engine (forward + eligibility + shadow update + repack) trains end-to-end.
        let seed = 0xE9_0B_0A17u64;
        let (mut net, entries) = make_ff(seed, 16, 4, 32, 3, 5, 6);
        let mut cfg = ff_cfg(0.004, 0.0);
        cfg.size = 16;
        let (best, _at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, 100, 3, 1000);
        assert!(best > 600, "wave_bitnet FF should train above chance: {best}");
    }

    #[test]
    #[ignore] // smoke: run manually in --release
    fn wave_bitnet_ff_depth8_smoke() {
        let seeds = [0xE9_0B_0A17u64, 0x1234_5678, 0xDEAD_BEEF];
        eprintln!("== wave_bitnet FF depth-8 pure ternary smoke (r4/c48, adapt=5) ==");
        let mut bests = Vec::new();
        for &s in &seeds {
            let (mut net, entries) = make_ff(s, 32, 8, 48, 4, 5, 6);
            let mut cfg = ff_cfg(0.004, 0.0);
            cfg.size = 32;
            cfg.present = 8;
            cfg.read = 8;
            cfg.holdout = 300;
            let (best, at) = train_and_eval_best(&mut net, &entries, s, s, &cfg, single_task, 300, 3, 3000);
            let (tr, rest) = bytes_per_synapse(&net);
            eprintln!("seed {s:#x}: best {best}@{at}, sparsity {:.0}%, {tr:.2} B/syn train, {rest:.2} B/syn at-rest", weight_sparsity(&net) * 100.0);
            bests.push(best);
        }
        let mean = bests.iter().sum::<u64>() / bests.len() as u64;
        let worst = *bests.iter().min().unwrap();
        eprintln!("mean {mean}, worst {worst} (target ~1000)");
        assert!(worst >= 900, "pure ternary FF depth-8 should hold ~1000 (worst {worst})");
    }
}
