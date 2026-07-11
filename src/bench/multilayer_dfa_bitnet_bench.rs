//! Int8-vs-ternary (BitNet) A/B benchmarks for `multilayer_dfa` — `#[ignore]`d, run manually in `--release`.
//! Same net/task/seeds trained under Int8 vs Ternary (per-row γ, pure ±1/0). Per the benchmark convention:
//! sweep fan-in × duration × seeds, read the top spiking layer, report accuracy, and (for ternary) the
//! weight-sparsity. Compare at the PEAK of the duration curve (the rate_reg over-training collapse is
//! documented). Question: does per-row pure-ternary reach the int8 baseline, and at what sparsity?

#[cfg(test)]
mod tests {
    use crate::bench::multilayer_dfa::harness::*;
    use crate::wave_net::neurons::WeightQuant;

    const SEEDS: [u64; 3] = [0xE9_0B_0A17, 0x1234_5678, 0xDEAD_BEEF];

    fn fmt_accs(a: &[u64]) -> String {
        a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("/")
    }
    fn fmt_rates(r: &[f64]) -> String {
        r.iter().map(|x| format!("{:.1}", x * 100.0)).collect::<Vec<_>>().join("/")
    }
    fn mean_curve(c: &[Vec<u64>]) -> Vec<u64> {
        (0..c[0].len()).map(|k| c.iter().map(|r| r[k]).sum::<u64>() / c.len() as u64).collect()
    }
    fn worst_curve(c: &[Vec<u64>]) -> Vec<u64> {
        (0..c[0].len()).map(|k| c.iter().map(|r| r[k]).min().unwrap()).collect()
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_ff_xor() {
        // FF temporal XOR (4 layers, size 32), Int8 vs Ternary, fan-in × duration × seeds. Reports each
        // quantizer's mean/worst acc curve; for ternary also the final weight-sparsity (% pruned to 0).
        let ckpts = [800usize, 2500];
        eprintln!("== BitNet A/B: FF temporal XOR (4L, size 32, read top; {} seeds) ==", SEEDS.len());
        eprintln!("r/c     int8 mean|worst @{ckpts:?}    ternary mean|worst @{ckpts:?}   tern.sparsity%");
        for &(ur, uc) in &[(4u32, 32u32), (4, 64)] {
            let run = |ternary: bool| -> (Vec<Vec<u64>>, f64) {
                let (mut curves, mut sparsity0) = (Vec::new(), 0.0);
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_ff(s, 32, 4, uc, ur, 20, 6);
                    if ternary {
                        net.set_weight_quant(WeightQuant::Ternary);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.4);
                    cfg.size = 32;
                    cfg.delay = 8;
                    cfg.holdout = 300;
                    let accs = train_and_eval_curve(&mut net, &entries, s, s, &cfg, xor_task, &ckpts);
                    if si == 0 {
                        sparsity0 = weight_sparsity(&net);
                    }
                    curves.push(accs);
                }
                (curves, sparsity0)
            };
            let (i8c, _) = run(false);
            let (tc, tsp) = run(true);
            eprintln!(
                "{ur}/{uc:<3}   {:>9} | {:<9}   {:>9} | {:<9}   {:.1}",
                fmt_accs(&mean_curve(&i8c)),
                fmt_accs(&worst_curve(&i8c)),
                fmt_accs(&mean_curve(&tc)),
                fmt_accs(&worst_curve(&tc)),
                tsp * 100.0
            );
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_ff_depth() {
        // Int8 vs Ternary depth revival: does ternary train as deep as int8? Fan-in held at the reviving
        // r4/c48; sweep depth × duration × seeds (present/read scaled to depth). Reports int8 vs ternary acc,
        // ternary sparsity, and ternary per-layer firing (seed 0) so we can see whether ternary keeps deep
        // layers alive.
        let ckpts = [300usize, 1500];
        let (ur, uc) = (4u32, 48u32);
        eprintln!("== BitNet A/B: FF depth revival (size 32, r{ur}/c{uc}, read top; {} seeds) ==", SEEDS.len());
        eprintln!("depth  int8 mean|worst @{ckpts:?}   ternary mean|worst @{ckpts:?}  tern.sp%  tern.rates(seed0)");
        for &depth in &[4usize, 8, 12] {
            let run = |ternary: bool| -> (Vec<Vec<u64>>, f64, Vec<f64>) {
                let (mut curves, mut sp0, mut rates0) = (Vec::new(), 0.0, Vec::new());
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_ff(s, 32, depth, uc, ur, 20, 6);
                    if ternary {
                        net.set_weight_quant(WeightQuant::Ternary);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.0);
                    cfg.size = 32;
                    cfg.present = depth.max(6);
                    cfg.read = depth.max(6);
                    let accs = train_and_eval_curve(&mut net, &entries, s, s, &cfg, single_task, &ckpts);
                    if si == 0 {
                        sp0 = weight_sparsity(&net);
                        rates0 = per_layer_rates(&mut net, s ^ 0x5A5A);
                    }
                    curves.push(accs);
                }
                (curves, sp0, rates0)
            };
            let (i8c, _, _) = run(false);
            let (tc, tsp, trates) = run(true);
            eprintln!(
                "{depth:>4}   {:>9} | {:<9}   {:>9} | {:<9}  {:>5.1}   {}",
                fmt_accs(&mean_curve(&i8c)), fmt_accs(&worst_curve(&i8c)),
                fmt_accs(&mean_curve(&tc)), fmt_accs(&worst_curve(&tc)), tsp * 100.0, fmt_rates(&trates)
            );
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_sidecar_parity() {
        // Int8 vs Ternary on the side-car parity N=3 (size 32, forward r4/c48). Sweep recurrent fan-in ×
        // duration × seeds. Does the ternary recurrence still train the non-monotone parity task?
        let (fur, fuc) = (4u32, 48u32);
        let ckpts = [500usize, 2000];
        eprintln!("== BitNet A/B: side-car parity N=3 (size 32, forward r{fur}/c{fuc}, read top; {} seeds) ==", SEEDS.len());
        eprintln!("rec r/c  int8 mean|worst @{ckpts:?}   ternary mean|worst @{ckpts:?}  tern.sp%");
        for &(rr, rc) in &[(4u32, 16u32), (4, 24)] {
            let run = |ternary: bool| -> (Vec<Vec<u64>>, f64) {
                let (mut curves, mut sp0) = (Vec::new(), 0.0);
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_sidecar(s, 32, fuc, fur, rc, rr, 20, 6);
                    if ternary {
                        net.set_weight_quant(WeightQuant::Ternary);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.4);
                    cfg.size = 32;
                    cfg.delay = 8;
                    cfg.read = 8;
                    cfg.holdout = 300;
                    cfg.elig.rec_tau = 20.0;
                    let accs = train_and_eval_curve(&mut net, &entries, s, s, &cfg, |sd, t| task_parity(sd, t, 3), &ckpts);
                    if si == 0 {
                        sp0 = weight_sparsity(&net);
                    }
                    curves.push(accs);
                }
                (curves, sp0)
            };
            let (i8c, _) = run(false);
            let (tc, tsp) = run(true);
            eprintln!(
                "{rr}/{rc:<3}  {:>9} | {:<9}   {:>9} | {:<9}  {:>5.1}",
                fmt_accs(&mean_curve(&i8c)), fmt_accs(&worst_curve(&i8c)),
                fmt_accs(&mean_curve(&tc)), fmt_accs(&worst_curve(&tc)), tsp * 100.0
            );
        }
    }
}
