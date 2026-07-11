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
}
