//! Int8 vs ternary (BitNet) A/B/C benchmarks for `multilayer_dfa` — `#[ignore]`d, run manually in
//! `--release`. Three quantizers on the SAME net/task/seeds: **int8**, **pure ternary** (±1/0), and
//! **scaled ternary** (±g/0, per-row integer gain). Uses early-exit best-checkpoint reporting
//! (`train_and_eval_best`), so each cell is `meanBest/worstBest@meanTrials` — the readable PEAK accuracy
//! (immune to the rate_reg over-training collapse) and the training SPEED (trials to reach it). Plus the
//! ternary weight-sparsity. Per the benchmark convention: fan-in/depth × seeds, read the top spiking layer.

#[cfg(test)]
mod tests {
    use crate::bench::multilayer_dfa::harness::*;
    use crate::wave_net::neurons::WeightQuant;

    const SEEDS: [u64; 3] = [0xE9_0B_0A17, 0x1234_5678, 0xDEAD_BEEF];
    const QUANTS: [(&str, Option<WeightQuant>); 3] = [
        ("int8", None),
        ("pure", Some(WeightQuant::Ternary)),
        ("scaled", Some(WeightQuant::TernaryScaled)),
    ];

    /// (meanBest, worstBest, meanTrialsToBest) over per-seed (best, trials) pairs.
    fn agg(b: &[(u64, usize)]) -> (u64, u64, usize) {
        let n = b.len();
        let mean_best = b.iter().map(|x| x.0).sum::<u64>() / n as u64;
        let worst_best = b.iter().map(|x| x.0).min().unwrap();
        let mean_at = b.iter().map(|x| x.1).sum::<usize>() / n;
        (mean_best, worst_best, mean_at)
    }
    fn cell(a: (u64, u64, usize)) -> String {
        format!("{}/{}@{}", a.0, a.1, a.2)
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_ff_xor() {
        // FF temporal XOR (4 layers, size 32). int8 vs pure vs scaled ternary × fan-in × seeds.
        let (ee, pat, max) = (400usize, 3usize, 5000usize);
        eprintln!("== BitNet A/B/C: FF temporal XOR (4L, size 32, read top; {} seeds; best/worst@trials) ==", SEEDS.len());
        eprintln!("r/c     int8              pure              scaled            sp pure/scaled%");
        for &(ur, uc) in &[(4u32, 32u32), (4, 64)] {
            let (mut cells, mut sp) = (Vec::new(), Vec::new());
            for &(_name, q) in &QUANTS {
                let (mut bests, mut sp0) = (Vec::new(), 0.0);
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_ff(s, 32, 4, uc, ur, 20, 6);
                    if let Some(qq) = q {
                        net.set_weight_quant(qq);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.4);
                    cfg.size = 32;
                    cfg.delay = 8;
                    cfg.holdout = 300;
                    let bt = train_and_eval_best(&mut net, &entries, s, s, &cfg, xor_task, ee, pat, max);
                    if si == 0 {
                        sp0 = weight_sparsity(&net);
                    }
                    bests.push(bt);
                }
                cells.push(cell(agg(&bests)));
                sp.push(sp0);
            }
            eprintln!("{ur}/{uc:<3}   {:<16}  {:<16}  {:<16}  {:.1}/{:.1}", cells[0], cells[1], cells[2], sp[1] * 100.0, sp[2] * 100.0);
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_ff_depth() {
        // FF depth revival (size 32, reviving fan-in r4/c48). Does scaled ternary revive as deep as int8?
        let (ur, uc) = (4u32, 48u32);
        let (ee, pat, max) = (300usize, 3usize, 3000usize);
        eprintln!("== BitNet A/B/C: FF depth revival (size 32, r{ur}/c{uc}, read top; {} seeds; best/worst@trials) ==", SEEDS.len());
        eprintln!("depth  int8              pure              scaled            sp pure/scaled%");
        for &depth in &[4usize, 8, 12] {
            let (mut cells, mut sp) = (Vec::new(), Vec::new());
            for &(_name, q) in &QUANTS {
                let (mut bests, mut sp0) = (Vec::new(), 0.0);
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_ff(s, 32, depth, uc, ur, 20, 6);
                    if let Some(qq) = q {
                        net.set_weight_quant(qq);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.0);
                    cfg.size = 32;
                    cfg.present = depth.max(6);
                    cfg.read = depth.max(6);
                    let bt = train_and_eval_best(&mut net, &entries, s, s, &cfg, single_task, ee, pat, max);
                    if si == 0 {
                        sp0 = weight_sparsity(&net);
                    }
                    bests.push(bt);
                }
                cells.push(cell(agg(&bests)));
                sp.push(sp0);
            }
            eprintln!("{depth:>4}   {:<16}  {:<16}  {:<16}  {:.1}/{:.1}", cells[0], cells[1], cells[2], sp[1] * 100.0, sp[2] * 100.0);
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_sidecar_parity() {
        // Side-car parity N=3 (size 32, forward r4/c48). Does scaled ternary's per-row gain let the
        // recurrent loop train where pure ternary failed?
        let (fur, fuc) = (4u32, 48u32);
        let (ee, pat, max) = (500usize, 3usize, 5000usize);
        eprintln!("== BitNet A/B/C: side-car parity N=3 (size 32, forward r{fur}/c{fuc}, read top; {} seeds; best/worst@trials) ==", SEEDS.len());
        eprintln!("rec r/c  int8              pure              scaled            sp pure/scaled%");
        for &(rr, rc) in &[(4u32, 16u32), (4, 24)] {
            let (mut cells, mut sp) = (Vec::new(), Vec::new());
            for &(_name, q) in &QUANTS {
                let (mut bests, mut sp0) = (Vec::new(), 0.0);
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_sidecar(s, 32, fuc, fur, rc, rr, 20, 6);
                    if let Some(qq) = q {
                        net.set_weight_quant(qq);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.4);
                    cfg.size = 32;
                    cfg.delay = 8;
                    cfg.read = 8;
                    cfg.holdout = 300;
                    cfg.elig.rec_tau = 20.0;
                    let bt = train_and_eval_best(&mut net, &entries, s, s, &cfg, |sd, t| task_parity(sd, t, 3), ee, pat, max);
                    if si == 0 {
                        sp0 = weight_sparsity(&net);
                    }
                    bests.push(bt);
                }
                cells.push(cell(agg(&bests)));
                sp.push(sp0);
            }
            eprintln!("{rr}/{rc:<3}  {:<16}  {:<16}  {:<16}  {:.1}/{:.1}", cells[0], cells[1], cells[2], sp[1] * 100.0, sp[2] * 100.0);
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_ff_depth_fanin() {
        // Ternary needs more fan-in than int8 (the XOR r4/c32→c64 jump). Sweep forward (radius, count) at
        // depths 8 & 12 to see how much fan-in each quantizer needs to train deep. SINGLE seed (fast; noisy).
        // Reports best@trials-to-peak per quantizer + ternary sparsity.
        let seed = 0xE9_0B_0A17u64;
        let (ee, pat, max) = (300usize, 3usize, 3000usize);
        eprintln!("== BitNet A/B/C: FF depth × fan-in (size 32, read top; seed {seed:#x}; best@trials) ==");
        eprintln!("depth  r/c      int8            pure            scaled          sp pure/scaled%");
        for &depth in &[8usize, 12] {
            for &(ur, uc) in &[(3u32, 48u32), (4, 64), (5, 96), (5, 128)] {
                let (mut cells, mut sp) = (Vec::new(), Vec::new());
                for &(_name, q) in &QUANTS {
                    let (mut net, entries) = make_ff(seed, 32, depth, uc, ur, 20, 6);
                    if let Some(qq) = q {
                        net.set_weight_quant(qq);
                    }
                    let mut cfg = ff_cfg(0, 0.004, 0.0);
                    cfg.size = 32;
                    cfg.present = depth.max(6);
                    cfg.read = depth.max(6);
                    let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, ee, pat, max);
                    cells.push(format!("{best}@{at}"));
                    sp.push(weight_sparsity(&net));
                }
                eprintln!("{depth:>4}  {ur}/{uc:<3}   {:<14}  {:<14}  {:<14}  {:.1}/{:.1}", cells[0], cells[1], cells[2], sp[1] * 100.0, sp[2] * 100.0);
            }
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_ff_depth8_scaled_band() {
        // Probe the scaled-ternary "bump" at depth 8 / r4/c64 (worked while r3/c48, r5/c96, r5/c128 failed —
        // but those vary radius too). Fine count sweep at FIXED radius 4: if scaled trains across a BAND, its
        // gain has a real fan-in window; if only c64, likely single-seed noise. depth 8, single seed.
        let seed = 0xE9_0B_0A17u64;
        let (ee, pat, max) = (300usize, 3usize, 3000usize);
        eprintln!("== BitNet: FF depth-8 scaled band, r4 × count (size 32, read top; seed {seed:#x}; best@trials) ==");
        eprintln!("r/c      int8            pure            scaled          sp pure/scaled%");
        for &uc in &[48u32, 56, 64, 72, 80, 96] {
            let (mut cells, mut sp) = (Vec::new(), Vec::new());
            for &(_name, q) in &QUANTS {
                let (mut net, entries) = make_ff(seed, 32, 8, uc, 4, 20, 6);
                if let Some(qq) = q {
                    net.set_weight_quant(qq);
                }
                let mut cfg = ff_cfg(0, 0.004, 0.0);
                cfg.size = 32;
                cfg.present = 8;
                cfg.read = 8;
                let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, single_task, ee, pat, max);
                cells.push(format!("{best}@{at}"));
                sp.push(weight_sparsity(&net));
            }
            eprintln!("4/{uc:<3}   {:<14}  {:<14}  {:<14}  {:.1}/{:.1}", cells[0], cells[1], cells[2], sp[1] * 100.0, sp[2] * 100.0);
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn bitnet_sidecar_parity_rec() {
        // Was PURE ternary's side-car failure just starvation? It failed at rec 4/16 & 4/24 — sweep MORE
        // recurrent count (radius 4, count 24→48), forward held r4/c48. single seed. best@trials + sparsity.
        let (fur, fuc) = (4u32, 48u32);
        let seed = 0xE9_0B_0A17u64;
        let (ee, pat, max) = (500usize, 3usize, 5000usize);
        eprintln!("== BitNet: side-car parity N=3, rec r4 × count (size 32, forward r{fur}/c{fuc}; seed {seed:#x}; best@trials) ==");
        eprintln!("rec r/c  int8            pure            scaled          sp pure/scaled%");
        for &rc in &[24u32, 32, 48] {
            let (mut cells, mut sp) = (Vec::new(), Vec::new());
            for &(_name, q) in &QUANTS {
                let (mut net, entries) = make_sidecar(seed, 32, fuc, fur, rc, 4, 20, 6);
                if let Some(qq) = q {
                    net.set_weight_quant(qq);
                }
                let mut cfg = ff_cfg(0, 0.004, 0.4);
                cfg.size = 32;
                cfg.delay = 8;
                cfg.read = 8;
                cfg.holdout = 300;
                cfg.elig.rec_tau = 20.0;
                let (best, at) = train_and_eval_best(&mut net, &entries, seed, seed, &cfg, |sd, t| task_parity(sd, t, 3), ee, pat, max);
                cells.push(format!("{best}@{at}"));
                sp.push(weight_sparsity(&net));
            }
            eprintln!("4/{rc:<3}   {:<14}  {:<14}  {:<14}  {:.1}/{:.1}", cells[0], cells[1], cells[2], sp[1] * 100.0, sp[2] * 100.0);
        }
    }
}
