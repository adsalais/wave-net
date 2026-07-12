//! Exploratory benchmarks for `multilayer_dfa` — `#[ignore]`d, print reports, run manually in `--release`:
//! `cargo test --release <name> -- --ignored --nocapture`. Per the AGENTS.md benchmark convention, each
//! sweeps **every** axis it studies — synapse radius/count (recurrent fan-in separate from forward), depth,
//! training duration — across **several seeds** (worst + mean). Reads the **top spiking layer** directly (no
//! dedicated readout layer) and reports, per config: fan-in density, σ branching ratio (untrained substrate,
//! mean over seeds), the mean/worst accuracy curve over duration, and the post-training per-layer firing
//! profile (seed 0). Uses the shared harness in `multilayer_dfa::harness`.

#[cfg(test)]
mod tests {
    use crate::bench::multilayer_dfa::harness::*;

    const SEEDS: [u64; 3] = [0xE9_0B_0A17, 0x1234_5678, 0xDEAD_BEEF];

    /// Fraction of the (2r+1)² receptive window actually sampled — the structural fan-in density.
    fn density(count: u32, radius: u32) -> f64 {
        let w = (2 * radius + 1) as f64;
        count as f64 / (w * w)
    }

    /// Per-layer firing rates → "L0/L1/…" percent string.
    fn fmt_rates(r: &[f64]) -> String {
        r.iter().map(|x| format!("{:.1}", x * 100.0)).collect::<Vec<_>>().join("/")
    }

    /// Accuracy curve → "a0/a1/…" string (one per training-duration checkpoint).
    fn fmt_accs(a: &[u64]) -> String {
        a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("/")
    }

    /// Per-checkpoint mean over seed curves.
    fn mean_curve(curves: &[Vec<u64>]) -> Vec<u64> {
        (0..curves[0].len()).map(|k| curves.iter().map(|c| c[k]).sum::<u64>() / curves.len() as u64).collect()
    }

    /// Per-checkpoint worst (min) over seed curves.
    fn worst_curve(curves: &[Vec<u64>]) -> Vec<u64> {
        (0..curves[0].len()).map(|k| curves.iter().map(|c| c[k]).min().unwrap()).collect()
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn ff_depth_revival() {
        // How deep can a 32×32 feed-forward stack be trained (simple 2-class task; readout = top spiking
        // layer), across depth × fan-in (radius/count) × training duration × seeds? "Comes to life" = deep
        // layers fire + acc ≫ chance (500). present/read waves scaled to depth (else deep layers never get
        // driven → zero eligibility → no revival).
        let ckpts = [300usize, 1500];
        eprintln!("== FF depth revival (size 32, single 2-class task, read top layer; {} seeds) ==", SEEDS.len());
        eprintln!("depth  r/c     dens   σ     mean@{ckpts:?} | worst   per-layer firing % (seed0, post-train)");
        for &depth in &[4usize, 8, 12] {
            for &(ur, uc) in &[(3u32, 16u32), (4, 48)] {
                let (mut curves, mut sig_sum, mut rates0) = (Vec::new(), 0.0f64, Vec::new());
                for (si, &s) in SEEDS.iter().enumerate() {
                    let (mut net, entries) = make_ff(s, 32, depth, uc, ur, DEFAULT_ADAPT_BUMP, 6);
                    sig_sum += sigma_ratio(&mut net, s ^ 0x5A5A);
                    let mut cfg = ff_cfg(0, 0.004, 0.0);
                    cfg.size = 32;
                    cfg.present = depth.max(6);
                    cfg.read = depth.max(6);
                    let accs = train_and_eval_curve(&mut net, &entries, s, s, &cfg, single_task, &ckpts);
                    if si == 0 {
                        rates0 = per_layer_rates(&mut net, s ^ 0x5A5A);
                    }
                    curves.push(accs);
                }
                let sigma = sig_sum / SEEDS.len() as f64;
                eprintln!("{depth:>4}  {ur}/{uc:<3}   {:>4.2}  {sigma:>4.2}  {:>9} | {:<9}  {}", density(uc, ur), fmt_accs(&mean_curve(&curves)), fmt_accs(&worst_curve(&curves)), fmt_rates(&rates0));
            }
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn ff_temporal_xor_depth4() {
        // Feed-forward temporal XOR, 4 layers, 32×32, readout = top spiking layer. Sweep fan-in
        // (radius/count) × training duration × seeds. A starved substrate (σ→0) can still revive with enough
        // training, so the acc curve over checkpoints is the point. (ALIF neurons hold the gap; εᵃ, β=0.4.)
        let ckpts = [800usize, 3000, 6000];
        eprintln!("== FF temporal XOR (4 layers, size 32, delay 8, read top layer; {} seeds) ==", SEEDS.len());
        eprintln!("r/c     dens   σ     mean@{ckpts:?} | worst        per-layer firing % (seed0, post-train)");
        for &(ur, uc) in &[(3u32, 16u32), (4, 64)] {
            let (mut curves, mut sig_sum, mut rates0) = (Vec::new(), 0.0f64, Vec::new());
            for (si, &s) in SEEDS.iter().enumerate() {
                let (mut net, entries) = make_ff(s, 32, 4, uc, ur, DEFAULT_ADAPT_BUMP, 6);
                sig_sum += sigma_ratio(&mut net, s ^ 0x5A5A);
                let mut cfg = ff_cfg(0, 0.004, 0.4);
                cfg.size = 32;
                cfg.delay = 8;
                cfg.holdout = 300;
                let accs = train_and_eval_curve(&mut net, &entries, s, s, &cfg, xor_task, &ckpts);
                if si == 0 {
                    rates0 = per_layer_rates(&mut net, s ^ 0x5A5A);
                }
                curves.push(accs);
            }
            let sigma = sig_sum / SEEDS.len() as f64;
            eprintln!("{ur}/{uc:<3}   {:>4.2}  {sigma:>4.2}  {:>14} | {:<14}  {}", density(uc, ur), fmt_accs(&mean_curve(&curves)), fmt_accs(&worst_curve(&curves)), fmt_rates(&rates0));
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_parity_recurrence() {
        // Parity N=3 on the backward-fed side-car (size 32), readout = top spiking layer. Forward/skip path
        // held at a generous fan-in; sweep the recurrent scratchpad fan-in (radius/count — separate from the
        // forward path) × training duration × seeds. (FF alone cannot do parity N≥3 — does trained recurrence
        // revive it, robustly?)
        let (fur, fuc) = (4u32, 48u32); // generous forward/skip fan-in (held fixed)
        let ckpts = [500usize, 1500, 4000];
        eprintln!("== Side-car parity N=3 (size 32, read top layer; forward r{fur}/c{fuc}; {} seeds) ==", SEEDS.len());
        eprintln!("rec r/c  dens   σ     mean@{ckpts:?} | worst      per-layer firing % (seed0, post-train)");
        for &(rr, rc) in &[(3u32, 8u32), (4, 16), (4, 24)] {
            let (mut curves, mut sig_sum, mut rates0) = (Vec::new(), 0.0f64, Vec::new());
            for (si, &s) in SEEDS.iter().enumerate() {
                let (mut net, entries) = make_sidecar(s, 32, fuc, fur, rc, rr, DEFAULT_ADAPT_BUMP, 6);
                sig_sum += sigma_ratio(&mut net, s ^ 0x5A5A);
                let mut cfg = ff_cfg(0, 0.004, 0.4);
                cfg.size = 32;
                cfg.delay = 8;
                cfg.read = 8;
                cfg.holdout = 300;
                cfg.elig.rec_tau = 20.0;
                let accs = train_and_eval_curve(&mut net, &entries, s, s, &cfg, |sd, t| task_parity(sd, t, 3), &ckpts);
                if si == 0 {
                    rates0 = per_layer_rates(&mut net, s ^ 0x5A5A);
                }
                curves.push(accs);
            }
            let sigma = sig_sum / SEEDS.len() as f64;
            eprintln!("{rr}/{rc:<3}  {:>4.2}  {sigma:>4.2}  {:>13} | {:<13}  {}", density(rc, rr), fmt_accs(&mean_curve(&curves)), fmt_accs(&worst_curve(&curves)), fmt_rates(&rates0));
        }
    }
}
