//! Exploratory benchmarks for `multilayer_dfa` — `#[ignore]`d, print reports, run manually in `--release`:
//! `cargo test --release <name> -- --ignored --nocapture`. Each sweeps synapse radius/count to find where
//! training "comes to life", reads the **top spiking layer** directly (no dedicated readout layer), and
//! reports the chosen radius/count + fan-in density, the σ branching ratio (untrained substrate), the
//! per-layer firing-rate profile (post-training), and held-out accuracy. Uses the shared harness in
//! `multilayer_dfa::harness`.

#[cfg(test)]
mod tests {
    use crate::bench::multilayer_dfa::harness::*;

    /// Fraction of the (2r+1)² receptive window actually sampled — the structural fan-in density.
    fn density(count: u32, radius: u32) -> f64 {
        let w = (2 * radius + 1) as f64;
        count as f64 / (w * w)
    }

    /// Per-layer firing rates → "L0/L1/…" percent string.
    fn fmt_rates(r: &[f64]) -> String {
        r.iter().map(|x| format!("{:.1}", x * 100.0)).collect::<Vec<_>>().join("/")
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn ff_depth_revival() {
        // How deep can a 32×32 feed-forward stack be trained (simple 2-class task; readout = top spiking
        // layer), and which fan-in (radius/count) revives it? Increase depth to find the wall; sweep fan-in
        // to find where it comes to life. Per (depth, radius, count): σ branching ratio (untrained
        // substrate), post-training per-layer firing %, held-out acc. "Comes to life" = deep layers fire +
        // acc ≫ chance (500).
        let seed = 0xE9_0B_0A17u64;
        eprintln!("== FF depth revival (size 32, single 2-class task, read top layer; present/read scaled to depth) ==");
        eprintln!("depth  r/c     dens   σ     acc   per-layer firing % (L0..Ltop)");
        let (mut deepest, mut chosen) = (0usize, (0u32, 0u32, 0u64));
        for &depth in &[4usize, 8, 12] {
            for &(ur, uc) in &[(3u32, 16u32), (4, 48), (6, 96)] {
                let (mut net, entries) = make_ff(seed, 32, depth, uc, ur, 20, 6);
                let sigma = sigma_ratio(&mut net, seed ^ 0x5A5A);
                let mut cfg = ff_cfg(300, 0.004, 0.0);
                cfg.size = 32;
                cfg.present = depth.max(6); // signal needs ~depth waves to climb the stack
                cfg.read = depth.max(6);
                let acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, single_task);
                let rates = per_layer_rates(&mut net, seed ^ 0x5A5A);
                eprintln!("{depth:>4}  {ur}/{uc:<3}   {:>4.2}  {sigma:>4.2}  {acc:>4}   {}", density(uc, ur), fmt_rates(&rates));
                if acc > 700 && depth >= deepest {
                    deepest = depth;
                    chosen = (ur, uc, acc);
                }
            }
        }
        if deepest > 0 {
            eprintln!("chosen: deepest revived = depth {deepest}, fan-in r{}/c{} (acc {})", chosen.0, chosen.1, chosen.2);
        } else {
            eprintln!("chosen: no config revived above acc 700");
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn ff_temporal_xor_depth4() {
        // Feed-forward temporal XOR, 4 layers, 32×32, readout = top spiking layer. Sweep fan-in
        // (radius/count) to find where training comes to life. Per config: σ, post-training per-layer
        // firing %, held-out acc. (ALIF neurons hold the gap; εᵃ eligibility, β=0.4.)
        let seed = 0xE9_0B_0A17u64;
        eprintln!("== FF temporal XOR (4 layers, size 32, delay 8, read top layer) ==");
        eprintln!("r/c     dens   σ     acc   per-layer firing % (L0..L3)");
        let mut chosen = (0u32, 0u32, 0u64);
        for &(ur, uc) in &[(3u32, 16u32), (4, 32), (4, 64), (4, 96)] {
            let (mut net, entries) = make_ff(seed, 32, 4, uc, ur, 20, 6);
            let sigma = sigma_ratio(&mut net, seed ^ 0x5A5A);
            let mut cfg = ff_cfg(800, 0.004, 0.4);
            cfg.size = 32;
            cfg.delay = 8;
            cfg.holdout = 300;
            let acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, xor_task);
            let rates = per_layer_rates(&mut net, seed ^ 0x5A5A);
            eprintln!("{ur}/{uc:<3}   {:>4.2}  {sigma:>4.2}  {acc:>4}   {}", density(uc, ur), fmt_rates(&rates));
            if acc > chosen.2 {
                chosen = (ur, uc, acc);
            }
        }
        eprintln!("chosen (best): r{}/c{} (acc {})", chosen.0, chosen.1, chosen.2);
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_parity_recurrence() {
        // Parity N=3 on the backward-fed side-car (size 32), readout = top spiking layer. The forward/skip
        // path is held at a generous fan-in; the recurrent scratchpad may want a DIFFERENT radius/count, so
        // sweep it. Per config: σ, post-training per-layer firing %, held-out acc. (FF alone cannot do
        // parity N≥3 — this asks whether the trained recurrence revives it.)
        let seed = 0xE9_0B_0A17u64;
        let (fur, fuc) = (4u32, 48u32); // generous forward/skip fan-in (held fixed)
        eprintln!("== Side-car parity N=3 (size 32, read top layer; forward r{fur}/c{fuc}) ==");
        eprintln!("rec r/c  dens   σ     acc   per-layer firing % (L0..L4)");
        let mut chosen = (0u32, 0u32, 0u64);
        for &(rr, rc) in &[(3u32, 8u32), (4, 16), (4, 24), (4, 32)] {
            let (mut net, entries) = make_sidecar(seed, 32, fuc, fur, rc, rr, 20, 6);
            let sigma = sigma_ratio(&mut net, seed ^ 0x5A5A);
            let mut cfg = ff_cfg(1000, 0.004, 0.4);
            cfg.size = 32;
            cfg.delay = 8;
            cfg.read = 8;
            cfg.holdout = 300;
            cfg.elig.rec_tau = 20.0; // eligibility trace spans the gap for the recurrent path
            let acc = train_and_eval(&mut net, &entries, seed, seed, &cfg, |s, t| task_parity(s, t, 3));
            let rates = per_layer_rates(&mut net, seed ^ 0x5A5A);
            eprintln!("{rr}/{rc:<3}  {:>4.2}  {sigma:>4.2}  {acc:>4}   {}", density(rc, rr), fmt_rates(&rates));
            if acc > chosen.2 {
                chosen = (rr, rc, acc);
            }
        }
        eprintln!("chosen (best rec fan-in): r{}/c{} (acc {})", chosen.0, chosen.1, chosen.2);
    }
}
