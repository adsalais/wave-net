//! Exploratory benchmarks for `multilayer_dfa` — `#[ignore]`d, print reports, run manually in `--release`:
//! `cargo test --release <name> -- --ignored --nocapture`. Each sweeps synapse radius/count AND training
//! duration to find where training "comes to life", reads the **top spiking layer** directly (no dedicated
//! readout layer), and reports the fan-in density, σ branching ratio (untrained substrate), an accuracy
//! **curve over training duration** (`acc@` each checkpoint), and the post-training per-layer firing profile.
//! Uses the shared harness in `multilayer_dfa::harness`.

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

    /// Accuracy curve → "a0/a1/…" string (one per training-duration checkpoint).
    fn fmt_accs(a: &[u64]) -> String {
        a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("/")
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn ff_depth_revival() {
        // How deep can a 32×32 feed-forward stack be trained (simple 2-class task; readout = top spiking
        // layer), and which fan-in (radius/count) revives it — and how does that trade against training
        // DURATION? Per (depth, radius, count): σ branching ratio (untrained substrate), an acc curve over
        // trial checkpoints, post-training per-layer firing %. "Comes to life" = deep layers fire + acc ≫
        // chance (500). present/read waves scaled to depth (else deep layers never get driven).
        let seed = 0xE9_0B_0A17u64;
        let ckpts = [300usize, 1500];
        eprintln!("== FF depth revival (size 32, single 2-class task, read top layer) ==");
        eprintln!("depth  r/c     dens   σ     acc@{ckpts:?}   per-layer firing % (L0..Ltop, post-train)");
        for &depth in &[4usize, 8, 12] {
            for &(ur, uc) in &[(3u32, 16u32), (4, 48)] {
                let (mut net, entries) = make_ff(seed, 32, depth, uc, ur, 20, 6);
                let sigma = sigma_ratio(&mut net, seed ^ 0x5A5A);
                let mut cfg = ff_cfg(0, 0.004, 0.0);
                cfg.size = 32;
                cfg.present = depth.max(6); // signal needs ~depth waves to climb the stack
                cfg.read = depth.max(6);
                let accs = train_and_eval_curve(&mut net, &entries, seed, seed, &cfg, single_task, &ckpts);
                let rates = per_layer_rates(&mut net, seed ^ 0x5A5A);
                eprintln!("{depth:>4}  {ur}/{uc:<3}   {:>4.2}  {sigma:>4.2}  {:>11}   {}", density(uc, ur), fmt_accs(&accs), fmt_rates(&rates));
            }
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn ff_temporal_xor_depth4() {
        // Feed-forward temporal XOR, 4 layers, 32×32, readout = top spiking layer. Sweep fan-in
        // (radius/count) AND training duration to find where training comes to life — a starved substrate
        // (σ→0) can still revive with enough training, so the acc curve over checkpoints is the point. Per
        // config: σ, acc curve, post-training per-layer firing %. (ALIF neurons hold the gap; εᵃ, β=0.4.)
        let seed = 0xE9_0B_0A17u64;
        let ckpts = [800usize, 2500, 6000];
        eprintln!("== FF temporal XOR (4 layers, size 32, delay 8, read top layer) ==");
        eprintln!("r/c     dens   σ     acc@{ckpts:?}   per-layer firing % (L0..L3, post-train)");
        for &(ur, uc) in &[(3u32, 16u32), (4, 64)] {
            let (mut net, entries) = make_ff(seed, 32, 4, uc, ur, 20, 6);
            let sigma = sigma_ratio(&mut net, seed ^ 0x5A5A);
            let mut cfg = ff_cfg(0, 0.004, 0.4);
            cfg.size = 32;
            cfg.delay = 8;
            cfg.holdout = 300;
            let accs = train_and_eval_curve(&mut net, &entries, seed, seed, &cfg, xor_task, &ckpts);
            let rates = per_layer_rates(&mut net, seed ^ 0x5A5A);
            eprintln!("{ur}/{uc:<3}   {:>4.2}  {sigma:>4.2}  {:>14}   {}", density(uc, ur), fmt_accs(&accs), fmt_rates(&rates));
        }
    }

    #[test]
    #[ignore] // expensive; run manually in --release
    fn sidecar_parity_recurrence() {
        // Parity N=3 on the backward-fed side-car (size 32), readout = top spiking layer. The forward/skip
        // path is held at a generous fan-in; the recurrent scratchpad may want a DIFFERENT radius/count, so
        // sweep it — and sweep training DURATION too (acc curve). Per config: σ, acc curve, post-training
        // per-layer firing %. (FF alone cannot do parity N≥3 — this asks whether trained recurrence revives it.)
        let seed = 0xE9_0B_0A17u64;
        let (fur, fuc) = (4u32, 48u32); // generous forward/skip fan-in (held fixed)
        let ckpts = [500usize, 1500, 4000];
        eprintln!("== Side-car parity N=3 (size 32, read top layer; forward r{fur}/c{fuc}) ==");
        eprintln!("rec r/c  dens   σ     acc@{ckpts:?}   per-layer firing % (L0..L4, post-train)");
        for &(rr, rc) in &[(3u32, 8u32), (4, 16), (4, 24)] {
            let (mut net, entries) = make_sidecar(seed, 32, fuc, fur, rc, rr, 20, 6);
            let sigma = sigma_ratio(&mut net, seed ^ 0x5A5A);
            let mut cfg = ff_cfg(0, 0.004, 0.4);
            cfg.size = 32;
            cfg.delay = 8;
            cfg.read = 8;
            cfg.holdout = 300;
            cfg.elig.rec_tau = 20.0; // eligibility trace spans the gap for the recurrent path
            let accs = train_and_eval_curve(&mut net, &entries, seed, seed, &cfg, |s, t| task_parity(s, t, 3), &ckpts);
            let rates = per_layer_rates(&mut net, seed ^ 0x5A5A);
            eprintln!("{rr}/{rc:<3}  {:>4.2}  {sigma:>4.2}  {:>13}   {}", density(rc, rr), fmt_accs(&accs), fmt_rates(&rates));
        }
    }
}
