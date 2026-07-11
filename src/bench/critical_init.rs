//! Criticality-init experiments + a fast sanity check. The init and σ diagnostic now live in
//! `wave_net::critical_init` (`Network::critical_init`, `forward_avalanche`); these exercise and
//! validate them. The full comparison arc (rate-init vs σ-init, density sweeps, the computational
//! verdict) is consolidated in `docs/experiments_results.md`; the end-to-end trained comparison is the
//! FF validation gate in `rsnn`.

#[cfg(test)]
mod tests {
    use crate::wave_net::config::{Config, LayerConfig};
    use crate::wave_net::critical_init::{forward_avalanche, random_l0_input, CriticalInitParams};
    use crate::wave_net::network::Network;
    use crate::wave_net::synapse::TopologyLevel;

    const SEED: u64 = 0xC0FFEE_1234_5678;

    fn ff_config(up_count: u32, up_radius: u32, adapt_bump: i16) -> Config {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: up_radius, count: up_count }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 32,
            baseline_init: 6,
            adapt_bump,
            adapt_decay: 6,
        };
        Config { seed: SEED, size: 32, layers: vec![layer; 5] }
    }

    /// Sanity (fast): a low-fan-out 4-layer stack starves with depth; `critical_init` should lift the
    /// top layer's firing well above the untrained ±1 net (it drives σ toward 1, reviving depth).
    #[test]
    fn critical_init_revives_a_starved_stack() {
        let layer = LayerConfig {
            topology: vec![TopologyLevel { level: 1, radius: 2, count: 6 }],
            leak: (3, 5),
            cooldown_base: 2,
            inhibitor_ratio: 0,
            threshold_jitter: 16,
            baseline_init: 6,
            adapt_bump: 5,
            adapt_decay: 6,
        };
        let cfg = Config { seed: SEED, size: 8, layers: vec![layer; 4] };
        let input = random_l0_input(SEED, 8, 20000);
        let top = cfg.layers.len() - 1;

        let mut untrained = Network::new(cfg.clone());
        let before = untrained.measure_layer_rates(16, 64, &input)[top];

        let mut net = Network::new(cfg);
        net.critical_init(SEED ^ 0xABCD, 20000, &CriticalInitParams { rounds: 12, n_perturb: 8, ..CriticalInitParams::default() });
        let after = net.measure_layer_rates(16, 64, &input)[top];

        assert!(after > before + 0.01, "critical_init should revive the top layer: {before:.3} -> {after:.3}");
    }

    /// Experiment (run manually): `critical_init`'s resulting per-hop σ (and emergent rate) across the
    /// density sweep. Expect σ_hop ≈ 1 with the rate self-selecting.
    ///   cargo test --release critical_init_vs_density -- --ignored --nocapture
    #[test]
    #[ignore]
    fn critical_init_vs_density() {
        let params = CriticalInitParams::default();
        let hops = |f: &[f64]| -> Vec<f64> {
            (1..f.len()).map(|z| if f[z - 1] > 0.0 { (f[z] / f[z - 1] * 100.0).round() / 100.0 } else { 0.0 }).collect()
        };
        let pct = |r: &[f64]| r.iter().map(|x| (x * 1000.0).round() / 10.0).collect::<Vec<_>>();
        for up_count in [8u32, 16, 24, 32, 48] {
            let mut net = Network::new(ff_config(up_count, 3, 5));
            net.critical_init(SEED ^ 0xABCD, 20000, &params);
            let input = random_l0_input(SEED, 32, 20000);
            let r = net.measure_layer_rates(32, 128, &input);
            let f = forward_avalanche(&mut net, SEED ^ 0xABCD, 20000, 32, 32, 16);
            println!("uc={up_count:<2}: rates={:?} σ_hop{:?}", pct(&r), hops(&f));
        }
    }
}
