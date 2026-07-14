//! (test-only) Bit-exact online-vs-dense HYPR eligibility oracle. The `Network`'s online accrual (active
//! set, 2-state ε recursion) must equal the dense reference (`training::dense_eligibility`, full-scan,
//! deterministic re-run) bit-for-bit — the correctness gate for the eligibility rule.

use crate::wave_resonate::config::{Config, LayerConfig};
use crate::wave_resonate::network::Network;
use crate::wave_resonate::synapse::{random_l0_input, TopologyLevel};
use crate::wave_resonate::training::{dense_eligibility, EligParams};

fn deep_cfg(size: u32) -> Config {
    let mk = |topology: Vec<TopologyLevel>| LayerConfig {
        topology,
        inhibitor_ratio: 6553,
        omega_init: (5.0, 10.0),
        b_offset_init: (0.1, 1.0),
        tau_out: 20.0,
    };
    Config {
        seed: 0x0E11,
        size,
        dt: 0.05,
        gamma: 0.9,
        theta_c: 1.0,
        layers: vec![
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
            mk(vec![]),
        ],
    }
}

#[test]
fn online_equals_dense_eligibility_bit_exact() {
    let size = 16u32;
    let cfg = deep_cfg(size);
    let p = EligParams::default();
    let input_gen = random_l0_input(0x0E11, size, 15000);
    let inputs: Vec<Vec<u32>> = (0..120).map(|w| input_gen(w)).collect();

    let mut net = Network::new(cfg.clone());
    net.set_elig_params(p);
    net.enable_training();
    for inp in &inputs {
        net.wave(inp);
    }

    let dense = dense_eligibility(&cfg, &inputs, &p);
    for z in 0..net.layer_count() {
        net.with_layer(z, |lz| {
            assert_eq!(lz.train.as_ref().unwrap().elig, dense[z], "layer {z} online == dense elig (bit-exact)");
        });
    }
}
