//! Equivalence oracles for the event-driven engine (test-only).
//!
//! 1. `sparse == dense`: the same config driven through `Network::new` (frontier) and
//!    `Network::new_dense` (all-neurons) must reach identical per-layer state every wave — proving the
//!    frontier never wrongly drops a neuron whose processing would not be a no-op.
//! 2. `adapt_bump == 0 ⟹ wave_driven == wave_bitnet`: with adaptation off, the one redefined dynamic
//!    vanishes, so both engines must produce identical potentials and fired sets — a cross-engine check
//!    of drain / decide / leak / generate and the copied topology + weight init.

use crate::wave_driven::config::{Config, LayerConfig};
use crate::wave_driven::network::Network;
use crate::wave_driven::synapse::{random_l0_input, TopologyLevel};

fn deep_cfg(size: u32, adapt_bump: i16) -> Config {
    let mk = |topology| LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump, adapt_decay: 6 };
    // 4 computational layers + empty top; forward +1, with a level-0 recurrent edge on L2 and a
    // level -1 feedback edge on L3 to exercise cross-layer waking and L0 feedback.
    let layers = vec![
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]),
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: 0, radius: 1, count: 3 }]),
        mk(vec![TopologyLevel { level: 1, radius: 2, count: 8 }, TopologyLevel { level: -1, radius: 1, count: 2 }]),
        mk(vec![]),
    ];
    Config { seed: 0xABCD_1234, size, layers }
}

fn assert_layers_eq(a: &Network, b: &Network, wave: usize) {
    for z in 0..a.layer_count() {
        a.with_layer(z, |la| {
            b.with_layer(z, |lb| {
                assert_eq!(la.potential, lb.potential, "wave {wave} layer {z} potential");
                assert_eq!(la.cooldown, lb.cooldown, "wave {wave} layer {z} cooldown");
                assert_eq!(la.adapt_ref, lb.adapt_ref, "wave {wave} layer {z} adapt_ref");
                assert_eq!(la.fire_wave, lb.fire_wave, "wave {wave} layer {z} fire_wave");
            })
        });
    }
}

#[test]
fn sparse_equals_dense_over_random_drive() {
    let size = 16u32;
    let mut sparse = Network::new(deep_cfg(size, 5));
    let mut dense = Network::new_dense(deep_cfg(size, 5));
    let input = random_l0_input(0xABCD_1234, size, 12000);
    for wv in 0..300 {
        let inp = input(wv);
        sparse.wave(&inp);
        dense.wave(&inp);
        assert_layers_eq(&sparse, &dense, wv);
    }
}

#[test]
fn sparse_equals_dense_with_bursty_gaps() {
    // Long input-free gaps stress lazy adaptation catch-up (large `gap` on wake).
    let size = 16u32;
    let mut sparse = Network::new(deep_cfg(size, 5));
    let mut dense = Network::new_dense(deep_cfg(size, 5));
    let input = random_l0_input(0x5151, size, 30000);
    for wv in 0..400 {
        let inp = if wv % 40 < 4 { input(wv) } else { Vec::new() }; // bursts then silence
        sparse.wave(&inp);
        dense.wave(&inp);
        assert_layers_eq(&sparse, &dense, wv);
    }
}

#[test]
fn adapt_bump_zero_matches_wave_bitnet() {
    use crate::wave_bitnet::config::{Config as BConfig, LayerConfig as BLayerConfig};
    use crate::wave_bitnet::network::Network as BNetwork;
    use crate::wave_bitnet::synapse::{random_l0_input as brandom, TopologyLevel as BTop};
    use std::sync::{Arc, Mutex};

    let size = 16u32;
    let seed = 0xFEED_BEEFu64;
    // field-identical layer configs, adapt_bump 0 (LIF; adaptation off)
    let dtop = |lvl, r, c| TopologyLevel { level: lvl, radius: r, count: c };
    let btop = |lvl, r, c| BTop { level: lvl, radius: r, count: c };
    let d_layers = vec![
        LayerConfig { topology: vec![dtop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        LayerConfig { topology: vec![dtop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        LayerConfig { topology: vec![], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
    ];
    let b_layers = vec![
        BLayerConfig { topology: vec![btop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        BLayerConfig { topology: vec![btop(1, 2, 8)], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
        BLayerConfig { topology: vec![], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 6553, threshold_jitter: 32, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 },
    ];
    let mut dn = Network::new(Config { seed, size, layers: d_layers });
    let mut bn = BNetwork::new(BConfig { seed, size, layers: b_layers });

    let dfire: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); 3]));
    let bfire: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); 3]));
    for z in 0..3 {
        let d = dfire.clone();
        dn.on_layer(z, Box::new(move |_w, f| d.lock().unwrap()[z] = f.to_vec()));
        let b = bfire.clone();
        bn.on_layer(z, Box::new(move |_w, f| b.lock().unwrap()[z] = f.to_vec()));
    }

    // both use their own random_l0_input; the generators are copies, so identical sites for equal args.
    let dinput = random_l0_input(seed, size, 12000);
    let binput = brandom(seed, size, 12000);
    for wv in 0..200 {
        let di = dinput(wv);
        let bi = binput(wv);
        assert_eq!(di, bi, "L0 drive generators agree at wave {wv}");
        dn.wave(&di);
        bn.wave(&bi);
        // fired sets per layer (as sorted sets, since frontier vs dense may differ in order)
        for z in 0..3 {
            let mut df = dfire.lock().unwrap()[z].clone();
            df.sort_unstable();
            let mut bf = bfire.lock().unwrap()[z].clone();
            bf.sort_unstable();
            assert_eq!(df, bf, "wave {wv} layer {z} fired set");
        }
        // potentials per computational layer
        for z in 0..3 {
            let dp = dn.with_layer(z, |l| l.potential.clone());
            let bp = bn.with_layer(z, |l| l.potential.clone());
            assert_eq!(dp, bp, "wave {wv} layer {z} potential");
        }
    }
}
