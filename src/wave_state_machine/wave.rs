//! `wave` — one layer's per-wave step: integrate (drain inbox) → inject → decide →
//! generate outgoing synapses → leak, then decay adaptation. Touches only this layer; the
//! Network routes the generated synapses into other layers' inboxes for the next wave. Firing
//! uses the ALIF effective threshold `threshold + (adapt >> ADAPT_SHIFT)` (i32); a fire bumps
//! `adapt` (Q8 fixed point, saturating at `ADAPT_MAX`) and every neuron's `adapt` decays
//! geometrically each wave, like the leak.

use crate::wave_state_machine::neurons::{Layer, ADAPT_MAX, ADAPT_SHIFT};
use crate::wave_state_machine::synapse::{generate_into, SynapseGroup};

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    seed: u64,
    size: u32,
    input: &[u32],
    acc: &mut [i32],
    out: &mut [SynapseGroup],
    fired: &mut Vec<u32>,
) {
    let ls = (size as usize) * (size as usize);

    // 1. cooldown decay
    for c in layer.cooldown.iter_mut() {
        *c = c.saturating_sub(1);
    }

    // 2. drain inbox: sum deliveries in i32, fold into potential, narrow to i16 (overflow guard)
    for a in acc[..ls].iter_mut() {
        *a = 0;
    }
    for s in layer.inbox.iter() {
        acc[s.target as usize] += s.weight as i32;
    }
    layer.inbox.clear();
    for i in 0..ls {
        let v = layer.potential[i] as i32 + acc[i];
        layer.potential[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }

    // 3. inject forced-fire input (L0 only; other layers get &[]). L0 is the input transducer
    // (baseline i16::MAX, no adaptation — forced in Network::new), so its effective threshold is
    // exactly i16::MAX and setting potential to i16::MAX fires precisely the injected neurons.
    // (In general the ALIF effective threshold `baseline + adapt` can exceed i16::MAX, which is
    // why the transducer must not adapt — otherwise a saturated neuron could swallow an input.)
    for &a in input {
        layer.potential[a as usize] = i16::MAX;
        layer.cooldown[a as usize] = 0;
    }

    // A readout layer is a non-spiking drain-only integrator: its potential (folded above) is the
    // clean cumulative ±1 input for the trial. No fire, no generate, no leak, no adapt — return now.
    fired.clear();
    if layer.readout {
        return;
    }

    // 4. decide (ALIF effective threshold = baseline + adapt, in i32; fire bumps adapt)
    for i in 0..ls {
        let eff = layer.threshold[i] as i32 + (layer.adapt[i] >> ADAPT_SHIFT);
        if layer.cooldown[i] == 0 && (layer.potential[i] as i32) >= eff {
            layer.potential[i] = 0;
            layer.cooldown[i] = layer.cooldown_base;
            let bumped = layer.adapt[i] + ((layer.adapt_bump as i32) << ADAPT_SHIFT);
            layer.adapt[i] = bumped.clamp(0, ADAPT_MAX);
            fired.push(i as u32);
        }
    }

    // 5. generate outgoing synapses, aggregated by relative level into `out`
    let base = layer_index as usize * ls;
    for &local in fired.iter() {
        generate_into(
            seed,
            (base + local as usize) as u32,
            local,
            size,
            &layer.topology,
            layer.inhibitor_ratio,
            out,
        );
    }

    // 6. leak survivors into the next wave. Floor the decay at 1 for positive membrane so small
    // potentials relax to 0 (finite membrane time constant) instead of freezing in the integer
    // shift dead zone (`(v>>la)+(v>>lb) == 0` for `0 < v < 2^la`). Negatives already relax —
    // arithmetic shift of a negative is <= -1 — so they keep the raw geometric decay.
    let (la, lb) = layer.leak;
    for p in layer.potential.iter_mut() {
        let v = *p;
        let decay = (v >> la) + (v >> lb);
        *p = v - if v > 0 { decay.max(1) } else { decay };
    }

    // 7. decay adaptation toward rest (geometric, like the potential leak)
    let d = layer.adapt_decay;
    for a in layer.adapt.iter_mut() {
        *a -= *a >> d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_state_machine::config::LayerConfig;
    use crate::wave_state_machine::neurons::Layer;
    use crate::wave_state_machine::synapse::{Synapse, SynapseGroup, TopologyLevel};

    // A layer with hand-set LOW thresholds so integration can actually cause firing.
    fn low_layer(size: u32, threshold: i16, cooldown_base: u8, topo: Vec<TopologyLevel>) -> Layer {
        let cfg = LayerConfig {
            topology: topo,
            leak: (3, 5),
            cooldown_base,
            inhibitor_ratio: 0,
            threshold_jitter: 0,
            baseline_init: 0,
            adapt_bump: 0,
            adapt_decay: 5,
        };
        let mut l = Layer::new(&cfg, 0, 0, size);
        for t in l.threshold.iter_mut() {
            *t = threshold;
        }
        l
    }

    fn groups_for(l: &Layer) -> Vec<SynapseGroup> {
        l.topology.iter().map(|e| SynapseGroup { level: e.level, synapses: Vec::new() }).collect()
    }

    #[test]
    fn fire_bumps_adaptation() {
        let mut l = low_layer(4, 3, 2, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        l.adapt_bump = 8;
        l.adapt_decay = 5;
        for _ in 0..3 {
            l.inbox.push(Synapse { target: 0, weight: 1 });
        }
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
        assert_eq!(fired, vec![0]);
        // a fire adds bump<<SHIFT in Q-fixed-point, then step 7 decays it once
        let bumped = 8i32 << ADAPT_SHIFT;
        assert_eq!(l.adapt[0], bumped - (bumped >> 5), "fire bumps adapt by bump<<SHIFT, then decays once");
    }

    #[test]
    fn adaptation_decays_each_wave() {
        let mut l = low_layer(1, 20_000, 2, vec![]); // threshold high -> no firing
        l.adapt_decay = 3;
        l.adapt[0] = 100;
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        // 100 - (100 >> 3) = 100 - 12 = 88
        assert_eq!(l.adapt[0], 88);
        assert!(fired.is_empty());
    }

    #[test]
    fn high_adaptation_blocks_firing() {
        // potential clears the baseline but not baseline + adapt.
        let drive = |eff_adapt: i32| {
            let mut l = low_layer(1, 5, 2, vec![]);
            l.adapt[0] = eff_adapt << ADAPT_SHIFT; // effective contribution eff_adapt
            for _ in 0..10 {
                l.inbox.push(Synapse { target: 0, weight: 1 });
            }
            let mut acc = vec![0i32; 1];
            let mut out: Vec<SynapseGroup> = Vec::new();
            let mut fired = Vec::new();
            process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
            fired
        };
        assert_eq!(drive(0), vec![0], "baseline 5, potential 10 -> fires with no adaptation");
        assert!(drive(100).is_empty(), "effective threshold 105 blocks potential 10");
    }

    #[test]
    fn bump_zero_leaves_adaptation_at_rest() {
        let mut l = low_layer(4, 3, 1, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        l.adapt_bump = 0; // plain LIF
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        for _ in 0..3 {
            l.inbox.clear();
            for _ in 0..3 {
                l.inbox.push(Synapse { target: 0, weight: 1 });
            }
            process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
            assert_eq!(l.adapt[0], 0, "adapt must stay 0 when adapt_bump is 0");
        }
    }

    #[test]
    fn readout_layer_integrates_and_never_fires() {
        use crate::wave_state_machine::config::{Config, LayerConfig};
        use crate::wave_state_machine::network::Network;
        use std::sync::{Arc, Mutex};
        let build = |readout: bool| -> (usize, i16) {
            let l0 = LayerConfig {
                topology: vec![TopologyLevel { level: 1, radius: 1, count: 4 }],
                leak: (3, 5),
                cooldown_base: 2,
                inhibitor_ratio: 0,
                threshold_jitter: 0,
                baseline_init: 2,
                adapt_bump: 0,
                adapt_decay: 5,
            };
            let l1 = LayerConfig { topology: vec![], ..l0.clone() };
            let cfg = Config { seed: 1, size: 4, layers: vec![l0, l1] };
            let mut net = if readout { Network::new_with_readout(cfg) } else { Network::new(cfg) };
            let fires = Arc::new(Mutex::new(0usize));
            {
                let f = fires.clone();
                net.on_layer(1, Box::new(move |_w, fired: &[u32]| *f.lock().unwrap() += fired.len()));
            }
            let all: Vec<u32> = (0..16).collect();
            for _ in 0..8 {
                net.wave(&all);
            }
            (*fires.lock().unwrap(), net.potential(1, 0))
        };
        let (normal_fires, _) = build(false);
        let (readout_fires, readout_pot) = build(true);
        assert!(normal_fires > 0, "control: a normal L1 fires under the drive");
        assert_eq!(readout_fires, 0, "readout L1 must never fire");
        assert!(readout_pot > 1, "readout L1 must integrate its input (potential {readout_pot})");
    }

    #[test]
    fn integration_fires_and_resets() {
        let mut l = low_layer(4, 3, 2, vec![TopologyLevel { level: 1, radius: 0, count: 1 }]);
        for _ in 0..3 {
            l.inbox.push(Synapse { target: 0, weight: 1 });
        }
        let mut acc = vec![0i32; 16];
        let mut out = groups_for(&l);
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 4, &[], &mut acc, &mut out, &mut fired);
        assert_eq!(fired, vec![0]);
        assert_eq!(l.potential[0], 0);
        assert_eq!(l.cooldown[0], 2);
        assert_eq!(out[0].synapses.len(), 1);
        assert!(l.inbox.is_empty());
    }

    #[test]
    fn refractory_blocks_refire() {
        let mut l = low_layer(1, 3, 2, vec![]);
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        // wave A: force fire via injection (potential=i16::MAX, cooldown=0)
        process_layer(&mut l, 0, 0, 1, &[0], &mut acc, &mut out, &mut fired);
        assert_eq!(fired, vec![0]);
        // wave B: strong drive but still refractory (cooldown 2 -> 1)
        for _ in 0..100 {
            l.inbox.push(Synapse { target: 0, weight: 1 });
        }
        process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        assert!(fired.is_empty(), "must not fire while refractory");
    }

    #[test]
    fn leak_decays_subthreshold_potential() {
        let mut l = low_layer(1, 20_000, 2, vec![]);
        l.potential[0] = 1000;
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        // leak (3,5): 1000 - 125 - 31 = 844
        assert_eq!(l.potential[0], 844);
    }

    #[test]
    fn leak_drives_small_positive_potential_to_zero() {
        // Small positive potentials must relax to 0 (finite membrane time constant), not freeze in
        // the integer shift dead zone where (v>>a)+(v>>b) == 0 for v < 2^a.
        let mut l = low_layer(1, 20_000, 2, vec![]); // high threshold: never fires
        l.potential[0] = 5;
        let mut acc = vec![0i32; 1];
        let mut out: Vec<SynapseGroup> = Vec::new();
        let mut fired = Vec::new();
        for _ in 0..6 {
            process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
        }
        assert_eq!(l.potential[0], 0, "small potential must leak to 0, not freeze in the dead zone");
    }

    #[test]
    fn inhibition_nets_once_order_independent() {
        // +100 excitatory, -10 inhibitory summed in one i32 pass -> net +90; order can't matter,
        // and with saturation gone there is no membrane clamp below the i16 bound.
        let run = |exc_first: bool| {
            let mut l = low_layer(1, 30_000, 2, vec![]);
            l.potential[0] = 40;
            if exc_first {
                for _ in 0..100 { l.inbox.push(Synapse { target: 0, weight: 1 }); }
                for _ in 0..10 { l.inbox.push(Synapse { target: 0, weight: -1 }); }
            } else {
                for _ in 0..10 { l.inbox.push(Synapse { target: 0, weight: -1 }); }
                for _ in 0..100 { l.inbox.push(Synapse { target: 0, weight: 1 }); }
            }
            let mut acc = vec![0i32; 1];
            let mut out: Vec<SynapseGroup> = Vec::new();
            let mut fired = Vec::new();
            process_layer(&mut l, 0, 0, 1, &[], &mut acc, &mut out, &mut fired);
            l.potential[0]
        };
        // 40 + 90 = 130 (no clamp), then leak (3,5): 130 - 16 - 4 = 110
        assert_eq!(run(true), 110);
        assert_eq!(run(false), 110); // identical regardless of delivery order
    }
}
