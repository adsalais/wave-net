//! `wave` — one layer's per-wave step. Neuron dynamics (drain → inject → decide/fire/ALIF/leak) are
//! identical to `wave_net`; only the synapse-generation step differs: instead of hashing targets on the
//! fly, it scans the occupancy bitset and decodes each wired cell, delivering the packed ±1/0 weight.
//! (No `seed` argument — targets are materialized at construction.)

use crate::wave_bitnet::neurons::{Layer, ADAPT_MAX, ADAPT_SHIFT};
use crate::wave_bitnet::synapse::{decode_cell, Synapse};

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    size: u32,
    input: &[u32],
    acc: &mut [i32],
    deliveries: &mut [Vec<Synapse>],
    fired: &mut Vec<u32>,
    record_elig: bool,
) {
    let ls = (size as usize) * (size as usize);

    // 1. drain inbox: sum deliveries in i32, fold into potential, narrow to i16 (overflow guard).
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

    // 2. inject forced-fire input (L0 only). L0 is the input transducer (baseline i16::MAX, no adapt).
    for &a in input {
        layer.potential[a as usize] = i16::MAX;
        layer.cooldown[a as usize] = 0;
    }

    // A readout layer is a non-spiking drain-only integrator: return after the fold.
    fired.clear();
    if layer.readout {
        return;
    }

    // 3. per-neuron step (fused): cooldown decay, decide against ALIF effective threshold, fire-reset +
    // adapt-bump, leak, adapt-decay; plus e-prop eligibility accrual. Identical to wave_net.
    const PSI_BAND: i32 = 8;
    let (la, lb) = layer.leak;
    let adapt_decay = layer.adapt_decay;
    let cooldown_base = layer.cooldown_base;
    let adapt_bump = layer.adapt_bump as i32;
    for i in 0..ls {
        let c = layer.cooldown[i].saturating_sub(1); // cooldown decay
        let p = layer.potential[i];
        let pi = p as i32;
        let eff = layer.threshold[i] as i32 + (layer.adapt[i] >> ADAPT_SHIFT);

        if record_elig {
            layer.decide_potential[i] = p; // snapshot pre fire-reset/leak
            layer.decide_eff[i] = eff; // pre-bump effective threshold
            if (pi - eff).abs() <= PSI_BAND {
                layer.elig_post[i] += 1;
            }
        }

        let mut pot = p;
        if c == 0 && pi >= eff {
            pot = 0;
            layer.cooldown[i] = cooldown_base;
            let bumped = layer.adapt[i] + (adapt_bump << ADAPT_SHIFT);
            layer.adapt[i] = bumped.clamp(0, ADAPT_MAX);
            if record_elig {
                layer.elig_pre[i] += 1;
            }
            fired.push(i as u32);
        } else {
            layer.cooldown[i] = c;
        }

        let decay = (pot >> la) + (pot >> lb);
        layer.potential[i] = pot - if pot > 0 { decay.max(1) } else { decay };

        layer.adapt[i] -= layer.adapt[i] >> adapt_decay;
    }

    // 4. generate outgoing synapses via the occupancy bitset scan — no hashing.
    let layer_count = deliveries.len() as i32;
    for &local in fired.iter() {
        let li_local = local as usize;
        for (lvl_idx, entry) in layer.topology.iter().enumerate() {
            let tl = layer_index as i32 + entry.level;
            if tl < 0 || tl >= layer_count {
                continue;
            }
            let n = layer.neigh[lvl_idx];
            let wbase = li_local * layer.total_slots + layer.slot_bases[lvl_idx];
            let radius = entry.radius;
            let mut r = 0usize;
            for c in layer.occupancy[lvl_idx].iter_set_in(li_local * n, n) {
                let w = layer.weight_at(wbase + r);
                r += 1;
                if w != 0 {
                    let target = decode_cell(c, local, radius, size);
                    deliveries[tl as usize].push(Synapse { target, weight: w as i16 });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::LayerConfig;
    use crate::wave_bitnet::neurons::Layer;
    use crate::wave_bitnet::synapse::{decode_cell, Synapse, TopologyLevel};

    #[test]
    fn firing_neuron_delivers_nonzero_synapses_to_decoded_targets() {
        let size = 4u32;
        let ls = (size * size) as usize;
        let cfg = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        let mut l = Layer::new(&cfg, 5, 0, size);
        // force neuron 0 to fire: low threshold, primed potential, cooldown 0
        l.threshold.iter_mut().for_each(|t| *t = 1);
        l.cooldown.iter_mut().for_each(|c| *c = 0);
        l.potential[0] = 100;
        // build expected: iterate neuron 0's wired cells (rank order), decode, skip weight 0
        let base = l.slot_base(0);
        let n = l.neigh[0];
        let mut expect: Vec<Synapse> = Vec::new();
        let mut r = 0;
        for c in l.occupancy[0].iter_set_in(0 * n, n) {
            let w = l.weight_at(0 * l.total_slots + base + r);
            r += 1;
            if w != 0 {
                expect.push(Synapse { target: decode_cell(c, 0, 1, size), weight: w as i16 });
            }
        }
        let mut acc = vec![0i32; ls];
        let mut deliveries: Vec<Vec<Synapse>> = vec![Vec::new(); 2];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, size, &[], &mut acc, &mut deliveries, &mut fired, true);
        assert_eq!(fired, vec![0], "only neuron 0 fires");
        assert_eq!(deliveries[1], expect, "delivers decoded nonzero synapses to layer 1");
    }
}
