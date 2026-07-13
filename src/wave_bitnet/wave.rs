//! `wave` — one layer's per-wave step: drain → inject → decide/fire/ALIF/leak, then synapse generation.
//! Synapse generation scans the per-neuron occupancy bitset and decodes each wired cell, delivering the
//! packed ±1/0 weight (no per-wave hashing; targets are materialized at construction, so no `seed`).

use crate::wave_bitnet::neurons::{Layer, ADAPT_MAX, ADAPT_SHIFT, WCODE};
use crate::wave_bitnet::synapse::{local_of, wrap, xy_of};

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    size: u32,
    input: &[u32],
    deliv: &mut [Vec<i32>],
    fired: &mut Vec<u32>,
    record_elig: bool,
) {
    let ls = (size as usize) * (size as usize);

    // Reslice the per-neuron arrays to exactly `ls` once. Each `[..ls]` is a SINGLE bounds check;
    // indexing `arr[i]` for i in 0..ls afterwards is check-free (the compiler proves i < len == ls),
    // eliding the per-access bounds checks in these hot loops. Disjoint fields => the borrows coexist,
    // and NLL drops them before the generate step reborrows `layer`. No `unsafe` needed.
    let pending = &mut layer.pending[..ls];
    let potential = &mut layer.potential[..ls];
    let cooldown = &mut layer.cooldown[..ls];

    // 1. drain: fold this wave's incoming per-target accumulator (`pending`, scatter-added into last
    // wave) into potential, then clear it. Deliveries are pre-summed by target — no per-synapse inbox.
    // Branchless (no `if d != 0`) so it auto-vectorizes: adding 0 then clamping an already-in-range
    // i16 is a no-op, and pending is zeroed unconditionally (it must end all-zero for the wave-end
    // swap). The i32→i16 saturating clamp maps to a packed pack-with-saturation.
    for i in 0..ls {
        let v = potential[i] as i32 + pending[i];
        potential[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        pending[i] = 0;
    }

    // 2. inject forced-fire input (L0 only). L0 is the input transducer (baseline i16::MAX, no adapt).
    // (`a` is input-sourced, not 0..ls, so these stay bounds-checked — negligible: L0 only, few sites.)
    for &a in input {
        potential[a as usize] = i16::MAX;
        cooldown[a as usize] = 0;
    }

    // A readout layer is a non-spiking drain-only integrator: return after the fold.
    fired.clear();
    if layer.readout {
        return;
    }

    // 3. per-neuron step, split into passes so the hot arithmetic
    // vectorizes and the inference path carries no eligibility branch: (A0) decide-time snapshot
    // [record only], (A) decide/fire/adapt-bump [scalar — diverges + compacts `fired`], (B) elementwise
    // leak + adapt-decay [auto-vectorized].
    let (la, lb) = layer.leak;
    let adapt_decay = layer.adapt_decay;
    let cooldown_base = layer.cooldown_base;
    let adapt_bump = layer.adapt_bump as i32;
    let threshold = &layer.threshold[..ls];
    let adapt = &mut layer.adapt[..ls];
    // (A0) decide-time snapshot — records decide-time potential/eff into the training scratch, and only
    // when training is enabled (train == Some). Reads potential/adapt BEFORE the fire loop mutates them,
    // capturing the pre-fire-reset state. `train` is a disjoint field, so it coexists with the borrows above.
    if record_elig {
        if let Some(t) = layer.train.as_mut() {
            let decide_potential = &mut t.decide_potential[..ls];
            let decide_eff = &mut t.decide_eff[..ls];
            for i in 0..ls {
                let p = potential[i];
                let eff = threshold[i] as i32 + (adapt[i] >> ADAPT_SHIFT);
                decide_potential[i] = p; // snapshot pre fire-reset/leak
                decide_eff[i] = eff; // pre-bump effective threshold
            }
        }
    }

    // (A) decide + fire (scalar: diverges on the fire branch and compacts `fired`). Writes the
    // POST-FIRE potential (0 if fired, else left as-is) so (B)'s leak reads it; adapt is bumped here
    // (pre-decay) and decayed in (B). No `record_elig` branch — eligibility lives in (A0)/(A2).
    for i in 0..ls {
        let c = cooldown[i].saturating_sub(1); // cooldown decay
        let pi = potential[i] as i32;
        let eff = threshold[i] as i32 + (adapt[i] >> ADAPT_SHIFT);
        if c == 0 && pi >= eff {
            potential[i] = 0;
            cooldown[i] = cooldown_base;
            let bumped = adapt[i] + (adapt_bump << ADAPT_SHIFT);
            adapt[i] = bumped.clamp(0, ADAPT_MAX);
            fired.push(i as u32);
        } else {
            cooldown[i] = c;
        }
    }

    // (B) elementwise tails, each in its own loop so LLVM auto-vectorizes them: the leak (each neuron
    // reads only its own post-fire potential) and the ALIF adapt-decay (pure elementwise).
    for i in 0..ls {
        let pot = potential[i];
        let decay = (pot >> la) + (pot >> lb);
        potential[i] = pot - if pot > 0 { decay.max(1) } else { decay };
    }
    for i in 0..ls {
        adapt[i] -= adapt[i] >> adapt_decay;
    }

    // 4. generate outgoing deliveries via a WORD-SCAN of the occupancy bitset — visit only set bits
    // (trailing_zeros + clear-lowest), decode via the offset LUT, and SCATTER-ADD each weight into the
    // target layer's per-target accumulator (no per-synapse Vec, no hashing, no div/mod).
    let layer_count = deliv.len() as i32;
    for &local in fired.iter() {
        let li = local as usize;
        let (sx, sy) = xy_of(local, size);
        for (lvl, entry) in layer.topology.iter().enumerate() {
            let tl = layer_index as i32 + entry.level;
            if tl < 0 || tl >= layer_count {
                continue;
            }
            let tl = tl as usize;
            let wpn = layer.occ_wpn[lvl];
            // SAFETY: li < ls and occ[lvl].len() == ls*wpn, so [li*wpn, li*wpn+wpn) is in bounds.
            let words = unsafe { layer.occ.get_unchecked(lvl).get_unchecked(li * wpn..li * wpn + wpn) };
            let wbase = li * layer.total_slots + layer.slot_bases[lvl];
            let lut = unsafe { layer.offsets.get_unchecked(lvl) };
            let flat = unsafe { layer.off_flat.get_unchecked(lvl) };
            let codes = &layer.codes;
            // SAFETY: tl was range-checked above (0 <= tl < layer_count == deliv.len()).
            let target_deliv = unsafe { deliv.get_unchecked_mut(tl) };
            // Interior source (>= radius from every toroidal edge) => no synapse wraps => the target is a
            // single add `li + flat[cell]`. `interior` is loop-invariant (perfect prediction / unswitchable).
            let r = entry.radius;
            let hi = size.saturating_sub(r);
            let interior = sx >= r && sx < hi && sy >= r && sy < hi;
            let li_i = li as i32;
            let mut rank = 0usize;
            for (wi, &w0) in words.iter().enumerate() {
                let mut word = w0;
                let cbase = wi * 64;
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    let cell = cbase + bit;
                    let widx = wbase + rank;
                    // SAFETY: widx < ls*total_slots => widx>>5 < codes.len() (ceil(../32) words).
                    let code = (unsafe { *codes.get_unchecked(widx >> 5) } >> ((widx & 31) * 2)) & 0b11;
                    let w = WCODE[code as usize];
                    // Branchless delivery: a pruned synapse (w == 0) scatter-adds 0 — a harmless no-op — so
                    // we drop the data-dependent `if w != 0`. This pays only while sparsity stays low
                    // (~6-7% at r4/c48, the trained default); a high-sparsity config wants the branch back
                    // (or an active-occupancy scan) so pruned synapses skip the decode + write entirely.
                    // SAFETY (both arms): `cell` is a SET occupancy bit => a sampled cell < neigh, which
                    // is the length of both `flat` and `lut`; the resulting target is < ls.
                    let target = if interior {
                        (li_i + unsafe { *flat.get_unchecked(cell) }) as usize
                    } else {
                        let (dx, dy) = unsafe { *lut.get_unchecked(cell) };
                        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize
                    };
                    // SAFETY: target < ls == target_deliv.len().
                    unsafe { *target_deliv.get_unchecked_mut(target) += w as i32 };
                    rank += 1;
                    word &= word - 1;
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
    use crate::wave_bitnet::synapse::TopologyLevel;

    #[test]
    fn firing_neuron_scatters_nonzero_weights_to_decoded_targets() {
        let size = 4u32;
        let ls = (size * size) as usize;
        let cfg = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count: 3 }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        let mut l = Layer::new(&cfg, 5, 0, size);
        // force neuron 0 to fire: low threshold, primed potential, cooldown 0
        l.threshold.iter_mut().for_each(|t| *t = 1);
        l.cooldown.iter_mut().for_each(|c| *c = 0);
        l.potential[0] = 100;
        // expected: sum decoded nonzero weights per target
        let base = l.slot_base(0);
        let mut expect = vec![0i32; ls];
        l.for_wired(0, 0, |r, cell| {
            let w = l.weight_at(0 * l.total_slots + base + r);
            if w != 0 {
                expect[l.decode(0, 0, cell, size) as usize] += w as i32;
            }
        });
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; ls]; 2];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, size, &[], &mut deliv, &mut fired, true);
        assert_eq!(fired, vec![0], "only neuron 0 fires");
        assert_eq!(deliv[1], expect, "scatter-adds decoded nonzero weights into layer 1's accumulator");
    }
}
