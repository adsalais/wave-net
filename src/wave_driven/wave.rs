//! `wave` — one layer's per-wave step, event-driven. Drain the sparse accumulator, (L0) inject,
//! decide/fire/leak with lazy fire-anchored adaptation, then scatter deliveries (the wave_bitnet
//! word-scan). `Work::Sparse` visits only a frontier and rebuilds the next one; `Work::Dense` visits
//! all neurons and is the equivalence oracle. Both share the per-neuron arithmetic (the `step!` macro),
//! so dense and sparse can only differ in *which neurons are visited* — the frontier-completeness
//! property the oracle checks.

use crate::wave_driven::frontier::Frontier;
use crate::wave_driven::neurons::{Layer, ADAPT_MAX, ADAPT_SHIFT, FRAC, WCODE};
use crate::wave_driven::synapse::{local_of, wrap, xy_of};

pub enum Work<'a> {
    Dense,
    Sparse { cur: &'a [u32], frontier_next: &'a mut [Frontier] },
}

pub fn process_layer(
    layer: &mut Layer,
    layer_index: u32,
    size: u32,
    input: &[u32],
    w: u32,
    work: Work,
    deliv: &mut [Vec<i32>],
    fired: &mut Vec<u32>,
) {
    let ls = (size as usize) * (size as usize);
    fired.clear();

    // Destructure `work` into independent bindings up front: `cur` (immutable) and `frontier_next`
    // (mutable) then no longer alias each other, so the drain/decide loops can iterate `cur` while the
    // carry/generate steps push into `frontier_next`. `dense` selects the iteration set (0..ls vs cur).
    let (dense, cur, mut frontier_next): (bool, &[u32], Option<&mut [Frontier]>) = match work {
        Work::Dense => (true, &[], None),
        Work::Sparse { cur, frontier_next } => (false, cur, Some(frontier_next)),
    };

    // --- 1. drain: fold pending into potential (i32), clamp to i16, clear pending ---
    macro_rules! drain {
        ($i:expr) => {{
            let i = $i;
            let v = layer.potential[i] as i32 + layer.pending[i];
            layer.potential[i] = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            layer.pending[i] = 0;
        }};
    }
    if dense {
        for i in 0..ls {
            drain!(i);
        }
    } else {
        for &iu in cur {
            drain!(iu as usize);
        }
    }

    // --- 2. inject (L0 only): override drained potential to forced-fire, clear cooldown ---
    for &a in input {
        layer.potential[a as usize] = i16::MAX;
        layer.cooldown[a as usize] = 0;
    }

    // --- readout: drain-only integrator; no decide/leak/generate/carry ---
    if layer.readout {
        return;
    }

    // --- 3. decide / fire / leak / carry (single arithmetic path; iteration set differs by mode) ---
    let (la, lb) = layer.leak;
    let cb = layer.cooldown_base;
    let bump = (layer.adapt_bump as i32) << ADAPT_SHIFT;
    let powlen = layer.pow_decay.len();
    macro_rules! step {
        ($iu:expr) => {{
            let iu = $iu;
            let i = iu as usize;
            let gap = w.wrapping_sub(layer.fire_wave[i]) as usize;
            let a = if gap >= powlen { 0 } else { ((layer.adapt_ref[i] as i64 * layer.pow_decay[gap]) >> FRAC) as i32 };
            let c = layer.cooldown[i].saturating_sub(1);
            let eff = layer.threshold[i] as i32 + (a >> ADAPT_SHIFT);
            if c == 0 && layer.potential[i] as i32 >= eff {
                layer.potential[i] = 0;
                layer.cooldown[i] = cb;
                layer.adapt_ref[i] = (a + bump).min(ADAPT_MAX);
                layer.fire_wave[i] = w;
                fired.push(iu);
            } else {
                layer.cooldown[i] = c;
            }
            let pot = layer.potential[i];
            let d = (pot >> la) + (pot >> lb);
            layer.potential[i] = pot - if pot > 0 { d.max(1) } else { d };
            if let Some(fn_) = frontier_next.as_deref_mut() {
                if layer.potential[i] != 0 || layer.cooldown[i] != 0 {
                    fn_[layer_index as usize].push(iu);
                }
            }
        }};
    }
    if dense {
        for i in 0..ls {
            step!(i as u32);
        }
    } else {
        for &iu in cur {
            step!(iu);
        }
    }

    // --- 4. generate: word-scan each firer's occupancy, decode, scatter weight into target accumulator ---
    let layer_count = deliv.len() as i32;
    for &local in fired.iter() {
        let li = local as usize;
        let (sx, sy) = xy_of(local, size);
        for (lvl, entry) in layer.topology.iter().enumerate() {
            let tl = layer_index as i32 + entry.level;
            if tl < 0 || tl >= layer_count {
                continue;
            }
            let tz = tl as usize;
            let wpn = layer.occ_wpn[lvl];
            let words = &layer.occ[lvl][li * wpn..li * wpn + wpn];
            let wbase = li * layer.total_slots + layer.slot_bases[lvl];
            let lut = &layer.offsets[lvl];
            let flat = &layer.off_flat[lvl];
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
                    let code = (layer.codes[widx >> 5] >> ((widx & 31) * 2)) & 0b11;
                    let wt = WCODE[code as usize] as i32;
                    let target = if interior {
                        (li_i + flat[cell]) as usize
                    } else {
                        let (dx, dy) = lut[cell];
                        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size) as usize
                    };
                    deliv[tz][target] += wt;
                    if let Some(fn_) = frontier_next.as_deref_mut() {
                        fn_[tz].push(target as u32);
                    }
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
    use crate::wave_driven::config::LayerConfig;
    use crate::wave_driven::neurons::Layer;
    use crate::wave_driven::synapse::TopologyLevel;

    fn one_up(size: u32, count: u32) -> Layer {
        let cfg = LayerConfig { topology: vec![TopologyLevel { level: 1, radius: 1, count }], leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 0, adapt_decay: 6 };
        Layer::new(&cfg, 5, 0, size)
    }

    #[test]
    fn dense_firer_scatters_decoded_weights() {
        let size = 4u32;
        let ls = (size * size) as usize;
        let mut l = one_up(size, 3);
        l.threshold.iter_mut().for_each(|t| *t = 1);
        l.cooldown.iter_mut().for_each(|c| *c = 0);
        l.potential[0] = 100;
        // expected: sum decoded nonzero weights per target for neuron 0
        let base = l.slot_base(0);
        let mut expect = vec![0i32; ls];
        l.for_wired(0, 0, |r, cell| {
            let wt = l.weight_at(0 * l.total_slots + base + r);
            if wt != 0 {
                expect[l.decode(0, 0, cell, size) as usize] += wt as i32;
            }
        });
        let mut deliv: Vec<Vec<i32>> = vec![vec![0i32; ls]; 2];
        let mut fired = Vec::new();
        process_layer(&mut l, 0, size, &[], 0, Work::Dense, &mut deliv, &mut fired);
        assert_eq!(fired, vec![0], "only neuron 0 fires (others at rest below threshold)");
        assert_eq!(deliv[1], expect, "scatter-adds decoded weights into layer 1's accumulator");
    }

    #[test]
    fn sparse_matches_dense_on_one_step() {
        // Same primed layer, processed dense vs sparse (cur = {0}); the visited neuron's post-state
        // and the deliveries must match.
        let size = 4u32;
        let ls = (size * size) as usize;
        let mut ld = one_up(size, 3);
        let mut lspx = one_up(size, 3);
        for l in [&mut ld, &mut lspx] {
            l.threshold.iter_mut().for_each(|t| *t = 1);
            l.potential[0] = 100;
        }
        let mut dd = vec![vec![0i32; ls]; 2];
        let mut fd = Vec::new();
        process_layer(&mut ld, 0, size, &[], 0, Work::Dense, &mut dd, &mut fd);
        let mut ds = vec![vec![0i32; ls]; 2];
        let mut fs = Vec::new();
        let mut fnext = vec![Frontier::new(ls), Frontier::new(ls)];
        let cur = vec![0u32];
        process_layer(&mut lspx, 0, size, &[], 0, Work::Sparse { cur: &cur, frontier_next: &mut fnext }, &mut ds, &mut fs);
        assert_eq!(fd, fs);
        assert_eq!(dd, ds);
        assert_eq!(ld.potential[0], lspx.potential[0]);
        assert_eq!(ld.cooldown[0], lspx.cooldown[0]);
    }
}
