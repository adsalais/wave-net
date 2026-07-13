//! `neurons` — a `Layer`'s per-neuron SoA state and its bitset topology substrate (copied from
//! wave_bitnet) plus the lazy fire-anchored adaptation state and its geometric decay table.

use crate::wave_driven::config::LayerConfig;
use crate::wave_driven::synapse::{key, local_of, map_range, mix, neigh_size, sample_distinct_cells, wrap, xy_of, TopologyLevel, P_TARGET, P_THRESHOLD};

/// Fixed-point scale for the adaptation contribution to the effective threshold (`adapt >> ADAPT_SHIFT`).
pub const ADAPT_SHIFT: u32 = 12;
/// Ceiling for the reconstructed adaptation (so its threshold contribution never exceeds i16::MAX).
pub const ADAPT_MAX: i32 = (i16::MAX as i32) << ADAPT_SHIFT;
/// Fixed-point fraction bits of the geometric decay table `pow_decay` (i64 math: adapt_ref≤2^27 · POW≤2^30 ⊂ i64).
pub const FRAC: u32 = 30;
/// Upper bound on `adapt_decay` (must stay < FRAC so ρ's fixed-point shift is valid; 24 leaves headroom).
pub const MAX_ADAPT_DECAY: u8 = 24;
/// Cap on the decay-table length (waves). Beyond it, reconstructed adaptation is 0.
pub const HORIZON_CAP: usize = 1 << 16;
/// 2-bit weight code decode LUT: 0b00→0, 0b01→+1, 0b11→−1.
pub(crate) const WCODE: [i8; 4] = [0, 1, 0, -1];

pub(crate) struct DerivedLayout {
    pub total_slots: usize,
    pub slot_bases: Vec<usize>,
    pub neigh: Vec<usize>,
    pub occ_wpn: Vec<usize>,
    pub offsets: Vec<Vec<(i8, i8)>>,
    pub off_flat: Vec<Vec<i32>>,
}

pub(crate) fn derive_layout(topology: &[TopologyLevel], size: u32) -> DerivedLayout {
    let n_levels = topology.len();
    let mut slot_bases = Vec::with_capacity(n_levels);
    let mut neigh = Vec::with_capacity(n_levels);
    let mut occ_wpn = Vec::with_capacity(n_levels);
    let mut offsets: Vec<Vec<(i8, i8)>> = Vec::with_capacity(n_levels);
    let mut off_flat: Vec<Vec<i32>> = Vec::with_capacity(n_levels);
    let mut total_slots = 0usize;
    for t in topology {
        slot_bases.push(total_slots);
        let n = neigh_size(t.radius);
        neigh.push(n);
        occ_wpn.push((n + 63) / 64);
        let span = 2 * t.radius + 1;
        let r = t.radius as i32;
        offsets.push((0..n).map(|c| (((c as u32 % span) as i32 - r) as i8, ((c as u32 / span) as i32 - r) as i8)).collect());
        off_flat.push((0..n).map(|c| { let dx = (c as u32 % span) as i32 - r; let dy = (c as u32 / span) as i32 - r; dy * size as i32 + dx }).collect());
        total_slots += t.count as usize;
    }
    DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat }
}

/// `pow_decay[k] = round(ρ^k · 2^FRAC)`, ρ = 1 − 2^−adapt_decay, `pow_decay[0] = 2^FRAC`. Grows until
/// even `ADAPT_MAX` decays to 0 through it, capped at `HORIZON_CAP`. Reconstructs adaptation exactly and
/// path-independently (a single jump from the last fire), so dense and sparse agree bit-for-bit.
pub fn build_pow_decay(adapt_decay: u8) -> Vec<i64> {
    let one = 1i64 << FRAC;
    let rho = one - (1i64 << (FRAC - adapt_decay as u32)); // ρ in fixed point
    let mut table = vec![one];
    let mut cur = one;
    while table.len() < HORIZON_CAP {
        // round(cur · ρ / 2^FRAC)
        let next = ((cur as i128 * rho as i128 + (1i128 << (FRAC - 1))) >> FRAC) as i64;
        if next <= 0 || ((ADAPT_MAX as i128 * next as i128) >> FRAC) == 0 {
            break; // even the largest possible adapt now reconstructs to 0 → horizon reached
        }
        table.push(next);
        cur = next;
    }
    table
}

pub struct Layer {
    // neuron state
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub threshold: Vec<i16>,
    pub adapt_ref: Vec<i32>, // adaptation value at the last fire (Q ADAPT_SHIFT)
    pub fire_wave: Vec<u32>, // wave index of the last fire
    pub pending: Vec<i32>,   // per-target incoming accumulator, drained (folded) each wave
    // config
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<TopologyLevel>,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub readout: bool,
    pub ternary_threshold: f32,
    // derived layout
    pub total_slots: usize,
    pub slot_bases: Vec<usize>,
    pub neigh: Vec<usize>,
    pub occ_wpn: Vec<usize>,
    pub occ: Vec<Vec<u64>>,
    pub offsets: Vec<Vec<(i8, i8)>>,
    pub off_flat: Vec<Vec<i32>>,
    pub codes: Vec<u64>, // 2-bit ±1/0 codes, 32 per u64
    // lazy adaptation decay table (per adapt_decay)
    pub pow_decay: Vec<i64>,
}

impl Layer {
    #[inline]
    pub fn slot_base(&self, level_idx: usize) -> usize {
        self.slot_bases[level_idx]
    }

    #[inline]
    pub fn weight_at(&self, widx: usize) -> i8 {
        WCODE[((self.codes[widx >> 5] >> ((widx & 31) * 2)) & 0b11) as usize]
    }

    #[inline]
    pub fn synapse_count(&self) -> usize {
        self.total_slots * self.threshold.len()
    }

    /// Iterate the wired cells of neuron `i` at level `lvl` in ascending cell order, calling `f(rank, cell)`.
    #[inline]
    pub fn for_wired(&self, lvl: usize, i: usize, mut f: impl FnMut(usize, usize)) {
        let wpn = self.occ_wpn[lvl];
        let words = &self.occ[lvl][i * wpn..i * wpn + wpn];
        let mut rank = 0usize;
        for (wi, &w0) in words.iter().enumerate() {
            let mut word = w0;
            let cbase = wi * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                f(rank, cbase + bit);
                rank += 1;
                word &= word - 1;
            }
        }
    }

    /// Decode neighborhood `cell` of a source at `src_local` to its target local index (offset LUT + wrap).
    #[inline]
    pub fn decode(&self, lvl: usize, src_local: u32, cell: usize, size: u32) -> u32 {
        let (sx, sy) = xy_of(src_local, size);
        let (dx, dy) = self.offsets[lvl][cell];
        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size)
    }

    /// Reconstruct neuron `i`'s adaptation at wave `w`: `(adapt_ref · ρ^(w − fire_wave)) >> FRAC`, or 0
    /// beyond the decay horizon. Pure function of the stored anchor — path-independent.
    #[inline]
    pub fn decayed_adapt(&self, i: usize, w: u32) -> i32 {
        let gap = w.wrapping_sub(self.fire_wave[i]) as usize;
        if gap >= self.pow_decay.len() {
            0
        } else {
            ((self.adapt_ref[i] as i64 * self.pow_decay[gap]) >> FRAC) as i32
        }
    }

    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;
        let DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat } = derive_layout(&cfg.topology, size);

        // thresholds: baseline_init + rand(0..threshold_jitter), clamp(1, i16::MAX)
        let mut threshold = vec![0i16; ls];
        for (local, th) in threshold.iter_mut().enumerate() {
            let global = (base + local) as u32;
            let h = mix(key(seed, global, 0, 0, P_THRESHOLD));
            let jitter = map_range(h as u32, cfg.threshold_jitter as u32) as i32;
            *th = (cfg.baseline_init as i32 + jitter).clamp(1, i16::MAX as i32) as i16;
        }

        // occupancy: `count` distinct cells per neuron per level, word-aligned
        let mut occ: Vec<Vec<u64>> = occ_wpn.iter().map(|&wpn| vec![0u64; ls * wpn]).collect();
        for (li, t) in cfg.topology.iter().enumerate() {
            let wpn = occ_wpn[li];
            for i in 0..ls {
                let sg = (base + i) as u32;
                for &cell in &sample_distinct_cells(seed, sg, t.level, t.radius, t.count) {
                    let c = cell as usize;
                    occ[li][i * wpn + c / 64] |= 1u64 << (c % 64);
                }
            }
        }

        // codes: init each wired synapse to the procedural ±1 sign (rank-indexed, wired-rank order)
        let mut codes = vec![0u64; (ls * total_slots + 31) / 32];
        for i in 0..ls {
            let sg = (base + i) as u32;
            for (li, t) in cfg.topology.iter().enumerate() {
                for r in 0..(t.count as usize) {
                    let h = mix(key(seed, sg, t.level, r as u32, P_TARGET));
                    let sign_code: u64 = if ((h & 0xFFFF) as u32) < cfg.inhibitor_ratio { 0b11 } else { 0b01 };
                    let idx = i * total_slots + slot_bases[li] + r;
                    let wshift = (idx & 31) * 2;
                    codes[idx >> 5] |= sign_code << wshift;
                }
            }
        }

        Layer {
            potential: vec![0i16; ls],
            cooldown: vec![0u8; ls],
            threshold,
            adapt_ref: vec![0i32; ls],
            fire_wave: vec![0u32; ls],
            pending: vec![0i32; ls],
            leak: cfg.leak,
            cooldown_base: cfg.cooldown_base,
            topology: cfg.topology.clone(),
            adapt_bump: cfg.adapt_bump,
            adapt_decay: cfg.adapt_decay,
            readout: false,
            ternary_threshold: 0.5,
            total_slots,
            slot_bases,
            neigh,
            occ_wpn,
            occ,
            offsets,
            off_flat,
            codes,
            pow_decay: build_pow_decay(cfg.adapt_decay),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_driven::config::LayerConfig;
    use crate::wave_driven::synapse::TopologyLevel;

    fn lc(topology: Vec<TopologyLevel>) -> LayerConfig {
        LayerConfig { topology, leak: (3, 5), cooldown_base: 2, inhibitor_ratio: 0, threshold_jitter: 0, baseline_init: 6, adapt_bump: 5, adapt_decay: 6 }
    }

    #[test]
    fn new_wires_exactly_count_distinct_cells_deterministically() {
        let size = 8u32;
        let ls = (size * size) as usize;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 16 }]);
        let a = Layer::new(&cfg, 7, 0, size);
        let b = Layer::new(&cfg, 7, 0, size);
        assert_eq!(a.total_slots, 16);
        for i in 0..ls {
            let mut cnt = 0usize;
            let mut cells = Vec::new();
            a.for_wired(0, i, |_r, c| { cnt += 1; cells.push(c); });
            assert_eq!(cnt, 16);
            assert!(cells.windows(2).all(|w| w[0] < w[1]));
        }
        assert_eq!(a.occ, b.occ, "deterministic occupancy");
        assert_eq!(a.codes, b.codes, "deterministic ±1 codes");
    }

    #[test]
    fn weight_at_decodes_pm1_from_codes() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]);
        let l = Layer::new(&cfg, 3, 0, size);
        // fresh net: every wired synapse is ±1 (procedural sign), never 0 (inhibitor_ratio 0 => all +1).
        for s in 0..l.synapse_count() {
            assert!(matches!(l.weight_at(s), 1 | -1), "fresh code is ±1, got {}", l.weight_at(s));
        }
    }

    #[test]
    fn decode_center_is_self() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let l = Layer::new(&cfg, 7, 0, size);
        let src = crate::wave_driven::synapse::local_of(3, 4, size);
        assert_eq!(l.decode(0, src, 12, size), src, "center cell (idx 12, span 5) maps to self");
    }
}
