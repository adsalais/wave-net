//! `neurons` — a `Layer`'s per-neuron state and its bitset weight representation. Topology is a
//! per-neuron neighborhood occupancy bitset, stored **word-aligned** (each neuron gets whole `u64`
//! words) so the forward pass can iterate only set bits via `trailing_zeros` instead of scanning the
//! whole neighborhood. Weights are 2-bit (a `nonzero` mask + a `sign` bit) with an `f32` training
//! shadow. A per-level offset LUT turns a cell index into a (dx, dy) with no integer div/mod.

use crate::wave_bitnet::config::LayerConfig;
use crate::wave_bitnet::synapse::{key, local_of, map_range, mix, neigh_size, sample_distinct_cells, wrap, xy_of, TopologyLevel, P_TARGET, P_THRESHOLD};

/// Fixed-point scale for `adapt`: it holds the effective threshold contribution × `2^ADAPT_SHIFT`.
/// Bounded by the i32 overflow limit on the bump-add (`2·ADAPT_MAX = i16::MAX << (SHIFT+1)` must fit
/// i32, i.e. `SHIFT <= 14`); 12 keeps ~8× margin and allows `adapt_decay` up to 12 (τ ≈ 4096 waves).
pub const ADAPT_SHIFT: u32 = 12;
/// Ceiling for `adapt`, so the effective contribution never exceeds `i16::MAX` (overflow guard).
pub const ADAPT_MAX: i32 = (i16::MAX as i32) << ADAPT_SHIFT;

/// 2-bit weight code decode LUT: `0b00`→0, `0b01`→+1, `0b11`→−1 (`0b10` unused → 0).
pub(crate) const WCODE: [i8; 4] = [0, 1, 0, -1];

pub struct Layer {
    // neuron state (identical to wave_net::neurons::Layer)
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub adapt: Vec<i32>,
    pub threshold: Vec<i16>,
    pub pending: Vec<i32>, // per-target incoming accumulator (scatter-add target); folded in step 1
    // eligibility / decide-step state (identical to wave_net)
    pub elig_pre: Vec<i32>,
    pub elig_post: Vec<i32>,
    pub decide_potential: Vec<i16>,
    pub decide_eff: Vec<i32>,
    // config
    pub leak: (u8, u8),
    pub cooldown_base: u8,
    pub topology: Vec<TopologyLevel>,
    pub adapt_bump: i16,
    pub adapt_decay: u8,
    pub readout: bool,
    pub ternary_threshold: f32,
    // derived layout
    pub total_slots: usize,          // Σ count
    pub slot_bases: Vec<usize>,      // per level: Σ_{ℓ'<ℓ} count
    pub neigh: Vec<usize>,           // per level: (2r+1)²  (logical neighborhood size)
    pub occ_wpn: Vec<usize>,         // per level: u64 words per neuron = ceil(neigh/64)
    // TOPOLOGY: per-neuron occupancy, word-aligned. occ[ℓ] has ls·occ_wpn[ℓ] words; neuron i at [i·wpn..].
    pub occ: Vec<Vec<u64>>,
    pub offsets: Vec<Vec<(i8, i8)>>, // per level: cell -> (dx, dy) LUT (length neigh[ℓ])
    // WEIGHTS (rank-indexed): 2-bit codes packed 32 per u64 — 0b00=0, 0b01=+1, 0b11=−1.
    pub codes: Vec<u64>,  // ceil(ls·total_slots / 32) words
    pub shadow: Vec<f32>, // ls·total_slots
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

    /// Iterate the wired cells of neuron `i` at level `lvl`, in ascending cell order, calling
    /// `f(rank, cell)`. Word-scan: only set bits are visited (`trailing_zeros` + clear-lowest-set).
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

    /// Decode a neighborhood `cell` of a source at `src_local` to its target local index, via the
    /// per-level (dx, dy) offset LUT + toroidal wrap (no integer div/mod on the hot path).
    #[inline]
    pub fn decode(&self, lvl: usize, src_local: u32, cell: usize, size: u32) -> u32 {
        let (sx, sy) = xy_of(src_local, size);
        let (dx, dy) = self.offsets[lvl][cell];
        local_of(wrap(sx, dx as i32, size), wrap(sy, dy as i32, size), size)
    }

    #[inline]
    fn set_code(&mut self, idx: usize, code: u64) {
        let w = idx >> 5;
        let shift = (idx & 31) * 2;
        self.codes[w] = (self.codes[w] & !(0b11u64 << shift)) | (code << shift);
    }

    /// Requantise neuron `i`'s row (all `total_slots` shadow values) into the 2-bit `codes` array:
    /// γ = mean(|shadow|) over the row; `|shadow|/γ < t → 0` (0b00), else sign(shadow) (+1=0b01, −1=0b11).
    pub fn repack_row(&mut self, i: usize) {
        let ts = self.total_slots;
        if ts == 0 {
            return;
        }
        let base = i * ts;
        let mut sum = 0.0f32;
        for s in 0..ts {
            sum += self.shadow[base + s].abs();
        }
        let gamma = sum / ts as f32;
        let t = self.ternary_threshold;
        for s in 0..ts {
            let sh = self.shadow[base + s];
            let x = if gamma <= 0.0 { 0.0 } else { sh / gamma };
            let code: u64 = if x.abs() < t { 0b00 } else if x > 0.0 { 0b01 } else { 0b11 };
            self.set_code(base + s, code);
        }
    }

    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;
        let n_levels = cfg.topology.len();

        // derived layout + per-level occupancy word count + offset LUT
        let mut slot_bases = Vec::with_capacity(n_levels);
        let mut neigh = Vec::with_capacity(n_levels);
        let mut occ_wpn = Vec::with_capacity(n_levels);
        let mut offsets: Vec<Vec<(i8, i8)>> = Vec::with_capacity(n_levels);
        let mut total_slots = 0usize;
        for t in &cfg.topology {
            slot_bases.push(total_slots);
            let n = neigh_size(t.radius);
            neigh.push(n);
            occ_wpn.push((n + 63) / 64);
            let span = 2 * t.radius + 1;
            let r = t.radius as i32;
            offsets.push(
                (0..n)
                    .map(|c| (((c as u32 % span) as i32 - r) as i8, ((c as u32 / span) as i32 - r) as i8))
                    .collect(),
            );
            total_slots += t.count as usize;
        }

        // threshold: baseline_init + rand(0..threshold_jitter), clamp(1, i16::MAX)  (verbatim wave_net)
        let mut threshold = vec![0i16; ls];
        for (local, th) in threshold.iter_mut().enumerate() {
            let global = (base + local) as u32;
            let h = mix(key(seed, global, 0, 0, P_THRESHOLD));
            let jitter = map_range(h as u32, cfg.threshold_jitter as u32) as i32;
            *th = (cfg.baseline_init as i32 + jitter).clamp(1, i16::MAX as i32) as i16;
        }

        // occupancy: fill `count` distinct cells per neuron per level, word-aligned
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

        // shadow init: ±1 sign from inhibitor_ratio (rank r = r-th wired synapse)
        let mut shadow = vec![0f32; ls * total_slots];
        for i in 0..ls {
            let sg = (base + i) as u32;
            for (li, t) in cfg.topology.iter().enumerate() {
                for r in 0..(t.count as usize) {
                    let h = mix(key(seed, sg, t.level, r as u32, P_TARGET));
                    let sign = if ((h & 0xFFFF) as u32) < cfg.inhibitor_ratio { -1.0 } else { 1.0 };
                    shadow[i * total_slots + slot_bases[li] + r] = sign;
                }
            }
        }

        let mut layer = Layer {
            potential: vec![0i16; ls],
            cooldown: vec![0u8; ls],
            adapt: vec![0i32; ls],
            threshold,
            pending: vec![0i32; ls],
            elig_pre: vec![0i32; ls],
            elig_post: vec![0i32; ls],
            decide_potential: vec![0i16; ls],
            decide_eff: vec![0i32; ls],
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
            codes: vec![0u64; (ls * total_slots + 31) / 32],
            shadow,
        };
        for i in 0..ls {
            layer.repack_row(i);
        }
        layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_bitnet::config::LayerConfig;
    use crate::wave_bitnet::synapse::TopologyLevel;

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
            a.for_wired(0, i, |_r, c| {
                cnt += 1;
                cells.push(c);
            });
            assert_eq!(cnt, 16, "neuron {i} wires exactly count cells");
            assert!(cells.windows(2).all(|w| w[0] < w[1]), "for_wired yields ascending cell order");
        }
        assert_eq!(a.occ, b.occ, "deterministic occupancy");
    }

    #[test]
    fn decode_matches_offset_lut() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let l = Layer::new(&cfg, 7, 0, size);
        // center cell (index 12 for span 5) decodes to self at dx=dy=0
        assert_eq!(l.decode(0, crate::wave_bitnet::synapse::local_of(3, 4, size), 12, size), crate::wave_bitnet::synapse::local_of(3, 4, size));
    }

    #[test]
    fn repack_roundtrips_shadow_to_ternary() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let mut l = Layer::new(&cfg, 7, 0, size);
        let ts = l.total_slots;
        l.shadow[0 * ts + 0] = 2.0;
        l.shadow[0 * ts + 1] = -3.0;
        l.shadow[0 * ts + 2] = 0.05;
        l.shadow[0 * ts + 3] = 0.0;
        l.repack_row(0);
        assert_eq!(l.weight_at(0 * ts + 0), 1);
        assert_eq!(l.weight_at(0 * ts + 1), -1);
        assert_eq!(l.weight_at(0 * ts + 2), 0);
        assert_eq!(l.weight_at(0 * ts + 3), 0);
    }
}
