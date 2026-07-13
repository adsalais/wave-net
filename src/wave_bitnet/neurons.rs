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

/// Layout quantities derived purely from `(topology, size)` — no seed, no RNG. Shared by
/// `Layer::new` (fresh build) and `Layer::from_parts` (load) so a loaded layer's LUTs are
/// byte-identical to a freshly-built one.
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
        offsets.push(
            (0..n)
                .map(|c| (((c as u32 % span) as i32 - r) as i8, ((c as u32 / span) as i32 - r) as i8))
                .collect(),
        );
        off_flat.push(
            (0..n)
                .map(|c| {
                    let dx = (c as u32 % span) as i32 - r;
                    let dy = (c as u32 / span) as i32 - r;
                    dy * size as i32 + dx
                })
                .collect(),
        );
        total_slots += t.count as usize;
    }
    DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat }
}

/// All per-layer TRAINING state — allocated only while training is enabled (see
/// `Network::enable_training`). Absent (`Layer.train == None`) on an inference-lean net.
/// `shadow` is the f32 training master requantized into `codes` by `repack_row`; the two
/// decide-time snapshots are the credit-assignment records the bench reads each wave.
pub struct TrainState {
    pub shadow: Vec<f32>,           // ls * total_slots
    pub decide_potential: Vec<i16>, // ls
    pub decide_eff: Vec<i32>,       // ls
}

pub struct Layer {
    // neuron state
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub adapt: Vec<i32>,
    pub threshold: Vec<i16>,
    pub pending: Vec<i32>, // per-target incoming accumulator (scatter-add target); folded in step 1
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
    pub offsets: Vec<Vec<(i8, i8)>>, // per level: cell -> (dx, dy) LUT (for the wrapping edge path)
    pub off_flat: Vec<Vec<i32>>,     // per level: cell -> dy·size + dx (flat delta, interior fast path)
    // WEIGHTS (rank-indexed): 2-bit codes packed 32 per u64 — 0b00=0, 0b01=+1, 0b11=−1.
    pub codes: Vec<u64>, // ceil(ls·total_slots / 32) words
    // TRAINING state — present only while training is enabled (None on an inference-lean net).
    pub train: Option<TrainState>,
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

    /// Number of stored synapses (`ls * total_slots`) — independent of whether training is enabled.
    #[inline]
    pub fn synapse_count(&self) -> usize {
        self.total_slots * self.threshold.len()
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
        let t = self.ternary_threshold;
        let gamma = {
            let shadow = &self.train.as_ref().expect("repack_row requires training enabled").shadow;
            let mut sum = 0.0f32;
            for s in 0..ts {
                sum += shadow[base + s].abs();
            }
            sum / ts as f32
        };
        for s in 0..ts {
            let sh = self.train.as_ref().unwrap().shadow[base + s];
            let x = if gamma <= 0.0 { 0.0 } else { sh / gamma };
            let code: u64 = if x.abs() < t { 0b00 } else if x > 0.0 { 0b01 } else { 0b11 };
            self.set_code(base + s, code);
        }
    }

    /// Build a `Layer` directly from persisted parts (bypassing seed-based generation): rebuild the
    /// derived LUTs from `topology`+`size`, reconstruct `shadow` as the per-slot decode of `codes`
    /// (codes are authoritative for inference), and zero all runtime arrays. Validates array shapes;
    /// returns `Err(msg)` on any mismatch.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        topology: Vec<TopologyLevel>,
        leak: (u8, u8),
        cooldown_base: u8,
        adapt_bump: i16,
        adapt_decay: u8,
        readout: bool,
        ternary_threshold: f32,
        threshold: Vec<i16>,
        occ: Vec<Vec<u64>>,
        codes: Vec<u64>,
        size: u32,
    ) -> Result<Layer, String> {
        let ls = (size as usize) * (size as usize);
        let DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat } =
            derive_layout(&topology, size);
        if threshold.len() != ls {
            return Err(format!("threshold length {} != ls {ls}", threshold.len()));
        }
        if occ.len() != topology.len() {
            return Err(format!("occ levels {} != topology levels {}", occ.len(), topology.len()));
        }
        for (li, t) in topology.iter().enumerate() {
            if t.count as usize > neigh[li] {
                return Err(format!("level {li}: count {} exceeds neighborhood {}", t.count, neigh[li]));
            }
            let want = ls * occ_wpn[li];
            if occ[li].len() != want {
                return Err(format!("occ[{li}] length {} != ls*occ_wpn {want}", occ[li].len()));
            }
        }
        let want_codes = (ls * total_slots + 31) / 32;
        if codes.len() != want_codes {
            return Err(format!("codes length {} != {want_codes}", codes.len()));
        }
        // shadow = per-slot decode of codes (inference-authoritative; not the training master)
        let n = ls * total_slots;
        let mut shadow = vec![0f32; n];
        for s in 0..n {
            shadow[s] = WCODE[((codes[s >> 5] >> ((s & 31) * 2)) & 0b11) as usize] as f32;
        }
        Ok(Layer {
            potential: vec![0i16; ls],
            cooldown: vec![0u8; ls],
            adapt: vec![0i32; ls],
            threshold,
            pending: vec![0i32; ls],
            leak,
            cooldown_base,
            topology,
            adapt_bump,
            adapt_decay,
            readout,
            ternary_threshold,
            total_slots,
            slot_bases,
            neigh,
            occ_wpn,
            occ,
            offsets,
            off_flat,
            codes,
            train: Some(TrainState {
                shadow,
                decide_potential: vec![0i16; ls],
                decide_eff: vec![0i32; ls],
            }),
        })
    }

    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;

        // derived layout + per-level occupancy word count + offset LUT (shared with load)
        let DerivedLayout { total_slots, slot_bases, neigh, occ_wpn, offsets, off_flat } =
            derive_layout(&cfg.topology, size);

        // threshold: baseline_init + rand(0..threshold_jitter), clamp(1, i16::MAX)
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
            codes: vec![0u64; (ls * total_slots + 31) / 32],
            train: Some(TrainState {
                shadow,
                decide_potential: vec![0i16; ls],
                decide_eff: vec![0i32; ls],
            }),
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
    fn derive_layout_matches_expected() {
        let topo = vec![
            TopologyLevel { level: 1, radius: 2, count: 8 },
            TopologyLevel { level: 0, radius: 1, count: 3 },
        ];
        let d = derive_layout(&topo, 8);
        assert_eq!(d.total_slots, 11);
        assert_eq!(d.slot_bases, vec![0, 8]);
        assert_eq!(d.neigh, vec![25, 9]);
        assert_eq!(d.occ_wpn, vec![1, 1]);
        assert_eq!(d.offsets[0].len(), 25);
        assert_eq!(d.off_flat[1].len(), 9);
        assert_eq!(d.offsets[1][4], (0, 0)); // center of a radius-1 (span-3) level
        assert_eq!(d.off_flat[1][4], 0);
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
    fn from_parts_reproduces_built_layer() {
        let size = 8u32;
        let cfg = lc(vec![
            TopologyLevel { level: 1, radius: 2, count: 8 },
            TopologyLevel { level: 0, radius: 1, count: 3 },
        ]);
        let mut built = Layer::new(&cfg, 7, 1, size);
        // make neuron 0's row non-trivial (+1 / -1 / 0 mix) then repack so codes differ from init
        let ts = built.total_slots;
        for s in 0..ts {
            built.train.as_mut().unwrap().shadow[s] = match s % 3 { 0 => 0.0, 1 => 2.0, _ => -2.0 };
        }
        built.repack_row(0);
        let rebuilt = Layer::from_parts(
            built.topology.clone(),
            built.leak,
            built.cooldown_base,
            built.adapt_bump,
            built.adapt_decay,
            built.readout,
            built.ternary_threshold,
            built.threshold.clone(),
            built.occ.clone(),
            built.codes.clone(),
            size,
        )
        .unwrap();
        assert_eq!(rebuilt.topology, built.topology);
        assert_eq!(rebuilt.threshold, built.threshold);
        assert_eq!(rebuilt.occ, built.occ);
        assert_eq!(rebuilt.codes, built.codes);
        assert_eq!(rebuilt.total_slots, built.total_slots);
        assert_eq!(rebuilt.slot_bases, built.slot_bases);
        assert_eq!(rebuilt.neigh, built.neigh);
        assert_eq!(rebuilt.occ_wpn, built.occ_wpn);
        assert_eq!(rebuilt.offsets, built.offsets);
        assert_eq!(rebuilt.off_flat, built.off_flat);
        // shadow is the decode of codes (inference-authoritative), runtime zeroed
        let rebuilt_shadow = &rebuilt.train.as_ref().unwrap().shadow;
        for s in 0..rebuilt_shadow.len() {
            assert_eq!(rebuilt_shadow[s], rebuilt.weight_at(s) as f32);
        }
        assert!(rebuilt.potential.iter().all(|&p| p == 0));
        assert!(rebuilt.adapt.iter().all(|&a| a == 0));
    }

    #[test]
    fn from_parts_rejects_bad_lengths() {
        let size = 8u32;
        let topo = vec![TopologyLevel { level: 1, radius: 2, count: 8 }];
        // threshold length 10 != ls 64
        let r = Layer::from_parts(topo, (3, 5), 2, 5, 6, false, 0.5, vec![0i16; 10], vec![vec![0u64; 64]], vec![0u64; 16], size);
        assert!(r.is_err());
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
        {
            let sh = &mut l.train.as_mut().unwrap().shadow;
            sh[0 * ts + 0] = 2.0;
            sh[0 * ts + 1] = -3.0;
            sh[0 * ts + 2] = 0.05;
            sh[0 * ts + 3] = 0.0;
        }
        l.repack_row(0);
        assert_eq!(l.weight_at(0 * ts + 0), 1);
        assert_eq!(l.weight_at(0 * ts + 1), -1);
        assert_eq!(l.weight_at(0 * ts + 2), 0);
        assert_eq!(l.weight_at(0 * ts + 3), 0);
    }
}
