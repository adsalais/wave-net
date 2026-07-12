//! `neurons` — a `Layer`'s per-neuron state and its bitset weight representation. Topology is a
//! per-neuron neighborhood occupancy bitset (filled once at startup); weights are 2-bit (a `nonzero`
//! mask + a `sign` bit) with an `f32` training shadow. Neuron/eligibility state mirrors `wave_net`.

use crate::wave_bitnet::bits::BitSet;
use crate::wave_bitnet::config::LayerConfig;
use crate::wave_bitnet::synapse::{key, map_range, mix, neigh_size, sample_distinct_cells, Synapse, TopologyLevel, P_TARGET, P_THRESHOLD};

/// Fixed-point scale for `adapt`: it holds the effective threshold contribution × `2^ADAPT_SHIFT`.
/// Bounded by the i32 overflow limit on the bump-add (`2·ADAPT_MAX = i16::MAX << (SHIFT+1)` must fit
/// i32, i.e. `SHIFT <= 14`); 12 keeps ~8× margin and allows `adapt_decay` up to 12 (τ ≈ 4096 waves).
pub const ADAPT_SHIFT: u32 = 12;
/// Ceiling for `adapt`, so the effective contribution never exceeds `i16::MAX` (overflow guard).
pub const ADAPT_MAX: i32 = (i16::MAX as i32) << ADAPT_SHIFT;

pub struct Layer {
    // neuron state (identical to wave_net::neurons::Layer)
    pub potential: Vec<i16>,
    pub cooldown: Vec<u8>,
    pub adapt: Vec<i32>,
    pub threshold: Vec<i16>,
    pub inbox: Vec<Synapse>,
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
    pub total_slots: usize,     // Σ count
    pub slot_bases: Vec<usize>, // per level: Σ_{ℓ'<ℓ} count
    pub neigh: Vec<usize>,      // per level: (2r+1)²
    // BITSET representation
    pub occupancy: Vec<BitSet>, // per level: ls·neigh[ℓ] bits
    pub w_nonzero: BitSet,      // ls·total_slots bits
    pub w_sign: BitSet,         // ls·total_slots bits
    pub shadow: Vec<f32>,       // ls·total_slots
}

impl Layer {
    #[inline]
    pub fn slot_base(&self, level_idx: usize) -> usize {
        self.slot_bases[level_idx]
    }

    #[inline]
    pub fn weight_at(&self, widx: usize) -> i8 {
        if !self.w_nonzero.get(widx) {
            0
        } else if self.w_sign.get(widx) {
            1
        } else {
            -1
        }
    }

    #[inline]
    fn set_weight_bits(&mut self, idx: usize, nonzero: bool, sign: bool) {
        self.w_nonzero.put(idx, nonzero);
        self.w_sign.put(idx, sign);
    }

    /// Requantise neuron `i`'s row (all `total_slots` shadow values) into `w_nonzero`/`w_sign`:
    /// γ = mean(|shadow|) over the row; `|shadow|/γ < t → 0`, else sign(shadow).
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
            let idx = base + s;
            if x.abs() < t {
                self.set_weight_bits(idx, false, false);
            } else {
                self.set_weight_bits(idx, true, x > 0.0);
            }
        }
    }

    pub fn new(cfg: &LayerConfig, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;
        let n_levels = cfg.topology.len();

        // derived layout
        let mut slot_bases = Vec::with_capacity(n_levels);
        let mut neigh = Vec::with_capacity(n_levels);
        let mut total_slots = 0usize;
        for t in &cfg.topology {
            slot_bases.push(total_slots);
            neigh.push(neigh_size(t.radius));
            total_slots += t.count as usize;
        }

        // threshold: baseline_init + rand(0..threshold_jitter), clamp(1, i16::MAX)  (verbatim wave_net)
        let mut threshold = vec![0i16; ls];
        for (local, th) in threshold.iter_mut().enumerate() {
            let global = (base + local) as u32;
            let h = mix(key(seed, global, 0, 0, P_THRESHOLD));
            let jitter = map_range(h as u32, cfg.threshold_jitter as u32) as i32; // [0, jitter)
            *th = (cfg.baseline_init as i32 + jitter).clamp(1, i16::MAX as i32) as i16;
        }

        // occupancy: fill `count` distinct cells per neuron per level
        let mut occupancy: Vec<BitSet> = neigh.iter().map(|&n| BitSet::zeros(ls * n)).collect();
        for (li, t) in cfg.topology.iter().enumerate() {
            let n = neigh[li];
            for i in 0..ls {
                let sg = (base + i) as u32;
                for &cell in &sample_distinct_cells(seed, sg, t.level, t.radius, t.count) {
                    occupancy[li].set(i * n + cell as usize);
                }
            }
        }

        // shadow init: ±1 sign from inhibitor_ratio (wave_net's rule; index the r-th wired synapse by r)
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
            inbox: Vec::new(),
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
            occupancy,
            w_nonzero: BitSet::zeros(ls * total_slots),
            w_sign: BitSet::zeros(ls * total_slots),
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
            let set = a.occupancy[0].iter_set_in(i * a.neigh[0], a.neigh[0]).count();
            assert_eq!(set, 16, "neuron {i} wires exactly count cells");
        }
        assert_eq!(a.occupancy[0].count_ones(), b.occupancy[0].count_ones());
    }

    #[test]
    fn repack_roundtrips_shadow_to_ternary() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let mut l = Layer::new(&cfg, 7, 0, size);
        // neuron 0 row shadow [2.0, -3.0, 0.05, 0.0]; γ = 1.2625; t=0.5
        // |x|/γ: 1.58, 2.38, 0.04, 0.0 -> nonzero [1,1,0,0], signs [+,-,.,.]
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
