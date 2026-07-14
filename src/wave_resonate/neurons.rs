//! `neurons` — a BRF `Layer`'s per-neuron SoA state (f32 x,y,q + per-neuron ω,b') and the copied bitset
//! topology substrate (occupancy + offset LUTs + 2-bit ±1/0 codes). Structure duplicated from
//! wave_driven::neurons; LIF/adaptation state is replaced by the resonator state.

use crate::wave_resonate::config::LayerConfig;
use crate::wave_resonate::synapse::{key, local_of, mix, neigh_size, sample_distinct_cells, wrap, xy_of, TopologyLevel, P_TARGET, P_THRESHOLD};

/// 2-bit weight code decode LUT: 0b00→0, 0b01→+1, 0b11→−1.
pub(crate) const WCODE: [i8; 4] = [0, 1, 0, -1];

/// Divergence boundary p(ω) = (−1 + √(1 − (δω)²)) / δ. Caller guarantees δ·ω ≤ 1 (Config::validate).
#[inline]
pub fn pw(omega: f32, dt: f32) -> f32 {
    (-1.0 + (1.0 - (dt * omega) * (dt * omega)).sqrt()) / dt
}

pub struct Layer {
    // BRF neuron state (readout layers reuse `x` as the leaky-integrator accumulator)
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub q: Vec<f32>,
    pub omega: Vec<f32>,
    pub b_off: Vec<f32>,
    pub pending: Vec<i32>,
    // dynamics constants
    pub dt: f32,
    pub gamma: f32,
    pub theta_c: f32,
    pub kappa: f32,
    // role
    pub transducer: bool,
    pub readout: bool,
    // topology substrate
    pub topology: Vec<TopologyLevel>,
    pub total_slots: usize,
    pub slot_bases: Vec<usize>,
    pub neigh: Vec<usize>,
    pub occ_wpn: Vec<usize>,
    pub occ: Vec<Vec<u64>>,
    pub offsets: Vec<Vec<(i8, i8)>>,
    pub off_flat: Vec<Vec<i32>>,
    pub codes: Vec<u64>,
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
        self.total_slots * self.x.len()
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

    pub fn new(cfg: &LayerConfig, dt: f32, gamma: f32, theta_c: f32, seed: u64, layer_index: u32, size: u32) -> Layer {
        let ls = (size as usize) * (size as usize);
        let base = layer_index as usize * ls;

        // derived layout (occupancy word counts + offset LUTs), copied from wave_driven::derive_layout
        let n_levels = cfg.topology.len();
        let mut slot_bases = Vec::with_capacity(n_levels);
        let mut neigh = Vec::with_capacity(n_levels);
        let mut occ_wpn = Vec::with_capacity(n_levels);
        let mut offsets: Vec<Vec<(i8, i8)>> = Vec::with_capacity(n_levels);
        let mut off_flat: Vec<Vec<i32>> = Vec::with_capacity(n_levels);
        let mut total_slots = 0usize;
        for t in &cfg.topology {
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

        // per-neuron ω, b' from the init ranges (deterministic hash streams; slot 0 = ω, slot 1 = b')
        let (olo, ohi) = cfg.omega_init;
        let (blo, bhi) = cfg.b_offset_init;
        let mut omega = vec![0f32; ls];
        let mut b_off = vec![0f32; ls];
        for local in 0..ls {
            let g = (base + local) as u32;
            let ho = mix(key(seed, g, 0, 0, P_THRESHOLD));
            let hb = mix(key(seed, g, 0, 1, P_THRESHOLD));
            let fo = ((ho >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
            let fb = ((hb >> 40) as f32) / ((1u64 << 24) as f32);
            omega[local] = olo + (ohi - olo) * fo;
            b_off[local] = blo + (bhi - blo) * fb;
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

        // codes: init each wired synapse to the procedural ±1 sign (wired-rank order)
        let mut codes = vec![0u64; (ls * total_slots + 31) / 32];
        for i in 0..ls {
            let sg = (base + i) as u32;
            for (li, t) in cfg.topology.iter().enumerate() {
                for r in 0..(t.count as usize) {
                    let h = mix(key(seed, sg, t.level, r as u32, P_TARGET));
                    let sign_code: u64 = if ((h & 0xFFFF) as u32) < cfg.inhibitor_ratio { 0b11 } else { 0b01 };
                    let idx = i * total_slots + slot_bases[li] + r;
                    codes[idx >> 5] |= sign_code << ((idx & 31) * 2);
                }
            }
        }

        let kappa = (-dt / cfg.tau_out).exp();
        Layer {
            x: vec![0f32; ls],
            y: vec![0f32; ls],
            q: vec![0f32; ls],
            omega,
            b_off,
            pending: vec![0i32; ls],
            dt,
            gamma,
            theta_c,
            kappa,
            transducer: false,
            readout: false,
            topology: cfg.topology.clone(),
            total_slots,
            slot_bases,
            neigh,
            occ_wpn,
            occ,
            offsets,
            off_flat,
            codes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_resonate::config::LayerConfig;
    use crate::wave_resonate::synapse::TopologyLevel;

    fn lc(topology: Vec<TopologyLevel>) -> LayerConfig {
        LayerConfig { topology, inhibitor_ratio: 0, omega_init: (5.0, 10.0), b_offset_init: (0.1, 1.0), tau_out: 20.0 }
    }

    #[test]
    fn new_wires_exactly_count_distinct_cells_and_is_deterministic() {
        let size = 8u32;
        let ls = (size * size) as usize;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 16 }]);
        let a = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        let b = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        assert_eq!(a.total_slots, 16);
        for i in 0..ls {
            let mut cells = Vec::new();
            a.for_wired(0, i, |_r, c| cells.push(c));
            assert_eq!(cells.len(), 16);
            assert!(cells.windows(2).all(|w| w[0] < w[1]), "ascending cell order");
        }
        assert_eq!(a.occ, b.occ, "deterministic occupancy");
        assert_eq!(a.codes, b.codes, "deterministic ±1 codes");
        assert_eq!(a.omega, b.omega, "deterministic omega");
        assert_eq!(a.b_off, b.b_off, "deterministic b_off");
    }

    #[test]
    fn omega_and_b_off_in_range() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let l = Layer::new(&cfg, 0.05, 0.9, 1.0, 3, 0, size);
        assert!(l.omega.iter().all(|&o| o >= 5.0 && o <= 10.0), "omega in [5,10]");
        assert!(l.b_off.iter().all(|&b| b >= 0.1 && b <= 1.0), "b_off in [0.1,1.0]");
    }

    #[test]
    fn weight_at_decodes_pm1_from_codes() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 8 }]);
        let l = Layer::new(&cfg, 0.05, 0.9, 1.0, 3, 0, size); // inhibitor_ratio 0 => all +1
        for s in 0..l.synapse_count() {
            assert!(matches!(l.weight_at(s), 1 | -1));
        }
    }

    #[test]
    fn decode_center_is_self() {
        let size = 8u32;
        let cfg = lc(vec![TopologyLevel { level: 1, radius: 2, count: 4 }]);
        let l = Layer::new(&cfg, 0.05, 0.9, 1.0, 7, 0, size);
        let src = crate::wave_resonate::synapse::local_of(3, 4, size);
        assert_eq!(l.decode(0, src, 12, size), src, "center cell (idx 12, span 5) maps to self");
    }

    #[test]
    fn pw_matches_formula() {
        let (omega, dt) = (10.0f32, 0.05f32);
        let want = (-1.0 + (1.0 - (dt * omega) * (dt * omega)).sqrt()) / dt;
        assert_eq!(pw(omega, dt), want);
    }
}
