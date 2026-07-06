use crate::wave_reservoir::hash::{key, map_range24, mix, P_TARGET};
use crate::wave_reservoir::index::Dims;
use crate::wave_reservoir::config::{IntConfig, IntLevel};

#[derive(Clone, Copy, Debug)]
pub struct IntSynapse {
    pub target: u32,
    /// `true` for an inhibitory synapse (delivers `-1`); `false` for excitatory (`+1`).
    /// Weights collapsed to binary {-1,+1}, so the sign is all that remains — one bool.
    pub inhibitory: bool,
}

/// A synapse addressed in per-layer coordinates: `(target_layer, local)` where
/// `local = ty*W + tx`. Used by the pipeline engine, whose state is stored per layer.
#[derive(Clone, Copy, Debug)]
pub struct LayeredSynapse {
    pub target_layer: u32,
    pub local: u32,
    pub inhibitory: bool,
}

/// Shared scatter-K core: for each synapse it derives the target coords `(tz, tx, ty)` and
/// the inhibitory flag from the hash, then hands them to `emit`. `scatter_into` (global `idx`)
/// and `scatter_layered` (per-layer) are thin emitters over this one hash, so their wiring is
/// identical by construction.
#[inline]
fn for_each_target(
    source: u32,
    seed: u64,
    topology: &[IntLevel],
    p_inh_q16: u32,
    dims: &Dims,
    mut emit: impl FnMut(u32, u32, u32, bool),
) {
    let (sx, sy, sz) = dims.coords(source);
    for entry in topology {
        let tz = sz as i32 + entry.level;
        if tz < 0 || tz >= dims.l as i32 {
            continue;
        }
        let span = 2 * entry.radius + 1;
        for k in 0..entry.count {
            // One hash per synapse (the per-spike hot path): dx from bits 63..40,
            // dy from bits 39..16, inhibitory from bits 15..0.
            let h = mix(key(seed, source, entry.level, k, P_TARGET));
            let dx = map_range24((h >> 40) as u32, span) as i32 - entry.radius as i32;
            let dy = map_range24(((h >> 16) as u32) & 0x00FF_FFFF, span) as i32 - entry.radius as i32;
            let tx = dims.wrap_x(sx, dx);
            let ty = dims.wrap_y(sy, dy);
            let inhibitory = ((h & 0xFFFF) as u32) < p_inh_q16;
            emit(tz as u32, tx, ty, inhibitory);
        }
    }
}

/// Integer scatter-K wiring, global `idx` targets. Each synapse carries an `inhibitory` flag
/// (probability `p_inh_q16 / 2^16`) — delivering `-1` when inhibitory, `+1` otherwise.
pub fn scatter_into(source: u32, cfg: &IntConfig, dims: &Dims, out: &mut Vec<IntSynapse>) {
    out.clear();
    let lc = &cfg.layers[dims.layer_of(source)];
    for_each_target(source, cfg.seed, &lc.topology, lc.p_inh_q16, dims, |tz, tx, ty, inh| {
        out.push(IntSynapse { target: dims.idx(tx, ty, tz), inhibitory: inh });
    });
}

/// Same wiring as `scatter_into`, but targets are emitted in per-layer coordinates
/// `(target_layer, local)` for the pipeline engine. `local = dims.idx(tx, ty, 0) = ty*W + tx`.
pub fn scatter_layered(
    source: u32,
    seed: u64,
    topology: &[IntLevel],
    p_inh_q16: u32,
    dims: &Dims,
    out: &mut Vec<LayeredSynapse>,
) {
    out.clear();
    for_each_target(source, seed, topology, p_inh_q16, dims, |tz, tx, ty, inh| {
        out.push(LayeredSynapse { target_layer: tz, local: dims.idx(tx, ty, 0), inhibitory: inh });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave_reservoir::index::Dims;
    use crate::wave_reservoir::config::IntConfig;

    #[test]
    fn flags_carry_both_excitatory_and_inhibitory() {
        // With p_inh ~15%, scanning a full layer must yield both flags.
        let cfg = IntConfig::demo();
        let dims = Dims::new(cfg.w, cfg.h, cfg.l);
        let mut out = Vec::new();
        let (mut exc, mut inh) = (false, false);
        for src in 0..cfg.n_total() as u32 {
            scatter_into(src, &cfg, &dims, &mut out);
            for s in &out {
                if s.inhibitory {
                    inh = true;
                } else {
                    exc = true;
                }
            }
        }
        assert!(exc && inh, "both excitatory and inhibitory synapses should appear");
    }

    #[test]
    fn layered_matches_global_targets() {
        let cfg = IntConfig::demo();
        let dims = Dims::new(cfg.w, cfg.h, cfg.l);
        let ls = (cfg.w * cfg.h) as usize;
        let src = dims.idx(8, 8, 1);
        let lc = &cfg.layers[1];
        let mut g = Vec::new();
        scatter_into(src, &cfg, &dims, &mut g);
        let mut ly = Vec::new();
        scatter_layered(src, cfg.seed, &lc.topology, lc.p_inh_q16, &dims, &mut ly);
        assert_eq!(g.len(), ly.len());
        for (a, b) in g.iter().zip(&ly) {
            assert_eq!(a.target as usize, b.target_layer as usize * ls + b.local as usize);
            assert_eq!(a.inhibitory, b.inhibitory);
        }
    }
}
