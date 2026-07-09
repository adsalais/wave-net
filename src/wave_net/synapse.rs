#[derive(Clone, Debug)]
pub struct TopologyLevel {
    pub level: i32,
    pub radius: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Synapse {
    // neuron that will receive the input
    pub target: u32,
    /// Signed weight delivered to `target` — the source layer's stored plastic weight, looked up from
    /// `out_weights` in `generate_into` (init `±1` sign from `inhibitor_ratio`, trained into the full
    /// int8 range via the `f32` shadow). Carried on the `Synapse` because the *target* layer folds it
    /// in at drain time, but the weight is owned by the *source* layer.
    pub weight: i16,
}

/// A firing neuron's outgoing synapses for one topology entry, tagged with the entry's
/// **relative** layer offset. The `Network` resolves the absolute target layer.
#[derive(Debug)]
pub struct SynapseGroup {
    pub level: i32,
    pub synapses: Vec<Synapse>,
}

/// Hash purpose tags (keep stable — they seed distinct hash streams).
/// (`wave_state_machine` also defines `P_WEIGHT` for its fire-time magnitude draw; `wave_net`
/// stores weights instead of hashing them, so it has no `P_WEIGHT`.)
pub const P_TARGET: u64 = 1;
pub const P_THRESHOLD: u64 = 3;
pub const P_INPUT: u64 = 5;

/// splitmix64 finalizer — the default integer mixer (dependency-free, deterministic).
#[cfg(not(feature = "strong_hash"))]
#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// BLAKE3 mixer (test-only, `--features strong_hash`) — rules out procedural-hash quality as a source of
/// seed-fragility by substituting a cryptographic hash at the single choke point every stream flows through.
#[cfg(feature = "strong_hash")]
#[inline]
pub fn mix(z: u64) -> u64 {
    let h = blake3::hash(&z.to_le_bytes());
    u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Pack coordinates + a purpose tag into a 64-bit key for the mixer.
#[inline]
pub fn key(seed: u64, idx: u32, dz: i32, slot: u32, purpose: u64) -> u64 {
    let mut k = seed;
    k = k.wrapping_mul(GOLDEN).wrapping_add(idx as u64);
    k = k.wrapping_mul(GOLDEN).wrapping_add((dz + 8) as u64);
    k = k.wrapping_mul(GOLDEN).wrapping_add(slot as u64);
    k = k.wrapping_mul(GOLDEN).wrapping_add(purpose);
    k
}

/// Map 32 random bits to `0..span` with no modulo bias (Lemire multiply-shift).
#[inline]
pub fn map_range(bits: u32, span: u32) -> u32 {
    (((bits as u64) * (span as u64)) >> 32) as u32
}

/// Map 24 random bits to `0..span` (multiply-shift; `span` must be < 2^24). Lets one 64-bit
/// hash feed dx (24 bits) + dy (24 bits) + a 16-bit attribute on the per-synapse hot path.
#[inline]
pub fn map_range24(bits: u32, span: u32) -> u32 {
    (((bits as u64) * (span as u64)) >> 24) as u32
}

/// (x, y) -> local index in a `size`-wide square layer (`size` is a power of two).
#[inline]
pub fn local_of(x: u32, y: u32, size: u32) -> u32 {
    (y << size.trailing_zeros()) | x
}

/// local index -> (x, y).
#[inline]
pub fn xy_of(local: u32, size: u32) -> (u32, u32) {
    (local & (size - 1), local >> size.trailing_zeros())
}

/// Toroidal shift of one coordinate by `off`, wrapped into `0..size`.
#[inline]
pub fn wrap(base: u32, off: i32, size: u32) -> u32 {
    ((base as i32 + off) as u32) & (size - 1)
}

/// Append one firing neuron's synapses into `groups` (one per topology entry, same order).
/// Emits **relative** levels only; the caller (Network) resolves absolute target layers.
/// Contract: `groups.len() == topology.len()` and `groups[i].level == topology[i].level`.
/// Appends (does not clear), so a whole layer's firers aggregate into one `groups` set.
pub fn generate_into(
    seed: u64,
    source_global: u32,
    src_local: u32,
    size: u32,
    topology: &[TopologyLevel],
    weights: &[i8],
    total_slots: usize,
    groups: &mut [SynapseGroup],
) {
    let (sx, sy) = xy_of(src_local, size);
    let mut slot = 0usize;
    for (entry, group) in topology.iter().zip(groups.iter_mut()) {
        let span = 2 * entry.radius + 1;
        for k in 0..entry.count {
            let h = mix(key(seed, source_global, entry.level, k, P_TARGET));
            let dx = map_range24((h >> 40) as u32, span) as i32 - entry.radius as i32;
            let dy = map_range24(((h >> 16) as u32) & 0x00FF_FFFF, span) as i32 - entry.radius as i32;
            let tx = wrap(sx, dx, size);
            let ty = wrap(sy, dy, size);
            let w = weights[src_local as usize * total_slots + slot] as i16;
            group.synapses.push(Synapse { target: local_of(tx, ty, size), weight: w });
            slot += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topo() -> Vec<TopologyLevel> {
        vec![
            TopologyLevel { level: 1, radius: 2, count: 6 },
            TopologyLevel { level: -1, radius: 0, count: 1 },
        ]
    }

    fn empty_groups(t: &[TopologyLevel]) -> Vec<SynapseGroup> {
        t.iter().map(|e| SynapseGroup { level: e.level, synapses: Vec::new() }).collect()
    }

    /// All-+1 stored weights for a size-8 layer, sized to index any src_local·total_slots + slot.
    fn ones(t: &[TopologyLevel]) -> (Vec<i8>, usize) {
        let tot: usize = t.iter().map(|e| e.count as usize).sum();
        (vec![1i8; 64 * tot], tot)
    }

    #[test]
    fn generate_counts_per_level() {
        let t = topo();
        let (w, tot) = ones(&t);
        let mut g = empty_groups(&t);
        generate_into(42, 0, 0, 8, &t, &w, tot, &mut g);
        assert_eq!(g[0].synapses.len(), 6);
        assert_eq!(g[1].synapses.len(), 1);
        // radius 0 targets the source cell itself
        assert_eq!(g[1].synapses[0].target, local_of(0, 0, 8));
    }

    #[test]
    fn generate_targets_within_radius() {
        let t = topo();
        let (w, tot) = ones(&t);
        let mut g = empty_groups(&t);
        let (sx, sy) = (3u32, 5u32);
        generate_into(7, 100, local_of(sx, sy, 8), 8, &t, &w, tot, &mut g);
        for s in &g[0].synapses {
            let (tx, ty) = xy_of(s.target, 8);
            let dx = ((tx + 8 - sx) & 7).min((sx + 8 - tx) & 7);
            let dy = ((ty + 8 - sy) & 7).min((sy + 8 - ty) & 7);
            assert!(dx <= 2 && dy <= 2, "target ({tx},{ty}) out of radius from ({sx},{sy})");
        }
    }

    #[test]
    fn generate_is_deterministic_and_appends() {
        let t = topo();
        let (w, tot) = ones(&t);
        let mut a = empty_groups(&t);
        let mut b = empty_groups(&t);
        generate_into(1, 9, 9, 8, &t, &w, tot, &mut a);
        generate_into(1, 9, 9, 8, &t, &w, tot, &mut b);
        assert_eq!(a[0].synapses.len(), b[0].synapses.len());
        for (x, y) in a[0].synapses.iter().zip(&b[0].synapses) {
            assert_eq!((x.target, x.weight), (y.target, y.weight));
        }
        // second call appends (aggregation across firers)
        generate_into(1, 9, 9, 8, &t, &w, tot, &mut a);
        assert_eq!(a[0].synapses.len(), 12);
    }

    #[test]
    fn index_roundtrip() {
        let size = 8;
        for y in 0..size {
            for x in 0..size {
                let l = local_of(x, y, size);
                assert_eq!(xy_of(l, size), (x, y));
            }
        }
    }

    #[test]
    fn wrap_is_toroidal() {
        assert_eq!(wrap(0, -1, 8), 7);
        assert_eq!(wrap(7, 1, 8), 0);
        assert_eq!(wrap(0, -3, 8), 5);
    }
}
