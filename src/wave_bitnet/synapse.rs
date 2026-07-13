//! `synapse` — hash helpers (copied verbatim from `wave_net::synapse`) plus the two routines that
//! replace procedural target generation: `decode_cell` (neighborhood cell → target local, arithmetic)
//! and `sample_distinct_cells` (startup fill of `count` distinct cells via partial Fisher-Yates).

#[derive(Clone, Debug, PartialEq)]
pub struct TopologyLevel {
    pub level: i32,
    pub radius: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Synapse {
    // neuron that will receive the input
    pub target: u32,
    /// Signed weight delivered to `target` — the source layer's stored plastic weight (±1/0 in this
    /// fork). Carried on the `Synapse` because the *target* layer folds it in at drain time.
    pub weight: i16,
}

/// Hash purpose tags (keep stable — they seed distinct hash streams).
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

/// BLAKE3 mixer (test-only, `--features strong_hash`).
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

/// Map 24 random bits to `0..span` (multiply-shift; `span` must be < 2^24).
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

/// Number of cells in a radius-`r` neighborhood: `(2r+1)²`.
#[inline]
pub fn neigh_size(radius: u32) -> usize {
    let span = (2 * radius + 1) as usize;
    span * span
}

/// Target local index for neighborhood `cell` of a source at `src_local`. Cell layout is row-major
/// over the `(2r+1)×(2r+1)` window centered on the source; pure arithmetic, no hash.
pub fn decode_cell(cell: usize, src_local: u32, radius: u32, size: u32) -> u32 {
    let (sx, sy) = xy_of(src_local, size);
    let span = 2 * radius + 1;
    let dx = (cell as u32 % span) as i32 - radius as i32;
    let dy = (cell as u32 / span) as i32 - radius as i32;
    local_of(wrap(sx, dx, size), wrap(sy, dy, size), size)
}

/// `count` DISTINCT cell indices in `0..neigh_size(radius)`, via a partial Fisher-Yates shuffle of the
/// cell indices seeded by the hash stream (one draw per swap). Deterministic; `count` must be
/// `<= neigh_size(radius)` (guaranteed by `Config::validate`).
pub fn sample_distinct_cells(seed: u64, source_global: u32, level: i32, radius: u32, count: u32) -> Vec<u32> {
    let n = neigh_size(radius);
    debug_assert!(count as usize <= n);
    let mut idx: Vec<u32> = (0..n as u32).collect();
    for k in 0..(count as usize) {
        let h = mix(key(seed, source_global, level, k as u32, P_TARGET));
        // pick j in [k, n) without modulo bias, swap into position k
        let j = k + map_range((h >> 32) as u32, (n - k) as u32) as usize;
        idx.swap(k, j);
    }
    idx.truncate(count as usize);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_center_is_self_and_corners_wrap() {
        let size = 8u32;
        let r = 2u32;
        let span = 2 * r + 1; // 5, so N = 25, center cell index = 12 (dx=dy=0)
        let src = local_of(3, 4, size);
        assert_eq!(decode_cell(12, src, r, size), src, "center cell maps to self");
        // cell 0 -> dx=-2, dy=-2 -> (3-2, 4-2) = (1, 2)
        assert_eq!(decode_cell(0, src, r, size), local_of(1, 2, size));
        // last cell (span*span-1 = 24) -> dx=+2, dy=+2 -> (5, 6)
        assert_eq!(decode_cell((span * span - 1) as usize, src, r, size), local_of(5, 6, size));
    }

    #[test]
    fn sample_is_distinct_bounded_and_deterministic() {
        let (seed, sg, level, r, count) = (0xABCDu64, 700u32, 1i32, 4u32, 48u32);
        let a = sample_distinct_cells(seed, sg, level, r, count);
        let b = sample_distinct_cells(seed, sg, level, r, count);
        assert_eq!(a, b, "deterministic");
        assert_eq!(a.len(), count as usize, "exactly count cells");
        let n = neigh_size(r);
        assert!(a.iter().all(|&c| (c as usize) < n), "all in range");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), count as usize, "all distinct");
    }
}
