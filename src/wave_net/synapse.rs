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
    /// `true` for an inhibitory synapse (delivers `-1`); `false` for excitatory (`+1`).
    /// Weights collapsed to binary {-1,+1}, so the sign is all that remains — one bool.
    pub inhibitory: bool,
}

/// A firing neuron's outgoing synapses for one topology entry, tagged with the entry's
/// **relative** layer offset. The `Network` resolves the absolute target layer.
#[derive(Debug)]
pub struct SynapseGroup {
    pub level: i32,
    pub synapses: Vec<Synapse>,
}

/// Hash purpose tags (keep stable — they seed distinct hash streams).
pub const P_TARGET: u64 = 1;
pub const P_THRESHOLD: u64 = 3;

#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

#[cfg(test)]
mod tests {
    use super::*;

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
