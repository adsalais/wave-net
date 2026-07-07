pub const P_TARGET: u64 = 1;
pub const P_THRESHOLD: u64 = 3; // 2 was a retired per-synapse attribute purpose; keep 3 stable

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_is_deterministic() {
        assert_eq!(mix(42), mix(42));
    }

    #[test]
    fn map_range_stays_in_bounds() {
        for bits in [0u32, 1, 12345, u32::MAX] {
            assert!(map_range(bits, 7) < 7);
        }
    }

    #[test]
    fn map_range24_stays_in_bounds_and_covers_span() {
        for bits in [0u32, 1, 12345, 0x00FF_FFFF] {
            assert!(map_range24(bits, 7) < 7);
        }
        assert_eq!(map_range24(0, 5), 0);
        assert_eq!(map_range24(0x00FF_FFFF, 5), 4);
    }
}
