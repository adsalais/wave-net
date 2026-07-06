//! Input/bit-stream harness: a deterministic fair-bit source and a bipolar bottom-layer input
//! encoder. Shared by the calibrator, the depth diagnostic, and task examples so they drive the
//! reservoir identically. Lifted from the `params_study` example.

use crate::wave_reservoir::hash::mix;
use crate::wave_reservoir::index::Dims;
use crate::wave_reservoir::input::InputMap;

/// Deterministic fair coin: `fair_bit(seed, t)` is a reproducible pseudo-random bit for step `t`.
pub fn fair_bit(seed: u64, t: u64) -> u8 {
    (mix(seed ^ t.wrapping_mul(0xD1B5_4A32)) & 1) as u8
}

/// A set of bottom-layer input sites driven bipolar (`+level` for bit 1, `-level` for bit 0).
#[derive(Clone, Debug)]
pub struct BipolarInput {
    pub sites: Vec<u32>,
    pub level: i16,
}

impl BipolarInput {
    /// Scatter `per_channel` input sites across the bottom layer (layer 0), driven at `±level`.
    pub fn scatter_bottom(dims: &Dims, seed: u64, per_channel: usize, level: i16) -> BipolarInput {
        let sites = InputMap::scatter_bottom(dims, seed, 1, per_channel).channels[0].clone();
        BipolarInput { sites, level }
    }

    /// Add this bit's bipolar drive into `buf` (assumed already zeroed — `run_stream` zeroes it,
    /// the sequential `wave()` caller must). Each site gets `+level` (bit 1) or `-level` (bit 0).
    pub fn drive_into(&self, buf: &mut [i16], bit: u8) {
        let v = if bit == 1 { self.level } else { -self.level };
        for &s in &self.sites {
            buf[s as usize] += v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fair_bit_is_deterministic_and_balanced() {
        assert_eq!(fair_bit(42, 7), fair_bit(42, 7));
        let ones = (0..2000u64).filter(|&t| fair_bit(0xABCD, t) == 1).count();
        assert!((900..=1100).contains(&ones), "fair bit unbalanced: {ones}/2000 ones");
    }

    #[test]
    fn drive_into_places_bipolar_current_at_sites() {
        let inp = BipolarInput { sites: vec![0, 2], level: 4 };
        let mut buf = vec![0i16; 5];
        inp.drive_into(&mut buf, 1);
        assert_eq!(buf, vec![4, 0, 4, 0, 0]);
        let mut buf = vec![0i16; 5];
        inp.drive_into(&mut buf, 0);
        assert_eq!(buf, vec![-4, 0, -4, 0, 0]);
    }

    #[test]
    fn scatter_bottom_sites_are_all_in_layer_zero() {
        let dims = Dims::new(16, 16, 4);
        let inp = BipolarInput::scatter_bottom(&dims, 99, 24, 4);
        assert_eq!(inp.sites.len(), 24);
        for &s in &inp.sites {
            assert!((s as usize) < 16 * 16, "site {s} not in layer 0");
        }
    }
}
