#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub w: u32,
    pub h: u32,
    pub l: u32,
    w_log2: u32,
    wh_log2: u32,
}

impl Dims {
    pub fn new(w: u32, h: u32, l: u32) -> Dims {
        debug_assert!(w.is_power_of_two() && h.is_power_of_two());
        // w, h are powers of two, so all index math is shifts/masks rather than div/mod.
        // The div/mod in `coords` was a per-spike cost on the scatter path.
        Dims { w, h, l, w_log2: w.trailing_zeros(), wh_log2: w.trailing_zeros() + h.trailing_zeros() }
    }

    #[inline]
    pub fn layer_size(&self) -> u32 {
        1 << self.wh_log2
    }

    #[inline]
    pub fn idx(&self, x: u32, y: u32, z: u32) -> u32 {
        (z << self.wh_log2) | (y << self.w_log2) | x
    }

    #[inline]
    pub fn coords(&self, idx: u32) -> (u32, u32, u32) {
        let z = idx >> self.wh_log2;
        let rem = idx & (self.layer_size() - 1);
        (rem & (self.w - 1), rem >> self.w_log2, z)
    }

    #[inline]
    pub fn layer_of(&self, idx: u32) -> usize {
        (idx >> self.wh_log2) as usize
    }

    #[inline]
    pub fn wrap_x(&self, base: u32, off: i32) -> u32 {
        ((base as i32 + off) as u32) & (self.w - 1)
    }

    #[inline]
    pub fn wrap_y(&self, base: u32, off: i32) -> u32 {
        ((base as i32 + off) as u32) & (self.h - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx_coords_roundtrip() {
        let d = Dims::new(8, 4, 3);
        for z in 0..3 {
            for y in 0..4 {
                for x in 0..8 {
                    let i = d.idx(x, y, z);
                    assert_eq!(d.coords(i), (x, y, z));
                    assert_eq!(d.layer_of(i), z as usize);
                }
            }
        }
    }

    #[test]
    fn wrap_is_toroidal() {
        let d = Dims::new(8, 8, 1);
        assert_eq!(d.wrap_x(0, -1), 7);
        assert_eq!(d.wrap_x(7, 1), 0);
        assert_eq!(d.wrap_y(0, -3), 5);
    }
}
