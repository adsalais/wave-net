//! Minimal fixed-size bitset over `Vec<u64>` (std-only).

#[derive(Clone, Debug)]
pub struct BitSet {
    words: Vec<u64>,
    n_bits: usize,
}

impl BitSet {
    pub fn zeros(n_bits: usize) -> BitSet {
        BitSet { words: vec![0u64; (n_bits + 63) / 64], n_bits }
    }
    #[inline]
    pub fn set(&mut self, i: usize) {
        debug_assert!(i < self.n_bits);
        self.words[i >> 6] |= 1u64 << (i & 63);
    }
    /// Set (`v == true`) or clear (`v == false`) bit `i`.
    #[inline]
    pub fn put(&mut self, i: usize, v: bool) {
        debug_assert!(i < self.n_bits);
        let mask = 1u64 << (i & 63);
        if v {
            self.words[i >> 6] |= mask;
        } else {
            self.words[i >> 6] &= !mask;
        }
    }
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.n_bits);
        (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
    /// Set-bit offsets **relative to `start`** within `[start, start+len)`, ascending.
    pub fn iter_set_in(&self, start: usize, len: usize) -> impl Iterator<Item = usize> + '_ {
        (0..len).filter(move |&o| self.get(start + o))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_count_and_iter() {
        let mut b = BitSet::zeros(200);
        assert_eq!(b.count_ones(), 0);
        assert!(!b.get(5));
        for &i in &[5usize, 63, 64, 130, 199] {
            b.set(i);
        }
        assert!(b.get(64) && b.get(199) && !b.get(0));
        assert_eq!(b.count_ones(), 5);
        // iterate the neighborhood slice [64, 64+80): global 64 and 130 -> offsets 0 and 66.
        let got: Vec<usize> = b.iter_set_in(64, 80).collect();
        assert_eq!(got, vec![0, 66]);
    }

    #[test]
    fn put_sets_and_clears() {
        let mut b = BitSet::zeros(70);
        b.put(3, true);
        b.put(69, true);
        assert!(b.get(3) && b.get(69));
        b.put(3, false);
        assert!(!b.get(3) && b.get(69));
        assert_eq!(b.count_ones(), 1);
    }
}
