//! `frontier` — the per-layer active-set worklist. A `Vec` gives ordered, cache-friendly iteration; a
//! 1-bit-per-neuron `mark` bitset makes insertion a test-and-set so no neuron is ever queued twice
//! (two firers hitting one target, or a target that is also a carryover). Cleared by walking `list`
//! (O(activity)), never by zeroing size². This is exactly the GPU unique-frontier-append primitive.

pub struct Frontier {
    pub list: Vec<u32>,
    pub mark: Vec<u64>, // ceil(ls / 64) words
}

impl Frontier {
    pub fn new(ls: usize) -> Frontier {
        Frontier { list: Vec::new(), mark: vec![0u64; (ls + 63) / 64] }
    }

    /// Test-and-set insert. Returns true iff `t` was newly added (was not already queued).
    #[inline]
    pub fn push(&mut self, t: u32) -> bool {
        let w = (t >> 6) as usize;
        let bit = 1u64 << (t & 63);
        if self.mark[w] & bit == 0 {
            self.mark[w] |= bit;
            self.list.push(t);
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn contains(&self, t: u32) -> bool {
        self.mark[(t >> 6) as usize] & (1u64 << (t & 63)) != 0
    }

    /// Empty the worklist and reset its marks by walking `list` (O(activity)).
    #[inline]
    pub fn clear(&mut self) {
        for &t in &self.list {
            self.mark[(t >> 6) as usize] &= !(1u64 << (t & 63));
        }
        self.list.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.list.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_dedups() {
        let mut f = Frontier::new(64);
        assert!(f.push(5));
        assert!(!f.push(5), "second push of 5 is a no-op");
        assert!(f.push(9));
        assert_eq!(f.list, vec![5, 9], "each neuron appears once, in insertion order");
        assert!(f.contains(5) && f.contains(9) && !f.contains(7));
    }

    #[test]
    fn clear_empties_list_and_resets_marks() {
        let mut f = Frontier::new(64);
        f.push(1);
        f.push(2);
        f.push(63);
        f.clear();
        assert!(f.is_empty());
        assert!(!f.contains(1) && !f.contains(2) && !f.contains(63));
        // reusable after clear
        assert!(f.push(1));
        assert_eq!(f.list, vec![1]);
    }

    #[test]
    fn handles_bit_boundaries() {
        let mut f = Frontier::new(128);
        for t in [0u32, 63, 64, 127] {
            assert!(f.push(t));
        }
        for t in [0u32, 63, 64, 127] {
            assert!(f.contains(t));
        }
        assert_eq!(f.len(), 4);
    }
}
