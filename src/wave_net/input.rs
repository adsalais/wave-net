use crate::wave_net::hash::{key, map_range, mix, P_TARGET};
use crate::wave_net::index::Dims;

#[derive(Clone, Debug)]
pub struct InputMap {
    pub channels: Vec<Vec<u32>>,
}

impl InputMap {
    /// Deterministically scatter `num_channels` channels across the bottom layer,
    /// `per_channel` neurons each (indices in `0 .. w*h`, i.e. layer 0).
    pub fn scatter_bottom(dims: &Dims, seed: u64, num_channels: usize, per_channel: usize) -> InputMap {
        let layer_size = dims.layer_size();
        let mut channels = Vec::with_capacity(num_channels);
        for c in 0..num_channels {
            let mut neurons = Vec::with_capacity(per_channel);
            for s in 0..per_channel {
                let h = mix(key(seed, c as u32, 0, s as u32, P_TARGET));
                neurons.push(map_range(h as u32, layer_size)); // layer 0 => idx == cell
            }
            channels.push(neurons);
        }
        InputMap { channels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_targets_bottom_layer() {
        let dims = Dims::new(16, 16, 4);
        let map = InputMap::scatter_bottom(&dims, 99, 2, 5);
        assert_eq!(map.channels.len(), 2);
        for ch in &map.channels {
            assert_eq!(ch.len(), 5);
            for &nrn in ch {
                assert_eq!(dims.layer_of(nrn), 0, "input neurons must be in layer 0");
            }
        }
    }

}
