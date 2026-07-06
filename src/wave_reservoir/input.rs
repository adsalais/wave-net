use crate::wave_reservoir::hash::{key, map_range, mix, P_TARGET};
use crate::wave_reservoir::index::Dims;

#[derive(Clone, Debug)]
pub struct InputMap {
    pub channels: Vec<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub struct CurrentInjector {
    pub map: InputMap,
    pub input_scale: f32,
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

impl CurrentInjector {
    pub fn new(map: InputMap, input_scale: f32) -> CurrentInjector {
        CurrentInjector { map, input_scale }
    }

    /// Build a per-neuron drive vector from per-channel `values`.
    pub fn drive(&self, values: &[f32], n_total: usize) -> Vec<f32> {
        let mut d = vec![0.0f32; n_total];
        for (ch, neurons) in self.map.channels.iter().enumerate() {
            let v = values[ch] * self.input_scale;
            for &nrn in neurons {
                d[nrn as usize] += v;
            }
        }
        d
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

    #[test]
    fn drive_scales_and_places_current() {
        let n = (16 * 16 * 4) as usize;
        let map = InputMap { channels: vec![vec![0, 1], vec![2]] };
        let inj = CurrentInjector::new(map, 2.0);
        let d = inj.drive(&[1.0, 3.0], n);
        assert_eq!((d[0], d[1], d[2], d[3]), (2.0, 2.0, 6.0, 0.0));
    }
}
