//! `network` — the top-level wave-net: owns layers and drives them.

use std::sync::Mutex;

use super::neurons::Layer;

// main role it to initialize the network an calibrate it
// then be able to and feed the input and orchestrate the wave execution
// input is a Vec<u32> u32 is the adress of the level 0 neuron that will be set to their fire_state before a wave.
//      it reset the cooldown of those neurons
pub struct Network {
    pub seed: u64,
    pub layers: Vec<Mutex<Layer>>,
}
