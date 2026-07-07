//! `layer` — a single layer of neurons within the wave-net.

use crate::wave_net::synapse::Synapse;

pub const TRESHOLD_BASE_RANGE: u16 = 128;

pub struct Layer {
    pub potential: Vec<u16>,

    // leak_a and leak_b
    pub leak: (u8, u8),

    // 1 means that that the potential needs to read 1 to fire, u16.max_value is the most refractory
    // initialisation: u16.max_value - random(0..TRESHOLD_BASE_RANGE)
    pub treshold: Vec<u16>,

    //the refractory cooldown
    pub cooldown: Vec<u8>,

    //after firing, set the cooldown to this value. drecreased by 1 after each wave, a neuron can fire at zero
    pub cooldown_base: u8,

    //the neurons that will need update (for lateral and backard layer topology), clear after consume to keep its capacity.
    pub update_buffer: Vec<Synapse>,

    pub size: u32, //the size of the layer, must be a power of two, the layer is a square of size*size
    pub topology: Vec<super::synapse::TopologyLevel>,

    //previously p_inh_q16?
    pub inibitor_synapse_ratio: u32,

    //please explain
    pub spread_log2: u8,

    //the listeners that will be notified of the layer spikes
    #[allow(clippy::type_complexity)]
    pub listeners: Vec<Option<Box<dyn Fn(usize, &[u32]) + Send + Sync>>>,
}
