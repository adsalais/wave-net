//process a wave tick:
// access layer
// 1- update neurons potential from Layer::update_buffer
// 2- parse neurons
//      - if neuron is ready to fire
//          proceduraly generate its synapse
//             for each layer from the topology add the synapse to the update_buffer
//          update the refractory to its cooldown_base
//          feed a spike_buffer
//      - if not, decrease the refractory period if not zero
//      - leak the potential
// 3- notify the spikes to the listeners using the spike_buffer
//pub fn process_wave(layer: &mut Layer, the otehr layers for topology) {}
