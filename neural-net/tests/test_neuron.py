import unittest
from neuron import Neuron
from synapse import Synapse

class TestNeuron(unittest.TestCase):
    def test_create_neuron_non_empty_i(self):
        neuron = Neuron("test_neuron")
        self.assertEqual(neuron.id, "test_neuron")
        self.assertEqual(neuron.value, 0)
        self.assertEqual(neuron.synapses_in, [])
        self.assertEqual(neuron.synapses_out, [])
        
    def test_create_neuron_emtpy_id(self):
        neuron = Neuron("")
        self.assertEqual(neuron.id, "")
        self.assertEqual(neuron.value, 0)
        self.assertEqual(neuron.synapses_in, [])
        self.assertEqual(neuron.synapses_out, [])

    def test_neuron_to_synapse_out(self):
        neuron1 = Neuron("neuron1")
        synapse_from_1_to_2 = Synapse("neuron1_to_neuron2")
        neuron1.synapses_out.append(synapse_from_1_to_2)
        
        self.assertEqual(len(neuron1.synapses_out), 1)
        self.assertEqual(neuron1.synapses_out[0].to_neuron, synapse_from_1_to_2.to_neuron)
    

if __name__ == "__main__":
    unittest.main()