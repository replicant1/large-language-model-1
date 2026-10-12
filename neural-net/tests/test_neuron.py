import unittest
from neuron import Neuron, identity, sigmoid
from synapse import Synapse


class TestNeuron(unittest.TestCase):
    def test_create_neuron_non_empty_id(self):
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
        
    def test_synapse_in_to_neuron(self):
        neuron2 = Neuron("neuron2")
        synapse_from_1_to_2 = Synapse("neuron1_to_neuron2")
        neuron2.synapses_in.append(synapse_from_1_to_2)

        self.assertEqual(len(neuron2.synapses_in), 1)
        self.assertEqual(neuron2.synapses_in[0].from_neuron, synapse_from_1_to_2.from_neuron)

    def test_default_activation_is_identity(self):
        self.assertIs(identity, Neuron("n").activation)

    def test_forward_example_from_comment(self):
        a, b, out = Neuron("a"), Neuron("b"), Neuron("out")
        a.value, b.value = 2.0, 3.0
        Synapse("a_to_out", a, out, 0.5)
        Synapse("b_to_out", b, out, -1.0)
        self.assertEqual(-2.0, out.forward())
        self.assertEqual(-2.0, out.value)

    def test_forward_applies_the_activation(self):
        a, out = Neuron("a"), Neuron("out", sigmoid)
        a.value = 1.0
        Synapse("a_to_out", a, out, 0.5)
        self.assertAlmostEqual(1 / (1 + 2.718281828459045 ** -0.5), out.forward())

    def test_forward_with_no_inputs_is_activation_of_zero(self):
        self.assertEqual(0.0, Neuron("n").forward())
        self.assertEqual(0.5, Neuron("n", sigmoid).forward())


class TestSynapseConstructor(unittest.TestCase):
    def test_connects_and_sets_weight(self):
        a, b = Neuron("a"), Neuron("b")
        synapse = Synapse("a_to_b", a, b, 0.5)
        self.assertIs(a, synapse.from_neuron)
        self.assertIs(b, synapse.to_neuron)
        self.assertEqual(0.5, synapse.weight)
        self.assertEqual([synapse], a.synapses_out)
        self.assertEqual([synapse], b.synapses_in)

    def test_id_only_leaves_it_unconnected_with_weight_one(self):
        synapse = Synapse("s")
        self.assertIsNone(synapse.from_neuron)
        self.assertIsNone(synapse.to_neuron)
        self.assertEqual(1.0, synapse.weight)


class TestActivations(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(-1.5, identity(-1.5))

    def test_sigmoid_middle_and_ends(self):
        self.assertEqual(0.5, sigmoid(0.0))
        self.assertEqual(1.0, sigmoid(6.5))
        self.assertEqual(0.0, sigmoid(-6.5))
        self.assertAlmostEqual(1 / (1 + 2.718281828459045 ** -2.0), sigmoid(2.0))
    

if __name__ == "__main__":
    unittest.main()