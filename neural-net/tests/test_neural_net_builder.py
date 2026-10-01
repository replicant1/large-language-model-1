from neural_net_builder import NeuralNetBuilder
import unittest

class TestNeuralNetBuilder(unittest.TestCase):
    def test_create_neural_net_builder_with_three_layers(self):
        builder = NeuralNetBuilder(3)
        self.assertIsInstance(builder, NeuralNetBuilder)
        
    def test_create_neural_net_builder_with_no_layers(self):
        builder = NeuralNetBuilder(0)
        self.assertIsInstance(builder, NeuralNetBuilder)
        
    def test_build_neural_net_with_three_layers(self):
        builder = NeuralNetBuilder([3, 2, 1])
        neural_net = builder.build()
        self.assertIsNotNone(neural_net)
        
        # Test has the correct number of neurons
        self.assertEqual(len(neural_net.neurons), 6)
        
        # Test the synaptic connections for layer 0 to layer 1
        layer_0_neurons = neural_net.neurons[:3]
        layer_1_neurons = neural_net.neurons[3:5]
        for from_neuron in layer_0_neurons:
            for to_neuron in layer_1_neurons:
                self.assertTrue(any(synapse for synapse in from_neuron.synapses_out if synapse in to_neuron.synapses_in))

        # Test the synaptic connections for layer 1 to layer 2
        layer_1_neurons = neural_net.neurons[3:5]
        layer_2_neurons = neural_net.neurons[5:6]
        for from_neuron in layer_1_neurons:
            for to_neuron in layer_2_neurons:
                self.assertTrue(any(synapse for synapse in from_neuron.synapses_out if synapse in to_neuron.synapses_in))

    def test_built_synapses_know_their_endpoints(self):
        builder = NeuralNetBuilder([3, 2, 1])
        neural_net = builder.build()

        # Every synapse must point back at the neurons it connects
        for neuron in neural_net.neurons:
            for synapse in neuron.synapses_out:
                self.assertIs(synapse.from_neuron, neuron)
            for synapse in neuron.synapses_in:
                self.assertIs(synapse.to_neuron, neuron)

        # Each neuron in layer n has one synapse to each neuron in layer n+1, with no duplicates
        layer_0_neurons = neural_net.neurons[:3]
        layer_1_neurons = neural_net.neurons[3:5]
        layer_2_neurons = neural_net.neurons[5:6]
        for neuron in layer_0_neurons:
            self.assertEqual(len(neuron.synapses_out), 2)
        for neuron in layer_1_neurons:
            self.assertEqual(len(neuron.synapses_in), 3)
            self.assertEqual(len(neuron.synapses_out), 1)
        for neuron in layer_2_neurons:
            self.assertEqual(len(neuron.synapses_in), 2)

if __name__ == "__main__":
    unittest.main()