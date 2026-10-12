from neural_net_builder import NeuralNetBuilder
from neuron import Neuron
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

    def test_default_names_and_weights(self):
        neural_net = NeuralNetBuilder([2, 1]).build()
        self.assertEqual(["neuron_0_layer_0", "neuron_1_layer_0", "neuron_0_layer_1"], [neuron.id for neuron in neural_net.neurons])
        self.assertEqual(["layer_0", "layer_1"], [layer.name for layer in neural_net.layers])
        self.assertEqual([1.0, 1.0], [synapse.weight for synapse in neural_net.output_layer.neurons[0].synapses_in])
        self.assertIs(Neuron.identity, neural_net.neurons[0].activation)

    def test_layers_hold_their_neurons_in_order(self):
        neural_net = NeuralNetBuilder([3, 2, 1]).build()
        self.assertEqual([3, 2, 1], [layer.size() for layer in neural_net.layers])
        self.assertEqual(neural_net.neurons[3:5], neural_net.layers[1].neurons)

    def test_names_activations_and_initial_weight(self):
        neural_net = NeuralNetBuilder([2, 1], layer_names=["input", "output"],
                                      neuron_names_per_layer=[["the", "cat"], ["sat"]],
                                      activation_per_layer=[Neuron.identity, Neuron.sigmoid],
                                      initial_weight=0.25).build()
        self.assertEqual(["input:the", "input:cat"], [neuron.id for neuron in neural_net.layer("input").neurons])
        self.assertEqual("output:sat", neural_net.layer("output").neurons[0].id)
        self.assertIs(Neuron.sigmoid, neural_net.output_layer.neurons[0].activation)
        self.assertEqual([0.25, 0.25], [synapse.weight for synapse in neural_net.output_layer.neurons[0].synapses_in])

    def test_synapses_are_in_neuron_order_of_the_next_layer(self):
        neural_net = NeuralNetBuilder([2, 3]).build()
        for neuron in neural_net.input_layer.neurons:
            self.assertEqual(neural_net.output_layer.neurons, [synapse.to_neuron for synapse in neuron.synapses_out])
        for neuron in neural_net.output_layer.neurons:
            self.assertEqual(neural_net.input_layer.neurons, [synapse.from_neuron for synapse in neuron.synapses_in])

if __name__ == "__main__":
    unittest.main()