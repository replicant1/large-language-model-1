import unittest

class TestNeuralNet(unittest.TestCase):
    def test_neural_net_initialization(self):
        from neural_net import NeuralNet
        from neuron import Neuron
        first_layer_of_neurons = [Neuron("n1"), Neuron("n2")]
        neural_net = NeuralNet(first_layer_of_neurons)
        self.assertEqual(neural_net.neurons, first_layer_of_neurons)
        self.assertEqual([], neural_net.layers)

    def test_layers_by_position_and_name(self):
        from neural_net import NeuralNet
        from neuron import Neuron
        from layer import Layer
        input_layer = Layer("input", [Neuron("i0"), Neuron("i1")])
        output_layer = Layer("output", [Neuron("o0")])
        neural_net = NeuralNet(input_layer.neurons + output_layer.neurons, [input_layer, output_layer])
        self.assertIs(input_layer, neural_net.input_layer)
        self.assertIs(output_layer, neural_net.output_layer)
        self.assertIs(output_layer, neural_net.layer("output"))
        with self.assertRaises(KeyError):
            neural_net.layer("hidden")

if __name__ == "__main__":
    unittest.main()