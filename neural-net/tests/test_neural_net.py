import unittest

class TestNeuralNet(unittest.TestCase):
    def test_neural_net_initialization(self):
        from neural_net import NeuralNet
        from neuron import Neuron
        first_layer_of_neurons = [Neuron("n1"), Neuron("n2")]
        neural_net = NeuralNet(first_layer_of_neurons)
        self.assertEqual(neural_net.neurons, first_layer_of_neurons)

if __name__ == "__main__":
    unittest.main()