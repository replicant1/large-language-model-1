import unittest
import math
from neuron import sigmoid
from skip_gram_network import SkipGramNetwork


# Builds a network with one token per embedding, and sets each token's embedding and context vector
def network_with(embeddings: list[list[float]], context_vectors: list[list[float]]) -> SkipGramNetwork:
    network = SkipGramNetwork([f"t{i}" for i in range(len(embeddings))], len(embeddings[0]))
    for token, (embedding, context_vector) in enumerate(zip(embeddings, context_vectors)):
        network.set_embedding(token, embedding)
        network.set_context_vector(token, context_vector)
    return network


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestStructure(unittest.TestCase):
    def test_layer_sizes(self):
        network = SkipGramNetwork(["the", "cat", "sat"], 4)
        self.assertEqual(["input", "hidden", "output"], [layer.name for layer in network.net.layers])
        self.assertEqual([3, 4, 3], [layer.size() for layer in network.net.layers])

    def test_neurons_are_named_after_tokens(self):
        network = SkipGramNetwork(["the", "cat"], 2)
        self.assertEqual(["input:the", "input:cat"], [neuron.id for neuron in network.input_layer.neurons])
        self.assertEqual(["output:the", "output:cat"], [neuron.id for neuron in network.output_layer.neurons])

    def test_fully_connected(self):
        network = SkipGramNetwork(["the", "cat", "sat"], 4)
        for neuron in network.input_layer.neurons:
            self.assertEqual(network.hidden_layer.neurons, [synapse.to_neuron for synapse in neuron.synapses_out])
        for neuron in network.output_layer.neurons:
            self.assertEqual(network.hidden_layer.neurons, [synapse.from_neuron for synapse in neuron.synapses_in])

    def test_activations(self):
        network = SkipGramNetwork(["the", "cat"], 2)
        self.assertEqual(0.25, network.hidden_layer.neurons[0].activation(0.25))
        self.assertEqual(sigmoid(0.25), network.output_layer.neurons[0].activation(0.25))

    def test_embedding_is_the_weights_out_of_the_input_neuron(self):
        network = SkipGramNetwork(["the", "cat"], 3)
        network.set_embedding(1, [0.1, 0.2, 0.3])
        self.assertEqual([0.1, 0.2, 0.3], [synapse.weight for synapse in network.input_layer.neurons[1].synapses_out])
        self.assertEqual([0.1, 0.2, 0.3], network.embedding(1))
        self.assertEqual([0.0, 0.0, 0.0], network.embedding(0))

    def test_context_vector_is_the_weights_into_the_output_neuron(self):
        network = SkipGramNetwork(["the", "cat"], 3)
        network.set_context_vector(0, [0.4, 0.5, 0.6])
        self.assertEqual([0.4, 0.5, 0.6], [synapse.weight for synapse in network.output_layer.neurons[0].synapses_in])
        self.assertEqual([0.4, 0.5, 0.6], network.context_vector(0))
        self.assertEqual([0.0, 0.0, 0.0], network.context_vector(1))


class TestForward(unittest.TestCase):
    def test_score_is_sigmoid_of_embedding_dot_context_vector(self):
        network = network_with([[0.5, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.2, -0.4]])
        self.assertAlmostEqual(sigmoid(0.5 * 0.2 + 1.0 * -0.4), network.forward(0, 1))

    def test_hidden_values_are_the_target_embedding(self):
        network = network_with([[0.5, 1.0], [0.3, 0.7]], [[0.0, 0.0], [0.0, 0.0]])
        network.forward(1, 0)
        self.assertEqual([0.3, 0.7], [neuron.value for neuron in network.hidden_layer.neurons])


class TestInitializeWeights(unittest.TestCase):
    def test_draws_alternate_between_embedding_and_context_vector(self):
        draws = iter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        network = SkipGramNetwork(["a", "b"], 2)
        network.initialize_weights(lambda: next(draws))
        scale = 0.5 / 2
        # token 0: embedding gets draws 1 and 3, context vector draws 2 and 4; then token 1
        self.assertEqual([(0.1 - 0.5) * scale, (0.3 - 0.5) * scale], network.embedding(0))
        self.assertEqual([(0.2 - 0.5) * scale, (0.4 - 0.5) * scale], network.context_vector(0))
        self.assertEqual([(0.5 - 0.5) * scale, (0.7 - 0.5) * scale], network.embedding(1))
        self.assertEqual([(0.6 - 0.5) * scale, (0.8 - 0.5) * scale], network.context_vector(1))


class TestTrainPair(unittest.TestCase):
    def test_positive_example_from_comment(self):
        network = network_with([[0.5, 0.5]], [[0.5, -0.5]])
        loss = network.train_pair(0, 0, 1, 0.1)
        for expected, actual in zip([0.525, 0.475], network.embedding(0)):
            self.assertAlmostEqual(expected, actual)
        for expected, actual in zip([0.525, -0.475], network.context_vector(0)):
            self.assertAlmostEqual(expected, actual)
        self.assertAlmostEqual(math.log(2), loss)

    def test_positive_pair_score_goes_up(self):
        # target is token 0's embedding, other is token 1's context vector
        network = network_with([[0.1, 0.2], [0.0, 0.0]], [[0.0, 0.0], [0.3, -0.1]])
        before = dot(network.embedding(0), network.context_vector(1))
        network.train_pair(0, 1, 1, 0.5)
        self.assertGreater(dot(network.embedding(0), network.context_vector(1)), before)

    def test_negative_pair_score_goes_down(self):
        network = network_with([[0.1, 0.2], [0.0, 0.0]], [[0.0, 0.0], [0.3, -0.1]])
        before = dot(network.embedding(0), network.context_vector(1))
        network.train_pair(0, 1, 0, 0.5)
        self.assertLess(dot(network.embedding(0), network.context_vector(1)), before)

    def test_only_the_two_vectors_change(self):
        network = network_with([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])
        network.train_pair(0, 2, 1, 0.5)
        self.assertEqual([[0.3, 0.4], [0.5, 0.6]], [network.embedding(1), network.embedding(2)])
        self.assertEqual([[0.6, 0.5], [0.4, 0.3]], [network.context_vector(0), network.context_vector(1)])

    def test_both_vectors_use_values_from_before_the_update(self):
        # Both vectors start at [1.0], so dot = 1.0. lr is chosen so that grad = 0.5.
        # Each vector should become 1.0 + 0.5 * 1.0 = 1.5. If the context vector were updated
        # with the already-updated embedding instead, it would become 1.0 + 0.5 * 1.5 = 1.75.
        network = network_with([[1.0]], [[1.0]])
        lr = 0.5 / (1 - sigmoid(1.0))
        network.train_pair(0, 0, 1, lr)
        self.assertAlmostEqual(1.5, network.embedding(0)[0])
        self.assertAlmostEqual(1.5, network.context_vector(0)[0])

    def test_loss_for_negative_pair(self):
        # dot = 0 -> score 0.5 -> loss = -log(1 - 0.5)
        network = network_with([[0.0, 0.0]], [[0.0, 0.0]])
        self.assertAlmostEqual(math.log(2), network.train_pair(0, 0, 0, 0.1))


if __name__ == "__main__":
    unittest.main()
