import unittest
import math
from neuron import Neuron
from embed import Embed
from random_numbers import Mulberry32
from skip_gram_network import SkipGramNetwork


class NetworkTestCase(unittest.TestCase):
    # Builds a network with one token per embedding, and sets each token's embedding and context vector
    @staticmethod
    def network_with(embeddings: list[list[float]], context_vectors: list[list[float]]) -> SkipGramNetwork:
        network = SkipGramNetwork([f"t{i}" for i in range(len(embeddings))], len(embeddings[0]))
        for token, (embedding, context_vector) in enumerate(zip(embeddings, context_vectors)):
            network.set_embedding(token, embedding)
            network.set_context_vector(token, context_vector)
        return network



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
        self.assertEqual(Neuron.sigmoid(0.25), network.output_layer.neurons[0].activation(0.25))

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


class TestForward(NetworkTestCase):
    def test_score_is_sigmoid_of_embedding_dot_context_vector(self):
        network = self.network_with([[0.5, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.2, -0.4]])
        self.assertAlmostEqual(Neuron.sigmoid(0.5 * 0.2 + 1.0 * -0.4), network.forward(0, 1))

    def test_hidden_values_are_the_target_embedding(self):
        network = self.network_with([[0.5, 1.0], [0.3, 0.7]], [[0.0, 0.0], [0.0, 0.0]])
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

    def test_rand_given_to_the_constructor_initialises_the_weights(self):
        draws = iter([0.1, 0.2, 0.3, 0.4])
        network = SkipGramNetwork(["a"], 2, lambda: next(draws))
        scale = 0.5 / 2
        self.assertEqual([(0.1 - 0.5) * scale, (0.3 - 0.5) * scale], network.embedding(0))
        self.assertEqual([(0.2 - 0.5) * scale, (0.4 - 0.5) * scale], network.context_vector(0))

    def test_without_rand_every_weight_starts_at_zero(self):
        network = SkipGramNetwork(["a", "b"], 3)
        self.assertEqual([[0.0] * 3] * 2, [network.embedding(0), network.embedding(1)])
        self.assertEqual([[0.0] * 3] * 2, [network.context_vector(0), network.context_vector(1)])


# The starting weights SkipGramNetwork sets when given the seeded generator
class TestNetworkStartingWeights(unittest.TestCase):
    def test_one_vector_per_token_of_the_right_size(self):
        network = SkipGramNetwork(["a", "b", "c", "d", "e"], 3, Mulberry32(1))
        self.assertEqual(5, network.vocab_size())
        for token in range(5):
            self.assertEqual(3, len(network.embedding(token)))
            self.assertEqual(3, len(network.context_vector(token)))

    def test_values_within_scale(self):
        dim = 4
        half_scale = 0.5 / dim / 2
        network = SkipGramNetwork([str(i) for i in range(10)], dim, Mulberry32(7))
        for token in range(10):
            for weight in network.embedding(token) + network.context_vector(token):
                self.assertLessEqual(abs(weight), half_scale)

    def test_every_synapse_is_a_separate_object(self):
        network = SkipGramNetwork(["a", "b", "c"], 2, Mulberry32(1))
        synapses = [synapse for neuron in network.net.neurons for synapse in neuron.synapses_out]
        self.assertEqual(3 * 2 + 2 * 3, len({id(synapse) for synapse in synapses}))

    def test_matches_typescript_for_seed_42(self):
        # Reference values from running the TypeScript initialisation with Mulberry32(42),
        # vocabSize = 2, dim = 2. The TypeScript stores them flat: token 0 then token 1.
        network = SkipGramNetwork(["a", "b"], 2, Mulberry32(42))
        self.assertEqual([[0.025275937980040908, 0.08811644837260246],
                          [-0.08129652531351894, -0.056693001417443156]],
                         [network.embedding(0), network.embedding(1)])
        self.assertEqual([[-0.012927360250614583, 0.04243351035984233],
                          [0.006648135546129197, 0.031186163483653218]],
                         [network.context_vector(0), network.context_vector(1)])


class TestTrainPair(NetworkTestCase):
    def test_positive_example_from_comment(self):
        network = self.network_with([[0.5, 0.5]], [[0.5, -0.5]])
        loss = network.train_pair(0, 0, 1, 0.1)
        for expected, actual in zip([0.525, 0.475], network.embedding(0)):
            self.assertAlmostEqual(expected, actual)
        for expected, actual in zip([0.525, -0.475], network.context_vector(0)):
            self.assertAlmostEqual(expected, actual)
        self.assertAlmostEqual(math.log(2), loss)

    def test_positive_pair_score_goes_up(self):
        # target is token 0's embedding, other is token 1's context vector
        network = self.network_with([[0.1, 0.2], [0.0, 0.0]], [[0.0, 0.0], [0.3, -0.1]])
        before = Embed.dot(network.embedding(0), network.context_vector(1))
        network.train_pair(0, 1, 1, 0.5)
        self.assertGreater(Embed.dot(network.embedding(0), network.context_vector(1)), before)

    def test_negative_pair_score_goes_down(self):
        network = self.network_with([[0.1, 0.2], [0.0, 0.0]], [[0.0, 0.0], [0.3, -0.1]])
        before = Embed.dot(network.embedding(0), network.context_vector(1))
        network.train_pair(0, 1, 0, 0.5)
        self.assertLess(Embed.dot(network.embedding(0), network.context_vector(1)), before)

    def test_only_the_two_vectors_change(self):
        network = self.network_with([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])
        network.train_pair(0, 2, 1, 0.5)
        self.assertEqual([[0.3, 0.4], [0.5, 0.6]], [network.embedding(1), network.embedding(2)])
        self.assertEqual([[0.6, 0.5], [0.4, 0.3]], [network.context_vector(0), network.context_vector(1)])

    def test_both_vectors_use_values_from_before_the_update(self):
        # Both vectors start at [1.0], so dot = 1.0. lr is chosen so that grad = 0.5.
        # Each vector should become 1.0 + 0.5 * 1.0 = 1.5. If the context vector were updated
        # with the already-updated embedding instead, it would become 1.0 + 0.5 * 1.5 = 1.75.
        network = self.network_with([[1.0]], [[1.0]])
        lr = 0.5 / (1 - Neuron.sigmoid(1.0))
        network.train_pair(0, 0, 1, lr)
        self.assertAlmostEqual(1.5, network.embedding(0)[0])
        self.assertAlmostEqual(1.5, network.context_vector(0)[0])

    def test_loss_for_negative_pair(self):
        # dot = 0 -> score 0.5 -> loss = -log(1 - 0.5)
        network = self.network_with([[0.0, 0.0]], [[0.0, 0.0]])
        self.assertAlmostEqual(math.log(2), network.train_pair(0, 0, 0, 0.1))


if __name__ == "__main__":
    unittest.main()
