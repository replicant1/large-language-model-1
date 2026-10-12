import math
from typing import Callable, Optional
from neuron import identity, sigmoid
from neural_net_builder import NeuralNetBuilder


class SkipGramNetwork:
    """The skip-gram model as a three-layer neural network:

        input layer:   one neuron per token. Feeding in a token sets its neuron to 1 and
                       every other to 0 (a "one-hot" input).
        hidden layer:  dim neurons, identity activation. Their values are the embedding.
        output layer:  one neuron per token, sigmoid activation. Output neuron c's value is
                       the model's guess that token c really appears near the input token.

    Every input neuron is connected to every hidden neuron, and every hidden neuron to every
    output neuron. A token's embedding is the weights on the synapses going out of its input
    neuron (what used to be called w_in); its context vector is the weights on the synapses
    coming into its output neuron (what used to be called w_out).

    If rand is given, every synapse starts with a small random weight (see initialize_weights);
    otherwise every weight starts at 0.

    Example, tokens ["the", "cat"] and dim = 2:

        input:the ──┬──▶ hidden:0 ──┬──▶ output:the
        input:cat ──┘    hidden:1 ──┘    output:cat     (each ─▶ fans out to every neuron in the next layer)

        embedding(1)       = weights of input:cat -> hidden:0 and input:cat -> hidden:1
        context_vector(0)  = weights of hidden:0 -> output:the and hidden:1 -> output:the
    """

    def __init__(self, token_names: list[str], dim: int, rand: Optional[Callable[[], float]] = None):
        vocab_size = len(token_names)
        self.dim = dim
        self.net = NeuralNetBuilder(
            [vocab_size, dim, vocab_size],
            layer_names=["input", "hidden", "output"],
            neuron_names_per_layer=[token_names, [str(d) for d in range(dim)], token_names],
            activation_per_layer=[identity, identity, sigmoid],
            initial_weight=0.0,
        ).build()
        self.input_layer = self.net.layer("input")
        self.hidden_layer = self.net.layer("hidden")
        self.output_layer = self.net.layer("output")
        if rand is not None:
            self.initialize_weights(rand)

    def vocab_size(self) -> int:
        return self.input_layer.size()

    # The token's embedding: the weights on the synapses from its input neuron to each hidden neuron
    def embedding(self, token: int) -> list[float]:
        return [synapse.weight for synapse in self.input_layer.neurons[token].synapses_out]

    # The token's context vector: the weights on the synapses from each hidden neuron to its output neuron
    def context_vector(self, token: int) -> list[float]:
        return [synapse.weight for synapse in self.output_layer.neurons[token].synapses_in]

    def set_embedding(self, token: int, values: list[float]) -> None:
        for synapse, value in zip(self.input_layer.neurons[token].synapses_out, values):
            synapse.weight = value

    def set_context_vector(self, token: int, values: list[float]) -> None:
        for synapse, value in zip(self.output_layer.neurons[token].synapses_in, values):
            synapse.weight = value

    # Gives every synapse a small random starting weight, between -scale/2 and +scale/2,
    # where scale = 0.5 / dim. Starting small and random keeps every token's vectors different
    # but close to zero, so no token starts out favoured.
    #
    # The rand() calls go token by token and, within a token, alternate between its embedding
    # and its context vector, value by value. That is the same order as the TypeScript, so the
    # same seed gives the same starting weights in both versions.
    #
    # Example, dim = 2: for each token t, the draws are
    #   input:t -> hidden:0,  hidden:0 -> output:t,  input:t -> hidden:1,  hidden:1 -> output:t
    def initialize_weights(self, rand: Callable[[], float]) -> None:
        scale = 0.5 / self.dim
        for token in range(self.vocab_size()):
            synapses_from_input = self.input_layer.neurons[token].synapses_out
            synapses_to_output = self.output_layer.neurons[token].synapses_in
            for d in range(self.dim):
                synapses_from_input[d].weight = (rand() - 0.5) * scale
                synapses_to_output[d].weight = (rand() - 0.5) * scale

    # Forward pass: feeds in the target token and computes one output neuron's value, the
    # model's guess that the other token really appears near the target.
    #
    # With a one-hot input, each hidden neuron's weighted sum is just the weight on the synapse
    # from the target's input neuron (every other input is 0), so the hidden values are set
    # directly rather than summing 0 x weight over every input neuron. Only the one output
    # neuron that is asked about is computed.
    def forward(self, target: int, other: int) -> float:
        for synapse in self.input_layer.neurons[target].synapses_out:
            synapse.to_neuron.value = synapse.to_neuron.activation(synapse.weight)
        return self.output_layer.neurons[other].forward()

    # Trains on one (target, other) pair and returns its loss: a forward pass, then a backward
    # pass that updates the two sets of synapse weights involved.
    #
    # label is 1 for a positive pair (other really is near target in the text) and 0 for a
    # negative sample (other is a random token). Training moves the score towards the label:
    #
    #   positive (label 1):  grad = lr * (1 - score)  ->  the two vectors move closer together
    #   negative (label 0):  grad = lr * (0 - score)  ->  the two vectors move apart
    #
    # Backward pass: label - score is the error at the output neuron. For each hidden neuron,
    # the synapse into it from the target is nudged by grad times the synapse out of it to the
    # output, and the synapse out of it is nudged by grad times the hidden neuron's value. Both
    # nudges use the weights from before either changes, as backpropagation requires.
    #
    # The loss measures how wrong the score was: -log(score) for a positive and
    # -log(1 - score) for a negative. It is 0 for a perfect score and grows as the score
    # gets worse. 1e-10 stops log(0) when a score is exactly 0 or 1.
    #
    # Example, positive pair, dim 2, lr = 0.1:
    #
    #   embedding(target) = [0.5, 0.5],  context_vector(other) = [0.5, -0.5]
    #   score = sigmoid(0.25 - 0.25) = 0.5  ->  grad = 0.1 * (1 - 0.5) = 0.05
    #   embedding(target)    becomes [0.5 + 0.05*0.5, 0.5 + 0.05*-0.5] = [0.525, 0.475]
    #   context_vector(other) becomes [0.5 + 0.05*0.5, -0.5 + 0.05*0.5] = [0.525, -0.475]
    #   loss = -log(0.5) = 0.693
    def train_pair(self, target: int, other: int, label: int, lr: float) -> float:
        score = self.forward(target, other)
        grad = lr * (label - score)

        synapses_from_target = self.input_layer.neurons[target].synapses_out
        synapses_to_other = self.output_layer.neurons[other].synapses_in
        for synapse_in, synapse_out in zip(synapses_from_target, synapses_to_other):
            hidden_value = synapse_in.to_neuron.value
            synapse_in.weight += grad * synapse_out.weight
            synapse_out.weight += grad * hidden_value

        if label == 1:
            return -math.log(score + 1e-10)
        return -math.log(1 - score + 1e-10)
