import math
from typing import Callable


# Activation functions: decide how strongly a neuron fires, given the weighted sum of its inputs.

# Passes the weighted sum straight through. Used where a layer only needs to carry values on,
# such as the hidden layer of the skip-gram network.
def identity(x: float) -> float:
    return x


# Squashes any number into the range 0 to 1, so it can be read as a probability.
# Returns exactly 1 above 6 and exactly 0 below -6, as the TypeScript does, to save time.
def sigmoid(x: float) -> float:
    if x > 6:
        return 1.0
    if x < -6:
        return 0.0
    return 1 / (1 + math.exp(-x))


class Neuron:
    """A single artificial neuron: its value is its activation function applied to the
    weighted sum of its inputs, that is, each incoming synapse's weight times the value
    of the neuron it comes from."""

    def __init__(self, id: str, activation: Callable[[float], float] = identity):
        self.id = id
        self.activation = activation
        self._synapses_in = []
        self._synapses_out = []
        self._value = 0.0
        return None

    @property
    def synapses_in(self):
        return self._synapses_in

    @property
    def synapses_out(self):
        return self._synapses_out

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    # Computes this neuron's value from its incoming synapses and returns it.
    # The products are added one at a time, in the order the synapses were connected.
    #
    # Example, two incoming synapses with weights 0.5 and -1.0, from neurons with values
    # 2.0 and 3.0, and the identity activation:
    #
    #   value = identity(0.5 * 2.0 + -1.0 * 3.0) = -2.0
    def forward(self) -> float:
        total = 0.0
        for synapse in self._synapses_in:
            total += synapse.weight * synapse.from_neuron.value
        self._value = self.activation(total)
        return self._value

    def trace(self):
        print(self.id)
