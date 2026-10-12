from neuron import Neuron


class Layer:
    """A named group of neurons at the same depth in a network, such as "input" or "hidden".
    Neurons keep the order they were added in, so neurons[i] is the i-th neuron of the layer."""

    def __init__(self, name: str, neurons: list[Neuron]):
        self.name = name
        self.neurons = neurons

    def size(self) -> int:
        return len(self.neurons)
