from typing import Optional
from neuron import Neuron
from layer import Layer

class NeuralNet:
    """A network of neurons connected by synapses.

    neurons: every neuron in the network, layer by layer
    layers: the same neurons grouped into layers, input layer first (optional)
    """
    def __init__(self, neurons: list[Neuron], layers: Optional[list[Layer]] = None):
        self.neurons = neurons
        self.layers = layers if layers is not None else []

    @property
    def input_layer(self) -> Layer:
        return self.layers[0]

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]

    # Raises KeyError if there is no layer with that name
    def layer(self, name: str) -> Layer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)
