from typing import Callable, Optional
from neuron import Neuron, identity
from neural_net import NeuralNet
from layer import Layer
from synapse import Synapse

class NeuralNetBuilder:
    """Builds a fully connected network: every neuron in layer n is connected to every
    neuron in layer n+1 by a synapse.

    neuron_count_per_layer: how many neurons in each layer, input layer first
    layer_names: a name for each layer (default "layer_0", "layer_1", ...)
    neuron_names_per_layer: a name for each neuron in each layer, used in its id
        (default: its index); for example the tokens, for a layer with one neuron per token
    activation_per_layer: each layer's activation function (default: identity)
    initial_weight: every synapse's starting weight (default 1.0)
    verbose: print each layer and neuron as it is created

    Example: NeuralNetBuilder([3, 2, 1]) builds 3 -> 2 -> 1 neurons, joined by 3*2 + 2*1 = 8 synapses.
    """
    def __init__(self, neuron_count_per_layer: list[int],
                 layer_names: Optional[list[str]] = None,
                 neuron_names_per_layer: Optional[list[list[str]]] = None,
                 activation_per_layer: Optional[list[Callable[[float], float]]] = None,
                 initial_weight: float = 1.0,
                 verbose: bool = False):
        self.neuron_count_per_layer = neuron_count_per_layer
        self.layer_names = layer_names
        self.neuron_names_per_layer = neuron_names_per_layer
        self.activation_per_layer = activation_per_layer
        self.initial_weight = initial_weight
        self.verbose = verbose
        self.neurons = []

    def build(self) -> NeuralNet:
        self.neurons = []
        layers: list[Layer] = []

        # Create all the neurons first
        for layer_index, neuron_count in enumerate(self.neuron_count_per_layer):
            layer_name = self.layer_names[layer_index] if self.layer_names else f"layer_{layer_index}"
            activation = self.activation_per_layer[layer_index] if self.activation_per_layer else identity
            if self.verbose:
                print(f"Building layer {layer_index} with {neuron_count} neurons")
            layer_neurons = []
            for neuron_index in range(neuron_count):
                if self.verbose:
                    print(f"  Creating neuron {neuron_index} in layer {layer_index}")
                if self.neuron_names_per_layer:
                    neuron_id = f"{layer_name}:{self.neuron_names_per_layer[layer_index][neuron_index]}"
                else:
                    neuron_id = f"neuron_{neuron_index}_layer_{layer_index}"
                neuron = Neuron(neuron_id, activation)
                self.neurons.append(neuron)
                layer_neurons.append(neuron)
            layers.append(Layer(layer_name, layer_neurons))

        # Connect each neuron in layer n to each neuron in layer n+1 via synapse.
        # Each neuron's synapses_out and synapses_in are therefore in neuron order of the
        # layer they connect to: neuron.synapses_out[j] goes to the j-th neuron of the next layer.
        for layer_index in range(len(self.neuron_count_per_layer) - 1):
            current_layer_neurons = self._neurons_in_layer(layer_index)
            next_layer_neurons = self._neurons_in_layer(layer_index + 1)
            for from_neuron in current_layer_neurons:
                for to_neuron in next_layer_neurons:
                    synapse = Synapse(f"{from_neuron.id}_to_{to_neuron.id}")
                    synapse.from_neuron = from_neuron
                    synapse.to_neuron = to_neuron
                    synapse.weight = self.initial_weight

        return NeuralNet(self.neurons, layers)

    def _neurons_in_layer(self, layer_index: int) -> list[Neuron]:
        start_index = sum(self.neuron_count_per_layer[:layer_index])
        end_index = start_index + self.neuron_count_per_layer[layer_index]
        return self.neurons[start_index:end_index]
