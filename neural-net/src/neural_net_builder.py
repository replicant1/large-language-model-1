from neuron import Neuron
from neural_net import NeuralNet
from synapse import Synapse

class NeuralNetBuilder:
    def __init__(self, neuron_count_per_layer: list[int]):
        self.neuron_count_per_layer = neuron_count_per_layer
        self.neurons = []
        
    def build(self) -> NeuralNet:
        self.neurons = []
        
        # Create all the neurons first
        for layer_index, neuron_count in enumerate(self.neuron_count_per_layer):
            print(f"Building layer {layer_index} with {neuron_count} neurons")
            for neuron_index in range(neuron_count):
                print(f"  Creating neuron {neuron_index} in layer {layer_index}")   
                neuron = Neuron(f"neuron_{neuron_index}_layer_{layer_index}")
                self.neurons.append(neuron)
                
        # Connect each neuron in layer n to each neuron in layer n+1 via synapse 
        for layer_index in range(len(self.neuron_count_per_layer) - 1):
            current_layer_neurons = self._neurons_in_layer(layer_index)
            next_layer_neurons = self._neurons_in_layer(layer_index + 1)
            for from_neuron in current_layer_neurons:
                for to_neuron in next_layer_neurons:
                    synapse = Synapse(f"{from_neuron.id}_to_{to_neuron.id}")
                    from_neuron.synapses_out.append(synapse)
                    to_neuron.synapses_in.append(synapse)
        
        return NeuralNet(self.neurons)
    
    def _neurons_in_layer(self, layer_index: int) -> list[Neuron]:
        start_index = sum(self.neuron_count_per_layer[:layer_index])
        end_index = start_index + self.neuron_count_per_layer[layer_index]
        return self.neurons[start_index:end_index]
                