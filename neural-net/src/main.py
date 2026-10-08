from neuron import Neuron
from synapse import Synapse
from neural_net_builder import NeuralNetBuilder

def main():
	# input_neuron_0 = Neuron("input_neuron_0")
	# input_neuron_1 =  Neuron("input_neuron_1")
	# input_neuron_2 = Neuron("inputNeuron_2")
 
	# hidden_neuron_0 = Neuron("hidden_neuron_0")
	# hidden_neuron_1 = Neuron("hidden_neuron_1")
	# hidden_neurons = [hidden_neuron_0, hidden_neuron_1]
 
	# output_neuron = Neuron("output_neuron")

	# synapse_0 = Synapse("synapse0")
	# synapse_1 = Synapse("synapse1")
	# synapse_2 = Synapse("synapse2")

	# synapse_3 = Synapse("synapse3")
	# synapse_4 = Synapse("synapse4")
	# synapse_5 = Synapse("synapse5")
 
	# synapse_6 = Synapse("synapse_6")
	# synapse_7 = Synapse("synapse_7")
 
	# synapse_6.weight = 1.0
	# synapse_7.weight = 1.0
 
	# synapse_0.from_neuron = input_neuron_0
	# synapse_0.to_neuron = hidden_neuron_0
 
	# synapse_1.from_neuron = input_neuron_0
	# synapse_1.to_neuron = hidden_neuron_1
 
	# synapse_2.from_neuron = input_neuron_1
	# synapse_2.to_neuron = hidden_neuron_0
 
	# synapse_3.from_neuron = input_neuron_1
	# synapse_3.to_neuron = hidden_neuron_1

	# synapse_4.from_neuron = input_neuron_2
	# synapse_4.to_neuron = hidden_neuron_0

	# synapse_5.from_neuron = input_neuron_2
	# synapse_5.to_neuron = hidden_neuron_1

	# input_neuron_0.value = 2.0
	# input_neuron_1.value = 3.0
	# input_neuron_2.value = 4.0
 
	# synapse_0.weight = 1.0
	# synapse_1.weight = 1.0
	# synapse_2.weight = 1.0	

	# synapse_3.weight = 1.0
	# synapse_4.weight = 1.0
	# synapse_5.weight = 1.0
 
	# print("Tracing hidden layer neurons:")
	# for hiddenNeuron in hidden_neurons:
	# 	print(f"Tracing hidden neuron {hiddenNeuron.id}")
	# 	for synapse_in in hiddenNeuron.synapses_in:
	# 		print(f"Processing synapse {synapse_in.id}")	
	# 		hiddenNeuron.value += synapse_in.weight * synapse_in.from_neuron.value
	# 		print(f"Updated {hiddenNeuron.id} value to {hiddenNeuron.value} using synapse {synapse_in.id}")
 
	# synapse_6.from_neuron = hidden_neurons[0]
	# synapse_6.to_neuron = output_neuron
 
	# synapse_7.from_neuron = hidden_neurons[1]
	# synapse_7.to_neuron = output_neuron
 
	# print("Tracing output neuron:")
	# print(f"Tracing output neuron {output_neuron.id}")
	# for synapse_in in output_neuron.synapses_in:
	# 	print(f"Processing synapse {synapse_in.id}")
	# 	output_neuron.value += synapse_in.weight * synapse_in.from_neuron.value # sigmoid will squash int the range 0 to 1
	# 	print(f"Updated value to {output_neuron.value} because synapse weight is {synapse_in.weight} from neuron {synapse_in.from_neuron.id} with value {synapse_in.from_neuron.value}")
 
	# builder = NeuralNetBuilder([3, 2, 1])
	# builder.build()
 
 
 

if __name__ == "__main__":
	main()
