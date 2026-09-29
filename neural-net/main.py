from neuron import Neuron
from synapse import Synapse


def main():
	inputLayer = [Neuron("input1"), Neuron("input2"), Neuron("input3")]
	hiddenLayer = [Neuron("hidden1"), Neuron("hidden2")]
	outputLayer = [Neuron("output1")]

	# Synapses from input[x] to hidden[1]
	firstSynapseLayer = [
		Synapse("syn1", inputLayer[0], hiddenLayer[0], 1.0), # input1 -> hidden1
	Synapse("syn2", inputLayer[1], hiddenLayer[0], 1.0), # input2 -> hidden1
	Synapse("syn3", inputLayer[2], hiddenLayer[0], 1.0)] # input3 -> hidden1

	# Synapses from input[0] to hidden[2]
	secondSynapseLayer = [
		Synapse("syn4", inputLayer[0], hiddenLayer[1], 1.0), # input1 -> hidden2
		Synapse("syn5", inputLayer[1], hiddenLayer[1], 1.0), # input2 -> hidden2
		Synapse("syn6", inputLayer[2], hiddenLayer[1], 1.0) # input3 -> hidden2
	]
	
	# Synapses from hidden[x] to output[0]
	thirdSynapseLayer = [
		Synapse("syn7", hiddenLayer[0], outputLayer[0], 1.0), # hidden1 -> output1
		Synapse("syn8", hiddenLayer[1], outputLayer[0], 1.0) # hidden2 -> output1
	]

	print("Tracing input layer neurons:")
	for inputNeuron in inputLayer:
		inputNeuron.trace()

	print("Tracing hidden layer neurons:")
	for hiddenNeuron in hiddenLayer:
		hiddenNeuron.trace()

	print("Tracing output layer neurons:")
	for outputNeuron in outputLayer:
		outputNeuron.trace()

	print("Tracing first synapse layer:")
	for synapse in firstSynapseLayer:
		synapse.trace()

	print("Tracing second synapse layer:")
	for synapse in secondSynapseLayer:
		synapse.trace()

	print("Tracing third synapse layer:")
	for synapse in thirdSynapseLayer:
		synapse.trace()


if __name__ == "__main__":
	main()
