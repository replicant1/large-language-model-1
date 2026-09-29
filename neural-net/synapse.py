from neuron import Neuron

class Synapse:
	
	def __init__(self, id: str, from_neuron : Neuron, to_neuron: Neuron, weight: float):
		self.id = id
		self.weight = weight
		self.from_neuron = from_neuron
		self.to_neuron = to_neuron
		return None

	def trace(self):
		print(f"{self.from_neuron.id} -> {self.to_neuron.id} (weight: {self.weight})")
