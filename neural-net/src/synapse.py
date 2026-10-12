from typing import Optional
from neuron import Neuron

class Synapse:
	
	# Optionally connects the synapse as it is created:
	#   Synapse("a_to_b", a, b, 0.5)  is the same as setting from_neuron = a, to_neuron = b, weight = 0.5
	def __init__(self, id: str, from_neuron: Optional[Neuron] = None, to_neuron: Optional[Neuron] = None,
				 weight: float = 1.0):
		self.id = id
		self._from_neuron = None
		self._to_neuron = None
		self._weight = weight
		if from_neuron is not None:
			self.from_neuron = from_neuron
		if to_neuron is not None:
			self.to_neuron = to_neuron
		return None

	@property
	def from_neuron(self) -> Neuron:
		return self._from_neuron

	@property
	def to_neuron(self) -> Neuron:
		return self._to_neuron

	@from_neuron.setter
	def from_neuron(self, neuron: Neuron):
		self._from_neuron = neuron
		neuron.synapses_out.append(self)

	@to_neuron.setter
	def to_neuron(self, neuron: Neuron):
		self._to_neuron = neuron
		neuron.synapses_in.append(self)

	@property
	def weight(self) -> float:
		return self._weight

	@weight.setter
	def weight(self, value: float):
		self._weight = value

	def trace(self):
		print(f"{self.from_neuron.id} -> {self.to_neuron.id} (weight: {self.weight})")
