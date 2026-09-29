
class Neuron:
	"""A single artificial neuron: a weighted sum of inputs plus a bias, passed through an activation."""

	def __init__(self, id: str):
		self.id = id
		return None

	def trace(self):
		print(self.id)
