
class Neuron:
    """A single artificial neuron: a weighted sum of inputs plus a bias, passed through an activation."""

    def __init__(self, id: str):
        self.id = id
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

    def trace(self):
        print(self.id)
