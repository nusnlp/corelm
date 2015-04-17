import theano.tensor as T

class Activation(object):

	def __init__(self, rng, input, func=None):
		self.input = input
		self.output = func(input)
