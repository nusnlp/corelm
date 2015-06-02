import numpy
import theano
import theano.tensor as T

class Linear():

	def __init__(self, rng, input, n_in, n_out, W=None, b=None):

		self.input = input

		if W is None:
			W_values = numpy.asarray(
				rng.uniform(
					low = -0.01, #low=-numpy.sqrt(6. / (n_in + n_out)),
					high = 0.01, #high=numpy.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		self.output = T.dot(input, self.W) + self.b

		# parameters of the model
		self.params = [self.W, self.b]
