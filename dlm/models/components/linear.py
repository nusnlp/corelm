import numpy
import theano
import theano.tensor as T
import dlm.io.logging as L

class Linear():

	def __init__(self, rng, input, n_in, n_out, W_values=None, b_values=None, no_bias=False, suffix=None):
		
		L.info("Linear layer, #inputs: %i, #outputs: %i" % (n_in, n_out))

		self.input = input

		if W_values is None:
			W_values = numpy.asarray(
				rng.uniform(
					low = -0.01, #low=-numpy.sqrt(6. / (n_in + n_out)),
					high = 0.01, #high=numpy.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)

		if b_values is None and not no_bias:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
		
		W_name = 'W'
		if suffix is not None:
			W_name += '.' + str(suffix)
		
		W = theano.shared(value=W_values, name=W_name, borrow=True)
		self.W = W

		if no_bias:
			self.output = T.dot(input, self.W)
			self.params = [self.W]
		else:
			b_name = 'b'
			if suffix is not None:
				b_name += '.' + str(suffix)
			b = theano.shared(value=b_values, name=b_name, borrow=True)
			self.b = b
			self.output = T.dot(input, self.W) + self.b
			self.params = [self.W, self.b]
