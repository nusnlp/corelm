import numpy
import theano
import theano.tensor as T
import dlm.io.logging as L
import dlm.utils as U

class Linear():

	def __init__(self, rng, input, n_in, n_out, W_values=None, init_method=0, b_values=None, no_bias=False, suffix=None, high=0.01):
		
		L.info("Linear layer, #inputs: %s, #outputs: %s" % (U.red(n_in), U.red(n_out)))

		self.input = input

		if W_values is None:
			if init_method == 0:	# Useful for Relu activation
				high = high
			elif init_method == 1:	# Useful for Tanh activation
				high = numpy.sqrt(6. / (n_in + n_out))
			elif init_method == 2:	# Useful for Sigmoid activation
				high = 4 * numpy.sqrt(6. / (n_in + n_out))
			else:
				L.error('Invalid initialization method')
			W_values = numpy.asarray(
				rng.uniform(
					low=-high,
					high=high,
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
