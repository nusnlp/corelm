from __future__ import division
import theano.tensor as T
import dlm.io.logging as L
import dlm.utils as U

class Activation():

	def __init__(self, input, func_name):
		L.info("Activation layer, function: " + U.red(func_name))
		self.input = input
		self.func = self.get_function(func_name)
		self.output = self.func(input)
	
	def get_function(self, func_name):
		if func_name == 'tanh':
			return T.tanh
		elif func_name == 'hardtanh':
			L.warning('Current hardTanh implementation is slow!')
			return lambda x: ((abs(x) <= 1) * x) + ((1 < abs(x)) * T.sgn(x))
		elif func_name == 'sigmoid':
			return T.nnet.sigmoid
		elif func_name == 'fastsigmoid':
			L.error('T.nnet.ultra_fast_sigmoid function has some problems')
		elif func_name == 'hardsigmoid':
			return T.nnet.hard_sigmoid
		elif func_name == 'softplus':
			return T.nnet.softplus
		elif func_name == 'relu':
			#return lambda x: T.maximum(x, 0)
			#return lambda x: x * (x > 0)
			return T.nnet.relu
		elif func_name == 'cappedrelu':
			return lambda x: T.minimum(x * (x > 0), 6)
		elif func_name == 'softmax':
			return T.nnet.softmax
		elif func_name == 'norm1':
			return lambda x: x / T.nlinalg.norm(x, 1)
		elif func_name == 'norm2':
			#return lambda x: x / T.nlinalg.norm(x, 2)
			return lambda x: x / T.dot(x, x)**0.5
		else:
			L.error('Invalid function name given: ' + func_name)
