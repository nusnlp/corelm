import theano.tensor as T
import dlm.utils as U

class Activation():

	def __init__(self, input, func_name):
		self.input = input
		self.func = self.get_function(func_name)
		self.output = self.func(input)
	
	def get_function(self, func_name):
		if func_name == 'tanh':
			return T.tanh
		elif func_name == 'hardtanh':
			U.warning('Current hardTanh implementation is slow!')
			return lambda x: ((abs(x) <= 1) * x) + ((1 < abs(x)) * T.sgn(x))
		elif func_name == 'sigmoid':
			return T.nnet.sigmoid
		elif func_name == 'fastsigmoid':
			U.error('T.nnet.ultra_fast_sigmoid function has some problems')
		elif func_name == 'hardsigmoid':
			return T.nnet.hard_sigmoid
		elif func_name == 'softplus':
			return T.nnet.softplus
		elif func_name == 'relu':
			#return lambda x: T.maximum(x, 0)
			return lambda x: x * (x > 0)
		elif func_name == 'cappedrelu':
			return lambda x: T.minimum(x * (x > 0), 6)
		else:
			U.error('Invalid function name given: ' + func_name)
