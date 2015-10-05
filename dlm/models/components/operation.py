import theano.tensor as T

class Operation():

	def __init__(self, input, op_name):
		self.input = input
		self.operate = self.get_operation(op_name)
		self.output = self.operate(input, axis=1)
	
	def get_operation(self, op_name):
		if op_name == 'sum':
			return T.sum
		elif op_name == 'mean':
			return T.mean
		elif op_name == 'max':
			return T.max
		else:
			L.error('Invalid operation name given: ' + op_name)
