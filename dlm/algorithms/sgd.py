import theano.tensor as T
import theano

class SGD:
	def __init__(self, classifier, criterion, learning_rate, trainset):
		self.eta = learning_rate
		
		gparams = [T.grad(criterion.cost, param) for param in classifier.params]

		lr = T.fscalar()
		
		updates = [
			(param, param - lr * gparam)
			for param, gparam in zip(classifier.params, gparams)
		]
	
		index = T.lscalar()		# index to a [mini]batch
		x = classifier.input
		y = criterion.y
		
		self.step_func = theano.function(
			inputs=[index, lr],
			outputs=criterion.cost,
			updates=updates,
			givens={
				x: trainset.get_x(index),
				y: trainset.get_y(index)
			}
		)

	def step(self, minibatch_index):
		return self.step_func(minibatch_index, self.eta)
	
	def set_learning_rate(self, eta):
		self.eta = eta
	
	def get_learning_rate(self):
		return self.eta
