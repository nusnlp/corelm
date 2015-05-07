import dlmutils.utils as U
import theano.tensor as T
import theano

class SGD:
	def __init__(self, classifier, criterion, learning_rate, trainset):
		gparams = [T.grad(criterion.cost, param) for param in classifier.params]

		updates = [
			(param, param - learning_rate * gparam)
			for param, gparam in zip(classifier.params, gparams)
		]
	
		index = T.lscalar()		# index to a [mini]batch
		x = classifier.input
		y = criterion.y
		
		self.step_func = theano.function(
			inputs=[index],
			outputs=criterion.cost,
			updates=updates,
			givens={
				x: trainset.get_x(index),
				y: trainset.get_y(index)
			}
		)

	def step(self, minibatch_index):
		return self.step_func(minibatch_index)
