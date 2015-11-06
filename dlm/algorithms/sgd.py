import theano.tensor as T
import theano

class SGD:
	def __init__(self, classifier, criterion, learning_rate, trainset, clip_threshold=0):
		self.eta = learning_rate
		self.is_weighted = trainset.is_weighted
		
		if clip_threshold > 0:
			gparams = [T.clip(T.grad(criterion.cost, param), -clip_threshold, clip_threshold) for param in classifier.params]
		else:
			gparams = [T.grad(criterion.cost, param) for param in classifier.params]
		
		lr = T.fscalar()
		
		updates = [
			(param, param - lr * gparam)
			for param, gparam in zip(classifier.params, gparams)
		]
	
		index = T.lscalar()		# index to a [mini]batch
		x = classifier.input
		y = criterion.y
		
		if self.is_weighted: 
			w = criterion.w
			self.step_func = theano.function(
				inputs=[index, lr],
				outputs=[criterion.cost] + gparams,
				updates=updates,
				givens={
					x: trainset.get_x(index),
					y: trainset.get_y(index),
					w: trainset.get_w(index)
				}
			)
		else:
			self.step_func = theano.function(
				inputs=[index, lr],
				outputs=[criterion.cost] + gparams,
				updates=updates,
				givens={
					x: trainset.get_x(index),
					y: trainset.get_y(index)
				}
			)

	def step(self, minibatch_index):
		outputs = self.step_func(minibatch_index, self.eta)
		step_cost, gparams = outputs[0], outputs[1:]
		return step_cost, gparams

	def set_learning_rate(self, eta):
		self.eta = eta
	
	def get_learning_rate(self):
		return self.eta
