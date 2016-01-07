import theano.tensor as T
import theano
import dlm.io.logging as L

class SGD:
	def __init__(self, classifier, criterion, learning_rate, trainset, clip_threshold=0):
		self.eta = learning_rate
		self.is_weighted = trainset.is_weighted
		self.trainset = trainset
		
		if clip_threshold > 0:
			gparams = [T.clip(T.grad(criterion.cost, param), -clip_threshold, clip_threshold) for param in classifier.params]
		else:
			gparams = [T.grad(criterion.cost, param) for param in classifier.params]
		
		lr = T.fscalar()
		
		updates = [
			(param, param - lr * gparam)
			for param, gparam in zip(classifier.params, gparams)
		]
		
		x = classifier.input
		y = criterion.y
		
		if self.is_weighted: 
			w = criterion.w
			self.step_func = theano.function(
				inputs=[x, y, w, lr],
				outputs=[criterion.cost] + gparams,
				updates=updates,
			)
		else:
			self.step_func = theano.function(
				inputs=[x, y, lr],
				outputs=[criterion.cost] + gparams,
				updates=updates,
			)

	def step(self, minibatch_index):
		if self.is_weighted:
			outputs = self.step_func(self.trainset.get_x(minibatch_index), self.trainset.get_y(minibatch_index), self.trainset.get_w(minibatch_index), self.eta)
		else:
			outputs = self.step_func(self.trainset.get_x(minibatch_index), self.trainset.get_y(minibatch_index), self.eta)
		step_cost, gparams = outputs[0], outputs[1:]
		return step_cost, gparams

	def set_learning_rate(self, eta):
		self.eta = eta
	
	def get_learning_rate(self):
		return self.eta
