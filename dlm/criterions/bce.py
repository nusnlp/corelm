import theano.tensor as T

class BinaryCrossEntropy():

	def __init__(self, classifier, args):
		
		self.y = T.matrix('y')
		
		self.cost = (
			classifier.mean_batch_cross_entropy(self.y)
			+ args.L1_reg * classifier.L1
			+ args.L2_reg * classifier.L2_sqr
		)
