import theano.tensor as T

class NegLogLikelihood():

	def __init__(self, classifier, args):
		
		self.y = T.ivector('y')
		
		self.cost = (
			classifier.negative_log_likelihood(self.y)
			+ args.L1_reg * classifier.L1
			+ args.L2_reg * classifier.L2_sqr
		)
		
		if args.alpha is not None and args.alpha > 0:
			self.cost = self.cost + args.alpha  * classifier.log_Z_sqr

		self.test = (
			T.mean(classifier.p_y_given_x(self.y))
		)
