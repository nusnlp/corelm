from dlm.models.components.lookuptable import LookupTableLayer
from dlm.models.components.hiddenlayer import HiddenLayer
from dlm.models.components.logisticregression import LogisticRegression
from dlm.models import classifier
import theano.tensor as T

class MLP(classifier.Classifier):

	def __init__(self, rng, input, vocab_size, emb_dim, ngram_size, num_hidden, n_out):


		self.input = input

		self.lookupTableLayer = LookupTableLayer(
			rng=rng,
			input=input,
			vocab_size=vocab_size,
			emb_dim=emb_dim
		)

		self.hiddenLayer = HiddenLayer(
			rng=rng,
			input=self.lookupTableLayer.output,
			n_in=(ngram_size - 1) * emb_dim,
			n_out=num_hidden,
			activation=T.tanh
		)

		self.logRegressionLayer = LogisticRegression(
			input=self.hiddenLayer.output,
			n_in=num_hidden,
			n_out=n_out
		)

		self.L1 = (
			abs(self.hiddenLayer.W).sum()
			+ abs(self.logRegressionLayer.W).sum()
		)

		self.L2_sqr = (
			(self.hiddenLayer.W ** 2).sum()
			+ (self.logRegressionLayer.W ** 2).sum()
		)

		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
		)

		self.errors = self.logRegressionLayer.errors

		self.params = self.lookupTableLayer.params + self.hiddenLayer.params + self.logRegressionLayer.params
