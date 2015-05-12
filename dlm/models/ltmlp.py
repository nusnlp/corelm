from dlm.models.components.lookuptable import LookupTableLayer
from dlm.models.components.hiddenlayer import HiddenLayer
from dlm.models.components.logisticregression import LogisticRegression
from dlm.models import classifier
import dlm.utils as U
import theano.tensor as T
import numpy

class MLP(classifier.Classifier):

	def __init__(self, args=None, model_path=None):

		U.xassert((args or model_path) and not (args and model_path), "args or model_path are mutually exclusive")
		
		if model_path:
			args, loaded_params = self.load_model(model_path)
		
		self.args = args
		emb_dim = args.emb_dim
		num_hidden_list = map(int, args.num_hidden.split(','))
		U.xassert(len(num_hidden_list) == 1, "More than one hidden layer is not supported yet")
		vocab_size = args.vocab_size
		self.ngram_size = args.ngram_size
		num_classes = args.num_classes
		
		# Creating model
		
		self.rng = numpy.random.RandomState(1234)
		self.input = T.imatrix('input')		# the data is presented as rasterized images

		self.lookupTableLayer = LookupTableLayer(
			rng=self.rng,
			input=self.input,
			vocab_size=vocab_size,
			emb_dim=emb_dim
		)

		self.hiddenLayer = HiddenLayer(
			rng=self.rng,
			input=self.lookupTableLayer.output,
			n_in=(self.ngram_size - 1) * emb_dim,
			n_out=num_hidden_list[0],
			activation=T.tanh
		)

		self.logRegressionLayer = LogisticRegression(
			input=self.hiddenLayer.output,
			n_in=num_hidden_list[-1],
			n_out=num_classes
		)

		self.L1 = (
			abs(self.hiddenLayer.W).sum()
			+ abs(self.logRegressionLayer.W).sum()
		)

		self.L2_sqr = (
			(self.hiddenLayer.W ** 2).sum()
			+ (self.logRegressionLayer.W ** 2).sum()
		)

		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
		
		self.p_y_given_x = self.logRegressionLayer.p_y_given_x

		self.errors = self.logRegressionLayer.errors

		self.params = self.lookupTableLayer.params + self.hiddenLayer.params + self.logRegressionLayer.params
		
		# Initializating parameters (if model file is given)
		if model_path:
			self.set_params(loaded_params)
		
		
		
		
