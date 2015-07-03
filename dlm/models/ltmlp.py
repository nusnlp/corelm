from dlm.models.components.lookuptable import LookupTable
from dlm.models.components.linear import Linear
from dlm.models.components.activation import Activation
from dlm.models import classifier
import dlm.utils as U
import dlm.io.logging as L
import theano.tensor as T
import numpy
import math

class MLP(classifier.Classifier):

	def __init__(self, args=None, model_path=None):

		######################################################################
		## Parameters
		#
		
		U.xassert((args or model_path) and not (args and model_path), "args or model_path are mutually exclusive")
		
		if model_path:
			args, loaded_params = self.load_model(model_path)
		
		emb_dim = args.emb_dim
		num_hidden_list = map(int, args.num_hidden.split(','))
		for i in range(len(num_hidden_list)):
			#L.info("Hidden Layer %i: %i" % (i+1, num_hidden_list[i]))
			L.info("Hidden Layer " + str(i+1) + ": " + U.BColors.RED + str(num_hidden_list[i]) + U.BColors.ENDC)
		vocab_size = args.vocab_size
		self.ngram_size = args.ngram_size
		num_classes = args.num_classes
		activation_name = args.activation_name
		self.args = args
		self.L1 = 0
		self.L2_sqr = 0
		self.params = []
		
		rng = numpy.random.RandomState(1234)
		self.input = T.imatrix('input')

		######################################################################
		## Lookup Table Layer
		#
		
		lookupTableLayer = LookupTable(
			rng=rng,
			input=self.input,
			vocab_size=vocab_size,
			emb_dim=emb_dim
		)
		last_layer_output = lookupTableLayer.output
		last_layer_output_size = (self.ngram_size - 1) * emb_dim
		self.params += lookupTableLayer.params
		
		######################################################################
		## Hidden Layer(s)
		#
		
		for i in range(0, len(num_hidden_list)):
			linearLayer = Linear(
				rng=rng,
				input=last_layer_output,
				n_in=last_layer_output_size,
				n_out=num_hidden_list[i]
			)
			last_layer_output = linearLayer.output
			last_layer_output_size = num_hidden_list[i]
			self.params += linearLayer.params
			
			activation = Activation(
				input=last_layer_output,
				func_name=activation_name
			)
			last_layer_output = activation.output
			
			self.L1 = self.L1 + abs(linearLayer.W).sum()
			self.L2_sqr = self.L2_sqr + (linearLayer.W ** 2).sum()
		
		######################################################################
		## Output Linear Layer
		#
		
		linearLayer = Linear(
			rng=rng,
			input=last_layer_output,
			n_in=last_layer_output_size,
			n_out=num_classes,
			b_values = numpy.zeros(num_classes) - math.log(num_classes) 
		)
		last_layer_output = linearLayer.output
		self.params += linearLayer.params
		
		self.L1 = self.L1 + abs(linearLayer.W).sum()
		self.L2_sqr = self.L2_sqr + (linearLayer.W ** 2).sum()
		
		######################################################################
		## Model Output
		#
		
		self.output = last_layer_output
		self.p_y_given_x_matrix = T.nnet.softmax(last_layer_output)
		
		self.log_Z_sqr = T.log(T.mean(T.sum(T.exp(last_layer_output), axis=1))) ** 2
		#self.log_Z_sqr = T.sum(T.log(T.sum(T.exp(last_layer_output), axis=1))) ** 2
		#self.log_Z_sqr = T.mean(T.log(T.sum(T.exp(last_layer_output), axis=1))) ** 2
		
		######################################################################
		## Model Predictions
		#
		
		self.y_pred = T.argmax(self.p_y_given_x_matrix, axis=1)
		
		######################################################################
		## Loading parameters from file (if given)
		#
		
		if model_path:
			self.set_params(loaded_params)
		
	######################################################################
	## Model Functions
	#
		
	def p_y_given_x(self, y):
		return self.p_y_given_x_matrix[T.arange(y.shape[0]), y]
	
	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x(y)))

		

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
		if y.dtype.startswith('int'):
			return T.sum(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

