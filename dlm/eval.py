from theano import *
import theano.tensor as T
from dlm.io.lmDatasetReader import LMDatasetReader
from dlm.io.irisDatasetReader import IRISDatasetReader
from dlm.models.ltmlp import MLP
import dlm.utils as U
import math
import numpy as np

class Evaluator():

def __init__(self, dataset, classifier):
		self.dataset = dataset							# Initializing the dataset
		self.num_batches = dataset.get_num_batches()	# Number of minibatches in the dataset

		index = T.lscalar()
		x = classifier.input
		y = T.ivector('y')

self.classifier = classifier
		self.output = self.classifier.negative_log_likelihood(y)


		self.batch_neg_log_likelihood = theano.function(
			inputs=[index],
			outputs=self.classifier.negative_log_likelihood(y),
			givens={
				x: self.dataset.get_x(index),
				y: self.dataset.get_y(index)
			}
		)

	def mean_neg_log_likelihood(self):
		return np.mean([self.batch_neg_log_likelihood(i) for i in xrange(self.num_batches)])

	def perplexity(self):
		return math.exp(self.mean_neg_log_likelihood())
