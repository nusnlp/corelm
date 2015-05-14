from __future__ import division
import sys
import time
from theano import *
import theano.tensor as T
from dlm.io.mmapReader import MemMapReader
from dlm.models.ltmlp import MLP
import dlm.utils as U
import math
import numpy as np

class Evaluator():

	def __init__(self, dataset, classifier):
		self.dataset = dataset							# Initializing the dataset
		self.num_batches = dataset.get_num_batches()	# Number of minibatches in the dataset
		self.num_samples = dataset._get_num_samples()	# Number of samples in the dataset

		index = T.lscalar()
		x = classifier.input
		y = T.ivector('y')
		
		self.neg_sum_batch_log_likelihood = theano.function(
			inputs=[index],
			outputs=-T.sum(T.log(classifier.p_y_given_x(y))),
			givens={
				x: self.dataset.get_x(index),
				y: self.dataset.get_y(index)
			}
		)
		
		self.sum_batch_error = theano.function(
			inputs=[index],
			outputs=classifier.errors(y),
			givens={
				x: dataset.get_x(index),
				y: dataset.get_y(index)
			}
		)
		
		# x: A matrix (N * (ngram - 1)) representing the sequence of length N
		# y: A vector of class labels
		self.sequence_log_prob = self.neg_sum_batch_log_likelihood
		
		self.denominator = theano.function(
			inputs=[index],
			outputs=classifier.log_Z_sqr,
			givens={
				x: dataset.get_x(index)
			}
		)

	def classification_error(self):
		return np.sum([self.sum_batch_error(i) for i in xrange(self.num_batches)]) / self.num_samples
		
	def mean_neg_log_likelihood(self):
		return math.fsum([self.neg_sum_batch_log_likelihood(i) for i in xrange(self.num_batches)]) / self.num_samples # np.sum() has some precision problems here
	
	def perplexity(self):
		return math.exp(self.mean_neg_log_likelihood())

	def get_sequence_log_prob(self, index):
		return - self.sequence_log_prob(index)
	
	def get_denominator(self):
		return np.mean([self.denominator(i) for i in xrange(self.num_batches)])





