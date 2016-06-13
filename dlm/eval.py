from __future__ import division
import sys
import time
from theano import *
import theano.tensor as T
from dlm.io.mmapReader import MemMapReader
from dlm.models.mlp import MLP
import dlm.utils as U
import math
import numpy as np

class Evaluator():

	def __init__(self, dataset, classifier):

		index = T.lscalar()
		x = classifier.input
		y = T.ivector('y')

		if dataset:
			self.dataset = dataset								# Initializing the dataset
			self.num_batches = self.dataset.get_num_batches()	# Number of minibatches in the dataset
			self.num_samples = self.dataset._get_num_samples()	# Number of samples in the dataset

			self.neg_sum_batch_log_likelihood = theano.function(
				inputs=[index],
				outputs=-T.sum(T.log(classifier.p_y_given_x(y))),
				givens={
					x: self.dataset.get_x(index),
					y: self.dataset.get_y(index)
				}
			)

			self.unnormalized_neg_sum_batch_log_likelihood = theano.function(
				inputs=[index],
				outputs=-T.sum(classifier.unnormalized_p_y_given_x(y)), # which is: -T.sum(T.log(T.exp(classifier.unnormalized_p_y_given_x(y))))
				givens={
					x: self.dataset.get_x(index),
					y: self.dataset.get_y(index)
				}
			)

			self.sum_batch_error = theano.function(
				inputs=[index],
				outputs=classifier.errors(y),
				givens={
					x: self.dataset.get_x(index),
					y: self.dataset.get_y(index)
				}
			)

			# x: A matrix (N * (ngram - 1)) representing the sequence of length N
			# y: A vector of class labels
			self.neg_sequence_log_prob = self.neg_sum_batch_log_likelihood

			self.denominator = theano.function(
				inputs=[index],
				outputs=classifier.log_Z_sqr,
				givens={
					x: self.dataset.get_x(index)
				}
			)

			self.get_p_matrix  = theano.function(
				inputs=[index],
				outputs=classifier.p_y_given_x_matrix,
				givens={
					x:self.dataset.get_x(index)
				}
			)
			self.get_y_pred = theano.function(
				inputs=[index],
				outputs=classifier.y_pred,
				givens={
					x:self.dataset.get_x(index)
				}
			)
		# End of if

		self.ngram_log_prob = theano.function(
			inputs=[x, y],
			outputs=T.log(classifier.p_y_given_x(y)),
		)


	def classification_error(self):
		return np.sum([self.sum_batch_error(i) for i in xrange(self.num_batches)]) / self.num_samples

	def mean_neg_log_likelihood(self):
		return math.fsum([self.neg_sum_batch_log_likelihood(i) for i in xrange(self.num_batches)]) / self.num_samples # np.sum() has some precision problems here

	def mean_unnormalized_neg_log_likelihood(self):
		return math.fsum([self.unnormalized_neg_sum_batch_log_likelihood(i) for i in xrange(self.num_batches)]) / self.num_samples # np.sum() has some precision problems here

	def perplexity(self):
		return math.exp(self.mean_neg_log_likelihood())

	def unnormalized_perplexity(self):
		return math.exp(self.mean_unnormalized_neg_log_likelihood())

	def get_sequence_log_prob(self, index):
		return - self.neg_sequence_log_prob(index)

	def get_unnormalized_sequence_log_prob(self, index):
		return - self.unnormalized_neg_sum_batch_log_likelihood(index)

	def get_ngram_log_prob(self, x, y):
		return self.ngram_log_prob(x, y)

	def get_denominator(self):
		return np.mean([self.denominator(i) for i in xrange(self.num_batches)])

	def get_class(self, index, restricted_ids = []):
		if restricted_ids != []:
			return restricted_ids[np.argmax(self.get_p_matrix(index)[:,restricted_ids])]
		else:
			return self.get_y_pred(index)[0]

	def get_batch_predicted_class(self, index):
		return self.get_y_pred(index)