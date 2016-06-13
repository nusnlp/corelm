from __future__ import division
import dlm.io.logging as L
import dlm.utils as U
import numpy as np
import theano
import theano.tensor as T
import math as M
import sys
import os

class FeaturesMemMapReader():

	#### Constructor

	def __init__(self, dataset_path, batch_size=500, instance_weights_path=None):

		L.info("Initializing dataset (with features) from: " + os.path.abspath(dataset_path))

		# Reading parameters from the mmap file
		fp = np.memmap(dataset_path, dtype='int32', mode='r')
		#print type(fp1)
		#fp = np.empty(fp1.shape, dtype='int32')
		#fp[:] = fp1
		#print type(fp)
		self.num_samples = fp[0]
		self.ngram = fp[1]

		fp = fp.reshape((len(fp)/self.ngram, self.ngram))

		num_header_lines = fp[1,0]


		self.features_info = []    # Format (vocab_size, num_of_elements)
		for i in xrange(num_header_lines-1):
			self.features_info.append( (fp[i+2,0], fp[i+2,1]) )


		self.num_classes = fp[(num_header_lines+2)-1,0]


		# Setting minibatch size and number of mini batches
		self.batch_size = batch_size
		self.num_batches = int(M.ceil(self.num_samples / self.batch_size))

		# Reading the matrix of samples
		# x is list
		'''
		self.shared_x_list = []
		last_start_pos = 0
		for i in xrange(len(self.features_info)):
			vocab_size, num_elems = self.features_info[i]
			x = fp[num_header_lines+2:,last_start_pos:last_start_pos + num_elems]			# Reading the context indices
			last_start_pos += num_elems
			shared_x = T.cast(theano.shared(x, borrow=True), 'int32')
			self.shared_x_list.append(shared_x)
		'''
		x = fp[num_header_lines+2:,0:self.ngram - 1]			# Reading the context indices
		self.shared_x = T.cast(theano.shared(x, borrow=True), 'int32')
		y = fp[num_header_lines+2:,self.ngram - 1]			# Reading the output word index
		self.shared_y = T.cast(theano.shared(y, borrow=True), 'int32')


		## Untested instance weighting
		self.is_weighted = False
		if instance_weights_path:
			instance_weights = np.loadtxt(instance_weights_path)
			U.xassert(instance_weights.shape == (self.num_samples,), "The number of lines in weights file must be the same as the number of samples.")
			self.shared_w = T.cast(theano.shared(instance_weights, borrow=True), theano.config.floatX)
			self.is_weighted = True

		L.info('  #samples: %s,  #classes: %s, batch size: %s, #batches: %s' % (
				U.red(self.num_samples),   U.red(self.num_classes), U.red(self.batch_size), U.red(self.num_batches)
			))
		for feature in enumerate(self.features_info):
			L.info("Feature %s: #ngrams= %s vocab_size= %s" %( U.red(feature[0]), U.red(feature[1][1]), U.red(feature[1][0])))




	#### Accessors

	def get_x(self, index):			## Get the  stacked x's
		#return T.concatenate(self.shared_x_list, axis=1)[index * self.batch_size : (index+1) * self.batch_size]
		return self.shared_x[index * self.batch_size : (index+1) * self.batch_size]

	def get_y(self, index):
		return self.shared_y[index * self.batch_size : (index+1) * self.batch_size]

	def get_x_from_list(self, index, feature_index):
		return self.shared_x_list[feature_index][index * self.batch_size : (index+1) * self.batch_size]

	def get_x_list(self, index):
		return self.shared_x_list

	def get_w(self, index):
		return self.shared_w[index * self.batch_size : (index+1) * self.batch_size]

	#### INFO

	def _get_num_samples(self):
		return self.num_samples

	def get_num_batches(self):
		return self.num_batches

	def get_ngram_size(self):
		return self.ngram

	def get_vocab_size(self, feature_index):
		return self.features_info[feature_index][0]

	def get_features_info(self):
		return self.features_info

	def get_num_classes(self):
		return self.num_classes

	def get_unigram_model(self):
		unigram_counts = np.bincount(self.shared_y.get_value())
		unigram_counts = np.append(unigram_counts, np.zeros(self.num_classes - unigram_counts.size, dtype='int32'))
		sum_unigram_counts = np.sum(unigram_counts)

		unigram_model = unigram_counts / sum_unigram_counts
		unigram_model = unigram_model.astype(theano.config.floatX)
		return theano.shared(unigram_model,borrow=True)
