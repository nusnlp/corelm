from __future__ import division
import dlm.io.logging as L
import dlm.utils as U
import numpy as np
import theano
import theano.tensor as T
import math as M
import sys
import os

class MemMapReader():
	
	#### Constructor
	
	def __init__(self, dataset_path, batch_size=500, instance_weights_path=None):
		
		L.info("Initializing dataset from: " + os.path.abspath(dataset_path))
		
		# Reading parameters from the mmap file
		fp = np.memmap(dataset_path, dtype='int32', mode='r')
		self.num_samples = fp[0]
		self.ngram = fp[1]
		fp = fp.reshape((self.num_samples + 3, self.ngram))
		self.vocab_size = fp[1,0]
		self.num_classes = fp[2,0]

		# Setting minibatch size and number of mini batches
		self.batch_size = batch_size
		self.num_batches = int(M.ceil(self.num_samples / self.batch_size))
		
		# Reading the matrix of samples
		x = fp[3:,0:self.ngram - 1]			# Reading the context indices
		y = fp[3:,self.ngram - 1]			# Reading the output word index
		self.shared_x = T.cast(theano.shared(x, borrow=True), 'int32')
		self.shared_y = T.cast(theano.shared(y, borrow=True), 'int32')
		
		self.is_weighted = False
		if instance_weights_path:
			instance_weights = np.loadtxt(instance_weights_path)
			U.xassert(instance_weights.shape == (self.num_samples,), "The number of lines in weights file must be the same as the number of samples.")
			self.shared_w = T.cast(theano.shared(instance_weights, borrow=True), theano.config.floatX)
			self.is_weighted = True
		
		L.info('  #samples: %s, ngram size: %s, vocab size: %s, #classes: %s, batch size: %s, #batches: %s' % (
				U.red(self.num_samples), U.red(self.ngram), U.red(self.vocab_size), U.red(self.num_classes), U.red(self.batch_size), U.red(self.num_batches)
			)
		)
	
	#### Accessors
	
	def get_x(self, index):
		return self.shared_x[index * self.batch_size : (index+1) * self.batch_size]
	
	def get_y(self, index):
		return self.shared_y[index * self.batch_size : (index+1) * self.batch_size]
	
	def get_w(self, index):
		return self.shared_w[index * self.batch_size : (index+1) * self.batch_size]
	
	#### INFO
	
	def _get_num_samples(self):
		return self.num_samples
	
	def get_num_batches(self):
		return self.num_batches
	
	def get_ngram_size(self):
		return self.ngram
	
	def get_vocab_size(self):
		return self.vocab_size
	
	def get_num_classes(self):
		return self.num_classes

	def get_unigram_model(self):
		unigram_counts = np.bincount(self.shared_y.get_value())
		unigram_counts = np.append(unigram_counts, np.zeros(self.num_classes - unigram_counts.size, dtype='int32'))
		sum_unigram_counts = np.sum(unigram_counts)

		unigram_model = unigram_counts / sum_unigram_counts
		unigram_model = unigram_model.astype(theano.config.floatX)
		return theano.shared(unigram_model,borrow=True)
