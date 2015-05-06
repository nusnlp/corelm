from __future__ import division
import dlm.utils as U
import reader
import numpy as np
import theano
import theano.tensor as T
import math as M
import sys

class LMDatasetReader(reader.Reader):
	
	#### Constructor
	
	def __init__(self, dataset_path, batch_size=500):
		
		print "Initializing dataset from: " + dataset_path
		
		# Reading parameters from the mmap file
		fp = np.memmap(dataset_path, dtype='int32', mode='r')
		self.num_samples = fp[0]
		self.ngram = fp[1]
		self.num_word_types = fp[2]

		# Setting minibatch size and number of mini batches
		self.batch_size = batch_size
		self.num_batches = int(M.ceil(self.num_samples / self.batch_size))
		
		# Reading the matrix of samples
		fp = fp.reshape((self.num_samples + 1, self.ngram))
		x = fp[1:,0:self.ngram - 1]			# Reading the context indices
		y = fp[1:,self.ngram - 1]			# Reading the output word index
		self.shared_x = T.cast(theano.shared(x, borrow=True), 'int32')
		self.shared_y = T.cast(theano.shared(y, borrow=True), 'int32')
	
	#### Accessors
	
	def get_x(self, index):
		return self.shared_x[index * self.batch_size : (index+1) * self.batch_size]
	
	def get_y(self, index):
		return self.shared_y[index * self.batch_size : (index+1) * self.batch_size]
	
	#### INFO
	
	def get_num_batches(self):
		return self.num_batches
	
	def get_ngram_size(self):
		return self.ngram
	
	def get_vocab_size(self):
		return self.num_word_types
	
	def get_num_classes(self):
		return self.num_word_types
