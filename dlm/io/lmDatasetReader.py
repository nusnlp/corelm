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
	
	def __init__(self, dataset_path, batch_size=10):
		
		print "Initializing dataset"
		fp = np.memmap(dataset_path, dtype='int32', mode='r')
		self.num_samples = fp[0]
		self.ngram = fp[1]
		self.num_word_types = fp[2]
		self.batch_size = batch_size
		fp = fp.reshape((self.num_samples + 1, self.ngram))
		features = fp[1:,0:self.ngram - 1]
		labels = fp[1:,self.ngram - 1]
		self.num_batches = int(M.ceil(self.num_samples / batch_size))
		self.shared_x = T.cast(theano.shared(features, borrow=True), theano.config.floatX)
		self.shared_y = T.cast(theano.shared(labels, borrow=True), 'int32')
		print "Dataset initialized"
	
	#### Accessors
	
	def get_features(self, index):
		return self.shared_x[index * self.batch_size : (index+1) * self.batch_size]
	
	def get_label(self, index):
		return self.shared_y[index * self.batch_size : (index+1) * self.batch_size]
	
	#### INFO
	
	def get_num_batches(self):
		return self.num_batches
	
	def get_num_features(self):
		return self.ngram - 1
	
	def get_num_classes(self):
		return self.num_word_types
