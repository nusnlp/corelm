from __future__ import division
import dlmutils.utils as U
import reader
import numpy as np
import theano
import theano.tensor as T
import math as M

class IRISDatasetReader(reader.Reader):
	
	#### Constructor
	
	def __init__(self, dataset_path, batch_size=10):
		
		data = np.loadtxt(dataset_path, dtype=theano.config.floatX)
		self.batch_size = batch_size
		self.num_samples = data.shape[0]
		self.num_features = data.shape[1] - 1
		features = data[:,0:self.num_features]
		labels = data[:,self.num_features]
		self.num_classes = len(set(labels))
		self.num_batches = int(M.ceil(self.num_samples / batch_size))
		self.shared_x = theano.shared(features)
		self.shared_y = T.cast(theano.shared(labels), 'int32')
	
	#### Accessors
	
	def get_features(self, index):
		return self.shared_x[index * self.batch_size : (index+1) * self.batch_size]
	
	def get_label(self, index):
		return self.shared_y[index * self.batch_size : (index+1) * self.batch_size]
	
	#### INFO
	
	def get_num_batches(self):
		return self.num_batches
	
	def get_num_features(self):
		return self.num_features
	
	def get_num_classes(self):
		return self.num_classes
