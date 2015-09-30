import sys
import dlm.utils as U
import dlm.io.logging as L
from dlm.io.vocabReader import VocabManager
from dlm.io.nbestReader import NBestList
import numpy as np
import codecs
import theano
import theano.tensor as T

class NgramsReader():

	def __init__(self, dataset_path, ngram_size, vocab_path):

		L.info("Initializing dataset from: " + dataset_path)

		vocab = VocabManager(vocab_path)

		curr_index = 0
		self.num_sentences = 0

		ngrams_list = []
		dataset = codecs.open(dataset_path, 'r', encoding="UTF-8")
		for line in dataset:
			tokens = line.split()
			ngrams = vocab.get_ids_given_word_list(tokens)
			ngrams_list.append(ngrams)
			curr_index += 1
		dataset.close()

		data = np.asarray(ngrams_list)

		x = data[:,0:-1]
		y = data[:,-1]
		self.num_samples = y.shape[0]

		self.shared_x = T.cast(theano.shared(x, borrow=True), 'int32')
		self.shared_y = T.cast(theano.shared(y, borrow=True), 'int32')

	def get_x(self, index):
		return self.shared_x[ index : index+1 ]

	def get_y(self, index):
		return self.shared_y[ index : index+1 ]

	def get_num_batches(self):
		return self.num_samples

	def _get_num_samples(self):
		return self.num_samples




