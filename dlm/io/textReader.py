import sys
import dlm.utils as U
from dlm.io.vocabReader import VocabManager
from dlm.io.nbestReader import NBestList
import numpy as np
import codecs
import theano
import theano.tensor as T

class TextReader():
	
	def __init__(self, dataset_path, is_nbest, ngram_size, vocab_path):
		
		U.info("Initializing dataset from: " + dataset_path)
		
		vocab = VocabManager(vocab_path)
		
		def get_ngrams(tokens):
			for i in range(ngram_size - 1):
				tokens.insert(0, '<s>')
			indices = vocab.get_ids_given_word_list(tokens)
			return U.get_all_windows(indices, ngram_size)
		
		starts_list = []
		curr_index = 0
		curr_start_index = 0
		self.num_sentences = 0
		
		ngrams_list = []
		if is_nbest == True:
			nbest = NBestList(dataset_path)
			for group in nbest:
				for item in group:
					tokens = item.hyp.split()
					starts_list.append(curr_start_index)
					curr_start_index += len(tokens)
					ngrams_list += get_ngrams(tokens)
		else:
			dataset = codecs.open(dataset_path, 'r', encoding="UTF-8")
			for line in dataset:
				tokens = line.split()
				starts_list.append(curr_start_index)
				curr_start_index += len(tokens)
				ngrams_list += get_ngrams(tokens)
			dataset.close()
		
		self.num_sentences = len(starts_list)
		
		data = np.asarray(ngrams_list)
		starts_list.append(curr_start_index)
		starts_array = np.asarray(starts_list)
		
		x = data[:,0:-1]
		y = data[:,-1]
		
		self.num_samples = y.shape[0]
		
		self.shared_starts = T.cast(theano.shared(starts_array, borrow=True), 'int64')
		self.shared_x = T.cast(theano.shared(x, borrow=True), 'int32')
		self.shared_y = T.cast(theano.shared(y, borrow=True), 'int32')
	
	def get_x(self, index):
		return self.shared_x[ self.shared_starts[index] : self.shared_starts[index+1] ]
	
	def get_y(self, index):
		return self.shared_y[ self.shared_starts[index] : self.shared_starts[index+1] ]

	def get_num_sentences(self):
		return self.num_sentences
	
	def get_num_batches(self):
		return self.num_sentences
	
	def _get_num_samples(self):
		return self.num_samples



