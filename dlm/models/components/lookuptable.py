import sys
import theano
import theano.tensor as T
import numpy
import dlm.utils as U
import dlm.io.logging as L
from dlm.io.vocabReader import VocabManager
from dlm.io.w2vEmbReader import W2VEmbReader

class LookupTable():
	
	def __init__(self, rng, input, vocab_size, emb_dim, emb_matrix=None, concat=True, emb_path=None, vocab_path=None):
		
		L.info("Lookup Table layer, #words: %i, #dims: %i" % (vocab_size, emb_dim))

		self.input = input
		
		self.emb_matrix = emb_matrix

		if self.emb_matrix is None:
			self.emb_matrix = numpy.asarray(
				rng.uniform(
					low=-0.01, #low=-1,
					high=0.01, #high=1,
					size=(vocab_size, emb_dim)
				),
				dtype=theano.config.floatX
			)
		
		if emb_path:
			U.xassert(vocab_path, 'When emb_path is given, vocab must be given too.')
			self.initialize(emb_path, vocab_path)
		
		self.embeddings = theano.shared(value=self.emb_matrix, name='embeddings', borrow=True) # Check if borrowing makes any problems

		if concat:
			self.output = self.embeddings[input].reshape((input.shape[0], emb_dim * input.shape[1]))
		else:
			self.output = self.embeddings[input]

		# parameters of the model
		self.params = [self.embeddings]
	
	
	
	def initialize(self, emb_path, vocab_path):
		L.info('Initializing lookup table')
		vm = VocabManager(vocab_path)
		w2v = W2VEmbReader(emb_path)
		U.xassert(w2v.get_emb_dim() == self.emb_matrix.shape[1], 'The embeddings dimension does not match with the given word embeddings')
		for i in range(self.emb_matrix.shape[0]):
			vec = w2v.get_emb_given_word(vm.get_word_given_id(i))
			if vec:
				self.emb_matrix[i] = vec
