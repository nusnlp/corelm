import sys
import theano
import theano.tensor as T
import numpy

class LookupTable():
	
	def __init__(self, rng, input, vocab_size, emb_dim, emb_matrix=None, concat=True):

		self.input = input

		if emb_matrix is None:
			emb_matrix = numpy.asarray(
				rng.uniform(
					low=-0.01, #low=-1,
					high=0.01, #high=1,
					size=(vocab_size, emb_dim)
				),
				dtype=theano.config.floatX
			)
		self.emb_matrix = emb_matrix
		
		self.embeddings = theano.shared(value=self.emb_matrix, name='embeddings', borrow=True) # Check if borrowing makes any problems

		if concat:
			self.output = self.embeddings[input].reshape((input.shape[0], emb_dim * input.shape[1]))
		else:
			self.output = self.embeddings[input]

		# parameters of the model
		self.params = [self.embeddings]
