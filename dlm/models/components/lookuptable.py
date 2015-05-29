import sys
import theano
import theano.tensor as T
import numpy

class LookupTable():
	
	def __init__(self, rng, input, vocab_size, emb_dim, embeddings=None):

		self.input = input

		if embeddings is None:
			emb_matrix = numpy.asarray(
				rng.uniform(
					low=-0.01, #low=-1,
					high=0.01, #high=1,
					size=(vocab_size, emb_dim)
				),
				dtype=theano.config.floatX
			)
			embeddings = theano.shared(value=emb_matrix, name='embeddings', borrow=True) # Check if borrowing makes any problems
		
		self.embeddings = embeddings

		self.output = self.embeddings[input].reshape((input.shape[0], emb_dim * input.shape[1]))

		# parameters of the model
		self.params = [self.embeddings]
