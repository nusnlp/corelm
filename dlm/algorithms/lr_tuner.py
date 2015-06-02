from __future__ import division
import theano.tensor as T
import theano
import numpy

class LRTuner:
	def __init__(self, low, high, inc):
		self.low = low
		self.high = high
		self.inc = inc
		self.prev_ppl = numpy.inf
	
	def adapt_lr(self, curr_ppl, curr_lr):
		if curr_ppl >= self.prev_ppl:
			lr = max(curr_lr / 2, self.low)
		else:
			lr = min(curr_lr + self.inc, self.high)
		self.prev_ppl = curr_ppl
		return lr
