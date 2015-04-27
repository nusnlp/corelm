#!/usr/bin/env python

from __future__ import division
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import time
import math
import argparse

from dlm.io.lmDatasetReader import LMDatasetReader
from dlm.io.irisDatasetReader import IRISDatasetReader
#from dlm.models.mlp import MLP
from dlm.models.ltmlp import MLP
import dlm.utils as U

def test_mlp(trainset, devset, testset, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100, n_hidden=10):

	n_train_batches = trainset.get_num_batches()
	n_dev_batches = devset.get_num_batches()
	n_test_batches = testset.get_num_batches()

	print '... building the model'

	index = T.lscalar()		# index to a [mini]batch
	x = T.imatrix('x')		# the data is presented as rasterized images
	y = T.ivector('y')		# the labels are presented as 1D vector of
							# [int] labels
	
	rng = numpy.random.RandomState(1234)
	
	# construct the MLP class
	classifier = MLP(
		rng=rng,
		input=x,
		vocab_size=trainset.get_vocab_size(),
		emb_dim=10,
		ngram_size=trainset.get_ngram_size(),
		n_hidden=n_hidden,
		n_out=trainset.get_num_classes()
	)

	cost = (
		classifier.negative_log_likelihood(y)
		#+ L1_reg * classifier.L1
		#+ L2_reg * classifier.L2_sqr
	)

	gparams = [T.grad(cost, param) for param in classifier.params]

	updates = [
		(param, param - learning_rate * gparam)
		for param, gparam in zip(classifier.params, gparams)
	]

	#f = theano.function([index], dataset.get_features(index))

	test_model = theano.function(
		inputs=[index],
		outputs=classifier.errors(y),
		givens={
			x: testset.get_x(index),
			y: testset.get_y(index)
		}
	)
	
	validate_model = theano.function(
		inputs=[index],
		outputs=classifier.errors(y),
		givens={
			x: devset.get_x(index),
			y: devset.get_y(index)
		}
	)

	train_model = theano.function(
		inputs=[index],
		outputs=cost,
		updates=updates,
		givens={
			x: trainset.get_x(index),
			y: trainset.get_y(index)
		}
	)

	dev_negative_log_likelihood = theano.function(
		inputs=[index],
		outputs=classifier.negative_log_likelihood(y),
		givens={
			x: devset.get_x(index),
			y: devset.get_y(index)
		}
	)


	print '... training'

	patience = 10000											# look as this many examples regardless
	patience_increase = 2										# wait this much longer when a new best is found
	improvement_threshold = 0.995								# a relative improvement of this much is considered significant
	validation_frequency = min(n_train_batches, patience / 2)	# go through this many minibatche before
																# checking the network on the validation set;
																# in this case we check every epoch

	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		print "Epoch " + str(epoch)
		minibatch_avg_cost_sum = 0
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			minibatch_avg_cost_sum += minibatch_avg_cost
			if minibatch_index % 1000 == 0:
				print time.ctime() + ", " + str(minibatch_index) + "/" + str(n_train_batches) + ", " + str(minibatch_avg_cost_sum/(minibatch_index+1))
			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i in xrange(n_dev_batches)]
				this_validation_loss = numpy.mean(validation_losses)

				dev_sum_neg_likelihood = sum([dev_negative_log_likelihood(i) for i in xrange(n_dev_batches)])
				dev_perplexity = math.exp(dev_sum_neg_likelihood / n_dev_batches)

				print(
					'epoch %i, minibatch %i/%i, validation error %f %% , perplexity %f' %
					(
						epoch,
						minibatch_index + 1,
						n_train_batches,
						this_validation_loss * 100.,
						dev_perplexity
					)
				)

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if (
						this_validation_loss < best_validation_loss *
						improvement_threshold
					):
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					best_iter = iter

				
				# test it on the test set
				test_losses = [test_model(i) for i in xrange(n_test_batches)]
				test_score = numpy.mean(test_losses)

				print(('epoch %i, minibatch %i/%i, test error of '
					   'best model %f %%') %
					  (epoch, minibatch_index + 1, n_train_batches,
					   test_score * 100.))

			if patience <= iter:
				#done_looping = True
				#break
				pass

	end_time = time.clock()
	print(('Optimization complete. Best validation score of %f %% '
		   'obtained at iteration %i, with test performance %f %%') %
		  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
	if len(sys.argv) != 4:
		print "USAGE: python " + sys.argv[0] + " train_mmap dev_mmap test_mmap"
		sys.exit()
	# Parsing arguments
	parser = argparse.ArgumentParser()

	train_path = sys.argv[1]
	dev_path = sys.argv[2]
	test_path = sys.argv[3]
	
	batch_size = 10
	
	trainset = LMDatasetReader(train_path, batch_size=batch_size)
	devset = LMDatasetReader(dev_path, batch_size=batch_size)
	testset = LMDatasetReader(test_path, batch_size=batch_size)
	
	#dataset = IRISDatasetReader("iris.txt", batch_size=5)
	# check OMP variable

	test_mlp(trainset, devset, testset)
	
	
	
	
