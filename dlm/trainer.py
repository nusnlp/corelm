from __future__ import division
import theano
import theano.tensor as T
from dlm import eval
import dlm.utils as U
import time
import numpy
import sys

def train(classifier, criterion, args, trainset, devset, testset=None):
	if args.algorithm == "sgd":
		from dlm.algorithms.sgd import SGD as Trainer
	else:
		U.error("Invalid training algorithm: " + args.algorithm)

	n_train_batches = trainset.get_num_batches()
	
	trainer = Trainer(classifier, criterion, args.learning_rate, trainset)
	
	dev_eval = eval.Evaluator(dataset=devset, classifier=classifier)
	if testset:
		test_eval = eval.Evaluator(dataset=testset, classifier=classifier)

	patience = 10000											# look as this many examples regardless
	patience_increase = 2										# wait this much longer when a new best is found
	improvement_threshold = 0.995								# a relative improvement of this much is considered significant
	validation_frequency = min(n_train_batches, patience / 2)	# go through this many minibatche before
																# checking the network on the validation set;
																# in this case we check every epoch
	best_dev_perplexity = numpy.inf
	best_test_perplexity = numpy.inf
	best_iter = 0

	epoch = 0
	done_looping = False
	
	U.info('Training')
	start_time = time.time()
	verbose_freq = 1000 # minibatches
	proc_time = start_time

	while (epoch < args.num_epochs) and (not done_looping):
		epoch = epoch + 1
		print "Epoch " + str(epoch)
		minibatch_avg_cost_sum = 0
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = trainer.step(minibatch_index)
			minibatch_avg_cost_sum += minibatch_avg_cost
			
			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index
			
			if minibatch_index % verbose_freq == 0:
				print time.ctime() + ", " + str(minibatch_index) + "/" + str(n_train_batches) + ", " + str(minibatch_avg_cost_sum/(minibatch_index+1))

			if (iter + 1) % validation_frequency == 0:
				dev_error = dev_eval.classification_error()
				dev_perplexity = dev_eval.perplexity()
				if args.testset:
					test_error = test_eval.classification_error()
					test_perplexity = test_eval.perplexity()

				# if we got the best validation score until now
				if dev_perplexity < best_dev_perplexity:
					best_dev_perplexity = dev_perplexity
					best_iter = iter
					if args.testset:
						best_test_perplexity = test_perplexity
					
					#improve patience if loss improvement is good enough
					#if (dev_perplexity < best_dev_perplexity * improvement_threshold):
					#	patience = max(patience, iter * patience_increase)

				rem_time = int((args.num_epochs * n_train_batches - iter) * (time.time() - proc_time) / (validation_frequency * 60))
				proc_time = time.time()
				
				print('epoch %i, minibatch %i/%i, dev error %f %%, perplexity %f (best: %f), %im' % (
						epoch,
						minibatch_index + 1,
						n_train_batches,
						dev_error * 100.,
						dev_perplexity,
						best_dev_perplexity,
						rem_time
					)
				)
				
				if args.testset:
					print('epoch %i, minibatch %i/%i, test error %f %%, perplexity %f (best: %f), %im' % (
							epoch,
							minibatch_index + 1,
							n_train_batches,
							test_error * 100.,
							test_perplexity,
							best_test_perplexity,
							rem_time
						)
					)

			if patience <= iter:
				#done_looping = True
				#break
				pass

	end_time = time.time()

	print('Optimization complete')
	print('Best dev perplexity: %f at iteration %i' % (best_dev_perplexity, best_iter + 1))
	if args.testset:
		print(('Test perplexity at iteration %i: %f') % (best_iter + 1, best_test_perplexity))
	print >> sys.stderr, 'Ran for %.2fm' % ((end_time - start_time) / 60.)
