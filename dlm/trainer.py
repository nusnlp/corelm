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

	best_dev_perplexity = numpy.inf
	best_test_perplexity = numpy.inf
	best_iter = 0
	epoch = 0
	verbose_freq = 1000 # minibatches
	validation_frequency = 5000 # minibatches
	start_time = time.time()
	proc_time = start_time
	
	U.info('Training')

	while (epoch < args.num_epochs):
		epoch = epoch + 1
		U.info("Epoch: " + U.BColors.RED + str(epoch) + U.BColors.ENDC)
		minibatch_avg_cost_sum = 0
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = trainer.step(minibatch_index)
			minibatch_avg_cost_sum += minibatch_avg_cost
			
			if minibatch_index % verbose_freq == 0:
				#U.info(U.BColors.BLUE + "[" + time.ctime() + "] " + U.BColors.ENDC + str(minibatch_index) + "/" + str(n_train_batches) + ", " + str(minibatch_avg_cost_sum/(minibatch_index+1)))
				U.info(U.BColors.BLUE + "[" + time.ctime() + "] " + U.BColors.ENDC + '%i/%i, %.2f' % (minibatch_index, n_train_batches, minibatch_avg_cost_sum/(minibatch_index+1)))

			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index
			
			if (iter+1) % validation_frequency == 0 or minibatch_index == (n_train_batches-1):
				denominator = dev_eval.get_denominator()
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

				rem_time = int((args.num_epochs * n_train_batches - iter) * (time.time() - proc_time) / (validation_frequency * 60))
				proc_time = time.time()
				
				U.info('DEV =>  Error=%.2f%%, PPL=%.2f (best=%.2f), Denom=%.3f, %im' % (dev_error * 100., dev_perplexity, best_dev_perplexity, denominator, rem_time))
				if args.testset:
					U.info('TEST => Error=%.2f%%, PPL=%.2f (best=%.2f)' % (test_error * 100., test_perplexity, best_test_perplexity))
		
		classifier.save_model(args.model_path + '.epoch' + str(epoch) + '.zip')

	end_time = time.time()
	
	dev_perplexity = dev_eval.perplexity()
	U.info('Final dev perplexity: ' + str(dev_perplexity))

	U.info('Optimization complete')
	U.info('Best dev perplexity: %f at iteration %i' % (best_dev_perplexity, best_iter + 1))
	if args.testset:
		U.info(('Test perplexity at iteration %i: %f') % (best_iter + 1, best_test_perplexity))
	U.info('Ran for %.2fm' % ((end_time - start_time) / 60.))
