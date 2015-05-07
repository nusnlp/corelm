#!/usr/bin/env python

import os
import numpy
import argparse
import dlmutils.utils as U

###############
## Arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train-mmap", dest="trainset", required=True, help="The memory-mapped training file")
parser.add_argument("-tu", "--tune-mmap", dest="devset", required=True, help="The memory-mapped development (tune) file")
parser.add_argument("-ts", "--test-mmap", dest="testset", help="The memory-mapped evaluation (test) file")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")
parser.add_argument("-e", "--emb-dim", dest="emb_dim", default=50, type=int, help="Word embeddings dimension")
parser.add_argument("-h1", "--h1-size", dest="num_hidden_1", default=512, type=int, help="Number of units in the 1st hidden layer")
parser.add_argument("-h2", "--h2-size", dest="num_hidden_2", default=0, type=int, help="Number of units in the 2nd hidden layer")
parser.add_argument("-alg", "--training-algorithm", dest="algorithm", default="sgd", help="The training algorithm (only sgd is supported for now)")
parser.add_argument("-b", "--batch-size", dest="batchsize", default=128, type=int, help="Minibatch size for training")
parser.add_argument("-l", "--learning-rate", dest="learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("-i", "--num-epochs", dest="num_epochs", default=50, type=int, help="Number of iterations (epochs)")
parser.add_argument("-L1", "--L1-regularizer", dest="L1_reg", default=0, type=float, help="L1 regularization coefficient")
parser.add_argument("-L2", "--L2-regularizer", dest="L2_reg", default=0, type=float, help="L2 regularization coefficient")
args = parser.parse_args()

U.xassert(args.device == "cpu" or args.device == "gpu", "The device can only be 'cpu' or 'gpu'")

os.environ['THEANO_FLAGS'] = 'device=' + args.device
os.environ['THEANO_FLAGS'] += ',force_device=True'
os.environ['THEANO_FLAGS'] += ',floatX=float32'
os.environ['THEANO_FLAGS'] += ',print_active_device=False'
os.environ['THEANO_FLAGS'] += ',mode=FAST_RUN'
os.environ['THEANO_FLAGS'] += ',nvcc.fastmath=True' # makes div and sqrt faster at the cost of precision

try:
	import theano
except EnvironmentError:
	U.exception()
import theano.tensor as T
import dlm.trainer
from dlm.io.lmDatasetReader import LMDatasetReader
from dlm.criterions.likelihood import NegLogLikelihood
from dlm.models.ltmlp import MLP

if theano.config.device == "gpu":
	U.info(
		"Device: " + theano.config.device.upper() + " "
		+ str(theano.sandbox.cuda.active_device_number())
		+ " (" + str(theano.sandbox.cuda.active_device_name()) + ")"
	)
else:
	U.info("Device: " + theano.config.device.upper())

#########################
## Loading datasets
#

trainset = LMDatasetReader(args.trainset, batch_size=args.batchsize)
devset = LMDatasetReader(args.devset)
testset = None
if args.testset:
	testset = LMDatasetReader(args.testset)

#########################
## Creating model
#

U.info('Building the model')

x = T.imatrix('x')		# the data is presented as rasterized images
rng = numpy.random.RandomState(1234)

classifier = MLP(
	rng=rng,
	input=x,
	vocab_size=trainset.get_vocab_size(),
	emb_dim=args.emb_dim,
	ngram_size=trainset.get_ngram_size(),
	num_hidden=args.num_hidden_1,
	n_out=trainset.get_num_classes()
)

#########################
## Training criterion
#

criterion = NegLogLikelihood(classifier, args)

#########################
## Training
#

dlm.trainer.train(classifier, criterion, args, trainset, devset, testset)








