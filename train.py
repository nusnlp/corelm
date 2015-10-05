#!/usr/bin/env python

import argparse
import dlm.utils as U
import dlm.io.logging as L

###############
## Arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train-mmap", dest="trainset", required=True, help="The memory-mapped training file")
parser.add_argument("-tu", "--tune-mmap", dest="devset", required=True, help="The memory-mapped development (tune) file")
parser.add_argument("-ts", "--test-mmap", dest="testset", help="The memory-mapped evaluation (test) file")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu). Default: gpu")
parser.add_argument("-E", "--emb-dim", dest="emb_dim", default=50, type=int, help="Word embeddings dimension. Default: 50")
parser.add_argument("-H", "--hidden-units", dest="num_hidden", default="512", help="A comma seperated list for the number of units in each hidden layer. Default: 512")
parser.add_argument("-A", "--activation", dest="activation_name", default="tanh", help="Activation function (tanh|hardtanh|sigmoid|fastsigmoid|hardsigmoid|softplus|relu|cappedrelu). Default: tanh")
parser.add_argument("-a", "--training-algorithm", dest="algorithm", default="sgd", help="The training algorithm (only sgd is supported for now). Default: sgd")
parser.add_argument("-b", "--batch-size", dest="batchsize", default=128, type=int, help="Minibatch size for training. Default: 128")
parser.add_argument("-l", "--learning-rate", dest="learning_rate", default=0.01, type=float, help="Learning rate. Default: 0.01")
parser.add_argument("-D", "--learning-rate-decay", dest="learning_rate_decay", default=0, type=float, help="Learning rate decay (e.g. 0.995) (TO DO). Default: 0")
parser.add_argument("-M", "--momentum", dest="momentum", default=0, type=float, help="Momentum (TO DO). Default: 0")
parser.add_argument("-lf","--loss-function", dest="loss_function", default="nll", help="Loss function (nll|nce). Default: nll (Negative Log Likelihood)")
parser.add_argument("-ns","--noise-samples", dest="num_noise_samples", default=100 ,type=int, help="Number of noise samples for noise contrastive estimation. Default:100")
parser.add_argument("-e", "--num-epochs", dest="num_epochs", default=50, type=int, help="Number of iterations (epochs). Default: 50")
parser.add_argument("-c", "--self-norm-coef", dest="alpha", default=0, type=float, help="Self normalization coefficient (alpha). Default: 0")
parser.add_argument("-L1", "--L1-regularizer", dest="L1_reg", default=0, type=float, help="L1 regularization coefficient. Default: 0")
parser.add_argument("-L2", "--L2-regularizer", dest="L2_reg", default=0, type=float, help="L2 regularization coefficient. Default: 0")
parser.add_argument("-dir", "--directory", dest="out_dir", help="The output directory for log file, model, etc.")
parser.add_argument("--threads", dest="threads", default=8, type=int, help="Number of threads when device is CPU. Default: 8")
#parser.add_argument("-m","--model-file", dest="model_path",  help="The file path to load the model from")

args = parser.parse_args()

if args.out_dir is None:
	args.out_dir = 'primelm-' + U.curr_time()
U.mkdir_p(args.out_dir)

L.set_file_path(args.out_dir + "/log.txt")

U.print_args(args)
U.set_theano_device(args.device, args.threads)

import dlm.trainer
from dlm.io.mmapReader import MemMapReader
from dlm.models.mlp import MLP

#########################
## Loading datasets
#

trainset = MemMapReader(args.trainset, batch_size=args.batchsize)
devset = MemMapReader(args.devset)
testset = None
if args.testset:
	testset = MemMapReader(args.testset)


#########################
## Creating model
#

L.info('Building the model')
args.vocab_size = trainset.get_vocab_size()
args.ngram_size = trainset.get_ngram_size()
args.num_classes = trainset.get_num_classes()

classifier = MLP(args)

#########################
## Training criterion
#
if args.loss_function == "nll":
	from dlm.criterions.nll import NegLogLikelihood
	criterion = NegLogLikelihood(classifier, args)
elif args.loss_function == "nce":
	from dlm.criterions.nce import NCELikelihood
	noise_dist = trainset.get_unigram_model()
	criterion = NCELikelihood(classifier, args, noise_dist)
else:
	L.error('Invalid loss function \'' + args.loss_function + '\'')

#########################
## Training
#

dlm.trainer.train(classifier, criterion, args, trainset, devset, testset)





