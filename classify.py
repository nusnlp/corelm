#!/usr/bin/env python

import sys
import time
import argparse
import dlm.utils as U
import dlm.io.logging as L
import numpy

###############
## Arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Input feature file")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", required=True, help="The vocabulary file for the model")
parser.add_argument("-rv", "--restricted-vocab-file", dest="restricted_vocab_path", help="Restricted vocab file to predict the word")
parser.add_argument("-m", "--model-file", dest="model_path", required=True, help="Input PrimeLM model file")
parser.add_argument("-o", "--output-file",dest="output_path", required=True, help="Output File path.")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")
args = parser.parse_args()

U.set_theano_device(args.device,1)

from dlm.models.ltmlp import MLP
from dlm import eval
import theano
import theano.tensor as T

#########################
## Loading model
#

classifier = MLP(model_path=args.model_path)

#########################
## Loading dataset
#

from dlm.io.ngramsReader import NgramsReader
from dlm.io.vocabReader import VocabManager
testset = NgramsReader(dataset_path=args.input_path, ngram_size=classifier.ngram_size, vocab_path=args.vocab_path)
vocab = VocabManager(args.vocab_path)

## Loading restricted vocab
restricted_ids = []
restricted_vocab = []
if args.restricted_vocab_path:
	with open(args.restricted_vocab_path) as restricted_vocab_file:
		for line in restricted_vocab_file:
			restricted_vocab.append(line.strip())
	restricted_ids = vocab.get_ids_given_word_list(restricted_vocab)


#########################
## Compiling theano function
#

evaluator = eval.Evaluator(testset, classifier)


if args.output_path:
	with open(args.output_path, 'w') as output:
		for i in xrange(testset._get_num_samples()):
			out = evaluator.get_class(i, restricted_ids) 
			output.write(vocab.get_word_given_id(out)+'\n')






