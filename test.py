#!/usr/bin/env python

import sys
import time
import argparse
import dlm.utils as U

###############
## Arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test-file", dest="test_path", required=True, help="The evaluation file (memory-mapped, nbest list or text file)")
parser.add_argument("-f", "--format", dest="format", required=True, help="The evaluation file format (mmap|nbest|text)")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", help="The vocabulary file that was used in training")
parser.add_argument("-m", "--model-file", dest="model_path", required=True, help="Input PrimeLM model file")
parser.add_argument("-lp", "--log-prob-file", dest="lp_path", help="Output file for sentence-level log-probabilities")
parser.add_argument("-ppl", "--perplexity", action='store_true', help="Compute perplexity")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")
args = parser.parse_args()

args.logger = Logger('primelm.test.log')
L = args.logger

U.set_theano_device(args)

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

U.xassert(args.format == "mmap" or args.format == "nbest" or args.format == "text", "Invalid file format given: " + args.format)
U.xassert(args.perplexity or (args.lp_path is not None), "You should use -ppl or -lp (or both)")

if args.format == "mmap":
	U.xassert(args.lp_path is None, "Cannot compute log-probabilities for an mmap file")
	from dlm.io.mmapReader import MemMapReader
	testset = MemMapReader(dataset_path=args.test_path, batch_size=500)
else:
	U.xassert(args.vocab_path, "Vocab file is required for non-mmap file formats")
	from dlm.io.textReader import TextReader
	is_nbest = False
	if args.format == "nbest":
		is_nbest = True
	testset = TextReader(dataset_path=args.test_path, is_nbest=is_nbest, ngram_size=classifier.ngram_size, vocab_path=args.vocab_path)

#########################
## Compiling theano function
#

evaluator = eval.Evaluator(testset, classifier)

#########################
## Testing
#

start_time = time.time()

if args.perplexity:
	L.info("Perplexity: %f" % (evaluator.perplexity()))

if args.lp_path:
	with open(args.lp_path, 'w') as output:
		for i in xrange(testset.get_num_sentences()):
			output.write(str(evaluator.sequence_log_prob(i)) + '\n')

L.info("Ran for %.2fs" % (time.time() - start_time))







