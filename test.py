#!/usr/bin/env python

import sys
import time
import argparse
import dlm.utils as U
import dlm.io.logging as L
from dlm.io.vocabReader import VocabManager
###############
## Arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test-file", dest="test_path", required=True, help="The evaluation file (memory-mapped, nbest list or text file)")
parser.add_argument("-f", "--format", dest="format", required=True, help="The evaluation file format (fmmap|mmap|nbest|text)")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", help="The vocabulary file that was used in training")
parser.add_argument("-m", "--model-file", dest="model_path", required=True, help="Input CoreLM model file")
parser.add_argument("-ulp", "--unnormalized-log-prob-file", dest="ulp_path", help="Output file for sentence-level UNNORMALIZED log-probabilities")
parser.add_argument("-nlp", "--normalized-log-prob-file", dest="nlp_path", help="Output file for sentence-level NORMALIZED log-probabilities")
parser.add_argument("-ppl", "--perplexity", action='store_true', help="Compute perplexity")
parser.add_argument("-op", "--output_path", dest="out_path",  help="Output classes path")
parser.add_argument("-un", "--unnormalized", action='store_true', help="Output need not be normalized")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")

args = parser.parse_args()

U.set_theano_device(args.device, 1)

from dlm.models.mlp import MLP
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

U.xassert(args.format == "mmap" or args.format == "nbest" or args.format == "text" or args.format == "fmmap", "Invalid file format given: " + args.format)
U.xassert(args.perplexity or args.nlp_path or args.ulp_path, "You should use one of (or more) -ppl, -nlp or -ulp")

if args.format == "mmap":
	U.xassert((args.nlp_path is None) and (args.ulp_path is None), "Cannot compute log-probabilities for an mmap file")
	from dlm.io.mmapReader import MemMapReader
	testset = MemMapReader(dataset_path=args.test_path, batch_size=500)
elif args.format == "fmmap":
	U.xassert((args.nlp_path is None) and (args.ulp_path is None), "Cannot compute log-probabilities for an features mmap file")
	from dlm.io.featuresmmapReader import FeaturesMemMapReader
	testset = FeaturesMemMapReader(dataset_path=args.test_path, batch_size=500)

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
	if args.unnormalized:
		L.info("Unnormalized Perplexity: %f" % (evaluator.unnormalized_perplexity()))

if args.nlp_path:
	with open(args.nlp_path, 'w') as output:
		for i in xrange(testset.get_num_sentences()):
			output.write(str(evaluator.get_sequence_log_prob(i)) + '\n')


if args.ulp_path:
	with open(args.ulp_path, 'w') as output:
		for i in xrange(testset.get_num_sentences()):
			output.write(str(evaluator.get_unnormalized_sequence_log_prob(i)) + '\n')

if args.out_path:
	with open(args.out_path, 'w') as output:
		for i in xrange(testset.get_num_batches()):
			batch_labels = evaluator.get_batch_predicted_class(i)
			for label in batch_labels:
				output.write(str(label) + '\n')


L.info("Ran for %.2fs" % (time.time() - start_time))







