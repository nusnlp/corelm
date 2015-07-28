#!/usr/bin/env python

import sys
import imp
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()

import dlm.io.logging as L
import dlm.utils as U
import argparse
from dlm.io.nbestReader import NBestList
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--command", dest="command", required=True, help="The command (topN|1best|featureN|correlN|augment)")
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Input n-best file")
parser.add_argument("-s", "--input-scores", dest="oracle", help="Input oracle scores  the n-best file")
parser.add_argument("-o", "--output-file", dest="output_path", required=True, help="Output file")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", help="The vocabulary file.")
parser.add_argument("-m", "--model-file", dest="model_path",  help="Input PrimeLM model file")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")
args = parser.parse_args()

input_nbest = NBestList(args.input_path, mode='r')

mode = -1

if args.command.startswith('top'):
	mode = 0
	N = int(args.command[3:]) # N in N-best
	output_nbest = NBestList(args.output_path, mode='w')
elif args.command == '1best':
	mode = 1
	output_1best = codecs.open(args.output_path, mode='w', encoding='UTF-8')
elif args.command.startswith('feature'):
	mode = 2
	N = int(args.command[7:]) # Nth feature
	output = open(args.output_path, mode='w')
elif args.command.startswith('correl'):
	mode = 3
	N = int(args.command[6:]) # Nth feature
	U.xassert(args.oracle, "correlN command needs a file (-s) containing oracle scores")
	with open(args.oracle, mode='r') as oracles_file:
		oracles = map(float, oracles_file.read().splitlines())
	#output = open(args.output_path, mode='w')
elif args.command.startswith('augment'):
	U.set_theano_device(args.device)
	from dlm.reranker import augmenter
	augmenter.augment(args.model_path, args.input_path, args.vocab_path, args.output_path)
else:
	L.error('Invalid command: ' + args.command)

counter = 0
features = []
for group in input_nbest:
	if mode == 0:
		for i in range(min(N, group.size())):
			output_nbest.write(group[i])
	elif mode == 1:
		output_1best.write(group[0].hyp + "\n")
	elif mode == 2:
		for i in range(group.size()):
			features = group[i].features.split()
			output.write(features[N] + "\n")
	elif mode == 3:
		for i in range(group.size()):
			features.append(float(group[i].features.split()[N]))
	counter += 1
	if counter % 100 == 0:
		L.info("%i groups processed" % (counter))
L.info("Finished processing %i groups" % (counter))

if mode == 0:
	output_nbest.close()
elif mode == 1:
	output_1best.close()
elif mode == 2:
	output.close()
elif mode == 3:
	import scipy.stats as S
	print 'PEARSON: ', S.pearsonr(features, oracles)
	print 'SPEARMAN:', S.spearmanr(features, oracles)





