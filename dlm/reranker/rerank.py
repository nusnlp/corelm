#!/usr/bin/env python

import sys
import os
import imp
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()

import dlm.utils as U
import dlm.io.logging as L
from dlm.io.nbestReader import NBestList
import argparse
import codecs
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-nbest", dest="input_nbest", required=True, help="Input n-best file")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", required=True, help="The vocabulary file that was used in training")
parser.add_argument("-m", "--model-file", dest="model_path", required=True, help="Input PrimeLM model file")
parser.add_argument("-w", "--weights", dest="weights", required=True, help="Input weights file")
parser.add_argument("-o", "--output-dir", dest="out_dir", required=True, help="Output directory")
args = parser.parse_args()

U.mkdir_p(args.out_dir)

from dlm.reranker import augmenter

output_nbest_path = args.out_dir + '/augmented.nbest'

augmenter.augment(args.model_path, args.input_nbest, args.vocab_path, output_nbest_path)

with open(args.weights, 'r') as input_weights:
	lines = input_weights.readlines()
	if len(lines) > 1:
		L.warning("Weights file has more than one line. I'll read the 1st and ignore the rest.")
	weights = np.asarray(lines[0].strip().split(" "), dtype=float)

prefix = os.path.basename(args.input_nbest)
input_aug_nbest = NBestList(output_nbest_path, mode='r')
output_nbest = NBestList(args.out_dir + '/' + prefix + '.reranked.nbest', mode='w')
output_1best = codecs.open(args.out_dir + '/' + prefix + '.reranked.1best', mode='w', encoding='UTF-8')

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

counter = 0
for group in input_aug_nbest:
	index = 0
	scores = dict()
	for item in group:
		features = np.asarray([x for x in item.features.split() if is_number(x)], dtype=float)
		try:
			scores[index] = np.dot(features, weights)
		except ValueError:
			L.error('Number of features in the nbest and the weights file are not the same')
		index += 1
	sorted_indices = sorted(scores, key=scores.get, reverse=True)
	for idx in sorted_indices:
		output_nbest.write(group[idx])
	output_1best.write(group[sorted_indices[0]].hyp + "\n")
	counter += 1
	if counter % 100 == 0:
		L.info("%i groups processed" % (counter))
L.info("Finished processing %i groups" % (counter))

output_nbest.close()
output_1best.close()
