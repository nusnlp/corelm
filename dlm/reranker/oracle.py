#!/usr/bin/env python

import sys
import imp
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()

import dlm.utils as U
import argparse
from dlm.io.nbestReader import NBestList
import dlm.reranker.bleu as B
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Input n-best file")
parser.add_argument("-r", "--reference-files", dest="ref_paths", required=True, help="A comma-seperated list of reference files")
parser.add_argument("-o", "--output-nbest-file", dest="out_nbest_path", required=True, help="Output oracle n-best file")
parser.add_argument("-b", "--output-1best-file", dest="out_1best_path", required=True, help="Output oracle 1-best file")
parser.add_argument("-m", "--smoothing-method", dest="method", required=True, help="Smoothing method (none|epsilon|lin|nist|chen)")
args = parser.parse_args()

methods = {
	'none'    : B.no_smoothing,
	'epsilon' : B.add_epsilon_smoothing,
	'lin'     : B.lin_smoothing,
	'nist'    : B.nist_smoothing,
	'chen'    : B.chen_smoothing
}

ref_path_list = args.ref_paths.split(',')

input_nbest = NBestList(args.input_path, mode='r', reference_list=ref_path_list)
output_nbest = NBestList(args.out_nbest_path, mode='w')
output_1best = codecs.open(args.out_1best_path, mode='w', encoding='UTF-8')

U.xassert(methods.has_key(args.method), "Invalid smoothing method: " + args.method)
scorer = methods[args.method]

for group in input_nbest:
	index = 0
	scores = dict()
	for item in group:
		scores[index] = scorer(item.hyp, group.refs)
		index += 1
	sorted_indices = sorted(scores, key=scores.get, reverse=True)
	for idx in sorted_indices:
		output_nbest.write(group[idx])
	output_1best.write(group[sorted_indices[0]].hyp + "\n")

output_nbest.close()
output_1best.close()
