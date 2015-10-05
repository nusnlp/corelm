#!/usr/bin/env python

import sys
import imp
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()

import dlm.utils as U
import dlm.io.logging as L
import argparse
from dlm.io.nbestReader import NBestList
import dlm.reranker.bleu as B
import codecs
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Input n-best file")
parser.add_argument("-r", "--reference-files", dest="ref_paths", required=True, help="A comma-seperated list of reference files")
parser.add_argument("-o", "--output-nbest-file", dest="out_nbest_path", help="Output oracle n-best file")
parser.add_argument("-b", "--output-1best-file", dest="out_1best_path", required=True, help="Output oracle 1-best file")
parser.add_argument("-s", "--output-scores", dest="out_scores_path", help="Output oracle scores file")
parser.add_argument("-m", "--smoothing-method", dest="method", required=True, help="Smoothing method (none|epsilon|lin|nist|chen)")
parser.add_argument("-t", "--threads", dest="threads", type=int, default=14, help="Number of threads")
parser.add_argument("-q", "--quiet", dest="quiet", action='store_true', help="Nothing will be printed in STDERR")
args = parser.parse_args()

if args.quiet:
	L.quiet = True

methods = {
	'none'    : B.no_smoothing,
	'epsilon' : B.add_epsilon_smoothing,
	'lin'     : B.lin_smoothing,
	'nist'    : B.nist_smoothing,
	'chen'    : B.chen_smoothing
}

ref_path_list = args.ref_paths.split(',')

input_nbest = NBestList(args.input_path, mode='r', reference_list=ref_path_list)
if args.out_nbest_path:
	output_nbest = NBestList(args.out_nbest_path, mode='w')
if args.out_scores_path:
	output_scores = open(args.out_scores_path, mode='w')
output_1best = codecs.open(args.out_1best_path, mode='w', encoding='UTF-8')

U.xassert(methods.has_key(args.method), "Invalid smoothing method: " + args.method)
scorer = methods[args.method]

L.info('Processing the n-best list')

def process_group(group):
	index = 0
	scores = dict()
	for item in group:
		scores[index] = scorer(item.hyp, group.refs)
		index += 1
	return scores

pool = Pool(args.threads)

counter = 0
group_counter = 0
flag = True
while (flag):
	group_list = []
	for i in range(args.threads):
		try:
			group_list.append(input_nbest.next())
		except StopIteration:
			flag = False
	if len(group_list) > 0:
		outputs = pool.map(process_group, group_list)
		for i in range(len(group_list)):
			scores = outputs[i]
			group = group_list[i]
			sorted_indices = sorted(scores, key=scores.get, reverse=True)
			if args.out_scores_path:
				for idx in scores:
					output_scores.write(str(group.group_index) + ' ' + str(idx) + ' ' + str(scores[idx]) + "\n")
			if args.out_nbest_path:
				for idx in sorted_indices:
					output_nbest.write(group[idx])
			output_1best.write(group[sorted_indices[0]].hyp + "\n")
		counter += 1
		group_counter += len(group_list)
		if counter % 5 == 0:
			L.info("%i groups processed" % (group_counter))
L.info("Finished processing %i groups" % (group_counter))

if args.out_scores_path:
	output_scores.close()
if args.out_nbest_path:
	output_nbest.close()
output_1best.close()
