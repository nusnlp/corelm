#!/usr/bin/env python

import sys
import imp
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()

import dlmutils.utils as U
import argparse
from dlmutils.reranker.nbestList import NBestList
import dlmutils.reranker.bleu as B
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Input n-best file")
parser.add_argument("-r", "--reference-files", dest="ref_paths", required=True, help="A comma-seperated list of reference files")
parser.add_argument("-o", "--output-nbest-file", dest="out_nbest_path", required=True, help="Output oracle n-best file")
parser.add_argument("-b", "--output-1best-file", dest="out_1best_path", required=True, help="Output oracle 1-best file")
args = parser.parse_args()

ref_path_list = args.ref_paths.split(',')

input_nbest = NBestList(args.input_path, mode='r', reference_list=ref_path_list)
output_nbest = NBestList(args.out_nbest_path, mode='w')
output_1best = codecs.open(args.out_1best_path, mode='w', encoding='UTF-8')

#scorer = B.no_smoothing
#scorer = B.add_epsilon_smoothing
scorer = B.add_one_smoothing

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




sys.exit()


print scorer(
	"it is a guide to action which ensures that the military always obeys the commands of the party",
	[
		"it is a guide to action that ensures that the military will forever heed Party commands",
		"it is the guiding principle which guarantees the military forces always being under the command of the Party",
		"it is the practical guide for the army always to heed the directions of the party"
	]
)
print "-----------------------------------------------------------"
print scorer(
	"it is to insure the troops forever hearing the activity guidebook that party direct",
	[
		"it is a guide to action that ensures that the military will forever heed Party commands",
		"it is the guiding principle which guarantees the military forces always being under the command of the Party",
		"it is the practical guide for the army always to heed the directions of the party"
	]
)
print "-----------------------------------------------------------"
print scorer(
	"the the the the the the the",
	[
		"the cat is on the mat",
		"there is a cat on the mat"
	]
)
