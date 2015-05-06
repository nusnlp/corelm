#!/usr/bin/env python

# INPUT: moses_nbest_list reference_file
# OUTPUT: reranked_nbest_list 1_best_list oracle_bleu

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
from dlmutils.reranker.bleu import BLEU
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Input n-best file")
parser.add_argument("-r", "--reference-file", dest="ref_path", required=True, help="Input reference file")
parser.add_argument("-o", "--output-nbest-file", dest="out_nbest_path", required=True, help="Output oracle n-best file")
parser.add_argument("-b", "--output-1best-file", dest="out_1best_path", required=True, help="Output oracle 1-best file")
args = parser.parse_args()
print args

input_nbest = NBestList(args.input_path, mode='r', reference_path=args.ref_path)
output_nbest = NBestList(args.out_nbest_path, mode='w')

for item in input_nbest:
	print BLEU.no_smoothing(item.hyp, item.ref)
	sys.exit()
	#output_nbest.write(item)
	
#print BLEU.no_smoothing("a b c d e f g", "a b c d e f g h")

output_nbest.close()
