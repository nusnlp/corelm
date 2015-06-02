#!/usr/bin/env python

import sys
import imp
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()

import dlm.io.logging as L
import argparse
from dlm.io.nbestReader import NBestList
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--command", dest="command", required=True, help="The command (topN|1best)")
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Input n-best file")
parser.add_argument("-o", "--output-file", dest="output_path", required=True, help="Output file")
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
else:
	L.error('Invalid command: ' + args.command)

counter = 0
for group in input_nbest:
	if mode == 0:
		for i in range(min(N, group.size())):
			output_nbest.write(group[i])
	elif mode == 1:
		output_1best.write(group[0].hyp + "\n")
	counter += 1
	if counter % 100 == 0:
		L.info("%i groups processed" % (counter))
L.info("Finished processing %i groups" % (counter))

if mode == 0:
	output_nbest.close()
elif mode == 1:
	output_1best.close()
