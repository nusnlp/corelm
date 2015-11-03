#!/usr/bin/env python

import numpy as np
import sys, os
import argparse

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-idx-file", dest="idx_path", required=True, help="Path to the input text (idx) file.")
parser.add_argument("-v", "--input-vocab-file", dest="vocab_path", help="Path to the vocab file.")
parser.add_argument("-o", "--output-file", dest="output_path", required=True, help="Path to output file.")
parser.add_argument("-n", "--no-header", dest="no_header", action='store_true', help="Use this flag to write a plain mmap file with no header information.")

args = parser.parse_args()

if not args.no_header:
	assert args.vocab_path, "[ERROR] Give a vocab file or use --no-header flag."

def verify_matrix_file(matrix_path):
	print "Verifying the input file"
	nrows = 0
	ncols = -1
	with open(matrix_path, 'r') as data:
		for line in data:
			tokens = line.split()
			if ncols <= 0:
				ncols = len(tokens)
			else:
				assert ncols == len(tokens)
			try:
				map(int, tokens)
			except ValueError:
				print "[ERROR] Matrix file format invalid @ line: " + line
				sys.exit()
			nrows += 1
			if nrows % 10000000 == 0:
				print nrows
	assert nrows > 0 and ncols > 0
	return nrows, ncols



if args.no_header:
	nrows, ncols = verify_matrix_file(args.idx_path)

	print "Number of rows: ", nrows
	print "Number of columns: ", ncols

	print "Creating the memory mapped file"
	print("Output file: " + os.path.abspath(args.output_path))

	with open(args.idx_path, 'r') as data:
		fp = np.memmap(args.output_path, dtype='int32', mode='w+', shape=(nrows, ncols))
		counter = 0
		for line in data:
			tokens = line.split()
			fp[counter] = tokens
			counter = counter + 1
			if counter % 10000000 == 0:
				print counter
		print counter, "samples mapped"
		fp.flush
		del fp
else:
	print "Reading the vocab file"

	vocab_size = 0
	with open(args.vocab_path, 'r') as vocab_file:
		for line in vocab_file:
			vocab_size += 1
	assert vocab_size > 0

	num_samples, ngram_size = verify_matrix_file(args.idx_path)

	print "Number of samples: ", num_samples
	print "Ngram size: ", ngram_size
	print "Vocab size: ", vocab_size

	print "Creating the memory mapped file"
	print("Output file: " + os.path.abspath(args.output_path))

	with open(args.idx_path, 'r') as data:
		fp = np.memmap(args.output_path, dtype='int32', mode='w+', shape=(num_samples + 3, ngram_size))
		fp[0,0] = num_samples
		fp[0,1] = ngram_size
		fp[1,0] = vocab_size		# vocab size
		fp[2,0] = vocab_size		# number of classes
		counter = 3
		for line in data:
			tokens = line.split()
			fp[counter] = tokens
			counter = counter + 1
			if counter % 10000000 == 0:
				print counter
		print str(counter - 3) + " samples mapped"
		fp.flush
		del fp
