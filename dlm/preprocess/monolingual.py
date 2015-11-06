#!/usr/bin/env python

import numpy as np
import sys, os
import tempfile
import shutil
import argparse
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()
import dlm.utils as U

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Path to the input text file.")
parser.add_argument("-n", "--ngram-size", dest="ngram_size", required=True, type=int, help="Ngram Size.")
parser.add_argument("-o", "--output-dir", dest="output_dir_path", required=True, help="Path to output directory.")
parser.add_argument("--text", dest="text_output", action='store_true', help="Add this flag to produce text output.")
parser.add_argument("--shuffle", dest="shuffle", action='store_true', help="Add this flag to shuffle the output.")
parser.add_argument("--endp", dest="endp", action='store_true', help="Add this flag to add sentence end padding </s>.")

# Mutually exculsive group of pruning arguments
prune_args = parser.add_mutually_exclusive_group(required=True)
prune_args.add_argument("--prune-vocab-size", dest="prune_vocab_size", type=int, help="Vocabulary size")
prune_args.add_argument("--prune-threshold",  dest="prune_threshold_count", type=int, help="Minimum number of occurances for a word to be added into vocabulary")
prune_args.add_argument("--input-vocab-file", dest="input_vocab_path", help="Path to an existing vocabulary file")

args = parser.parse_args()


if (not os.path.exists(args.output_dir_path)):
	os.makedirs(args.output_dir_path)
print("Output directory: " + os.path.abspath(args.output_dir_path))

prefix = args.output_dir_path + "/" + os.path.basename(args.input_path)

if args.shuffle:
	output_path = prefix + ".idx.shuf.mmap"
	output_text_path = prefix + ".idx.shuf.txt"
else:
	output_path = prefix + ".idx.mmap"
	output_text_path = prefix + ".idx.txt"

word_to_id_dict = dict()			# Word to Index Dictionary

if args.input_vocab_path is None:
	# Counting the frequency of the words.
	word_to_freq_dict = dict()		# Word Frequency Dictionary
	with open(args.input_path, 'r') as input_file:
		for line in input_file:
			line = line.strip()
			if len(line) == 0:
				continue
			tokens = line.split()
			for token in tokens:
				if not word_to_freq_dict.has_key(token):
					word_to_freq_dict[token] = 1
				else:
					word_to_freq_dict[token] += 1
	
	# Prune based on threshold
	if args.prune_threshold_count:
		for token, freq in word_to_freq_dict.items():
			if freq < args.prune_threshold_count:
				del word_to_freq_dict[token]

	# Writing the vocab file and creating a word to id dictionary.
	vocab_path = prefix + ".vocab"
	word_to_id_dict['<unk>'] = 0
	word_to_id_dict['<null>'] = 1
	word_to_id_dict['<s>'] = 2
	added_tokens = '<unk>\n<null>\n<s>\n'
	if args.endp:
		word_to_id_dict['</s>'] = 3
		added_tokens += '</s>\n'
	with open(vocab_path, 'w') as f_vocab:
		curr_index = len(word_to_id_dict)
		f_vocab.write(added_tokens)
		tokens_freq_sorted = sorted(word_to_freq_dict, key=word_to_freq_dict.get, reverse=True)
		if args.prune_vocab_size is not None and args.prune_vocab_size < len(tokens_freq_sorted):
			tokens_freq_sorted = tokens_freq_sorted[0:args.prune_vocab_size]
		for token in tokens_freq_sorted:
			f_vocab.write(token+"\n")
			word_to_id_dict[token] = curr_index
			curr_index = curr_index + 1
else:
	with open(args.input_vocab_path, 'r') as f_vocab:
		curr_index = 0
		for line in f_vocab:
			token = line.strip()
			if not word_to_id_dict.has_key(token):
				word_to_id_dict[token] = curr_index
			curr_index = curr_index + 1
		U.xassert(word_to_id_dict.has_key('<s>') and word_to_id_dict.has_key('<unk>') and word_to_id_dict.has_key('<null>'), "Missing <s> or <unk> or <null> in given vocab file")
		if args.endp:
			U.xassert(word_to_id_dict.has_key('</s>'), "Missing </s> in given vocab file while --endp flag is used")
		if word_to_id_dict.has_key('</s>'):
			U.xassert(args.endp, "Given vocab file has </s> but --endp flag is not activated")

_, tmp_path = tempfile.mkstemp(prefix='dlm.tmp.')

# For shuffling only
samples = []			# List of samples
nsamples = 0

# Reading input text file to create IDX file
with open(args.input_path, 'r') as input_file, open(tmp_path, 'w') as tmp_file:
	next_id = 0
	for line in input_file:
		line = line.strip()
		if len(line) == 0:
			continue
		tokens = line.split()
		for i in range(args.ngram_size - 1):
			tokens.insert(0, '<s>')
		if args.endp:
			tokens.append('</s>')
		indices = []
		for token in tokens:
			if not word_to_id_dict.has_key(token):
				token = "<unk>"
			indices.append(str(word_to_id_dict[token]))
		for i in range(args.ngram_size - 1, len(indices)):
			sample = ' '.join(indices[i - args.ngram_size + 1 : i + 1]) + "\n"
			if args.shuffle:
				samples.append(sample)
			else:
				tmp_file.write(sample)
			nsamples += 1

# Shuffling the data and writing to tmp file
if args.shuffle:
	permutation_arr = np.random.permutation(nsamples)
	with open(tmp_path, 'w') as tmp_file:
		for index in permutation_arr:
			tmp_file.write(samples[index])


# Creating the memory-mapped file
with open(tmp_path, 'r') as data:
	fp = np.memmap(output_path, dtype='int32', mode='w+', shape=(nsamples + 3, args.ngram_size))
	fp[0,0] = nsamples					# number of samples
	fp[0,1] = args.ngram_size			# n-gram size
	fp[1,0] = len(word_to_id_dict)		# vocab size (MLP classes)
	fp[2,0] = len(word_to_id_dict)		# number of word types (MLP classes)
	counter = 3
	for line in data:
		tokens = line.split()
		fp[counter] = tokens
		counter = counter + 1
		if counter % 10000000 == 0:
			print counter
	print str(counter-1) + " samples mapped"
	fp.flush
	del fp

if args.text_output:
	shutil.move(tmp_path, output_text_path)
else:
	os.remove(tmp_path)
