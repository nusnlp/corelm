#!/usr/bin/env python

import numpy as np
import theano
import sys, os
import tempfile
import shutil

if len(sys.argv) != 5:
	# U.usage()
	print "USAGE: python " + sys.argv[0] + " input_txt ngram output_idx output_txt('-' for null)"
	sys.exit()

input_path = sys.argv[1]		# Input text file
ngram = int(sys.argv[2])		# Order of Ngram
output_path = sys.argv[3]		# Binarized output filename of ngram indices
output_path_txt = sys.argv[4]	# Text output filename of ngram indices (optional)
vocab_path = "vocab.txt"		# Vocab path (hard coded temporarily)

nsamples = 0
word_to_id_dict = dict()		# Word to Index Dictionary
word_to_freq_dict = dict()		# Word Frequency Dictionary

_, tmp_path = tempfile.mkstemp(prefix='dlm.tmp.')

with open(input_path, 'r') as input_file, open(tmp_path, 'w') as tmp_file:
	next_id = 0
	for line in input_file:
		line = line.strip()
		if len(line) == 0:
			continue
		tokens = line.strip().split()
		for i in range(ngram - 1):
			tokens.insert(0, '<s>')
		indices = []
		for token in tokens:
			if not word_to_id_dict.has_key(token):
				word_to_id_dict[token] = str(next_id)
				next_id += 1
				word_to_freq_dict[token] = 1 
			else:
				word_to_freq_dict[token] += 1
			indices.append(word_to_id_dict[token])
		for i in range(ngram - 1, len(indices)):
			tmp_file.write(' '.join(indices[i - ngram + 1 : i + 1]) + "\n")
			nsamples += 1

# Writing to the vocabulary file in decreasing order of frequency
f_vocab = open(vocab_path,'w')
for token in sorted(word_to_freq_dict, key=word_to_freq_dict.get, reverse=True):
	f_vocab.write(token+"\n")

with open(tmp_path, 'r') as data:
	fp = np.memmap(output_path, dtype='int32', mode='w+', shape=(nsamples + 1, ngram))
	fp[0,0] = nsamples					# number of samples
	fp[0,1] = ngram						# n-gram size
	fp[0,2] = len(word_to_id_dict)		# number of word types (MLP classes)
	counter = 1
	for line in data:
		tokens = line.split()
		fp[counter] = tokens
		counter = counter + 1
		if counter % 10000000 == 0:
			print counter
	print str(counter-1) + " samples mapped"
	fp.flush
	del fp

if output_path_txt != '-':
	shutil.move(tmp_path, output_path_txt)
else:
	os.remove(tmp_path)
