#!/usr/bin/env python

import numpy as np
import sys, os
import tempfile
import shutil
import argparse
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add CoreLM root directory to your PYTHONPATH"
	sys.exit()
import dlm.utils as U
import dlm.io.logging as L


def process_vocab(input_path, vocab_size, vocab_path, has_null):
	word_to_id_dict = dict()			# Word to Index Dictionary
	word_to_freq_dict = dict()			# Word Frequency Dictionary
	with open(input_path, 'r') as input_file:
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
	
	# Writing the vocab file and creating a word to id dictionary.
	curr_index = 0
	word_to_id_dict['<unk>'] = curr_index
	added_tokens = '<unk>\n'
	curr_index += 1
	if has_null:
		word_to_id_dict['<null>'] = curr_index
		added_tokens += '<null>\n'
		curr_index += 1
	word_to_id_dict['<s>'] = curr_index
	added_tokens += '<s>\n'
	curr_index += 1
	
	if args.endp:
		word_to_id_dict['</s>'] = curr_index
		added_tokens += '</s>\n'
		curr_index += 1
	with open(vocab_path, 'w') as f_vocab:
		f_vocab.write(added_tokens)
		tokens_freq_sorted = sorted(word_to_freq_dict, key=word_to_freq_dict.get, reverse=True)
		if vocab_size < len(tokens_freq_sorted):
			tokens_freq_sorted = tokens_freq_sorted[0:vocab_size]
		for token in tokens_freq_sorted:
			f_vocab.write(token+"\n")
			word_to_id_dict[token] = curr_index
			curr_index = curr_index + 1
	return word_to_id_dict

def read_vocab(vocab_path, endp, has_null):
	word_to_id_dict = dict()
	with open(vocab_path,'r') as f_vocab:
		curr_index = 0
		for line in f_vocab:
			token = line.strip()
			if not word_to_id_dict.has_key(token):
				word_to_id_dict[token] = curr_index
			curr_index = curr_index + 1
		U.xassert(word_to_id_dict.has_key('<s>') and word_to_id_dict.has_key('<unk>'), "Missing <s> or <unk> in given vocab file")
		if has_null:
			U.xassert(word_to_id_dict.has_key('<null>'), "Missing <null> in given target vocab file")
		if endp:
			U.xassert(word_to_id_dict.has_key('</s>'), "Missing </s> in given vocab file while --endp flag is used")
		if word_to_id_dict.has_key('</s>'):
			U.xassert(args.endp, "Given vocab file has </s> but --endp flag is not activated")
	return word_to_id_dict

def replace_unks(tokens, word_to_id_dict):
	replaced_tokens = []
	for token in tokens:
		if not word_to_id_dict.has_key(token):
			token = "<unk>"
		replaced_tokens.append(token)
	return replaced_tokens

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("-is", "--input-source-text", dest="src_input_path", required=True, help="Path to the source langauge training text file")
parser.add_argument("-it", "--input-target-text", dest="trg_input_path", required=True, help="Path to the target language training text file")
parser.add_argument("-ia", "--alignment-file", dest="alignment_path", required=True, help="Alignment file for training text")

parser.add_argument("-cs", "--source-context", dest="src_context", required=True, type=int, help="(Size of source context window - 1)/ 2")
parser.add_argument("-ct", "--target-context", dest="trg_context", required=True, type=int, help="Size of target ngram (including the output)")

parser.add_argument("-o", "--output-dir", dest="output_dir_path", required=True, help="Path to output directory")

parser.add_argument("--shuffle", dest="shuffle", action='store_true', help="Add this flag to shuffle the output")
parser.add_argument("--endp", dest="endp", action='store_true', help="Add this flag to add sentence end padding </s>")
parser.add_argument("--word-output", dest="word_out", action='store_true', help="Get output in non-index format, i.e. as ngrams")

src_prune_args = parser.add_mutually_exclusive_group(required=True)
src_prune_args.add_argument("-vs","--prune-source-vocab", dest="src_vocab_size",  type=int, help="Source vocabulary size")
src_prune_args.add_argument("--source-vocab-file", dest="src_vocab_path",  help="Source vocabulary file path")

trg_prune_args = parser.add_mutually_exclusive_group(required=True)
trg_prune_args.add_argument("-vt","--prune-target-vocab", dest="trg_vocab_size", type=int, help="Target vocabulary size")
trg_prune_args.add_argument("--target-vocab-file", dest="trg_vocab_path", help="Target vocabulary file path")

output_prune_args = parser.add_mutually_exclusive_group(required=True)
output_prune_args.add_argument("-vo","--prune-output-vocab", dest="output_vocab_size", type=int, help="Output vocabulary size. Defaults to target vocabulary size.")
output_prune_args.add_argument("--output-vocab-file", dest="output_vocab_path", help="Output vocabulary file")

args = parser.parse_args()

# Format of the memmap file does not support less than 5 because the first row consists of parameters for the neural network
U.xassert(args.trg_context + args.src_context*2 + 1 > 3, "Total ngram size must be greater than 3. ngrams < 3 are not supported by the current memmap format.")

L.info("Source Window Size: " + str(args.src_context * 2 + 1))
L.info("Target Window Size: " + str(args.trg_context - 1))
L.info("Total Sample Size: " + str(args.trg_context + args.src_context * 2 + 1))

if (args.output_vocab_size is None):
	args.output_vocab_size = args.trg_vocab_size

# The output directory is 
if (not os.path.exists(args.output_dir_path)):
	os.makedirs(args.output_dir_path)
L.info("Output directory: " + os.path.abspath(args.output_dir_path))

# Prefix of files
src_prefix = args.output_dir_path + "/" + os.path.basename(args.src_input_path)
trg_prefix = args.output_dir_path + "/" + os.path.basename(args.trg_input_path)

prefix = os.path.basename(args.src_input_path).split('.')[0]

output_prefix = args.output_dir_path + "/output"

# File paths
if args.shuffle:
	raise NotImplementedError
	output_mmap_path = args.output_dir_path + "/" + prefix + ".idx.shuf.mmap"
	output_idx_path = args.output_dir_path + "/" + prefix + ".idx.shuf.txt"
	output_ngrams_path = args.output_dir_path + "/" + prefix + ".shuf.txt"
else:
	output_mmap_path = args.output_dir_path + "/" + prefix + ".idx.mmap"
	output_idx_path = args.output_dir_path + "/" +  prefix + ".idx.txt"
	output_ngrams_path = args.output_dir_path + "/" + prefix + ".txt"

tune_output_path = "tune.idx.mmap"

if args.src_vocab_path is None:
	src_word_to_id = process_vocab(args.src_input_path, args.src_vocab_size, src_prefix+'.vocab', has_null=False)	# Word to index dictionary of source langauge
else:
	src_word_to_id = read_vocab(args.src_vocab_path,args.endp, has_null=False)

if args.trg_vocab_path is None:
	trg_word_to_id = process_vocab(args.trg_input_path, args.trg_vocab_size, trg_prefix+'.vocab', has_null=True)	# Word to index dictionary of target langauge
else:
	trg_word_to_id = read_vocab(args.trg_vocab_path, args.endp, has_null=True)

if args.output_vocab_path is None:
	output_word_to_id = process_vocab(args.trg_input_path, args.output_vocab_size, output_prefix+'.vocab', has_null=True) # Word to index dictionary of vocab
else:
	output_word_to_id = read_vocab(args.output_vocab_path, args.endp, has_null=True)

svocab = len(src_word_to_id)
tvocab = len(trg_word_to_id)
ovocab = len(output_word_to_id)

## Generating the mmap file
_, tmp_path = tempfile.mkstemp(prefix='dlm.tmp.')

# Word output
if args.word_out:
	f_ngrams = open(output_ngrams_path, 'w')

# For shuffling only
samples = []			# List of samples
samples_idx = []		# For sample indices
nsamples= 0

sentence_count=0

with open(args.src_input_path,'r') as src_file, open(args.trg_input_path, 'r') as trg_file, open(args.alignment_path, 'r') as align_file, open(tmp_path,'w') as tmp_file:
	for sline,tline,aline in zip(src_file,trg_file,align_file):
		stokens = sline[:-1].split()
		ttokens = tline[:-1].split()
		atokens = aline[:-1].split()
		sentence_count += 1
		
		if args.endp:
			stokens.append('</s>')
			ttokens.append('</s>')
		
		stokens = replace_unks(stokens, src_word_to_id)
		otokens = replace_unks(ttokens, output_word_to_id)
		ttokens = replace_unks(ttokens, trg_word_to_id)
		
		trg_aligns = [[] for t in range(len(ttokens))]
		for atoken in atokens:
			sindex,tindex = atoken.split("-")
			sindex,tindex = int(sindex), int(tindex)
			trg_aligns[tindex].append(sindex)
		trg_aligns[-1] = [len(stokens)-1] # Alignment for </s>
				
		for tindex, sindex_list in enumerate(trg_aligns):
			if sindex_list == []: 		# No Alignment for the target token, look at nearby tokens, giving preference to right
				r_tindex = tindex + 1
				l_tindex = tindex - 1
				while r_tindex < len(ttokens) or l_tindex >=0:
					if r_tindex < len(ttokens) and trg_aligns[r_tindex]:
						sindex_list = trg_aligns[r_tindex]
						break
					if l_tindex >= 0 and trg_aligns[l_tindex]:
						sindex_list = trg_aligns[l_tindex]
						break
					r_tindex = r_tindex + 1
					l_tindex = l_tindex - 1

				if sindex_list == []:
					L.error("No alignments in line " + sentence_count)
			
			mid = (len(sindex_list)-1)/2   # Middle of the source alignments
			sindex_align = sorted(sindex_list)[mid]
				
			src_ngrams = []
			trg_ngrams = []
			
			ngram_idx = []
		
			# Get source context
			for i in range(max(0, args.src_context - sindex_align)):
				src_ngrams.append("<s>")
			src_ngrams = src_ngrams + stokens[max(0, sindex_align - args.src_context): sindex_align + args.src_context + 1]
			for i in range(max(0, sindex_align + args.src_context + 1 - len(stokens))):
				src_ngrams.append("</s>")

			# Get target context and predicted word
			for i in range(max(0, args.trg_context - (tindex + 1 ))):
				trg_ngrams.append("<s>")
			trg_ngrams = trg_ngrams +  ttokens[max(0, tindex + 1 - args.trg_context): tindex]

			output_word = otokens[tindex]
			
			sample = " ".join(src_ngrams) + " " + " ".join(trg_ngrams) + " " + output_word + "\n"
			sample_idx = " ".join([str(src_word_to_id[stoken] + tvocab) for stoken in src_ngrams]) 
			sample_idx += " " + " ".join([str(trg_word_to_id[ttoken]) for ttoken in trg_ngrams])
			sample_idx += " " + str(output_word_to_id[output_word]) + "\n"

			if args.shuffle:
				samples.append(sample)
				samples_idx.append(sample_idx)
			else:
				tmp_file.write(sample_idx)
				if args.word_out:
					f_ngrams.write(sample)

			nsamples += 1
			if nsamples % 10000000 == 0:
				L.info( str(nsamples) + " samples processed.")
				
# Shuffling the data and writing to tmp file
if args.shuffle:
	permutation_arr = np.random.permutation(nsamples)
	with open(tmp_path, 'w') as tmp_file:
		for index in permutation_arr:
			tmp_file.write(samples_idx[index])
			if args.word_out:
				f_ngrams.write(samples[index])

ngram_size = args.trg_context + args.src_context * 2 + 1

# Creating the memory-mapped file
with open(tmp_path, 'r') as data:
	fp = np.memmap(output_mmap_path, dtype='int32', mode='w+', shape=(nsamples + 3, ngram_size))
	fp[0,0] = nsamples												# number of samples
	fp[0,1] = ngram_size											# n-gram size
	fp[1,0] = svocab + tvocab										# context vocab size
	fp[2,0] = ovocab												# output vocab size
	counter = 3
	for line in data:
		tokens = line.split()
		fp[counter] = tokens
		counter = counter + 1
		if counter % 10000000 == 0:
			L.info(str(counter) + " samples mapped")
	L.info(str(counter-3) + " samples mapped")
	fp.flush
	del fp

shutil.move(tmp_path, output_idx_path)

if args.word_out:
	f_ngrams.close()
