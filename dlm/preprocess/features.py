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
import dlm.io.logging as L
import numpy as np

def read_vocab(vocab_path):
	word_to_id_dict = dict()
	found_sent_marker = False
	with open(vocab_path,'r') as f_vocab:
		curr_index = 0
		for line in f_vocab:
			token = line.strip().split()[0]
			U.xassert((not word_to_id_dict.has_key(token)), "Given vocab file has duplicate entry for '" + token + "'.")
			word_to_id_dict[token] = curr_index
			curr_index = curr_index + 1
	return word_to_id_dict		

def replace_unk(word, dict):
	if word in dict:
		return word
	else:
		return "<unk>" 


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", dest="input_path", required=True, help="Path to the input text file, words and features separated by underscorre(_) e.g. word_feature .")
parser.add_argument("-l", "--labels-file", dest="labels_path", required=True, help="Path to the labels text file")
parser.add_argument("-n", "--context", dest="context_size", required=True, type=int, help="Context Size.")
parser.add_argument("-o", "--output-dir", dest="output_dir_path", required=True, help="Path to output directory.")
parser.add_argument("--text", dest="text_output", action='store_true', help="Add this flag to produce text output.")
parser.add_argument("--input-vocab-file", dest="input_vocab_path", help="Path to an input(words) vocabulary file")
parser.add_argument("--labels-vocab-file", dest="labels_vocab_path", help="Path to an labels (POS, NER etc.) vocabulary file")
parser.add_argument("--features-vocab-file", dest="features_vocab_path", help="Path to an features vocabulary file")
parser.add_argument("--shuffle", dest="shuffle", action='store_true', help="Add this flag to shuffle the output.")
parser.add_argument("--word-output", dest="word_out", action='store_true', help="Get output in non-index format, i.e. as words and features")
parser.add_argument("--no-features", dest="no_features", action='store_true', help="Don't include features in the mmap file")

args = parser.parse_args()

if (not os.path.exists(args.output_dir_path)):
	os.makedirs(args.output_dir_path)
print("Output directory: " + os.path.abspath(args.output_dir_path))


prefix = args.output_dir_path + "/" + os.path.basename(args.input_path)

if args.shuffle:
	output_mmap_path = prefix + ".idx.shuf.mmap"
	output_text_path = prefix + ".idx.shuf.txt"
	output_words_path = prefix + ".shuf.txt"

else:
	output_mmap_path = prefix + ".idx.mmap"
	output_text_path = prefix + ".idx.txt"
	output_words_path = prefix + ".txt"

if args.word_out:
	f_words = open(output_words_path, 'w')


input_word_to_id = read_vocab(args.input_vocab_path)
feature_to_id = read_vocab(args.features_vocab_path)
label_to_id = read_vocab(args.labels_vocab_path)
input_vocab_size = len(input_word_to_id)
feature_vocab_size = len(feature_to_id)
label_vocab_size = len(label_to_id)


half_context = args.context_size/2
U.xassert(input_word_to_id.has_key("<s>"), "Sentence marker <s> not found in input vocabulary!")
U.xassert(feature_to_id.has_key("<s>"), "Sentence marker <s> not found in feature vocabulary!")


_, tmp_path = tempfile.mkstemp(prefix='dlm.tmp.')
# For shuffling only
samples = []			# List of samples
samples_idx = []
nsamples = 0


# Read lines and write to the mmap file
line_num=0
nsamples= 0

with open(args.input_path, 'r') as input_file, open(args.labels_path, 'r') as labels_file, open(tmp_path, 'w') as tmp_file:
	next_id = 0
	for line,labels_line in zip(input_file,labels_file):
		line_num += 1			# Increment the line number

		line = line.strip()
		labels_line = labels_line.strip()		# Target labels line
		if len(line) == 0:
			continue

		tokens = line.split()
		ltokens = labels_line.split()
		U.xassert(len(tokens) == len(ltokens), "The number of labels does not match the input sentence does not match in line " + str(line_num) )
		#for i in range(num_markers):
		#	tokens.insert(0, '<s>_<s>')
		#	tokens.append('<s>_<s>')

		indices = []
		f_indices = []
		for token_idx in xrange(len(ltokens)):
			word, feature = tokens[token_idx].split('_')
			label = ltokens[token_idx]
			U.xassert(feature_to_id.has_key(feature), "Feature " + feature + " not present in feature vocab!")
		
			sample = []
			sample_idx = []


			#### Add words to the sample #####
			# Add sentence padding for words if it is at beginning of sentence
			for i in xrange(max(0, half_context - token_idx )):
				sample.append("<s>")
				sample_idx.append(input_word_to_id["<s>"])
			
			sample_words = [replace_unk(token.split('_')[0],input_word_to_id) for token in tokens[max(0, token_idx - half_context): token_idx + half_context + 1]]
			sample 	= sample + sample_words
			sample_idx = sample_idx + [input_word_to_id[word] for word in sample_words]

			for i in xrange(max(0, token_idx + half_context + 1 - len(tokens))):
				sample.append("<s>")
				sample_idx.append(input_word_to_id["<s>"])

			if not args.no_features:
				#### Add features to the sample #####
				# Add sentence padding for features it is at beginning of sentence
				for i in xrange(max(0, half_context - token_idx )):
					sample.append("<s>")
					sample_idx.append(feature_to_id["<s>"])

				sample_features = [token.split('_')[1] for token in tokens[max(0, token_idx - half_context): token_idx + half_context + 1]]
				sample = sample + sample_features
				sample_idx = sample_idx + [feature_to_id[feature] for feature in sample_features]

				for i in xrange(max(0, token_idx + half_context + 1 - len(tokens))):
					sample.append("<s>")			
					sample_idx.append(feature_to_id["<s>"])

			#### Add POS tag to the sample ####
			sample.append(label)
			sample_idx.append(label_to_id[label])

			if args.shuffle:
				samples.append(sample)
				samples_idx.append(sample_idx)
			else:
				tmp_file.write(" ".join([str(idx) for idx in sample_idx]) + "\n")
				if args.word_out:
					f_words.write(" ".join([word for word in sample]) + "\n")
			
			nsamples += 1
			if nsamples % 100000 == 0:
				L.info( str(nsamples) + " samples processed.")


		
			#print word, feature, label

			#if not input_word_to_id.has_key(word):
			#	word = "<unk>"
			#indices.append(str(input_word_to_id[word]))
			#f_indices.append(str(feature_to_id[feature]))
		
# Shuffling the data and writing to tmp file
if args.shuffle:
	L.info("Shuffling data.")
	permutation_arr = np.random.permutation(nsamples)
	with open(tmp_path, 'w') as tmp_file:
		for index in permutation_arr:
			tmp_file.write(" ".join([str(idx) for idx in samples_idx[index]]) + "\n")
			if args.word_out:
				f_words.write(" ".join([word for word in samples[index]]) + "\n")

L.info("Writing to MMap")
# Creating the memory-mapped file

if not args.no_features:
	num_rows = nsamples + 5
	num_cols = args.context_size * 2 + 1
else: 
	num_rows = nsamples + 4
	num_cols = args.context_size + 1

with open(tmp_path, 'r') as data:
	fp = np.memmap(output_mmap_path, dtype='int32', mode='w+', shape=(nsamples + 5, num_cols))
	
	fp[0,0] = nsamples												# number of samples
	fp[0,1] = num_cols												# No. of words + POS tag
	if not args.no_features:
		fp[1,0] = 3 												# No. of header lines
		fp[2,0] = input_vocab_size
		fp[2,1] = args.context_size												
		fp[3,0] = feature_vocab_size
		fp[3,1] = args.context_size												
		fp[4,0] = label_vocab_size
		fp[4,1] = 1
		counter = 5
	else:
		fp[1,0] = 2 												# No. of header lines
		fp[2,0] = input_vocab_size
		fp[2,1] = args.context_size												
		fp[3,0] = label_vocab_size
		fp[4,1] = 1
		counter = 4

	for line in data:
		tokens = line.split()
		fp[counter] = tokens
		counter = counter + 1
		if counter % 100000 == 0:
			L.info(str(counter) + " samples mapped")
	L.info(str(counter-5) + " samples mapped")
	fp.flush
	del fp


shutil.move(tmp_path, output_text_path)

if args.word_out:
	f_words.close()
