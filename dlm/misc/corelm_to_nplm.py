#!/usr/bin/env python

import numpy as np
import argparse
import os
import dlm.utils as U
import dlm.io.logging as L


def write_matrix(f, matrix):
	for row in matrix:
		f.write(str(row[0]))
		for val in row[1:]:
			f.write("\t"+str(val))
		f.write("\n")

def write_biases(f, biases):
	for bias in biases:
		f.write(str(bias) + "\n")

# Arguments for this script
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--primelm-model", dest="primelm_model", required=True, help="The input NPLM model file")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", required=True, help="The input vocabulary")
parser.add_argument("-dir", "--directory", dest="out_dir", help="The output directory for log file, model, etc.")

args = parser.parse_args()

U.set_theano_device('cpu',1)
from dlm.models.mlp import MLP

if args.out_dir is None:
	args.out_dir = 'primelm_convert-' + U.curr_time()
U.mkdir_p(args.out_dir)

# Loading PrimeLM model and creating classifier class
L.info("Loading PrimeLM model")
classifier = MLP(model_path=args.primelm_model)
args_nn = classifier.args
params_nn = classifier.params
U.xassert(len(params_nn)==7, "PrimeLM model is not compatible with NPLM architecture. 2 hidden layers and an output linear layer is required.")

embeddings = params_nn[0].get_value()
W1 = params_nn[1].get_value()
W1 = np.transpose(W1)
b1 = params_nn[2].get_value()
W2 = params_nn[3].get_value()
W2 = np.transpose(W2)
b2 = params_nn[4].get_value()
W3 = params_nn[5].get_value()
W3 = np.transpose(W3)
b3 = params_nn[6].get_value()


# Storing vocabulary into an array
has_null = False
has_sentence_end = False
vocab_list = []
with open(args.vocab_path,'r') as f_vocab:
	for word in f_vocab:
		word = word.strip()
		vocab_list.append(word)
		if word == "<null>":
			has_null = True
		if word == "</s>":
			has_sentence_end = True

U.xassert(has_sentence_end, "End-of-sentence marker (</s>) has to be present in PrimeLM model.")

# adding null if it is not present
if has_null == False:
	vocab_list.append("<null>")	


# Writing to NPLM model
model_file = args.out_dir + "/" + os.path.basename(args.primelm_model) + ".nplm"
L.info("Writing NPLM Model: " + model_file)
with open(model_file,'w') as f_model:
	
	# Writing the config parameters for the NPLM model
	f_model.write("\config\n")
	f_model.write("version 1\n")
	f_model.write("ngram_size " + str(args_nn.ngram_size) + "\n")
 	if has_null == True:
		f_model.write("input_vocab_size " + str(args_nn.vocab_size)+"\n")
	else:
		f_model.write("input_vocab_size " + str(args_nn.vocab_size + 1)+"\n") # +1 is used to add the <null> token which is not in primelm
	if has_null == True:
		f_model.write("output_vocab_size " + str(args_nn.num_classes)+"\n")
	else:
		f_model.write("output_vocab_size " + str(args_nn.num_classes + 1)+"\n")
	f_model.write("input_embedding_dimension " + str(args_nn.emb_dim) + "\n")
	f_model.write("num_hidden " + args_nn.num_hidden.split(',')[0] + "\n")
	f_model.write("output_embedding_dimension " + args_nn.num_hidden.split(',')[1] + "\n")

	act_func = args_nn.activation_name
	U.xassert(act_func in ['relu','tanh','hardtanh'], "Invalid activation function: " + act_func + " (NPLM supports relu, tanh and hardtanh)")
	if act_func == "relu":
		act_func = "rectifier"
	f_model.write("activation_function " + act_func + "\n")

	f_model.write("\n")
	
	# Writing the input vocabulary
	f_model.write("\input_vocab\n")
	for word in vocab_list:
		f_model.write(word+"\n")

	f_model.write("\n")

	# Writing the output vocabulary ( Currently it is same as input vocabulary)
	f_model.write("\output_vocab\n")
	for word in vocab_list:
		f_model.write(word+"\n")

	f_model.write("\n")

	np.set_printoptions(precision=8, suppress=True)
	rng = np.random.RandomState(1234)
	
	# Writing the input embeddings
	f_model.write("\input_embeddings\n")
	if has_null == False:
		null_row = np.asarray(rng.uniform(low=-0.01, high=0.01, size=(1,embeddings.shape[1])), dtype=embeddings.dtype)
		embeddings = np.append(embeddings, null_row, axis=0)
	write_matrix(f_model, embeddings)

	f_model.write("\n")
	
	# Writing the hidden layer weights and biases
	f_model.write("\hidden_weights 1\n")
	write_matrix(f_model, W1)

	f_model.write("\n")
	f_model.write("\hidden_biases 1\n")
	write_biases(f_model, b1)

	f_model.write("\n")
	f_model.write("\hidden_weights 2\n")
	write_matrix(f_model, W2)

	f_model.write("\n")
	f_model.write("\hidden_biases 2\n")
	write_biases(f_model, b2)

	f_model.write("\n")
	
	# Writing the output linear layer and biases
	f_model.write("\output_weights\n")
	if has_null == False:
		null_row = np.asarray(rng.uniform(low=-0.01, high=0.01, size=(1,W3.shape[1])), dtype=W3.dtype)
		W3 = np.append(W3, null_row, axis=0)
	write_matrix(f_model, W3)
	
	f_model.write("\n")
	f_model.write("\output_biases\n")
	write_biases(f_model, b3)
	if has_null == False:
		f_model.write("0.0\n")
	f_model.write("\n")

	f_model.write("\end\n")







