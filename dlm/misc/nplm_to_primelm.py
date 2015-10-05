#!/usr/bin/env python

import numpy as np
import argparse
import os
import dlm.utils as U
import dlm.io.logging as L


def convert_type(param):
	return np.float32(param)



# Arguments for this script
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--nplm-model", dest="nplm_model", required=True, help="The input NPLM model file")
parser.add_argument("-dir", "--directory", dest="out_dir", help="The output directory for log file, model, etc.")

args = parser.parse_args()

U.set_theano_device('cpu',1)
from dlm.models.mlp import MLP


if args.out_dir is None:
	args.out_dir = 'nplm_convert-' + U.curr_time()
U.mkdir_p(args.out_dir)


# Reading the NPLM Model
args_nn = argparse.Namespace()
model_dict = dict()
lines = []
req_attribs = ['\config','\\vocab', '\input_vocab', '\output_vocab', '\input_embeddings',  '\hidden_weights 1', '\hidden_biases 1', '\hidden_weights 2', '\hidden_biases 2', '\output_weights', '\output_biases','\end']
attrib = ''

with open(args.nplm_model,'r') as f_model:
	for line in f_model:
		line = line.strip()
		if(line in req_attribs):
			if attrib != '':
				model_dict[attrib] = lines
			attrib = line
			lines = []
		elif(line):
			lines.append(line)
		else:
			continue;


# Storing the config parameters of the NPLM model
config_dict = dict()
for config_line in model_dict['\config']:
	config_arg,value = config_line.split()
	config_dict[config_arg] = value


# Setting the args for the classifier
args_nn.emb_dim = int(config_dict['input_embedding_dimension'])
args_nn.num_hidden = config_dict['num_hidden'] + ',' + config_dict['output_embedding_dimension']
args_nn.vocab_size = int(config_dict['input_vocab_size'])
args_nn.ngram_size = int(config_dict['ngram_size'])
args_nn.num_classes = int(config_dict['output_vocab_size'])

act_func = config_dict['activation_function']
if act_func == 'rectifier':
	act_func = 'relu'

args_nn.activation_name = act_func

# Creating the classifier with the arguments read
L.info("Creating PrimeLM model")
classifier = MLP(args_nn)


# Loading matrices
embeddings = np.loadtxt(model_dict['\input_embeddings'])
W1 = np.loadtxt(model_dict['\hidden_weights 1'])
W1 = np.transpose(W1)
b1 = np.loadtxt(model_dict['\hidden_biases 1'])
W2 = np.loadtxt(model_dict['\hidden_weights 2'])
W2 = np.transpose(W2)
b2 = np.loadtxt(model_dict['\hidden_biases 2'])
W3 = np.loadtxt(model_dict['\output_weights'])
W3 = np.transpose(W3)
b3 = np.loadtxt(model_dict['\output_biases'])
params_nn =[embeddings, W1, b1, W2, b2, W3, b3]

#Type Conversion
params_nn = [convert_type(param) for param in params_nn]

# Setting the classifier parameters
classifier.set_params(params_nn)

#Debugging
#print [np.array_equal(x.get_value(),y) for x,y in zip(classifier.params,params_nn)]

# Saving the vocab file
vocab_file = args.out_dir + "/vocab"
if '\input_vocab' in model_dict:
	with open(vocab_file,'w') as f_vocab:
		for word in model_dict['\input_vocab']:
			f_vocab.write(word+'\n')


# Saving the PrimeLM model
model_file = args.out_dir + "/" + os.path.basename(args.nplm_model) + ".primelm"
L.info("Saving PrimeLM model: " + model_file)
classifier.save_model(model_file)

