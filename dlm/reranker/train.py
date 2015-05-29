#!/usr/bin/env python

import sys
import os
import shutil
import imp
try:
	import dlm
except ImportError:
	print "[ERROR] dlm module not found. Add PrimeLM root directory to your PYTHONPATH"
	sys.exit()

import dlm.utils as U
import dlm.io.logging as L
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-nbest", dest="input_nbest", required=True, help="Input n-best file")
parser.add_argument("-v", "--vocab-file", dest="vocab_path", required=True, help="The vocabulary file that was used in training")
parser.add_argument("-m", "--model-file", dest="model_path", required=True, help="Input PrimeLM model file")
parser.add_argument("-r", "--reference-files", dest="ref_paths", required=True, help="A comma-seperated list of reference files")
parser.add_argument("-c", "--config", dest="input_config", required=True, help="Input moses config (ini) file")
parser.add_argument("-o", "--output-dir", dest="out_dir", required=True, help="Output directory")
parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu)")
parser.add_argument("-t", "--threads", dest="threads", default = 14, type=int, help="Number of MERT threads")
args = parser.parse_args()

U.set_theano_device(args.device)

from dlm.reranker import augmenter

if os.environ.has_key('MOSES_ROOT'):
	moses_root = os.environ['MOSES_ROOT']
else:
	L.error("Set MOSES_ROOT variable to your moses root directory")

U.mkdir_p(args.out_dir)

output_nbest_path = args.out_dir + '/augmented.nbest'

augmenter.augment(args.model_path, args.input_nbest, args.vocab_path, output_nbest_path)

L.info('Extracting stats and features')
#L.warning('The optional arguments of extractor are not used yet')
cmd = moses_root + '/bin/extractor -r ' + args.ref_paths + ' -n ' + output_nbest_path + ' --scfile ' + args.out_dir + '/statscore.data --ffile ' + args.out_dir + '/features.data'
U.capture(cmd)

cmd = moses_root + '/bin/moses -show-weights -f ' + args.input_config + ' 2> /dev/null'
features = U.capture(cmd).strip()

with open(args.out_dir + '/init.opt', 'w') as init_opt:
	init_list = []
	for line in features.split('\n'):
		tokens = line.split(" ")
		try:
			float(tokens[1])
			init_list += tokens[1:]
		except ValueError:
			pass
	dim = len(init_list)
	init_opt.write(' '.join(init_list) + '\n')
	init_opt.write(' '.join(['0' for i in range(dim)]) + '\n')
	init_opt.write(' '.join(['1' for i in range(dim)]) + '\n')
	
# MERT
#L.warning('The optional arguments of mert are not used yet')
L.info('Running MERT')
cmd = moses_root + '/bin/mert -d ' + str(dim) + ' -S ' + args.out_dir + '/statscore.data -F ' + args.out_dir + '/features.data --ifile ' + args.out_dir + '/init.opt --threads ' + str(args.threads)
U.capture(cmd)

U.xassert(os.path.isfile('weights.txt'), 'Optimization failed')

shutil.move('weights.txt', args.out_dir + '/weights.txt')

L.warning('PRO is not supported yet')
