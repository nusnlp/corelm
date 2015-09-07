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
parser.add_argument("-iv", "--init-value", dest="init_value", default = '0.05', help="The initial value of the feature")
parser.add_argument("-n", "--no-aug", dest="no_aug", action='store_true', help="Augmentation will be skipped, if this flag is set")
parser.add_argument("-a", "--tuning-algorithm", dest="alg", default = 'mert', help="Tuning Algorithm (mert|pro|wpro)")
parser.add_argument("-w", "--instance-weights", dest="instance_weights_path", help="Instance weights for wpro algorithm")
parser.add_argument("-s", "--predictable-seed", dest="pred_seed", action='store_true', help="Tune with predictable seed to avoid randomness")
args = parser.parse_args()

U.set_theano_device(args.device)

from dlm.reranker import augmenter
from dlm.reranker import mosesIniReader as iniReader

if os.environ.has_key('MOSES_ROOT'):
	moses_root = os.environ['MOSES_ROOT']
else:
	L.error("Set MOSES_ROOT variable to your moses root directory")

U.mkdir_p(args.out_dir)

#cmd = moses_root + '/bin/moses -show-weights -f ' + args.input_config + ' 2> /dev/null'
#features = U.capture(cmd).strip().split('\n')
features = iniReader.parseIni(args.input_config)

output_nbest_path = args.out_dir + '/augmented.nbest'

if args.no_aug:
	shutil.copy(args.input_nbest, output_nbest_path)
else:
	augmenter.augment(args.model_path, args.input_nbest, args.vocab_path, output_nbest_path)

L.info('Extracting stats and features')
#L.warning('The optional arguments of extractor are not used yet')
cmd = moses_root + '/bin/extractor -r ' + args.ref_paths + ' -n ' + output_nbest_path + ' --scfile ' + args.out_dir + '/statscore.data --ffile ' + args.out_dir + '/features.data'
U.capture(cmd)

with open(args.out_dir + '/init.opt', 'w') as init_opt:
	init_list = []
	for line in features:
		tokens = line.split(" ")
		try:
			float(tokens[1])
			init_list += tokens[1:]
		except ValueError:
			pass
	if not args.no_aug:
		init_list.append(args.init_value)
	dim = len(init_list)
	init_opt.write(' '.join(init_list) + '\n')
	init_opt.write(' '.join(['0' for i in range(dim)]) + '\n')
	init_opt.write(' '.join(['1' for i in range(dim)]) + '\n')

seed_arg = ''
if args.pred_seed:
	seed_arg = ' -r 1234 '

if (args.alg == 'pro' or args.alg == 'wpro'):
	# PRO
	if args.alg == 'pro':
		L.info("Running PRO")
		cmd = moses_root + '/bin/pro' + ' -S ' + args.out_dir + '/statscore.data -F ' + args.out_dir + '/features.data -o ' + args.out_dir +'/pro.data' + seed_arg
	else:
		L.info("Running WEIGHTED PRO")
		U.xassert(args.instance_weights_path, 'Instance weights are not given to wpro')
		cmd = moses_root + '/bin/proWeighted' + ' -S ' + args.out_dir + '/statscore.data -F ' + args.out_dir + '/features.data -o ' + args.out_dir +'/pro.data' + seed_arg + ' -w ' + args.instance_weights_path
	U.capture(cmd)
	cmd = moses_root + '/bin/megam_i686.opt -fvals -maxi 30 -nobias binary ' + args.out_dir + '/pro.data'
	pro_weights = U.capture(cmd)

	pro_weights_arr = pro_weights.strip().split('\n')
	weights_dict = dict()
	sum = 0.0
	highest_feature_index = 0

	for elem in pro_weights_arr:
		feature_index,weight = elem[1:].split()
		feature_index = int(feature_index)
		weight = float(weight)
		weights_dict[feature_index] = weight
		sum = sum + weight
		if feature_index >= highest_feature_index:
			highest_feature_index = feature_index

	# Write normalized weights to the file
	f_weights = open('weights.txt','w')
	for feature_index in xrange(highest_feature_index+1):
		weight = weights_dict[feature_index]
		f_weights.write(str(weight/sum) + ' ');
		#f_weights.write(str(weight) + ' ');
elif (args.alg == 'mert'):
	# MERT
	#L.warning('The optional arguments of mert are not used yet')
	L.info('Running MERT')
	cmd = moses_root + '/bin/mert -d ' + str(dim) + ' -S ' + args.out_dir + '/statscore.data -F ' + args.out_dir + '/features.data --ifile ' + args.out_dir + '/init.opt --threads ' + str(args.threads) + seed_arg
	U.capture(cmd)
else:
	L.error('Invalid tuning algorithm: ' + args.alg)

U.xassert(os.path.isfile('weights.txt'), 'Optimization failed')

shutil.move('weights.txt', args.out_dir + '/weights.txt')

