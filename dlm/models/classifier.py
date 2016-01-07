import sys
from theano.misc.pkl_utils import pickle
import gzip
import dlm.io.logging as L
import time
import dlm.utils as U

class Classifier:

	def __init__(self):
		self.params = []

	def get_params(self):
		return self.params

	def set_params(self, params):
		U.xassert(len(self.params) == len(params), 'The given model file is consistent with the architecture')
		for param, loaded_param in zip(self.params, params):
			param.set_value(loaded_param)

	def load_model(self, model_path):
		L.info('Loading model from ' + model_path)
		t0 = time.time()
		if model_path.endswith('.gz'):
			with gzip.open(model_path, 'rb') as model_file:
				args, params = pickle.load(model_file)
		else:
			with open(model_path, 'r') as model_file:
				args, params = pickle.load(model_file)
		L.info('  |-> took %.2f seconds' % (time.time() - t0))
		return args, params

	def save_model(self, model_path, zipped=True, compress_level=5):
		L.info('Saving model to ' + model_path)
		t0 = time.time()
		if zipped:
			with gzip.open(model_path, 'wb', compresslevel=compress_level) as model_file:
				params = self.get_params()
				pickle.dump((self.args, [param.get_value() for param in params]), model_file)
		else:
			with open(model_path, 'w') as model_file:
				params = self.get_params()
				pickle.dump((self.args, [param.get_value() for param in params]), model_file)
		L.info('  |-> took %.2f seconds' % (time.time() - t0))
