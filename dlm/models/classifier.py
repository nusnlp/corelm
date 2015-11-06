import sys
from theano.misc.pkl_utils import pickle
import gzip
import dlm.io.logging as L

class Classifier:

	def __init__(self):
		self.params = []

	def get_params(self):
		return self.params

	def set_params(self, params):
		for param, loaded_param in zip(self.params, params):
			param.set_value(loaded_param)

	def load_model(self, model_path):
		with open(model_path, 'r') as model_file:
			args, params = pickle.load(model_file)
		return args, params

	def save_model(self, model_path, zipped=True):
		L.info('Saving model to ' + model_path)
		if zipped:
			compress_level = 5
			with gzip.open(model_path, 'wb', compresslevel=compress_level) as model_file:
				params = self.get_params()
				pickle.dump((self.args, [param.get_value() for param in params]), model_file)
		else:
			with open(model_path, 'w') as model_file:
				params = self.get_params()
				pickle.dump((self.args, [param.get_value() for param in params]), model_file)

