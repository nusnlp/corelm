from theano.misc.pkl_utils import pickle
import theano

class Classifier:

	def __init__(self):
		self.params = []		

	def get_params(self):
		return self.params

	def set_params(self, params):
		for param, loaded_param in zip(self.params, params):
			param.set_value(loaded_param.get_value())

	def load_model(self, model_path):
		with open(model_path, 'r') as model_file:
			params = pickle.load(model_file)
			self.set_params(params)

	def save_model(self, model_path):
		with open(model_path, 'w') as model_file:
			params = self.get_params()
			pickle.dump(params, model_file)

