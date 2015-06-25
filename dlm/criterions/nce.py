import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import numpy as np
import theano

class NCELikelihood():

	def __init__(self, classifier, args, noise_dist):
		theano.config.exception_verbosity = 'high'	
		self.y = T.ivector('y')
		
		## Generating noise samples 
		srng = RandomStreams(seed=1234)  
		#args.num_noise_samples = 1
		#noise_samples = T.nonzero(srng.multinomial(size=(args.num_noise_samples,y.shape[0]), pvals=noise_dist))[1]
		noise_samples = T.nonzero(srng.multinomial(size=(self.y.shape[0],args.num_noise_samples), pvals=noise_dist))[2].reshape((self.y.shape[0],args.num_noise_samples))
		
		self.noise_dist = noise_dist
		self.batchsize = args.batchsize
		self.num_noise_samples = args.num_noise_samples
		self.noise_samples = np.zeros((args.batchsize,args.num_noise_samples))
		
		## Cost function
		#  Sum over minibatch instances (log ( u(w|c) / (u(w|c) + k * p_n(w)) ) + sum over noise samples ( log ( u(x|c) / ( u(x|c) + k * p_n(x) ) )))

		data_scores = T.exp(classifier.output[T.arange(self.y.shape[0]),self.y]) # u(w|c) for all w in y (Shape: 1 x #instances)
		log_data = T.log(data_scores / ( data_scores + args.num_noise_samples * noise_dist[self.y] )) # log ( u(w|c) / (u(w|c) + k * p_n(w))
		
		noise_prob = noise_dist[noise_samples]									   # p_n(x) for all noise samples (Shape: #instances x k)
		noise_mass = args.num_noise_samples * noise_prob								   # k * p_n(x) for all noise samples (Shape: #instance x k)
		noise_scores = 	T.exp(classifier.output[T.arange(noise_samples.shape[0]).reshape((-1,1)),noise_samples]) # u(x|c) (Shape: #instances x k)
		noise_log_sum = T.sum( T.log(noise_mass / (noise_scores + noise_mass)) , axis=1) #Sum(log ( u(x|c) /(u(x|c) + k * p_n(x))) (Shape: 1 x #instances)
		
		self.cost = (
			T.mean(log_data + noise_log_sum) 
	
		)
	
	def generate_noise(self):
		self.noise_samples = np.nonzero(np.random.multinomial(1,self.noise_dist.get_value(), size=(self.batchsize,self.num_noise_samples)))[2].reshape(self.batchsize,self.num_noise_samples)



