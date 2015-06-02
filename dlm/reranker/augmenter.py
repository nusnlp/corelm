import time
import dlm.utils as U
import dlm.io.logging as L
from dlm.models.ltmlp import MLP
from dlm import eval
from dlm.io.nbestReader import NBestList
from dlm.io.vocabReader import VocabManager
import numpy as np

def augment(model_path, input_nbest_path, vocab_path, output_nbest_path):
	L.info('Augmenting the given n-best list')

	classifier = MLP(model_path=model_path)
	evaluator = eval.Evaluator(None, classifier)

	vocab = VocabManager(vocab_path)

	ngram_size = classifier.ngram_size

	def get_ngrams(tokens):
		for i in range(ngram_size - 1):
			tokens.insert(0, '<s>')
		if vocab.has_end_padding:
			tokens.append('</s>')
		indices = vocab.get_ids_given_word_list(tokens)
		return U.get_all_windows(indices, ngram_size)

	input_nbest = NBestList(input_nbest_path, mode='r')
	output_nbest = NBestList(output_nbest_path, mode='w')

	start_time = time.time()

	counter = 0
	cache = dict()
	for group in input_nbest:
		ngram_list = []
		for item in group:
			tokens = item.hyp.split()
			ngrams = get_ngrams(tokens)
			for ngram in ngrams:
				if not cache.has_key(str(ngram)):
					ngram_list.append(ngram)
					cache[str(ngram)] = 1000
		ngram_array = np.asarray(ngram_list, dtype='int32')
		ngram_log_prob_list = evaluator.get_ngram_log_prob(ngram_array[:,0:-1], ngram_array[:,-1])
		for i in range(len(ngram_list)):
			cache[str(ngram_list[i])] = ngram_log_prob_list[i]
		for item in group:
			tokens = item.hyp.split()
			ngrams = get_ngrams(tokens)
			sum_ngram_log_prob = 0
			for ngram in ngrams:
				sum_ngram_log_prob += cache[str(ngram)]
			item.append_feature(sum_ngram_log_prob)
			output_nbest.write(item)
		#print counter
		counter += 1
	output_nbest.close()

	L.info("Ran for %.2fs" % (time.time() - start_time))
