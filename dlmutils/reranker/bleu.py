from __future__ import division
import sys
import math
import dlmutils.utils as U

class BLEU():
		
	@staticmethod
	def get_ngrams(tokens):
		dicts = [{}, {}, {}, {}]
		
		for token in tokens:
			if dicts[0].has_key(token):
				dicts[0][token] += 1
			else:
				dicts[0][token] = 1
		
		for k in range(1,4):
			for i in range(len(tokens) - k):
				segment = ' '.join(tokens[i:i+k+1])
				if dicts[k].has_key(segment):
					dicts[k][segment] += 1
				else:
					dicts[k][segment] = 1
		
		return dicts
	
	@staticmethod
	def no_smoothing(hyp, ref):
		l = [0, 0, 0, 0]
		m = [0, 0, 0, 0]
		log_p = [0, 0, 0, 0]
		
		hyp_tokens = hyp.split()
		ref_tokens = ref.split()
		
		hyp_dicts = BLEU.get_ngrams(hyp_tokens)
		ref_dicts = BLEU.get_ngrams(ref_tokens)
		
		sum_log_p = 0
		for k in range(0,4):
			l[k] = len(hyp_dicts[k])
			for w in hyp_dicts[k]:
				if ref_dicts[k].has_key(w):
					m[k] += hyp_dicts[k][w]
			try:
				log_p[k] = math.log(m[k]) - math.log(l[k])
			except ValueError:
				return 0
			
			sum_log_p += log_p[k]
		
		log_brevity = min(0, 1-len(ref_tokens)/len(hyp_tokens))
		return math.exp(1/4 * sum_log_p + log_brevity)
		
	@staticmethod
	def add_epsilon_smoothing(hyp, ref, eps):
		l = [0, 0, 0, 0]
		m = [0, 0, 0, 0]
		log_p = [0, 0, 0, 0]
		
		hyp_tokens = hyp.split()
		ref_tokens = ref.split()
		
		hyp_dicts = BLEU.get_ngrams(hyp_tokens)
		ref_dicts = BLEU.get_ngrams(ref_tokens)
		
		sum_log_p = 0
		for k in range(0,4):
			l[k] = len(hyp_dicts[k])
			for w in hyp_dicts[k]:
				if ref_dicts[k].has_key(w):
					m[k] += hyp_dicts[k][w]
			try:
				log_p[k] = math.log(m[k]) - math.log(l[k])
			except ValueError:
				m[k] = eps
				log_p[k] = math.log(m[k]) - math.log(l[k])
				
			sum_log_p += log_p[k]
		
		log_brevity = min(0, 1-len(ref_tokens)/len(hyp_tokens))
		return math.exp(1/4 * sum_log_p + log_brevity)
		
	# Lin and Och, 2004
	@staticmethod
	def add_one_smoothing(hyp, ref, eps):
		l = [0, 0, 0, 0]
		m = [0, 1, 1, 1]
		log_p = [0, 0, 0, 0]
		
		hyp_tokens = hyp.split()
		ref_tokens = ref.split()
		
		hyp_dicts = BLEU.get_ngrams(hyp_tokens)
		ref_dicts = BLEU.get_ngrams(ref_tokens)
		
		sum_log_p = 0
		for k in range(0,4):
			l[k] = len(hyp_dicts[k])
			for w in hyp_dicts[k]:
				if ref_dicts[k].has_key(w):
					m[k] += hyp_dicts[k][w]

			log_p[k] = math.log(m[k]) - math.log(l[k])
			sum_log_p += log_p[k]
		
		log_brevity = min(0, 1-len(ref_tokens)/len(hyp_tokens))
		return math.exp(1/4 * sum_log_p + log_brevity)
		
		
		
		
		
		
		
		
		

