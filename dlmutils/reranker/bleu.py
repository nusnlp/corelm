from __future__ import division
import sys
import math
import dlmutils.utils as U

###################################################################
## BLEU utility functions
#

def get_ngram_counts(tokens):
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
	
def get_max_ngram_counts(refs_list, hyp_len):
	max_counts = [{}, {}, {}, {}]
	closest_ref_len = 1000
	closest_ref_diff = 1000
	for ref in refs_list:
		ref_tokens = ref.split()
		abs_diff = abs(len(ref_tokens) - hyp_len)
		if abs_diff < closest_ref_diff:
			closest_ref_len = len(ref_tokens)
			closest_ref_diff = abs_diff
		dicts = get_ngram_counts(ref_tokens)
		for k in range(0,4):
			for ngram in dicts[k]:
				if not max_counts[k].has_key(ngram) or max_counts[k][ngram] < dicts[k][ngram]:
					max_counts[k][ngram] = dicts[k][ngram]
	return max_counts, closest_ref_len
	
def clip_ngram_counts(hyp_dicts, ref_dicts):
	for k in range(0,4):
		for ngram in hyp_dicts[k].keys():
			org_count = hyp_dicts[k][ngram]
			if ref_dicts[k].has_key(ngram):
				hyp_dicts[k][ngram] = min(org_count, ref_dicts[k][ngram])
			else:
				hyp_dicts[k][ngram] = 0

###################################################################
## Sentence-level BLEU metrics
#

def no_smoothing(hyp, refs_list):
	l = [0, 0, 0, 0]
	m = [0, 0, 0, 0]
	log_p = [0, 0, 0, 0]
	
	hyp_tokens = hyp.split()
	
	hyp_dicts = get_ngram_counts(hyp_tokens)
	ref_dicts, closest_ref_len = get_max_ngram_counts(refs_list, len(hyp_tokens))
	
	clip_ngram_counts(hyp_dicts, ref_dicts)
	
	sum_log_p = 0
	for k in range(0,4):
		l[k] = max(len(hyp_tokens) - k, 0)
		if l[k] == 0: # sentence length is less than 4
			log_p[k] = 0
		else:
			for w in hyp_dicts[k]:
				if ref_dicts[k].has_key(w):
					m[k] += hyp_dicts[k][w]
			if (m[k] == 0):
				return 0
			else:
				log_p[k] = math.log(m[k]) - math.log(l[k])
		sum_log_p += log_p[k]
	log_brevity = min(0, 1 - closest_ref_len/len(hyp_tokens))
	return math.exp(1/4 * sum_log_p + log_brevity)

###################################################################

def add_epsilon_smoothing(hyp, refs_list, eps=0.01):
	l = [0, 0, 0, 0]
	m = [0, 0, 0, 0]
	log_p = [0, 0, 0, 0]
	
	hyp_tokens = hyp.split()
	
	hyp_dicts = get_ngram_counts(hyp_tokens)
	ref_dicts, closest_ref_len = get_max_ngram_counts(refs_list, len(hyp_tokens))
	
	clip_ngram_counts(hyp_dicts, ref_dicts)
	
	sum_log_p = 0
	for k in range(0,4):
		l[k] = max(len(hyp_tokens) - k, 0)
		if l[k] == 0: # sentence length is less than 4
			log_p[k] = 0
		else:
			for w in hyp_dicts[k]:
				if ref_dicts[k].has_key(w):
					m[k] += hyp_dicts[k][w]
			if (m[k] == 0):
				log_p[k] = math.log(eps) - math.log(l[k])
			else:
				log_p[k] = math.log(m[k]) - math.log(l[k])
		sum_log_p += log_p[k]
	log_brevity = min(0, 1 - closest_ref_len/len(hyp_tokens))
	return math.exp(1/4 * sum_log_p + log_brevity)

###################################################################
	
# Lin and Och, 2004
def add_one_smoothing(hyp, refs_list):
	l = [0, 0, 0, 0]
	m = [0, 1, 1, 1]
	log_p = [0, 0, 0, 0]
	
	hyp_tokens = hyp.split()
	
	hyp_dicts = get_ngram_counts(hyp_tokens)
	ref_dicts, closest_ref_len = get_max_ngram_counts(refs_list, len(hyp_tokens))
	
	clip_ngram_counts(hyp_dicts, ref_dicts)
	
	sum_log_p = 0
	for k in range(0,4):
		l[k] = max(len(hyp_tokens) - k, 0)
		if l[k] == 0: # sentence length is less than 4
			log_p[k] = 0
		else:
			for w in hyp_dicts[k]:
				if ref_dicts[k].has_key(w):
					m[k] += hyp_dicts[k][w]
			if (m[k] == 0): # It can happen when unigram count m[0] is zero
				return 0
			else:
				log_p[k] = math.log(m[k]) - math.log(l[k])
		sum_log_p += log_p[k]
	log_brevity = min(0, 1 - closest_ref_len/len(hyp_tokens))
	return math.exp(1/4 * sum_log_p + log_brevity)
	
	
	
	
	
	
	
	
	

