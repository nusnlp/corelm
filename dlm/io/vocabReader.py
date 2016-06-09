import codecs
import dlm.utils as U
import dlm.io.logging as L

class VocabManager:
	def __init__(self, input_path, markers=True):
		L.info("Initializing vocabulary from: " + input_path)
		self.word_to_id_dict = dict()
		self.id_to_word_dict = dict()
		curr_id = 0
		with codecs.open(input_path, 'r', encoding='UTF-8') as input_file:
			for line in input_file:
				word = line.strip()
				self.word_to_id_dict[word] = curr_id
				self.id_to_word_dict[curr_id] = word
				curr_id += 1
		if markers == True:
			try:
				self.unk_id = self.word_to_id_dict['<unk>']
				self.padding_id = self.word_to_id_dict['<s>']
			except KeyError:
				L.error("Given vocab file does not include <unk> or <s>")
		self.has_end_padding = self.word_to_id_dict.has_key('</s>')
		
	def get_word_given_id(self, id):
		try:
			return self.id_to_word_dict[id]
		except KeyError:
			raise KeyError
	
	def get_id_given_word(self, word):
		try:
			return self.word_to_id_dict[word]
		except KeyError:
			return self.unk_id
	
	def get_ids_given_word_list(self, word_list):
		output = []
		for word in word_list:
			output.append(self.get_id_given_word(word))
		return output
	
	def get_words_given_id_list(self, id_list):
		output = []
		for id in id_list:
			output.append(self.get_word_given_id(id))
		return output
