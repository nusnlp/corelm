import sys
import dlm.utils as U
import dlm.io.logging as L
import codecs

class NBestList():
	def __init__(self, nbest_path, mode='r', reference_list=None):
		U.xassert(mode == 'r' or mode == 'w', "Invalid mode: " + mode)
		self.mode = mode
		self.nbest_file = codecs.open(nbest_path, mode=mode, encoding='UTF-8')
		self.prev_index = -1
		self.curr_item = None
		self.curr_index = 0
		self.eof_flag = False
		self.ref_manager = None
		if reference_list:
			U.xassert(mode == 'r', "Cannot accept a reference_list in 'w' mode")
			self.ref_manager = RefernceManager(reference_list)

	def __iter__(self):
		U.xassert(self.mode == 'r', "Iteration can only be done in 'r' mode")
		return self
	
	def next_item(self):
		U.xassert(self.mode == 'r', "next() method can only be used in 'r' mode")
		try:
			segments = self.nbest_file.next().split("|||")
		except StopIteration:
			self.close()
			raise StopIteration
		try:
			index = int(segments[0])
		except ValueError:
			L.error("The first segment in an n-best list must be an integer")
		hyp = segments[1].strip()
		features = segments[2].strip()
		score = segments[3].strip()
		phrase_alignments = segments[4].strip()
		word_alignments = segments[5].strip()
		return NBestItem(index, hyp, features, score, phrase_alignments, word_alignments)
	
	def next(self): # Returns a group of NBestItems with the same index
		if self.eof_flag == True:
			raise StopIteration
		U.xassert(self.mode == 'r', "next_group() method can only be used in 'r' mode")
		group = NBestGroup(self.ref_manager)
		group.add(self.curr_item) # add the item that was read in the last next() call
		try:
			self.curr_item = self.next_item()
		except StopIteration:
			self.eof_flag = True
			return group
		if self.curr_index != self.curr_item.index:
			self.curr_index = self.curr_item.index
			return group
		while self.curr_index == self.curr_item.index:
			group.add(self.curr_item)
			try:
				self.curr_item = self.next_item()
			except StopIteration:
				self.eof_flag = True
				return group
		self.curr_index = self.curr_item.index
		return group
	
	def write(self, item):
		U.xassert(self.mode == 'w', "write() method can only be used in 'w' mode")
		self.nbest_file.write(unicode(item) + "\n")
	
	def close(self):
		self.nbest_file.close()



class NBestItem:
	def __init__(self, index, hyp, features, score, phrase_alignments, word_alignments):
		self.index = index
		self.hyp = hyp
		self.features = features
		self.score = score
		self.phrase_alignments = phrase_alignments
		self.word_alignments = word_alignments
	
	def __unicode__(self):
		return ' ||| '.join([unicode(self.index), self.hyp, self.features, self.score, self.phrase_alignments, self.word_alignments])
	
	def append_feature(self, feature):
		self.features += ' ' + str(feature)


class NBestGroup:
	def __init__(self, refrence_manager=None):
		self.group_index = -1
		self.group = []
		self.ref_manager = refrence_manager
	
	def __unicode__(self):
		return '\n'.join([unicode(item) for item in self.group])
	
	def __iter__(self):
		self.item_index = 0
		return self
	
	def __getitem__(self, index):
		return self.group[index]

	def add(self, item):
		if item is None:
			return
		if self.group_index == -1:
			self.group_index = item.index
			if self.ref_manager:
				self.refs = self.ref_manager.get_all_refs(self.group_index)
		else:
			U.xassert(item.index == self.group_index, "Cannot add an nbest item with an incompatible index")
		self.group.append(item)
		
	def next(self):
		#if self.item_index < len(self.group):
		try:
			item = self.group[self.item_index]
			self.item_index += 1
			return item
		#else:
		except IndexError:
			raise StopIteration
	
	def size(self):
		return len(self.group)
	
	def append_features(self, features_list):
		U.xassert(len(features_list) == len(self.group), 'Number of features and number of items in this group do not match')
		for i in range(len(self.group)):
			self.group[i].append_feature(features_list[i])



class RefernceManager:
	def __init__(self, paths_list):
		U.xassert(type(paths_list) is list, "The input to a RefernceManager class must be a list")
		self.ref_list = []
		self.num_lines = -1
		self.num_refs = 0
		for path in paths_list:
			with codecs.open(path, mode='r', encoding='UTF-8') as f:
				self.num_refs += 1
				sentences = f.readlines()
				if self.num_lines == -1:
					self.num_lines = len(sentences)
				else:
					U.xassert(self.num_lines == len(sentences), "Reference files must have the same number of lines")
				self.ref_list.append(sentences)
	
	def get_all_refs(self, index):
		U.xassert(index < self.num_lines, "Index out of bound")
		return [self.ref_list[k][index] for k in range(self.num_refs)]





























