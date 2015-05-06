import sys
import dlmutils.utils as U
import codecs

class NBestList():

	def __init__(self, nbest_path, mode='r', reference_path=None):
		U.xassert(mode == 'r' or mode == 'w', "Invalid mode: " + mode)
		self.mode = mode
		self.nbest_file = codecs.open(nbest_path, mode=mode, encoding='UTF-8')
		self.prev_index = -1
		self.curr_ref = "NULL"
		self.ref_given = False
		if reference_path:
			U.xassert(mode == 'r', "Cannot accept a reference_path in 'w' mode")
			self.ref_file = codecs.open(reference_path, mode='r', encoding='UTF-8')
			self.ref_given = True

	def __iter__(self):
		U.xassert(self.mode == 'r', "Iteration can only be done in 'r' mode")
		return self
	
	def next(self):
		U.xassert(self.mode == 'r', "next() method can only be used in 'r' mode")
		try:
			segments = self.nbest_file.next().split("|||")
		except StopIteration:
			self.close()
			raise StopIteration
		try:
			index = int(segments[0])
		except ValueError:
			U.error("The first segment in an n-best list must be an integer")
		hyp = segments[1].strip()
		if self.ref_given and index != self.prev_index:
			try:
				self.curr_ref = self.ref_file.next().strip()
			except StopIteration:
				U.error("The reference file is not compatible with the nbest list. Check number of lines.")
			self.prev_index = index
		return NBestItem(index, hyp, self.curr_ref)
	
	def write(self, item):
		U.xassert(self.mode == 'w', "write() method can only be used in 'w' mode")
		self.nbest_file.write(unicode(item) + "\n")
	
	def close(self):
		self.nbest_file.close()



class NBestItem:
	
	def __init__(self, index, hyp, ref=None):
		self.index = index
		self.hyp = hyp
		self.ref = ref
	
	def __unicode__(self):
		return unicode(self.index) + " ||| " + self.hyp





