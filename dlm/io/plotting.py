import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import sys
import math
import numpy as np

class Plotter:
	
	def __init__(self, path, title=None, xlabel=None, ylabel=None, xspace=1, yspace=0.5):
		self.path = path
		self.title = title
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.xspace = xspace
		self.yspace = yspace
		self.series_list = []
		#self.tix_list = ['b*-', 'ro--']
		self.tix_list = ['b-', 'r--']
	
	def plot(self):
		plt.title(self.title, y=1.01, fontsize='medium')
		plt.xlabel(self.xlabel)
		plt.ylabel(self.ylabel)
		plt.grid('on')
		plt.margins(0.1)
		for i in range(len(self.series_list)):
			x_list = []
			y_list = []
			for x in sorted(self.series_list[i].keys()):
				x_list.append(x)
				y_list.append(self.series_list[i][x])
			#xmin, xmax = min(self.x_list), max(self.x_list) + 1
			#ymin, ymax = min(self.y_list), max(self.y_list) + 1
			#plt.xticks(np.arange(xmin, xmax, 1.0), np.arange(xmin, xmax, 1.0), fontsize='x-small')
			#plt.yticks(np.arange(ymin, ymax, 0.5), np.arange(ymin, ymax, 0.5), fontsize='x-small')
			plt.plot(x_list, y_list, self.tix_list[i], label='S' + str(i))
		plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center right', borderaxespad=0.2, fontsize='x-small')
		plt.savefig(self.path, format='pdf', bbox_inches='tight', pad_inches=0.3)
		plt.cla()
	
	def add(self, series_index, x, y):
		assert series_index < len(self.tix_list)
		assert series_index <= len(self.series_list)
		if series_index == len(self.series_list):
			self.series_list.append(dict())
		series = self.series_list[series_index]
		series[x] = y
		self.plot()
	
	def add_list(self, series_index, x_list, y_list):
		assert series_index < len(self.tix_list)
		assert series_index <= len(self.series_list)
		if series_index == len(self.series_list):
			self.series_list.append(dict())
		series = self.series_list[series_index]
		for x, y in zip(x_list, y_list):
			series[x] = y
		self.plot()
	
	def set_tix_list(self, tix_list):
		self.tix_list = tix_list
	
