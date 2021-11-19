import numpy as np
import scipy
import tensorflow as tf

from constants import *

class RandomGenerator(tf.keras.utils.Sequence):
	def __init__(self, names, values, batch_size = BATCH_SIZE, window = WINDOW, n_bins = N_BINS):
		self.names = names
		self.values = values
		self.batch_size = batch_size
		self.window = window
		self.n_bins = n_bins
		self.samples = self.values[-1]
	
	def __len__(self):
		return self.samples // self.batch_size
	
	def getRandomSample(self, data_x, data_y):
		pad = int((self.window-1)/2)
		output_x = np.zeros((self.batch_size, self.n_bins, self.window, 1))
		output_y = np.zeros((self.batch_size, 88))
		sample = np.random.randint(0, data_x.shape[1]-self.window-1, self.batch_size)

		i = 0
		for j in sample:
			output_x[i, :, :, :] = np.reshape(data_x[:, j:j+self.window], (self.n_bins, self.window, 1))
			output_y[i, :] = data_y[:88, j+pad]*1
			i += 1

		output_x = np.float32(output_x)
		output_y = np.float32(output_y) # int8
		return output_x, output_y
	
	def __getitem__(self, index):
		filename = self.names[np.argmin(self.values <= index*self.batch_size)]
		# load data
		data_wav = np.load(filename+"_wav.npy")
		data_mid = scipy.sparse.load_npz(filename+"_mid.npz").toarray()
		
		return self.getRandomSample(data_wav, data_mid)


class SequentialGenerator(tf.keras.utils.Sequence):
	def __init__(self, names, values, batch_size = BATCH_SIZE, window = WINDOW, n_bins = N_BINS):
		""" Initialization for sequential generator. Returns the requested sample from specified file file.

		Args:
			names (np.array): array of names of files
			values (np.array): array of cumulative sample sizes
			batch_size (int, optional): Batch size of samples. Defaults to BATCH_SIZE.
			window (int, optional): Window size of sample spectrogram. Defaults to WINDOW.
			n_bins (int, optional): Number of bins for sample spectrogram. Defaults to N_BINS.
		"""
		self.names = names
		self.values = values
		self.batch_size = batch_size
		self.window = window
		self.n_bins = n_bins
		self.samples = self.values[-1]
	
	def __len__(self):
		""" Return the number of batches per data

		Returns:
			int: the number of batches per data
		"""
		return self.samples//self.batch_size
	
	def __getitem__(self, index):
		""" Get the batch with requested index, where f(x) = y

		Args:
			index (int): index of the batch

		Returns:
			np.float32[batch_size, n_bins, window, 1]: input data (x)
			np.float32[batch_size, 88]: output data (y)
		"""
		# which file has index
		idx = np.argmin(self.values <= index*self.batch_size)

		# offset sample start to start of file
		if idx == 0:
			sample = index*self.batch_size
			# if sample overflows to next file push it back a little bit
			if sample+self.batch_size+self.window > self.values[idx]:
				sample = self.values[idx]-(self.batch_size+self.window)
		else:
			sample = index*self.batch_size-self.values[idx-1]
			# if sample overflows to next file push it back a little bit
			if sample+self.batch_size+self.window > self.values[idx]-self.values[idx-1]:
				sample = self.values[idx]-self.values[idx-1]-(self.batch_size+self.window)

		# open appropriate files
		filename = self.names[np.argmin(self.values <= index*self.batch_size)]
		data_wav = np.load(filename+"_wav.npy")
		data_mid = scipy.sparse.load_npz(filename+"_mid.npz").toarray()

		# prepare output data
		pad = int((self.window-1)/2)
		output_x = np.zeros((self.batch_size, self.n_bins, self.window, 1))
		output_y = np.zeros((self.batch_size, 88))

		# copy correct data to x and y
		for i in range(self.batch_size):
			try:
				output_x[i, :, :, :] = np.reshape(data_wav[:, sample+i:sample+i+self.window], (self.n_bins, self.window, 1))
				output_y[i, :] = data_mid[:88, sample+i+pad]*1
			except:
				print("")
				print(i, index, idx, sample)
				exit(1)
		
		# convert to float32
		output_x = np.float32(output_x)
		output_y = np.float32(output_y)
		return output_x, output_y