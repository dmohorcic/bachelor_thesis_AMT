import os
import numpy as np
import pickle
import scipy

from tqdm import tqdm
from constants import *
from generators import RandomGenerator, SequentialGenerator

def absPath(file_name):
	"""Creates an absoulte path of file requested

	Args:
		file_name (str): relative file name (supports ..)

	Returns:
		str: absolute path of file_name
	"""
	return os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

def joinDicts(dict_1, dict_2):
	j = dict()
	for key, val in dict_1.items():
		j[key] = val + dict_2[key]
	return j

def prepareForGenerator(data):
	""" Converts dictionary to array of names and cumulative sample sums

	Args:
		data (dict[str] = int): dictionary of names and corresponding sample sizes

	Returns:
		np.array: names - array of names of files
		np.array: values - array of cumulative sample sizes
	"""
	names = np.array([k for k in data.keys()])
	values = np.cumsum([v for v in data.values()])
	return names, values

def createGenerators(name_train, name_test, use_random = True):
	""" Creates generators for training and testing

	Args:
		name_train (String): name of file containing training data
		name_test (String): name of file containing testing data

	Returns:
		tf.keras.utils.Sequence: gen_train - generator with training examples
		tf.keras.utils.Sequence: gen_test - generator with testing examples
	"""
	name_train = absPath(name_train)
	name_test = absPath(name_test)

	data_train = pickle.load(open(name_train, "rb"))
	data_test = pickle.load(open(name_test, "rb"))

	names_train, values_train = prepareForGenerator(data_train)
	names_test, values_test = prepareForGenerator(data_test)

	if use_random:
		gen_train = RandomGenerator(names_train, values_train)
		gen_test = RandomGenerator(names_test, values_test)
	else:
		gen_train = SequentialGenerator(names_train, values_train)
		gen_test = SequentialGenerator(names_test, values_test)

	return gen_train, gen_test

def dataToMatrix1(data_x):
	shape_x = data_x.shape
	size = shape_x[1]-WINDOW+1
	pad = int((WINDOW-1)/2)
	output_x = np.zeros((size, N_BINS, WINDOW, 1))

	for i in range(size):
		output_x[i, :, :] = np.reshape(data_x[:, i:i+WINDOW], (N_BINS, WINDOW, 1))

	output_x = np.float32(output_x)
	return output_x

def dataToMatrix(data_x, data_y):
	shape_x = data_x.shape
	size = shape_x[1]-WINDOW+1
	pad = int((WINDOW-1)/2)
	out_x = np.zeros((size, N_BINS, WINDOW, 1))
	out_y = np.zeros((size, 88))

	for i in range(size):
		out_x[i, :, :] = np.reshape(data_x[:, i:i+WINDOW], (N_BINS, WINDOW, 1))
		out_y[i, :] = data_y[:88, i+pad]*1

	out_x = np.float32(out_x)
	out_y = np.float32(out_y)
	return out_x, out_y

def createTrainTest(name_train, name_test):
	"""Creates matrices for training and testing

	Args:
		name_train (String): name of file containing training data
		name_test (String): name of file containing testing data

	Returns:
		np.array: train - matrix with training examples
		np.array: test - matrix with testing examples
	"""
	name_train = absPath(name_train)
	name_test = absPath(name_test)

	data_train = pickle.load(open(name_train, "rb"))
	data_test = pickle.load(open(name_test, "rb"))

	names_train, values_train = prepareForGenerator(data_train)
	names_test, values_test = prepareForGenerator(data_test)

	train_x = np.zeros((values_train[-1], N_BINS, WINDOW, 1))
	train_y = np.zeros((values_train[-1], 88))

	test_x = np.zeros((values_test[-1], N_BINS, WINDOW, 1))
	test_y = np.zeros((values_test[-1], 88))

	prev = 0
	num = 0
	for name in tqdm(names_train):
		try:
			data_wav = np.load(absPath("../"+name+"_wav.npy"))
			data_mid = scipy.sparse.load_npz(absPath("../"+name+"_mid.npz")).toarray()
			out_x, out_y = dataToMatrix(data_wav, data_mid)
			num += out_x.shape[0]
			train_x[prev:num, :, :, :] = out_x
			train_y[prev:num, :] = out_y
			prev = num
		except Exception as e:
			print(name)
			print(e)

	prev = 0
	num = 0
	for name in tqdm(names_test):
		data_wav = np.load(absPath("../"+name+"_wav.npy"))
		data_mid = scipy.sparse.load_npz(absPath("../"+name+"_mid.npz")).toarray()
		out_x, out_y = dataToMatrix(data_wav, data_mid)
		num += out_x.shape[0]
		test_x[prev:num, :, :, :] = out_x
		test_y[prev:num, :] = out_y
		prev = num

	return train_x, train_y, test_x, test_y

FRAMES_PER_BIN = HOP_LENGTH / 44100
def output_to_events(data_y):
	tracing = np.array([False for i in range(88)])
	onsets = np.array([0.0 for i in range(88)])

	intervals = list()
	pitch = list()

	for frame in range(data_y.shape[1]):
		for note in range(88):
			if data_y[note, frame]:
				if not tracing[note]:
					tracing[note] = True
					onsets[note] = (frame+(WINDOW-1)/2)*FRAMES_PER_BIN
			elif tracing[note]:
				tracing[note] = False
				intervals.append(np.array([onsets[note], (frame+(WINDOW-1)/2)*FRAMES_PER_BIN]))
				pitch.append(note+21)
	return np.array(intervals), np.array(pitch)
