from constants import *
import pretty_midi
import numpy as np
import scipy
import os
import sys

def processWav(name):
	""" Function for preprocessing .wav files into Constant Q Transform
	spectrogram

	Args:
		name (String): file name (ending with .wav)

	Returns:
		np.float32[..., N_BINS]: CQT spectrogram of .wav file
	"""
	rate, data = scipy.io.wavfile.read(name)
	data = data.astype(float)

	# downsample
	#data = librosa.resample(data.T, rate, rate/2, res_type = "kaiser_fast")
	#rate = rate/2

	if data.shape[0] > data.shape[1]:
		data = data.T

	data_mono = librosa.to_mono(data)
	cq_data = np.abs(librosa.cqt(data_mono, sr = rate, hop_length = HOP_LENGTH,
								 fmin = FMIN, bins_per_octave = BINS_PER_OCTAVE,
								 n_bins = N_BINS, pad_mode = "wrap"))
	cq_data = np.float32(cq_data) # save space
	return cq_data, rate

def processMidi(name, rate, shape):
	""" Function for preprocessing .mid files into exact spectrograms

	Args:
		name (String): name of .mid file (ending with .mid)
		rate (int): sample rate of corresponding .wav file
		shape (np.shape(2)): shape of corresponding .wav spectrogram

	Returns:
		np.bool(..., 96): exact spectrogram
	"""
	frames_per_bin = HOP_LENGTH/rate
	of_data = np.zeros(shape).astype(np.bool)
	with pretty_midi.PrettyMIDI(name) as midi_data:
		for instrument in midi_data.instruments: # imamo samo 1 instrument
			for note in instrument.notes:
				onset_bin = int(float(note.start)//frames_per_bin)
				ofset_bin = int(float(note.end)//frames_per_bin)
				note = int(note.pitch)-21
				of_data_1_midi[note, onset_bin:ofset_bin+1] = True
	return of_data

def processTxt(name, rate, shape):
	""" Function for preprocessing .txt files into exact spectrograms

	Args:
		name (String): name of .txt file (ending with .txt)
		rate (int): sample rate of corresponding .wav file
		shape (np.shape(2)): shape of corresponding .wav spectrogram

	Returns:
		np.bool(..., 96): exact spectrogram
	"""
	frames_per_bin = HOP_LENGTH/rate
	of_data = np.zeros((96, shape[1])).astype(np.bool)
	with open(name) as f:
		next(f)
		try:
			for line in f:
				if line == '\n': # ignore last line
					continue
				args = line.rstrip().split("\t")
				onset_bin = int(float(args[0])//frames_per_bin)
				ofset_bin = int(float(args[1])//frames_per_bin)
				note = int(args[2])-21
				of_data[note, onset_bin:ofset_bin+1] = True
		except:
			print(name)
			print(sys.exc_info()[0])
	return of_data

def processAllFiles(inputpath):
	""" Function for preprocessing entire data folder

	Args:
		inputpath (String): location of data
	"""
	#dstart = time.time()
	for dirpath, dirnames, filenames in os.walk(inputpath):
		if len(filenames) > 2: # .mid, .txt, .wav
			# get working direcotry
			pwd_read = "MAPS"+dirpath[len(inputpath):]
			pwd_write = "MAPS_processed"+dirpath[len(inputpath):]

			# check if folder has been processed
			#if len(os.listdir(pwd_write)) == (len(filenames)//3)*2:
			#    continue
			#print(pwd_read)

			# get unique file names
			names = {file[:-4] for file in filenames}
			names.discard("desktop")
			lst = list(names)

			# process those files and save them
			for name in names:

				data_wav, rate = processWav(os.path.join(pwd_read, name)+".wav")
				data_mid = processTxt(os.path.join(pwd_read, name)+".txt", rate, data_wav.shape)

				# save CQT and spectrogram
				np.save(os.path.join(pwd_write, name)+"_wav", data_wav) # save CQT
				tmp = scipy.sparse.csc_matrix(data_mid)
				scipy.sparse.save_npz(os.path.join(pwd_write, name)+"_mid", tmp) # save as sparse

	#end = time.time()
	#print(end-start)

def prepareData(inputpath):
	""" Prepares input data as a dictionary of names and sizes

	Args:
		inputpath (String): path of data folder

	Returns:
		dict: train - dictionary of names and sizes of training examples
		dict: test - dictionary of names and sizes of testing examples
	"""
	data_train = dict()
	data_test = dict()

	# start = time.time()
	for dirpath, dirnames, filenames in os.walk(inputpath):
		if len(filenames) >= 2 and "MUS" in dirpath: # .npy, .npz
			pwd_read = "MAPS_processed"+dirpath[len(inputpath):]
			# print(pwd_read)
			
			names = {file[:-8] for file in filenames}
			names.discard("desktop")
			lst = list(names)
			
			if "ENSTDkAm" in dirpath or "ENSTDkCl" in dirpath: # test data
				for name in names:
					n = os.path.join(pwd_read, name)
					data_wav = np.load(n+"_wav.npy")
					data_test[n] = data_wav.shape[1]-WINDOW+1
			else: # train data
				for name in names:
					n = os.path.join(pwd_read, name)
					data_wav = np.load(n+"_wav.npy")
					data_train[n] = data_wav.shape[1]-WINDOW+1

	# end = time.time()
	# print(end-start)
	return data_train, data_test
