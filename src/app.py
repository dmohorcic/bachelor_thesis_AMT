# spectrogram representation
import librosa

# handling .mid files
import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10

from midi2audio import FluidSynth

# official evaluation methods
# import mir_eval

# for vectors
import numpy as np
# import matplotlib.pyplot as plt
import scipy

# file management
import os
import sys
import time
import pickle
import json

# deep learning
import tensorflow as tf

# constants
from constants import *
import generators
from models import model_5
from postprocess import *

# midi to wav
"""
Spremenjene vrstice v midi2audio:
- 37: spremenjena pot do sound fontov
- 46: spremenjena pot do fluidsynth, dodan shell=True
- 49: spremenjena pot do fluidsynth, dodan shell=True
"""
from midi2audio import FluidSynth
from midiutil import MIDIFile

def absPath(file_name):
	return os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

def test_tf():
	print(tf.test.is_built_with_cuda())
	print(tf.config.list_physical_devices())

bce = tf.keras.losses.BinaryCrossentropy()
def custom_binaryCrossentropy(y_true, y_pred):
	w = y_true*5 + (1-y_true)*1
	out = bce(y_true, y_pred, sample_weight=w).numpy()
	return tf.reduce_mean(out)

def prepareForGenerator(data):
	""" Converts dictionary to array of names and cumulative sample sums

	Args:
		data (dict): dictionary of names and corresponding sample sizes

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
		[type]: [description]
	"""
	name_train = absPath(name_train)
	name_test = absPath(name_test)

	data_train = pickle.load(open(name_train, "rb"))
	data_test = pickle.load(open(name_test, "rb"))

	names_train, values_train = prepareForGenerator(data_train)
	names_test, values_test = prepareForGenerator(data_test)

	if use_random:
		gen_train = generators.RandomGenerator(names_train, values_train)
		gen_test = generators.RandomGenerator(names_test, values_test)
	else:
		gen_train = generators.SequentialGenerator(names_train, values_train)
		gen_test = generators.SequentialGenerator(names_test, values_test)

	return gen_train, gen_test

def train():
	gen_train, gen_test = createGenerators("../tt_files/MUSlh_train_log.pickle", "../tt_files/MUS_test_log.pickle")

	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	#config.gpu_options.allow_growth = True
	session = tf.compat.v1.Session(config=config)

	model, model_name = model_5()

	checkpoint = tf.keras.callbacks.ModelCheckpoint(absPath(model_name)+"_{epoch}", verbose=1, save_weights_only=False, save_freq=len(gen_train)*5)
	#earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_recall", verbose=1, mode="max", patience=10)

	history = model.fit(gen_train, steps_per_epoch=len(gen_train), validation_data=gen_test, validation_steps=len(gen_test), epochs=20,
	                    verbose=1, callbacks=[checkpoint])
	model.save(absPath(model_name))
	json.dump(history.history, open(absPath(model_name)+"_history", "w"))

def continueTraining():
	gen_train, gen_test = createGenerators("../tt_files/_MUS_train_log.pickle", "../tt_files/MUS_test_log.pickle", use_random=True)

	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	session = tf.compat.v1.Session(config=config)
	
	model = tf.keras.models.load_model(absPath("../cnnModels/model5__MUS_log_15"))
	#model.load_weights(model_name)
	model.summary()

	#checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name, verbose=1, save_weights_only=True)
	history = model.fit(gen_train, steps_per_epoch=len(gen_train), validation_data=gen_test, validation_steps=len(gen_test), initial_epoch=15,
	                    epochs=20, verbose=1) # , callbacks=[checkpoint]
	model.save(absPath("../cnnModels/model5__MUS_log_20"))

def test():
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	session = tf.compat.v1.Session(config=config)

	gen_train, gen_test = createGenerators("../tt_files/_MUS_train_log.pickle", "../tt_files/MUS_test_log.pickle")

	for i in [i*5 for i in range(1, 5)]:
		model = tf.keras.models.load_model(absPath("../cnnModels/model5_MUSlh_log_")+str(i))
		e = model.evaluate(gen_test, batch_size=BATCH_SIZE, verbose=1)

def toWav():
	midi_data = pretty_midi.PrettyMIDI("MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.mid")
	for instrument in midi_data.instruments:
		for note in instrument.notes:
			prev = int(note.pitch)
			note.pitch = ((note.pitch-21)+58)%88+21
			#print("{} -> {}".format(prev, note.pitch))
	midi_data.write("MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht_lower.mid")

	FluidSynth().midi_to_audio("MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht_lower.mid", "MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht_lower.wav")

if __name__ == "__main__":
	#train()
	#continueTraining()
	test()

	#toWav()

	pass
