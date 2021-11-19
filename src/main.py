#!/usr/bin/python

import sys
import json

import numpy as np
import scipy
import tensorflow as tf

import kerastuner as kt

import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1e10
import mir_eval
import collections

from constants import *
from models import Models
from support import *
from argparser import ArgParser
from experiment import Experiment, ORDER

def main(argv):
	ap = ArgParser(argv)

	""" config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	config.gpu_options.allow_growth = True
	session = tf.compat.v1.Session(config=config) """

	""" try:
		gpu = tf.config.experimental.list_physical_devices('GPU')[0]
		tf.config.experimental.set_memory_growth(gpu, True)
	except Exception as e:
		print(e)
		exit(1) """

	# Prepare data for training and testing
	if ap.mode == 9 or ap.data_supply == 0:
		train_x, train_y, test_x, test_y = createTrainTest(ap.train_file, ap.test_file)
	else:
		gen_train, gen_test = createGenerators(ap.train_file, ap.test_file)

	# Get model
	if ap.mode == 0:
		model = getattr(Models, "model_"+str(ap.model_version))()
	elif ap.mode < 3:
		model = tf.keras.models.load_model(absPath(ap.name))

	# Setup checkpoints for saving model
	if ap.save_freq > 0:
		if ap.data_supply == 1:
			checkpoint = tf.keras.callbacks.ModelCheckpoint(absPath(ap.name+"_{epoch}"), verbose=1, save_weights_only=False,
			                                                save_freq=len(gen_train)*ap.save_freq)
		else:
			checkpoint = tf.keras.callbacks.ModelCheckpoint(absPath(ap.name+"_{epoch}"), verbose=1, save_weights_only=False,
			                                                save_freq=train_x.shape[0]*ap.save_freq)
	else:
		checkpoint = None

	if ap.mode < 2:
		if ap.data_supply == 1:
			history = model.fit(gen_train, steps_per_epoch=len(gen_train), validation_data=gen_test, validation_steps=len(gen_test),
		                        initial_epoch=ap.init_epoch, epochs=ap.epoch, verbose=1, callbacks=[checkpoint] if checkpoint else None)
		else:
			history = model.fit(x=train_x, y=train_y, batch_size=BATCH_SIZE, validation_data=(test_x, test_y),
			                    initial_epoch=ap.init_epoch, epochs=ap.epoch, verbose=1, callbacks=[checkpoint] if checkpoint else None)

		# Save history of training
		if ap.history:
			json.dump(history.history, open(absPath(ap.name+"_history"), "w"))

		# Save model
		model.save(absPath(ap.name))
	elif ap.mode == 2:
		final = None
		eval_num = 10
		for i in range(eval_num):
			if ap.data_supply == 1:
				e = np.array(model.evaluate(gen_test, batch_size=BATCH_SIZE, verbose=1))
			else:
				e = np.array(model.evaluate(x=test_x, y=test_y, batch_size=BATCH_SIZE, verbose=1))

			if not isinstance(final, np.ndarray):
				final = e
			else:
				final = final+e
		print(final/eval_num)
	elif ap.mode == 3:
		tuner = kt.Hyperband(Models.model_builder, objective=kt.Objective("val_recall", direction="max"), max_epochs=10)
		stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_recall', patience=5)

		tuner.search(gen_train, validation_data=gen_test, epochs=50, callbacks=[stop_early], verbose=1)
		b_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

		print(f"Layer 1: {b_hps.get('num_filters_1')}, {b_hps.get('kernel_1')}, {b_hps.get('dropout_1')}")
		print(f"Layer 2: {b_hps.get('num_filters_2')}, {b_hps.get('kernel_2')}, {b_hps.get('dropout_2')}")
		print(f"Layer 3: {b_hps.get('num_dense_3')}")
		print(f"Learning rate: {b_hps.get('lr')}")
	elif ap.mode == 9:
		exp = Experiment(train_x, train_y, test_x, test_y, name="hte_z")
		exp.start()
	elif ap.mode == 10:
		# open test files' names
		data_test = pickle.load(open(absPath(ap.test_file), "rb"))

		# loop through all models
		for ver in ORDER:
			for idx in range(5):
				# get model name and load it
				mname = f"l_{ver}_{idx}"
				model = tf.keras.models.load_model(absPath("../hteModels/"+mname))
				print(mname)

				metrics = None

				# open test files
				for nwav in data_test.keys():
					nmid = "../MAPS/"+nwav.split("/", 1)[1]

					# get model output, converted to events
					data_x = dataToMatrix1(np.load(absPath("../"+nwav+"_wav.npy")))
					out_y = model.predict(data_x, verbose=1).T > THRESHOLD
					onsets, pitch = output_to_events(out_y)

					# get ground truth from midi files
					midi_data = pretty_midi.PrettyMIDI(absPath(nmid+".mid"))
					intervals = list()
					notes = list()
					for instrument in midi_data.instruments:
						for note in instrument.notes:
							intervals.append(np.array([note.start, note.end]))
							notes.append(note.pitch)
					onsets_real = np.array(intervals)
					pitch_real = np.array(notes)

					# evaluate model
					scores = mir_eval.transcription.evaluate(onsets_real, pitch_real, onsets, pitch)

					if metrics == None:
						metrics = scores
					else:
						counter = collections.Counter()
						counter.update(metrics)
						counter.update(scores)
						metrics = dict(counter)

				total = len(data_test.keys())
				metrics = {key: value/total for key, value in metrics.items()}

				# save metrics
				with open("hte_events_l.txt", "a+") as f:
					f.write("{};{}\n".format(mname, [(k, v) for k, v in metrics.items()]))

if __name__ == "__main__":
	main(sys.argv[1:])
