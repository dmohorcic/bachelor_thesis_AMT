import pretty_midi
import numpy as np
import os
import sys
import time
import pickle
import json
from constants import *
import tensorflow as tf

from midi2audio import FluidSynth
from midiutil import MIDIFile

""" fs = FluidSynth()
	fs.play_midi("test.mid") """

""" midi = MIDIFile()
	midi.addTempo(0, 0, 90)
	
	degrees = [60, 62, 64, 65, 67, 69, 71, 72] # MIDI note number
	track = 0
	channel = 0
	time = 0
	duration = 2
	volume = 100

	for pitch in degrees:
		midi.addNote(track, channel, pitch, time, duration, pitch)
		time = time + 1
	
	with open("test.mid", "wb") as output_file:
		midi.writeFile(output_file) """

FRAMES_PER_BIN = HOP_LENGTH / 44100

def data_test(data_x):
	shape_x = data_x.shape
	size = shape_x[1]-WINDOW+1
	pad = int((WINDOW-1)/2)
	output_x = np.zeros((size, N_BINS, WINDOW, 1))

	for i in range(size):
		output_x[i, :, :] = np.reshape(data_x[:, i:i+WINDOW], (N_BINS, WINDOW, 1))

	output_x = np.float32(output_x)
	return output_x

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
					onsets[note] = frame*FRAMES_PER_BIN
			elif tracing[note]:
				tracing[note] = False
				intervals.append(np.array([onsets[note], frame*FRAMES_PER_BIN]))
				pitch.append(note+21)
	return np.array(intervals), np.array(pitch)

def clean_events(intervals, pitch):
	intervals_true = list()
	pitch_true = list()

	for i in range(len(pitch)):
		interval = intervals[i]
		if interval[1]-interval[0] > 0.05:
			intervals_true.append(interval)
			pitch_true.append(pitch[i])
	return np.array(intervals_true), np.array(pitch_true)

def get_tempo(intervals):
	return 1/(min(intervals[:, 1]-intervals[:, 0]))

class PostProcess():
	def __init__(self):
		self.fs = FluidSynth()
	
	def toMidi(self, data_y):
		onsets, pitch = output_to_events(data_y)
		onsets, pitch = clean_events(onsets, pitch)

		track = 0
		channel = 0
		time = 0
		duration = 2
		volume = 90

		tempo = get_tempo(onsets)

		self.midi = MIDIFile()
		self.midi.addTempo(track, time, tempo)

		for e, p in zip(onsets, pitch):
			l = int(round(tempo*(e[1]-e[0])))
			t = int(round(tempo*e[0]))
			self.midi.addNote(track, channel, p, t, l, volume)
		
	def writeMidi(self, file):
		with open(file, "wb") as output_file:
			midi.writeFile(output_file)
		
	def midiToWav(self, file_midi, file_wav):
		fs.midi_to_audio(file_midi, file_wav)