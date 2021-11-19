import numpy as np

import time

from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from support import absPath, joinDicts
from constants import BATCH_SIZE, N_BINS, WINDOW

METRICS = [Precision(name="precision"), Recall(name="recall")]
ORDER = [21, 22, 11, 24, 12, 13, 43, 32, 44, 45, 33, 34, 35]

class Timecallback(Callback):
	times = []
	def __init__(self):
		self.start = 0
	def on_epoch_begin(self, epoch, logs = {}):
		self.start = time.time()
	def on_epoch_end(self, epoch, logs = {}):
		self.times.append((epoch+1, time.time()-self.start))
	def on_train_begin(self, logs = {}):
		self.start = time.time()
	def on_train_end(self, logs = {}):
		pass

class Experiment:
	name = ""
	num_models = 0

	train_x = None
	train_y = None
	test_x = None
	test_y = None

	def __init__(self, train_x, train_y, test_x, test_y, name="experiment_l", num_models=5):
		self.name = name
		self.num_models = num_models

		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y

	def start(self):
		for i in ORDER:
			mname = "z_"+str(i)
			for idx in range(self.num_models):
				model = getattr(Experiment, "m_"+str(i))()
				self.train_model(model, mname, idx)

	def train_model(self, model, mname, i):
		# Create callback for time management
		tc = Timecallback()
		tc.times = []

		# Train model for 10 epochs
		hist_1 = model.fit(x=self.train_x, y=self.train_y, batch_size=BATCH_SIZE,
		                   validation_data=(self.test_x, self.test_y), epochs=10,
		                   verbose=1, callbacks=[tc])

		# Callback for early stopping
		es = EarlyStopping(monitor="val_recall", patience=10, verbose=1,
		                   mode="max", restore_best_weights=True)

		# Train model until it has not improved for 10 epochs
		hist_2 = model.fit(x=self.train_x, y=self.train_y, batch_size=BATCH_SIZE,
		                   validation_data=(self.test_x, self.test_y),
		                   initial_epoch=10, epochs=200, verbose=1,
		                   callbacks=[es, tc])

		# Save time
		with open(self.name+"_time.txt", "a+") as f:
			f.write("{};{};{}\n".format(mname, i, tc.times))
		del tc

		# Save mid evaluations
		h = joinDicts(hist_1.history, hist_2.history)
		with open(self.name+"_mid.txt", "a+") as f:
			f.write("{};{};{}\n".format(mname, i, h))

		# Save model
		model.save(absPath("../hteModels/"+mname+"_"+str(i)))

		# Evaluate model and save its score
		e = np.array(model.evaluate(x=self.test_x, y=self.test_y,
		                            batch_size=BATCH_SIZE, verbose=1))
		with open(self.name+".txt", "a+") as f:
			f.write("{};{};{}\n".format(mname, i, e))

	@staticmethod
	def m_11():
		# 178,080
		model = Sequential(name="m_11")
		model.add(Conv2D(8, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model

	@staticmethod
	def m_12():
		# 356,072 params
		model = Sequential(name="m_12")
		model.add(Conv2D(16, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model
	
	@staticmethod
	def m_13():
		# 534,064 params
		model = Sequential(name="m_13")
		model.add(Conv2D(24, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model

	@staticmethod
	def m_21():
		# 55,984 params
		model = Sequential(name="m_21")
		model.add(Conv2D(8, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Conv2D(16, (12, 3), activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model

	@staticmethod
	def m_22():
		# 121,096 params
		model = Sequential(name="m_22")
		model.add(Conv2D(16, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Conv2D(32, (12, 3), activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model

	@staticmethod
	def m_24():
		# 278,968 params
		model = Sequential(name="m_24")
		model.add(Conv2D(32, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Conv2D(64, (12, 3), activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model

	@staticmethod
	def m_32():
		# 1,056,232 params
		model = Sequential(name="m_32")
		model.add(Conv2D(16, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(256, activation="relu"))
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model
	
	@staticmethod
	def m_33():
		# 2,358,448 params
		model = Sequential(name="m_33")
		model.add(Conv2D(24, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(384, activation="relu"))
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model
	
	@staticmethod
	def m_34():
		# 3,133,176 params
		model = Sequential(name="m_34")
		model.add(Conv2D(32, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(384, activation="relu"))
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model

	@staticmethod
	def m_35():
		# 4,176,760 params
		model = Sequential(name="m_35")
		model.add(Conv2D(32, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(512, activation="relu"))
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model
	
	@staticmethod
	def m_43():
		# 741,088 params
		model = Sequential(name="m_43")
		model.add(Conv2D(24, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Conv2D(48, (12, 3), activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(384, activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model

	@staticmethod
	def m_44():
		# 1,301,432 params
		model = Sequential(name="m_44")
		model.add(Conv2D(32, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Conv2D(64, (12, 3), activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(512, activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model
	
	@staticmethod
	def m_45():
		# 1,956,384 params
		model = Sequential(name="m_45")
		model.add(Conv2D(40, (24, 3), activation="relu",
		                 input_shape=(N_BINS, WINDOW, 1),
		                 data_format="channels_last"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Conv2D(96, (12, 3), activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(MaxPooling2D((2, 1)))
		model.add(Flatten())
		model.add(Dense(512, activation="relu"))
		model.add(Dropout(rate=0.1))
		model.add(Dense(88, activation="sigmoid"))

		model.compile(optimizer=Adam(learning_rate=0.0006),
		              loss=BinaryCrossentropy(), metrics=METRICS)
		return model
