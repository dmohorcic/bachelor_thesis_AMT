from constants import *
import tensorflow as tf

import kerastuner as kt

def model_1():
	"""
	HOP_LENGTH = 512
	BINS_PER_OCTAVE = 48
	WINDOW = 7

	max precision
	"""
	model = tf.keras.models.Sequential(name="model1")
	model.add(tf.keras.layers.Conv2D(32, (48, 3), activation="relu", input_shape=(384, 7, 1), data_format="channels_last"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Conv2D(32, (24, 3), activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(88, activation="relu"))
	model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

	model.summary()

	# tf.keras.losses.BinaryCrossentropy(from_logits=True)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
	
	model_name = "cnnModels/conv32_48_3_mp2_1_conv32_24_3_mp2_1_flat_den88_den88"
	return model, model_name

def model_2():
	"""
	HOP_LENGTH = 512
	BINS_PER_OCTAVE = 48
	WINDOW = 7

	max precision
	"""
	model = tf.keras.models.Sequential(name="model2")
	model.add(tf.keras.layers.Conv2D(8, (23, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Conv2D(88, (6, 3), activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Conv2D(44, (6, 3), activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(88, activation="relu"))
	model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

	model.summary()

	# tf.keras.losses.BinaryCrossentropy(from_logits=True)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
	model_name = "cnnModels/conv8_23_3_mp2_1_conv88_6_3_mp2_1_conv44_6_3_mp2_1_flat_den88_den88"
	return model, model_name

def model_3():
	"""
	HOP_LENGTH = 512
	BINS_PER_OCTAVE = 48
	WINDOW = 7

	max auc
	"""
	model = tf.keras.models.Sequential(name="model3")
	model.add(tf.keras.layers.Conv2D(8, (12, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
	model.add(tf.keras.layers.Conv2D(32, (12, 3), activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Conv2D(88, (6, 3), activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(64, activation="relu"))
	model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

	model.summary()

	metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	              metrics=metrics)
	model_name = "cnnModels/3_conv8_12_3conv32_12_3mp2_1conv88_6_3mp_2_1flat_den64_den88"
	return model, model_name

def model_4():
	"""
	HOP_LENGTH = 1024
	BINS_PER_OCTAVE = 24
	WINDOW = 5

	max recall
	"""
	model = tf.keras.models.Sequential(name="model4")
	model.add(tf.keras.layers.Conv2D(32, (6, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Conv2D(64, (6, 3), activation="relu"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

	model.summary()

	metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(),
	              metrics=metrics)
	model_name = "cnnModels/4_conv32_6_3mp2_1conv64_6_3mp_2_1flat_den88"
	return model, model_name

def model_5():
	"""
	HOP_LENGTH = 1024
	BINS_PER_OCTAVE = 24
	WINDOW = 5

	max recall
	"""
	model = tf.keras.models.Sequential(name="model5_log")
	model.add(tf.keras.layers.Conv2D(16, (24, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
	model.add(tf.keras.layers.MaxPooling2D((2, 1)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

	model.summary()

	metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006), loss=tf.keras.losses.BinaryCrossentropy(),
	              metrics=metrics)
	model_name = "../cnnModels/model5_MUSlh_log"
	return model, model_name

class Models:
	@staticmethod
	def model_5():
		"""
		HOP_LENGTH = 1024
		BINS_PER_OCTAVE = 24
		WINDOW = 5

		max recall
		"""
		model = tf.keras.models.Sequential(name="model5")
		model.add(tf.keras.layers.Conv2D(16, (24, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

		model.summary()

		metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
		           tf.keras.metrics.AUC(name="auc")]
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006), loss=tf.keras.losses.BinaryCrossentropy(),
		              metrics=metrics)
		return model
	
	@staticmethod
	def model_6():
		"""
		HOP_LENGTH = 1024
		BINS_PER_OCTAVE = 24
		WINDOW = 5

		max recall
		"""
		model = tf.keras.models.Sequential(name="model6")
		model.add(tf.keras.layers.Conv2D(16, (24, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Conv2D(32, (12, 3), activation="relu"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Conv2D(64, (6, 1), activation="relu"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(88, activation="relu"))
		model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

		model.summary()

		metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
		           tf.keras.metrics.AUC(name="auc")]
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006), loss=tf.keras.losses.BinaryCrossentropy(),
		              metrics=metrics)
		return model
	
	@staticmethod
	def model_7():
		"""
		HOP_LENGTH = 1024
		BINS_PER_OCTAVE = 24
		WINDOW = 7

		max recall
		"""
		model = tf.keras.models.Sequential(name="model7")

		model.add(tf.keras.layers.Conv2D(32, (24, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Dropout(rate=0.1))
		model.add(tf.keras.layers.Conv2D(64, (12, 3), activation="relu"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Dropout(rate=0.1))
		model.add(tf.keras.layers.Conv2D(128, (6, 3), activation="relu"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Dropout(rate=0.1))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(256, activation="relu"))
		model.add(tf.keras.layers.Dense(128, activation="relu"))
		model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

		model.summary()

		metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
		           tf.keras.metrics.AUC(name="auc")]
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004), loss=tf.keras.losses.BinaryCrossentropy(),
		              metrics=metrics)
		
		return model

	@staticmethod
	def model_1337():
		"""
		HOP_LENGTH = 1024
		BINS_PER_OCTAVE = 24
		WINDOW = 5

		max recall
		"""
		model = tf.keras.models.Sequential(name="model1337")

		model.add(tf.keras.layers.Conv2D(32, (24, 3), activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Dropout(rate=0.05))
		model.add(tf.keras.layers.Conv2D(144, (24, 3), activation="relu"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))
		model.add(tf.keras.layers.Dropout(rate=0.15))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(416, activation="relu"))
		model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

		model.summary()

		metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
		           tf.keras.metrics.AUC(name="auc")]
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004), loss=tf.keras.losses.BinaryCrossentropy(),
		              metrics=metrics)
		
		return model
	
	@staticmethod
	def model_builder(hp):
		model = tf.keras.models.Sequential(name="model1000")

		hp_conv_num_1 = hp.Int("num_filters_1", min_value=16, max_value=128, step=16, default=32)
		hp_conv_kernel_1 = hp.Choice("kernel_1", values=[6, 12, 24, 48], default=24)
		kernel_1 = (hp_conv_kernel_1, 3)
		model.add(tf.keras.layers.Conv2D(hp_conv_num_1, kernel_1, activation="relu", input_shape=(N_BINS, WINDOW, 1), data_format="channels_last"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))

		hp_dropout_1 = hp.Float("dropout_1", min_value=0.0, max_value=0.3, step=0.05, default=0.1)
		model.add(tf.keras.layers.Dropout(rate=hp_dropout_1))

		hp_conv_num_2 = hp.Int("num_filters_2", min_value=32, max_value=256, step=16, default=64)
		hp_conv_kernel_2 = hp.Choice("kernel_2", values=[3, 6, 12, 24], default=12)
		kernel_2 = (hp_conv_kernel_2, 3)
		model.add(tf.keras.layers.Conv2D(hp_conv_num_2, kernel_2, activation="relu"))
		model.add(tf.keras.layers.MaxPooling2D((2, 1)))

		hp_dropout_2 = hp.Float("dropout_2", min_value=0.0, max_value=0.3, step=0.05, default=0.1)
		model.add(tf.keras.layers.Dropout(rate=hp_dropout_2))

		model.add(tf.keras.layers.Flatten())

		hp_dense_3 = hp.Int("num_dense_3", min_value=64, max_value=1024, step=32, default=128)
		model.add(tf.keras.layers.Dense(hp_dense_3, activation="relu"))

		model.add(tf.keras.layers.Dense(88, activation="sigmoid"))

		metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
		           tf.keras.metrics.AUC(name="auc")]

		hp_lr = hp.Choice("lr", values=[2e-4 * i for i in range(1, 6)])
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr), loss=tf.keras.losses.BinaryCrossentropy(),
		              metrics=metrics)

		return model
