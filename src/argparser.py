import sys
import getopt

class ArgParser:

	mode: int = 0
	name: str = "default"
	model_version: int = 5
	epoch: int = 20
	init_epoch: int = 0

	data_supply: int = 1

	train_file = "../tt_files/train_l5.pickle"
	test_file = "../tt_files/test_l5.pickle"

	history: bool = False
	save_freq: int = 0

	def __init__(self, argv):
		self.argv = argv
		self._parseArgv()
	
	@staticmethod
	def _errorExit(e, c):
		print(e)
		sys.exit(c)

	def _parseArgv(self):
		"""Parses arguments

		-m mode         Operates in mode 0 (train), 1 (continue training), 2 (test), 9 (train and test a lot of models), or 10 (note-test a lot of models) [default 0]
		-n name         Model name (either for training or testing) [default 'default']
		-e epoch        Number of epochs to train for [default 20]
		-a init_epoch   Number of already trained epochs (only for mode 1) [default 0]
		-t              Save history (only for mode 0 and 1) [default False]
		-s num          Save model every num epochs (only for mode 0 and 1) [default 0]
		-r ver          Model version (int) [default 5]
		-h              Display help
		-d data_supply  How to supply data to model, data_supply can be 0 (without generators) or 1 (with generators) [default 1]
		--train=file    Which tt file is for training purposes. Looks for files in tt folder [default train_l5]
		--test=file     Which tt file is for testing purposes. Looks for files in tt folder [default test_l5]
		TODO kontrola nad parametri v constants.py
		"""
		try:
			opts, args = getopt.getopt(self.argv,"m:n:e:a:ts:r:hg:d:",["train=", "test="])
		except getopt.GetoptError as e:
			print(e)
			sys.exit(1)

		for arg, val in opts:
			if arg == '-m':
				try:
					self.mode = int(val)
				except Exception as e:
					ArgParser._errorExit(e, 1)
			elif arg == '-n':
				self.name = "../cnnModels/"+str(val)
			elif arg == '-e':
				try:
					self.epoch = int(val)
				except Exception as e:
					ArgParser._errorExit(e, 1)
			elif arg == '-a':
				try:
					self.init_epoch = int(val)
				except Exception as e:
					ArgParser._errorExit(e, 1)
			elif arg == '-t':
				self.history = True
			elif arg == '-s':
				try:
					self.save_freq = int(val)
					if self.save_freq <= 0:
						raise Exception("Save frequency must be positive number!")
				except Exception as e:
					ArgParser._errorExit(e, 1)
			elif arg == '-r':
				try:
					self.model_version = int(val)
				except Exception as e:
					ArgParser._errorExit(e, 1)
			elif arg == '-h':
				print(self._parseArgv.__doc__)
				sys.exit(0)
			elif arg == '-d':
				try:
					self.data_supply = int(val)
				except Exception as e:
					ArgParser._errorExit(e, 1)
			elif arg == "--train":
				self.train_file = "../tt_files/"+val+".pickle"
			elif arg == "--test":
				self.test_file = "../tt_files/"+val+".pickle"
			else:
				print(f"Unknown argument: {arg} {val} {type(val)}")
