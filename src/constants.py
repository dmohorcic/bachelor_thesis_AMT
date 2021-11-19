import librosa

HOP_LENGTH = 1024 # 128*n
BINS_PER_OCTAVE = 24 # 12*m
N_BINS = BINS_PER_OCTAVE*8
FMIN = librosa.note_to_hz(librosa.midi_to_note(21))

BATCH_SIZE = 256 # 512 gives smallest training time
WINDOW = 5

THRESHOLD = 0.5