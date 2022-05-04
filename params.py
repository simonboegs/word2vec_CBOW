# -- DATA HYPERPARAMETERS --
DATASET = "WikiText2"

# num of dimensions of embedding vectors
EMBED_DIMENSION = 300

EMBED_MAX_NORM = 1

# min amount of times a word must show up in dataset to be included in vocabulary
MIN_WORD_FREQ = 50

# num of words before/after middle word
N_WORDS = 5

# max number of words in paragraph - no more words in the paragraph after this cutoff are considered
MAX_PARAGRAPH_LEN = 256


# -- TRAINING HYPERPARAMETERS --

BATCH_SIZE = 100
EPOCHS = 5
LR = .003
