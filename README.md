# word2vec-CBOW

Pytorch implementation of word2vec "continuous bag of words" model from the paper:
https://arxiv.org/pdf/1301.3781.pdf

## Usage
---

### Step 1. Clone this repository

	$ git clone https://github.com/simonboegs/word2vec_CBOW
	$ cd word2vec_CBOW/

### Step 2. Install dependancies

	$ pip3 install -r requirements.txt

### Step 3. Adjust parameters

Parameters are adjustable in `params.py`
- dataset (`WikiText2` or `WikiText103`)
- vocab construction
- data processing
- model training

### Step 4. Train model

	python3 train.py

Saves vocabulary object, model weights, and embeddings in saves/ folder.

### Step 5. Use embeddings
Similarity between words is measured by cosine similarity.
```
$ python3 test.py [COMMAND]

# cosine similarity between 2 words
$ python3 test.py cos-sim [word1] [word2]

# top n similar words
$ python3 test.py top [word]

# Optional arguments:
	# -n [N]       size of list
	# --farthest   get least similar words instead
$ python3 test.py [word] -n 10 --farthest
```
