# word2vec-CBOW

Pytorch implementation of word2vec "continuous bag of words" model from the paper:
https://arxiv.org/pdf/1301.3781.pdf

## Usage
---

### Step 1. Clone this repository

	git clone https://github.com/simonboegs/word2vec_CBOW

### Step 2. Install dependancies

	pip3 install -r requirements.txt

### Step 3. Adjust parameters

Parameters are adjustable in `params.py'
- dataset (`WikiText2`)
- vocab construction
- data processing
- model training

### Step 4. Train model

	python3 train.py

Saves vocabulary object, model weights, and embeddings in saves/ folder.

### Step 5. Use embeddings

	python3 test.py distance word1 word2

	python3 test.py closest word

	python3 test.py farthest word
