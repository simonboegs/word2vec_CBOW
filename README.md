#word2vec-CBOW
Pytorch implementation of word2vec "continuous bag of words" model from the paper:
https://arxiv.org/pdf/1301.3781.pdf

## Usage
---

### Step 1. Clone this repository
`git clone https://github.com/simonboegs/word2vec_CBOW'

### Step 2. Install requirements

### Step 3. Adjust parameters
Parameters are adjustable in `params.py'
1. dataset (`WikiText2`)
2. vocab construction
3. data processing
4. model training

### Step 4. Train model
`python3 train.py`
Saves model weights in `model_weights.pth`
Saves embeddings in `embeddings.pth`
Will 
### Step 5. Use embeddings

