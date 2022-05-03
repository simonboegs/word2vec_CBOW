import torch.nn as nn
from params import EMBED_DIMENSION, EMBED_MAX_NORM

class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
    
    def forward(self, inputs_):
        pass

#model = CBOW_Model((len(vocab))
#where is len(vocab) going to come from??
#actually i should probs not define that in this file
