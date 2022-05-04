import argparse
import torch
from torch.nn import CosineSimilarity

parser = argparse.ArgumentParser()
parser.add_argument("method")
args = parser.parse_args()

# seems like a huge complicated process to get these commands to work.
# subparseres and shit, lots of technicalities
# we will get it done tho

def distance(word1, word2, cos_fn):
    word1_idx = vocab(word1)
    word2_idx = vocab(word2)
    word1_embed = embeddings[word1_idx]
    word2_embed = embeddings[word2_idx]
    return cos(word1_embed, word2_embed).item()

def top_n(word, cos_fn, n=5, closest=False):
    word_idx = vocab(word)
    word_embed = embeddings[word_idx]
    word_repeat_tensor = word_embed.repeat(len(vocab), 1)
    similarity_tensor = cos_fn(embeddings, word_repeat_tensor) 
    top = torch.topk(similarity_tensor, k=n+1, largest=closest, sorted=True)
    top_values, top_indices = top
    words_values = [(vocab.lookup_token(top_indices[i], top_values[i])) for i in range(1, len(top_values))]
    return words_values

if __name__ == "__main__":
    vocab = torch.load("vocab.pth")
    cos = nn.CosineSimilarity(dim=0)
    embeddings = torch.load("embeddings.pth")
