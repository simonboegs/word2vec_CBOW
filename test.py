import argparse
import torch
from torch.nn import CosineSimilarity
from params import VOCAB_SAVE, EMBEDDINGS_SAVE

class Test():
    def __init__(self):
        self.cos_fn = CosineSimilarity(dim=0)
        try:
            self.vocab = torch.load(VOCAB_SAVE)
        except FileNotFoundError:
            raise FileNotFoundError("could not find vocabulary file " + VOCAB_SAVE)
        try:
            self.embeddings = torch.load(EMBEDDINGS_SAVE)
        except FileNotFoundError:
            raise FileNotFoundError("could not find embeddings file " + EMBEDDINGS_SAVE)

    def cos_sim(self, word1, word2):
        if not self.vocab.__contains__(word1):
            raise ValueError("word \"" + word1 + "\" not in vocabulary")
        if not self.vocab.__contains__(word2):
            raise ValueError("word \"" + word2 + "\" not in vocabulary")
        word1_idx, word2_idx = self.vocab([word1,word2])
        word1_embed = self.embeddings[word1_idx]
        word2_embed = self.embeddings[word2_idx]
        return self.cos_fn(word1_embed, word2_embed).item()

    def top_n(self, word, n, farthest):
        if not self.vocab.__contains__(word):
            raise ValueError("word \"" + word + "\" not in vocabulary")
        word_idx = self.vocab.__getitem__(word)
        word_embed = self.embeddings[word_idx]
        word_repeat_tensor = word_embed.repeat(len(self.vocab), 1)
        similarity_tensor = self.cos_fn(self.embeddings, word_repeat_tensor) 
        top = torch.topk(similarity_tensor, k=n+1, largest=(not farthest), sorted=True)
        top_values, top_indices = top
        words_values = [(self.vocab.lookup_token(top_indices[i]), top_values[i].item()) for i in range(1, len(top_values))]
        return words_values

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command",help="command")

    dist_parser = subparsers.add_parser("cos-sim", help="cosine similarity between two words. values between -1 (far) and 1 (close)")
    dist_parser.add_argument("word1")
    dist_parser.add_argument("word2")

    top_parser = subparsers.add_parser("top", help="top-n (default n=5) closest words to given word by cosine similarity")
    top_parser.add_argument("word")
    top_parser.add_argument("-n", required=False, dest="n", type=int, default=5)
    top_parser.add_argument("--farthest", required=False, action="store_true", help="returns bottom-n words instead")

    args = parser.parse_args()

    test = Test()

    if args.command == "cos-sim":
        word1 = args.word1
        word2 = args.word2
        result = test.cos_sim(word1, word2)
        print(result)
    elif args.command == "top":
        word = args.word
        n = args.n
        farthest = args.farthest
        result = test.top_n(word, n, farthest)
        for i in range(len(result)):
            word = result[i][0]
            similarity = result[i][1]
            print("%-12s %12s" % (word, str(similarity)))
