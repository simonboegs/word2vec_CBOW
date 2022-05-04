from utils import tokenizer, create_data_maps
from params import MIN_WORD_FREQ, DATASET
import argparse

def create_vocab(train_data_map, filename=None):
    data_tokenized = map(tokenizer, train_data_map)
    vocab = build_vocab_from_iterator(
        data_tokenized,
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQ
    )
    vocab.set_default_index(vocab["<unk>"])
    torch.save(vocab, "vocab.pth"
    return vocab

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--filename", type=str, required=False)
    # args = parser.parse_args()
    train_data_map, test_data_map = create_data_maps(DATASET)
    create_vocab(train_data_map)
