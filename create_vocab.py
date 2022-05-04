from utils import tokenizer

def create_vocab(train_data_map):
    data_tokenized = map(tokenizer, train_data_map)
    vocab = build_vocab_from_iterator(
        data_tokenized,
        specials=["<unk>"],
        min_freq=50
    )
    vocab.set_default_index(vocab["<unk>"])
    torch.save(vocab, "vocab.pth")
    return vocab

