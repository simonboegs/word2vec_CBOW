from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from params import DATASET, N_WORDS, MAX_PARAGRAPH_LEN, BATCH_SIZE
from functools import partial

tokenizer = get_tokenizer("basic_english",language="en")

def collate(batch, vocab):
    batch_input, batch_output = [], []
    # iterate over paragraphs in batch
    for paragraph in batch:
        tokens = tokenizer(paragraph)
        token_ids = vocab(tokens)
        # skip paragraph if too small
        if len(token_ids) < params.N_WORDS * 2 + 1:
            continue
        # cut off paragraph if too long
        elif len(token_ids) > MAX_PARAGRAPH_LEN:
            token_ids = token_ids[:MAX_PARAGRAPH_LEN]
        # iterate over windows (of length N_WORDS * 2 + 1)
        for idx in range(len(token_ids) - N_WORDS * 2):
            sequence = token_ids[idx : (idx + N_WORDS * 2 + 1)]
            # target is middle word in sequence
            target = sequence.pop(N_WORDS)
            batch_input.append(sequence)
            batch_output.append(target)
    batch_input_tensor = torch.tensor(batch_input, dtype=torch.long)
    batch_output_tensor = torch.tensor(batch_output, dtype=torch.long)
    return batch_input_tensor, batch_output_tensor

def create_data_maps(dataset: str):
    if dataset == "WikiText2":
        train_data, test_data = WikiText2(root="data", download=True)
    else:
        raise ValueError("unknown dataset " + DATASET)
    train_data_map = to_map_style_dataset(train_data)
    test_data_map = to_map_style_dataset(test_data)
    return (train_data_map, test_data_map)

def create_dataloaders(dataset, vocab):
    train_data_map, test_data_map = create_data_maps(dataset)
    train_dl = DataLoader(
        train_data_map,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate, vocab=vocab)
    )
    test_dl = DataLoader(
        test_data_map,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate, vocab=vocab)
    )
    return (train_dl, test_dl)

