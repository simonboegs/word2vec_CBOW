from params import EPOCHS, LR
import torch
from model import CBOW_Model
from utils import create_dataloaders
import argparse

# parse args and see if they want to give file for vocab
# if not, just make another one
parser = argparse.ArgumentParser()
parser.add_argument("--vocab", type=str, required=True)
args = parser.parse_args()
if args.vocab is None:
    vocab = create_vocab()
else:
    try:
        vocab = torch.load(args.vocab)
    except:
        print("Vocab file not found")

model = CBOW_Model(len(vocab))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train(model, loss_fn, optimizer, epochs, scheduler=None):
    train_dl, test_dl = create_dataloaders()
    for epoch in range(epochs):
        for i, data in enumerate(train_dl):
            inputs, labels = data
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10
                running_loss = 0
        if scheduler is not None:
            scheduler.step()


