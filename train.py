from params import EPOCHS, LR
import torch
from model import CBOW_Model
from utils import create_dataloader, create_vocab, create_data_maps
from params import DATASET, EPOCHS, LR
import argparse

def train(model, loss_fn, optimizer, train_dl, test_dl, epochs, scheduler=None):
    print(f"training started - epochs: {epochs}, batch_size: {BATCH_SIZE}, LR: {LR}")
    for epoch in range(epochs):
        print(f"epoch #{epoch+1}") 
        for i, data in enumerate(train_dl):
            inputs, labels = data
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10
                print(f"loss: {last_loss}")
                running_loss = 0
        if scheduler is not None:
            scheduler.step()
    torch.save(model.state_dict(), "model_weights.pth")
    embeddings_raw = list(model.parameters())[0]
    embeddings = nn.functional.normalize(embeddings_raw)
    torch.save(embeddings, "embeddings.pth")
    print(f"training finished")

if __name__ == "__main__":
    train_data, test_data = create_data_maps(DATASET)
    vocab = create_vocab(train_data)
    print(f"created vocab - size: {len(vocab)}")
    train_dl = create_dataloader(train_data)
    test_dl = create_dataloader(test_data)
    model = CBOW_Model(len(vocab))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train(model, loss_fn, optimizer, EPOCHS)
