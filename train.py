import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model.cnn import CNN
from data import load_dataloader


def train_model(epochs, lr, dataloader):
    model = CNN()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss/len(dataloader)}")
    
    return model