import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model.cnn import CNN
from train import train_model
from eval import eval_model
from data import load_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script argument parser")

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    args = parser.parse_args()

    # data
    train_dataloader, test_dataloader = load_dataloader()

    # train
    model = train_model(args.epochs, args.lr, train_dataloader)
    
    # test
    eval_model(model, test_dataloader)