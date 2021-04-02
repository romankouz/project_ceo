"""This is just so that pylint will stop fucking yelling at me."""

import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.datasets
from torchvision import transforms
from model import MNISTPredictor, CIFAR10Predictor

def train(dataset="MNIST", optim="Adam"):
    """This is just so that pylint will stop fucking yelling at me."""
    data = getattr(torchvision.datasets, dataset)('./', train=True, 
                        download=not os.path.isdir(dataset),
                        transform=transforms.ToTensor())
    n_data = len(data)
    train_n = round(11*n/12)
    train_data, val_data = random_split(data, [train_n, n_data-train_n])
    # model
    if dataset=="MNIST":
        model = MNISTPredictor(optim)
        train_loader = DataLoader(train_data, batch_size=32)
        val_loader = DataLoader(val_data, batch_size=32)
    if dataset=="CIFAR10":
        model = CIFAR10Predictor(optim)
        train_loader = DataLoader(train_data, batch_size=32)
        val_loader = DataLoader(val_data, batch_size=32)
    # training
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=2)
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False)
    parser.add_argument('--optim', required=False)
    args = parser.parse_args()
    if args.optim == "All":
        train(args.dataset if args.dataset is not None else "MNIST", "Adam")
        train(args.dataset if args.dataset is not None else "MNIST", "Adagrad")
        train(args.dataset if args.dataset is not None else "MNIST", "Adamax")
        train(args.dataset if args.dataset is not None else "MNIST", "RMSprop")
        train(args.dataset if args.dataset is not None else "MNIST", "SGD")
    else:
        train(args.dataset if args.dataset is not None else "CIFAR10",
              args.optim if args.optim is not None else "Adam")
        