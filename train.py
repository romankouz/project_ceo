"""This is just so that pylint will stop fucking yelling at me."""

import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from model import MNISTPredictor

def train(dataset="MNIST", optim="Adam"):
    """This is just so that pylint will stop fucking yelling at me."""
    if dataset == "MNIST":
        dataset = MNIST('', train=True, download=os.path.isdir('MNIST'),
                            transform=transforms.ToTensor())
        mnist_train, mnist_val = random_split(dataset, [55000, 5000])
        train_loader = DataLoader(mnist_train, batch_size=32)
        val_loader = DataLoader(mnist_val, batch_size=32)
        # model
        model = MNISTPredictor(optim)
        # training
        trainer = pl.Trainer(gpus=1, precision=16, max_epochs=500)
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
        train(args.dataset if args.dataset is not None else "MNIST", "RMSProp")
        train(args.dataset if args.dataset is not None else "MNIST", "SGD")
    else:
        train(args.dataset if args.dataset is not None else "MNIST",
              args.optim if args.optim is not None else "Adam")
        