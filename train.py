"""Script of training models from command line."""

import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.datasets
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from model import MNISTPredictor, CIFAR10Predictor
from images import metrics_image


def file_removal(dataset, optim):
    """Deletes previous logs. At the moment we don't support custom file names."""

    try:
        os.remove('manual_logs/' + optim.lower() + '_train_loss_' + dataset.lower() + '.txt')
    except FileNotFoundError:
        print("No previous training loss logs for this optimizer detected.")

    try:
        os.remove('manual_logs/' + optim.lower() + '_val_loss_' + dataset.lower() + '.txt')
    except FileNotFoundError:
        print("No previous validation loss logs for this optimizer detected.")

    try:
        os.remove('manual_logs/' + optim.lower() + '_train_acc_' + dataset.lower() + '.txt')
    except FileNotFoundError:
        print("No previous training accuracy logs for this optimizer detected.")

    try:
        os.remove('manual_logs/' + optim.lower() + '_val_acc_' + dataset.lower() + '.txt')
    except FileNotFoundError:
        print("No previous validation accuracy logs for this optimizer detected.")


def train(dataset="MNIST", optim="Adam", overwrite_logs=True, n_epochs=500):

    """Training Function"""
    if overwrite_logs:
        file_removal(dataset, optim)

    data = getattr(torchvision.datasets, dataset)('./', train=True,
                        download=not os.path.isdir(dataset),
                        transform=transforms.ToTensor())
    n_data = len(data)
    train_n = round(11*n_data/12)
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
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/' + optim + '/' + dataset + '/',
        monitor='val_loss',
        save_top_k=10,
        mode="min",
    )
    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=n_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False)
    parser.add_argument('--optim', required=False)
    parser.add_argument('--overwrite_logs', required=False)
    parser.add_argument('--n_epochs', required=False)
    args = parser.parse_args()

    best_model_dict = {}

    dataset_in = args.dataset if args.dataset is not None else "MNIST"
    optim_in= args.optim if args.optim is not None else "Adam"
    overwrite_logs_in = bool(args.optim) if args.optim is not None else True
    n_epochs_in = int(args.n_epochs) if args.n_epochs is not None else 500
    if args.optim == "All":
        # pylint: disable=line-too-long
        best_model_dict[(dataset_in, "Adam", n_epochs_in)] = train(dataset_in, "Adam", overwrite_logs_in, n_epochs_in)
        best_model_dict[(dataset_in, "Adagrad", n_epochs_in)] = train(dataset_in, "Adagrad", overwrite_logs_in, n_epochs_in)
        best_model_dict[(dataset_in, "Adamax", n_epochs_in)] = train(dataset_in, "Adamax", overwrite_logs_in, n_epochs_in)
        best_model_dict[(dataset_in, "RMSprop", n_epochs_in)] = train(dataset_in, "RMSprop", overwrite_logs_in, n_epochs_in)
        best_model_dict[(dataset_in, "SGD", n_epochs_in)] = train(dataset_in, "SGD", overwrite_logs_in, n_epochs_in)
        best_model_dict[(dataset_in, "Adabelief", n_epochs_in)] = train(dataset_in, "Adabelief", overwrite_logs_in, n_epochs_in)
        metrics_image(dataset_in, n_epochs_in)
    else:
        best_model_dict[(dataset_in, optim_in, n_epochs_in)] = train(dataset_in, optim_in, overwrite_logs_in, n_epochs_in)
        # pylint: enable=line-too-long
    best_model_txt = open('checkpoints/best_model_dict.txt', 'w')
    best_model_txt.write(str(best_model_dict))
    best_model_txt.close()
    # insert code here that deletes trailing newlines if need be
        