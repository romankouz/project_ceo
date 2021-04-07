"""This is just so that pylint will stop fucking yelling at me."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from adabelief_pytorch import AdaBelief

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


class MNISTPredictor(pl.LightningModule):

    """This is just so that pylint will stop fucking yelling at me."""

    def __init__(self, optim="Adam"):
        super().__init__()
        self.optim = optim
        self.forward_pass = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.forward_pass(x)

    def configure_optimizers(self):
        if self.optim == "Adabelief":
            optimizer = AdaBelief(self.parameters(), lr=1e-3)
        else:
            optimizer = getattr(torch.optim, self.optim)(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, target = train_batch
        inputs = inputs.mean(1)
        inputs = inputs.view(inputs.size(0), -1)
        out = self(inputs)
        loss = F.cross_entropy(out, target)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, target)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        file_train_loss_mnist = 'manual_logs/' + self.optim.lower() + '_train_loss_mnist.txt'
        file_train_acc_mnist = 'manual_logs/' + self.optim.lower() + '_train_acc_mnist.txt'
        with open(file_train_loss_mnist, 'a') as current_file:
            current_file.write(str(loss) + '\n')
            current_file.close()
        with open(file_train_acc_mnist, 'a') as current_file:
            current_file.write(str(acc) + '\n')
            current_file.close()
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, target = val_batch
        inputs = inputs.mean(1)
        inputs = inputs.view(inputs.size(0), -1)
        out = self(inputs)
        loss = F.cross_entropy(out, target)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, target)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        file_val_loss_mnist = 'manual_logs/' + self.optim.lower() + '_val_loss_mnist.txt'
        file_val_acc_mnist = 'manual_logs/' + self.optim.lower() + '_val_acc_mnist.txt'
        with open(file_val_loss_mnist, 'a') as current_file:
            current_file.write(str(loss) + '\n')
            current_file.close()
        with open(file_val_acc_mnist, 'a') as current_file:
            current_file.write(str(acc) + '\n')
            current_file.close()
        return loss


class CIFAR10Predictor(pl.LightningModule):

    """This is just so that pylint will stop fucking yelling at me."""

    def __init__(self, optim="Adam"):
        super().__init__()
        self.optim = optim
        self.forward_pass = nn.Sequential(
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.forward_pass(x)

    def configure_optimizers(self):
        if self.optim == "Adabelief":
            optimizer = AdaBelief(self.parameters(), lr=1e-3)
        else:
            optimizer = getattr(torch.optim, self.optim)(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, target = train_batch
        inputs = torch.mean(inputs, 1)
        inputs = inputs.view(inputs.size(0), -1)
        out = self(inputs)
        loss = F.cross_entropy(out, target)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, target)
        file_train_loss_cifar10 = 'manual_logs/' + self.optim.lower() + '_train_loss_cifar10.txt'
        file_train_acc_cifar10 = 'manual_logs/' + self.optim.lower() + '_train_acc_cifar10.txt'
        with open(file_train_loss_cifar10, 'a') as current_file:
            current_file.write(str(loss) + '\n')
            current_file.close()
        with open(file_train_acc_cifar10, 'a') as current_file:
            current_file.write(str(acc) + '\n')
            current_file.close()
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, target = val_batch
        inputs = torch.mean(inputs, 1)
        inputs = inputs.view(inputs.size(0), -1)
        out = self(inputs)
        loss = F.cross_entropy(out, target)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, target)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        file_val_loss_cifar10 = 'manual_logs/' + self.optim.lower() + '_val_loss_cifar10.txt'
        file_val_acc_cifar10 = 'manual_logs/' + self.optim.lower() + '_val_acc_cifar10.txt'
        with open(file_val_loss_cifar10, 'a') as current_file:
            current_file.write(str(loss) + '\n')
            current_file.close()
        with open(file_val_acc_cifar10, 'a') as current_file:
            current_file.write(str(acc) + '\n')
            current_file.close()
        return loss
