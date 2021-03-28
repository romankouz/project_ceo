"""This is just so that pylint will stop fucking yelling at me."""

from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        optimizer = literal_eval("torch.optim." + self.optim + "(self.parameters(), lr=1e-3)")
        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, target = train_batch
        inputs = inputs.view(inputs.size(0), -1)
        out = self(inputs)
        loss = F.cross_entropy(out, target)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, target)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, target = val_batch
        inputs = inputs.view(inputs.size(0), -1)
        out = self(inputs)
        loss = F.cross_entropy(out, target)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, target)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
