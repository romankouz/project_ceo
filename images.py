"""This is just so pylint stops fucking yelling at me."""
import os
import re

import numpy as np
import matplotlib.pyplot as plt

def metrics_image(dataset="MNIST"):
    """Creates plot of loss and accuracy analytics."""
    quantity = re.compile(r'\d\.\d*,')
    scientific = re.compile(r'\d\.\d*e-\d*,')
    logger_dict = {}
    for file in os.listdir("manual_logs"):
        if dataset.lower() in file:
            with open("manual_logs/" + file, 'r') as f:

                def float_note(x_value):
                    """Converts the list of string logs into a
                    list of strings that are float convertible."""
                    if len(re.findall(quantity, x_value)) == 1:
                        return re.findall(quantity, x_value)[0][:-1]
                    return re.findall(scientific, x_value)[0][:-1]

                raw_values = [float(float_note(x_value)) for x_value in f.read().split('\n')[:-1]]
                # this is wrong
                n = len(raw_values)
                batch_means = [float(np.mean(raw_values[(n//500)*i:(n//500)*(i+1)]))
                               for i in range(500)]
                logger_dict[file] = batch_means
                print(file[:-4] + " logged successfully!")

    fig, ax = plt.subplots(2,2)
    fig.suptitle(dataset + " Metrics")
    fig.set_size_inches(7,7)
    for key, value in logger_dict.items():
        if "train_loss" in key:
            ax[0,0].plot(range(len(value)), value, label=key.split('_')[0])
            ax[0,0].set_title('Training Loss')
            ax[0,0].set_xlabel('Epochs', fontsize=10)
            ax[0,0].set_ylabel('Loss', fontsize=10)
            plt.close()
        if "train_acc" in key:
            ax[0,1].plot(range(len(value)), value, label=key.split('_')[0])
            ax[0,1].set_title('Training Accuracy')
            ax[0,1].set_xlabel('Epochs', fontsize=10)
            ax[0,1].set_ylabel('Accuracy', fontsize=10)
            plt.close()
        if "val_loss" in key:
            ax[1,0].plot(range(len(value)), value, label=key.split('_')[0])
            ax[1,0].set_title('Validation Loss')
            ax[1,0].set_xlabel('Epochs', fontsize=10)
            ax[1,0].set_ylabel('Loss', fontsize=10)
            plt.close()
        if "val_acc" in key:
            ax[1,1].plot(range(len(value)), value, label=key.split('_')[0])
            ax[1,1].set_title('Validation Accuracy')
            ax[1,1].set_xlabel('Epochs', fontsize=10)
            ax[1,1].set_ylabel('Accuracy', fontsize=10)
            plt.close()
    ax[0,0].legend(fontsize='small')
    ax[0,1].legend(fontsize='small')
    ax[1,0].legend(fontsize='x-small')
    ax[1,1].legend(fontsize='small')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if dataset == "MNIST":
        fig.savefig('images/mnist_metrics')
    else:
        fig.savefig('images/cifar10_metrics.png')
