"""This is just so pylint stops fucking yelling at me."""
import os
import re

import numpy as np
import matplotlib.pyplot as plt


quantity = re.compile(r'\d\.\d*,')
scientific = re.compile(r'\d\.\d*e-\d*,')
logger_dict = {}
for file in os.listdir("manual_logs"):
    with open("manual_logs/" + file, 'r') as f:

        def float_note(x_value):
            """Converts the list of string logs into a
            list of strings that are float convertible."""
            if len(re.findall(quantity, x_value)) == 1:
                return re.findall(quantity, x_value)[0][:-1]
            return re.findall(scientific, x_value)[0][:-1]

        print(file[:-4] + " logged successfully!")
        raw_values = [float(float_note(x_value)) for x_value in f.read().split('\n')[:-1]]
        # this is wrong
        n = len(raw_values)
        batch_means = [float(np.mean(raw_values[(n//500)*i:(n//500)*(i+1)])) for i in range(500)]
        logger_dict[file] = batch_means

fig, ax = plt.subplots(2,2)
fig.set_size_inches(10,10)
for key, value in logger_dict.items():
    if "train_loss" in key:
        ax[0,0].plot(range(len(value)), value)
        ax[0,0].set_title('Training Loss')
        ax[0,0].set_xlabel('Epochs')
        ax[0,0].set_ylabel('Loss')
        plt.close()
    if "train_acc" in key:
        ax[0,1].plot(range(len(value)), value)
        ax[0,1].set_title('Training Accuracy')
        ax[0,1].set_xlabel('Epochs')
        ax[0,1].set_ylabel('Accuracy')
        plt.close()
    if "val_loss" in key:
        ax[1,0].plot(range(len(value)), value)
        ax[1,0].set_title('Validation Loss')
        ax[1,0].set_xlabel('Epochs')
        ax[1,0].set_ylabel('Loss')
        plt.close()
    if "val_acc" in key:
        ax[1,1].plot(range(len(value)), value)
        ax[1,1].set_title('Validation Accuracy')
        ax[1,1].set_xlabel('Epochs')
        ax[1,1].set_ylabel('Accuracy')
        plt.close()
fig.savefig('images/metrics.png')
