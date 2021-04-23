"""Script of testing models from command line."""

import os
import argparse
import ast
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.datasets
from torchvision import transforms
from model import MNISTPredictor, CIFAR10Predictor
from images import metrics_image


def file_removal(dataset, optim):
    """Deletes previous logs. At the moment we don't support custom file names."""

    try:
        os.remove('manual_logs/' + optim.lower() + '_test_loss_' + dataset.lower() + '.txt')
    except FileNotFoundError:
        print("No previous training loss logs for this optimizer detected.")

    try:
        os.remove('manual_logs/' + optim.lower() + '_test_acc_' + dataset.lower() + '.txt')
    except FileNotFoundError:
        print("No previous validation loss logs for this optimizer detected.")

# def generate_test_results():
#   This should create the test results for all of the optimizer runs and report it in a neat table.

def test(dataset="MNIST", optim="Adam", overwrite_logs=True, n_epochs=500):
    """Testing Function"""

    test_data = getattr(torchvision.datasets, dataset)('./', train=False,
                        download=not os.path.isdir(dataset),
                        transform=transforms.ToTensor())
    # model
    with open('checkpoints/best_model_dict.txt', 'r') as best_model_txt:
        filepath = best_model_txt.read()
    best_model_dict = ast.literal_eval(filepath)
    if dataset=="MNIST":
        model = MNISTPredictor.load_from_checkpoint(
            checkpoint_path=best_model_dict[(dataset, optim, n_epochs)]
        )
        model.eval()
        test_loader = DataLoader(test_data, batch_size=32)
    if dataset=="CIFAR10":
        model = CIFAR10Predictor.load_from_checkpoint(
            checkpoint_path=best_model_dict[(dataset, optim, n_epochs)]
        )
        model.eval()
        test_loader = DataLoader(test_data, batch_size=32)

    trainer = pl.Trainer()
    model.optim = optim
    avg_test_metrics = trainer.test(model, test_loader)
    return avg_test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False)
    parser.add_argument('--optim', required=False)
    parser.add_argument('--overwrite_logs', required=False)
    parser.add_argument('--n_epochs', required=False)
    args = parser.parse_args()

    dataset_in = args.dataset if args.dataset is not None else "MNIST"
    optim_in = args.optim if args.optim is not None else "Adam"
    overwrite_logs_in = bool(args.optim) if args.optim is not None else True
    n_epochs_in = int(args.n_epochs) if args.n_epochs is not None else 500

    test_result_dict = {}
    if args.optim == "All":
        # pylint: disable=line-too-long
        test_result_dict[(dataset_in, "Adam", n_epochs_in)] = test(dataset_in, "Adam", overwrite_logs_in, n_epochs_in)
        test_result_dict[(dataset_in, "Adagrad", n_epochs_in)] = test(dataset_in, "Adagrad", overwrite_logs_in, n_epochs_in)
        test_result_dict[(dataset_in, "Adamax", n_epochs_in)] = test(dataset_in, "Adamax", overwrite_logs_in, n_epochs_in)
        test_result_dict[(dataset_in, "RMSprop", n_epochs_in)] = test(dataset_in, "RMSprop", overwrite_logs_in, n_epochs_in)
        test_result_dict[(dataset_in, "SGD", n_epochs_in)] = test(dataset_in, "SGD", overwrite_logs_in, n_epochs_in)
        test_result_dict[(dataset_in, "Adabelief", n_epochs_in)] = test(dataset_in, "Adabelief", overwrite_logs_in, n_epochs_in)
        metrics_image(dataset_in)
    else:
        test_result_dict[(dataset_in, optim_in, n_epochs_in)] = test(dataset_in, optim_in, overwrite_logs_in, n_epochs_in)
        # pylint: enable=line-too-long
    test_results = open('manual_logs/testing_results.txt', 'w')
    test_results.write(str(test_result_dict))
    test_results.close()
