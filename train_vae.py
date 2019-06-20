import argparse

import matplotlib.pyplot as plt

import numpy as np
from itrl.models.vae import VAE
from itrl.models.classifiers import MLP
from itrl.data.datasets import load_cleveland_heart_disease, ADULT_DATASET_COLUMNS
from pyrsistent import m

from torch import optim
from torch.utils.data.dataloader import DataLoader

from itrl.data.datasets import load_dataset

from copy import deepcopy

from sklearn.metrics import accuracy_score


def impute(train_data_loader, vae):
    x_vae = vae.predict(train_data_loader)
    train_vae_dataset = deepcopy(train_data_loader.dataset)

    # only replace the missing values
    for column, mask in train_vae_dataset.masks.items():
        train_vae_dataset.x[mask, column] = x_vae[mask, column]

    train_vae_data_loader = DataLoader(train_vae_dataset)

    return train_vae_data_loader

def parse_list(arg_type):
    def _parse_list(arg: str):
        if "," in arg:
            return list(map(lambda s: arg_type(s.strip()), arg.split(",")))

    return _parse_list

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data-path", default=None)
    arg_parser.add_argument("--dataset")
    arg_parser.add_argument("--encoder-size", default=5, type=int)
    arg_parser.add_argument("--z-size", default=5, type=int)
    arg_parser.add_argument("--nb-epochs", default=10, type=int)
    arg_parser.add_argument("--columns", default="age,hours-per-week", type=parse_list(str))
    arg_parser.add_argument("--percentages", default="0.1,0.2,0.3,0.4", type=parse_list(float))
    args = arg_parser.parse_args()

    accuracies, f1_scores = [], []
    vae_accuracies, vae_f1_scores = [], []

    columns = {c: i for i, c in enumerate(ADULT_DATASET_COLUMNS[:len(ADULT_DATASET_COLUMNS) - 1])}


    for percent in args.percentages:
        print("Training on data with {} corruption".format(percent))
        train_dataset, valid_dataset = load_dataset(args.dataset)(columns=[columns[c] for c in args.columns],
                                                                  percent=percent)

        train_params = m(
            learning_rate=0.00001,
            minibatch_size=64,
            nb_epochs=args.nb_epochs
        )
        vae = VAE(train_dataset.nb_features, args.encoder_size, args.z_size)
        network_optimizer = optim.Adam(vae.parameters(), lr=train_params.learning_rate)

        train_data_loader = DataLoader(train_dataset, batch_size=train_params.minibatch_size)
        valid_data_loader = DataLoader(valid_dataset, batch_size=train_params.minibatch_size)

        print("Training VAE...")
        print("=" * 100)
        vae.fit(train_params, network_optimizer, train_data_loader, valid_data_loader)

        print("\nTraining classifier on regular data...")
        classifier = MLP(train_dataset.nb_features, train_dataset.nb_classes, hidden_layer_sizes=None)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=train_params.learning_rate)

        train_params = m(
            learning_rate=0.0001,
            minibatch_size=64,
            nb_epochs=args.nb_epochs
        )

        classifier.fit(train_params, classifier_optimizer, train_data_loader)
        preds_classifier = classifier.predict(valid_data_loader)
        accuracy = accuracy_score(valid_dataset.y, preds_classifier.argmax(axis=1))
        accuracies.append(accuracy)

        print("Accuracy: {}%".format(accuracy * 100))
        print("=" * 100)
        print("Training classifer on VAE transformed data...")
        train_vae_data_loader = impute(train_data_loader, vae)
        classifier = MLP(train_vae_data_loader.dataset.nb_features, train_vae_data_loader.dataset.nb_classes,
                         hidden_layer_sizes=None)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=train_params.learning_rate)

        train_params = m(
            learning_rate=0.0001,
            minibatch_size=64,
            nb_epochs=args.nb_epochs
        )

        classifier.fit(train_params, classifier_optimizer, train_vae_data_loader)


        valid_vae_data_loader = impute(valid_data_loader, vae)

        # valid_vae_dataset = deepcopy(valid_dataset)
        # valid_vae_dataset.x = vae.predict(valid_data_loader)
        # valid_vae_data_loader = DataLoader(valid_vae_dataset)
        preds_classifier = classifier.predict(valid_vae_data_loader)



        accuracy = accuracy_score(valid_vae_data_loader.dataset.y, preds_classifier.argmax(axis=1))
        vae_accuracies.append(accuracy)

        print("Accuracy: {}%".format(accuracy * 100))
        print("=" * 100)
        print("*" * 100)


    plt.plot(args.percentages, accuracies, label="corrupt")
    plt.plot(args.percentages, vae_accuracies, label="vae")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
