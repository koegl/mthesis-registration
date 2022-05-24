import argparse
import pathlib
import glob

import tensorflow as tf

from logic.cnn_classifier import Classifier
from utils import get_datasets, convert_cmd_args_to_correct_type
from train import train


def main(params):

    params = convert_cmd_args_to_correct_type(params)

    # create the datasets
    train_ds, val_ds, test_ds = get_datasets(params['train_and_val_dir'], params['test_dir'], params['batch_size'],
                                             params['over_fit_images'])

    # create an instance of the classifier
    model = Classifier()

    # loss function
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimiser = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    # train the model
    train(model, optimiser, loss_function, train_ds, val_ds, params['epochs'], params['validate'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-tvd", "--train_and_val_dir", default="/Users/fryderykkogl/Data/ViT_training_data/renamed_data/train",
                        help="path to the training data")
    parser.add_argument("-td", "--test_dir", default="//Users/fryderykkogl/Data/ViT_training_data/renamed_data/test",
                        help="path to the test data")

    parser.add_argument("-bs", "--batch_size", default=1)
    parser.add_argument("-e", "--epochs", default=10)

    parser.add_argument("-lr", "--learning_rate", default=0.001)

    parser.add_argument("-off", "--over_fit_images", default=50, type=int,
                        help="amount of images to be used for over-fit-training")

    parser.add_argument("-v", "--validate", default=True, type=bool,
                        help="if true, the model will be validated")

    args = parser.parse_args()

    main(args)
