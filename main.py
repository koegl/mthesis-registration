import argparse
import wandb

import tensorflow as tf

from logic.cnn_classifier import Classifier2
from utils import get_datasets, convert_cmd_args_to_correct_type, initialise_wandb
from train import train


def main(params):

    params = convert_cmd_args_to_correct_type(params)

    # create the datasets
    train_ds, val_ds, test_ds = get_datasets(params['train_and_val_dir'], params['test_dir'], params['batch_size'],
                                             params['over_fit_images'])

    # create an instance of the classifier
    model = Classifier2()

    # loss function
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimiser = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    initialise_wandb(params, len(train_ds), len(val_ds),
                     project="Classification-tf", entity="fryderykkogl")

    # train the model
    train(model, optimiser, loss_function, train_ds, val_ds, params['epochs'], params['validate'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-tvd", "--train_and_val_dir", default="/Users/fryderykkogl/Data/ViT_training_data/renamed_data/train",
                        help="path to the training data")
    parser.add_argument("-td", "--test_dir", default="//Users/fryderykkogl/Data/ViT_training_data/renamed_data/test",
                        help="path to the test data")

    parser.add_argument("-bs", "--batch_size", default=256)
    parser.add_argument("-e", "--epochs", default=20)

    parser.add_argument("-lr", "--learning_rate", default=0.0001)

    parser.add_argument("-off", "--over_fit_images", default=12499, type=int,
                        help="amount of images to be used for over-fit-training")

    parser.add_argument("-v", "--validate", default=True, type=bool,
                        help="if true, the model will be validated")

    args = parser.parse_args()

    main(args)
