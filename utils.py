import os
import glob
import wandb
import pathlib

import tensorflow as tf


def get_labels(params):

    train_and_val_dir = params.train_and_val_dir
    train_and_val_list = glob.glob(os.path.join(train_and_val_dir, '*.jpg'))

    # get the labels which are the first part of each file name
    labels = [path.split('/')[-1].split('.')[0] for path in train_and_val_list]

    return labels


def initialise_wandb(params, len_train, len_val, project="Classification", entity="fryderykkogl"):
    """
    Initialise everything for wand
    :param params: the user arguments
    :param len_train: the length of the training set
    :param len_val: the length of the validation set
    :param project: the project name
    :param entity: the entity name
    :return:
    """

    wandb.init(project=project, entity=entity)
    os.environ["WANDB_NOTEBOOK_NAME"] = "Classification"

    config_dict = {
        "learning_rate": float(params.learning_rate),
        "epochs": int(params.epochs),
        "batch_size": int(params.batch_size),
        "training_data": params.train_and_val_dir,
        "test_data": params.test_dir,
        "network_type": params.network_type,
        "device": params.device,
    }
    wandb.config = config_dict
    wandb.log(config_dict)
    wandb.log({"Training size": len_train,
               "Validation size": len_val})


def get_image_and_label(path):

    temp = tf.strings.split(path, '/')[-1]
    label = tf.strings.split(temp, '.')[0]

    if label == 'dog':
        label = 1
    else:
        label = 0

    image_string = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [224, 224])

    # # todo check if this rescaling works
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    image = datagen.standardize(image)

    return image, label


def configure_dataset_for_performance(ds, batch_size, autotune):
    # To train a model with this dataset you will want the data:
    # To be well shuffled.
    # To be batched.
    # Batches to be available as soon as possible.

    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=autotune)

    return ds


def get_datasets(train_data_dir, test_data_dir, batch_size, val_size=0.2):
    """
    Returns the train, val and test datasets
    :param train_data_dir: path to the training data
    :param test_data_dir: path to the test data
    :param batch_size: the batch size (neede for optimisation)
    :param val_size: the size of the validation set as a percentage
    :return: train, val and test datasets
    """

    train_data_dir = pathlib.Path(train_data_dir)
    test_data_dir = pathlib.Path(test_data_dir)

    # list all files in the directories
    train_image_count = len(list(train_data_dir.glob('*.jpg')))  # we need this for the shuffling and train/val split
    list_ds_train = tf.data.Dataset.list_files(str(train_data_dir / "*.jpg"), shuffle=False)
    list_ds_train = list_ds_train.shuffle(train_image_count, reshuffle_each_iteration=False)

    test_image_count = len(list(test_data_dir.glob('*.jpg')))
    list_ds_test = tf.data.Dataset.list_files(str(test_data_dir / "*.jpg"), shuffle=False)
    list_ds_test = list_ds_test.shuffle(test_image_count, reshuffle_each_iteration=False)

    # split into train and validation with val_size=0.2
    val_size = int(0.2 * train_image_count)
    train_ds = list_ds_train.skip(val_size)
    val_ds = list_ds_train.take(val_size)
    test_ds = list_ds_test

    # optimise datasets for performance
    autotune = tf.data.AUTOTUNE
    train_ds = configure_dataset_for_performance(train_ds, batch_size, autotune)
    val_ds = configure_dataset_for_performance(val_ds, batch_size, autotune)
    test_ds = configure_dataset_for_performance(test_ds, batch_size, autotune)

    return train_ds, val_ds, test_ds


def convert_cmd_args_to_correct_type(params):
    """
    All args are stored in string, but some of them need to be converted - will be saved as a dict
    :param params: the cmd line args
    :return: a dict with them stored correctly
    """

    params_dict = {'learning_rate': float(params.learning_rate),
                   'epochs': int(params.epochs),
                   'batch_size': int(params.batch_size),
                   'train_and_val_dir': params.train_and_val_dir,
                   'test_dir': params.test_dir,
                   }

    return params_dict
