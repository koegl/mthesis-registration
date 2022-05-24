import os
import glob
import wandb


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
