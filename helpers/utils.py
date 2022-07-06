import numpy as np
import random
import os
import wandb
import glob

import torch

from architectures.densenet3d import DenseNet
from architectures.vit_standard_3d import ViTStandard3D


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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

    config_dict = {
        "batch_size": params.batch_size,
        "epochs": params.epochs,
        "learning_rate": params.learning_rate,
        "seed": params.seed,
        "train_dir": params.train_dir,
        "val_dir": params.val_dir,
        "device": str(params.device),
        "architecture_type": params.architecture_type,
        "dropout": params.dropout,
        "early_stopping": params.early_stopping,
        "lr_scheduler": params.lr_scheduler,
        "lr_scheduler_patience": params.lr_scheduler_patience,
        "lr_scheduler_min_lr": params.lr_scheduler_min_lr,
        "lr_scheduler_factor": params.lr_scheduler_factor,
        "training_size": len_train,
        "validation_size": len_val
    }

    wandb.init(config=config_dict, project=project, entity=entity)
    os.environ["WANDB_NOTEBOOK_NAME"] = "Classification"


def get_architecture(params):

    architecture_type = params.architecture_type

    patch_size = get_patch_size_from_data_folder(params.train_dir)

    if "densenet" in architecture_type.lower():

        dense_dict = {"densenet121": (6, 12, 24, 16),
                      "densenet169": (6, 12, 32, 32),
                      "densenet201": (6, 12, 48, 32),
                      "densenet264": (6, 12, 64, 48)}

        model = DenseNet(
            growth_rate=32,
            block_config=dense_dict[architecture_type.lower()],
            num_init_features=64,
            bn_size=4,
            drop_rate=float(params.dropout),
            num_classes=20,
            memory_efficient=False)

    elif architecture_type.lower() == "vit":
        model = ViTStandard3D(
            dim=128,
            volume_size=patch_size,
            patch_size=4,
            num_classes=20,
            channels=2,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=float(params.dropout),
            device="cpu"
        )
    else:
        raise NotImplementedError("Architecture not supported. Only DenseNet and ViTStandard are supported.")

    return model


def calculate_accuracy(output, label):
    """
    Calculate the accuracy between the output and the label
    :param output: output of the network
    :param label: label/target/ground truth
    :return: accuracy
    """

    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output)

    # get the index of the maximal value in the output and the label
    output_argmax = output.argmax(dim=1)
    label_argmax = label.argmax(dim=1)

    # compare the maximal indices - a list of booleans
    correct_bool = (output_argmax == label_argmax)

    # transform the list of bools to a list of floats
    correct_float = correct_bool.float()

    # calculate the mean of the list of floats (the accuracy)
    accuracy = correct_float.mean()

    return accuracy


def get_label_from_label_id(label_id):
    """
    Get a dictionary that maps the label id to the label
    :return:
    """
    empty = np.zeros(20)
    empty_list = []
    for i in range(20):
        empty[i] = 1
        empty_list.append(empty.copy())
        empty[i] = 0

    label_id_to_label_dict = {'{0:05b}'.format(i): empty_list[i] for i in range(20)}

    return label_id_to_label_dict[label_id]


def get_label_id_from_label(label):
    """
    Get the label id from the label
    :param label: the label
    :return: the label id
    """
    label_id = '{0:05b}'.format(label.argmax())

    return label_id


def patch_inference(model, patches, offsets):
    """
    This function does inference for a patch_pair
    :param model: the pretrained model
    :param patches: the patch pair
    :param offsets: the possible offsets
    :return: the expected displacement
    """
    from helpers.visualisations import visualise_per_class_accuracies

    with torch.no_grad():

        softmax = torch.nn.Softmax(dim=1)

        # transform patches into a batch
        patches = torch.from_numpy(patches).float().unsqueeze(0)

        # get model predictions on patches
        model_output = model(patches)
        # visualise_per_class_accuracies(model_output.detach().squeeze().numpy(),
        #                                class_names=[np.array2string(offset) for offset in offsets],
        #                                title="Predicted probabilities per class", lim=False)
        predicted_probabilities = softmax(model_output).detach().cpu().numpy().squeeze()

        # calculate expected displacement
        e_d = np.matmul(predicted_probabilities, offsets)
        # set numpy print options to 2 signfigures
        np.set_printoptions(precision=2)
        # visualise_per_class_accuracies(predicted_probabilities,
        #                                class_names=[np.array2string(offset) for offset in offsets],
        #                                title=f"Softmaxed probabilities per class.\nE(d) = {e_d}", lim=False)

        result = e_d.squeeze()

        return result, model_output.detach().squeeze().numpy(), predicted_probabilities


def get_patch_size_from_data_folder(data_path):
    """
    This function takes in a path to a folder with patches and returns the patch size - it can do that because the bs is
    encoded in the file names
    :param data_path:
    :return: patch_size
    """

    patch_file_path_list = glob.glob(os.path.join(data_path, "*_patch.npy"))

    one_patch_path = patch_file_path_list[0].split("_")

    patch_size = int(one_patch_path[-2][2:])

    return patch_size


def softmax_sq(array):
    """
    This function takes in an array and returns the square softmax of the array
    """
    return array ** 2 / np.sum(array ** 2)


def load_model_for_inference(model_path, init_features=64):

    model = DenseNet(num_init_features=init_features)
    model_params = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_params['model_state_dict'])
    model.eval()

    return model
