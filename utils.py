import numpy as np
import nibabel as nib
import random
import os
import wandb

import torch

import architectures.densenet3d as densenet3d
from architectures.vit_standard_3d import ViTStandard3D


def save_np_array_as_nifti(array, path, affine, header=None):
    """
    Save an nd array as a nifti file.
    :param array: the nd array to save
    :param path: the path to save the nifti file
    :param header: the header of the nifti file
    :param affine: the affine of the nifti file
    """

    img = nib.Nifti1Image(array, affine=affine, header=header)

    nib.save(img, path)


def create_radial_gradient(width, height, depth):
    """
    Create a radial gradient.
    :param width: width of the volume
    :param height: height of the volume
    :param depth: depth of the volume
    :return: the gradient volume as a nd array
    """

    x, y, z = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height), np.linspace(-1, 1, depth))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    r = r / np.max(r)

    return r


def crop_volume_borders(volume):
    """
    Removes as much surrounding black pixels from a volume as possible
    :param volume: The entire volume
    :return: volume_cropped
    """

    shape = volume.shape

    assert len(shape) == 3, "Volume must be 3D"

    # set maximum and minimum boundaries in case the volume touches the sides of the image
    min_x = 0
    min_y = 0
    min_z = 0
    max_x = shape[0] - 1
    max_y = shape[1] - 1
    max_z = shape[2] - 1

    # find first plane in the x-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[0]):
        if np.count_nonzero(volume[i, :, :]) > 0:
            min_x = i
            break
    # find first plane in the x-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[0] - 1, -1, -1):
        if np.count_nonzero(volume[i, :, :]) > 0:
            max_x = i
            break

    # find first plane in the y-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[1]):
        if np.count_nonzero(volume[:, i, :]) > 0:
            min_y = i
            break
    # find first plane in the y-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[1] - 1, -1, -1):
        if np.count_nonzero(volume[:, i, :]) > 0:
            max_y = i
            break

    # find first plane in the z-direction that contains a non-black pixel - from the 'left'
    for i in range(shape[2]):
        if np.count_nonzero(volume[:, :, i]) > 0:
            min_z = i
            break
    # find first plane in the z-direction that contains a non-black pixel - from the 'right'
    for i in range(shape[2] - 1, -1, -1):
        if np.count_nonzero(volume[:, :, i]) > 0:
            max_z = i
            break

    # crop the volume
    volume_cropped = volume[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

    return volume_cropped


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

    wandb.init(project=project, entity=entity)
    os.environ["WANDB_NOTEBOOK_NAME"] = "Classification"

    config_dict = {
        "learning_rate": params.learning_rate,
        "epochs": params.epochs,
        "batch_size": params.batch_size,
        "training_data": params.train_and_val_dir,
        "architecture_type": params.architecture_type,
        "device": str(params.device),
    }
    wandb.config = config_dict
    wandb.log(config_dict)
    wandb.log({"Training size": len_train,
               "Validation size": len_val})


def get_architecture(architecture_type):

    if architecture_type.lower() == "densenet":
        model = densenet3d.DenseNet()
    elif architecture_type.lower() == "vit":
        model = ViTStandard3D(
            dim=128,
            volume_size=32,
            patch_size=4,
            num_classes=20,
            channels=2,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
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
