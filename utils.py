import numpy as np
import nibabel as nib
import random
import os
import wandb
import glob
import matplotlib.pyplot as plt

import torch

import architectures.densenet3d as densenet3d
from architectures.vit_standard_3d import ViTStandard3D


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience and delta"
    """

    def __init__(self, patience=5, min_delta=0.02):
        """
        :param patience: number of epochs to wait before early stopping
        :param min_delta: minimum difference between new loss and old loss for new loss to be considered as an
        improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


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
        "train_dir": params.train_dir,
        "val_dir": params.val_dir,
        "architecture_type": params.architecture_type,
        "device": str(params.device),
    }
    wandb.config = config_dict
    wandb.log(config_dict)
    wandb.log({"Training size": len_train,
               "Validation size": len_val})


def get_architecture(params):

    architecture_type = params.architecture_type

    patch_size = get_patch_size_from_data_folder(params.train_dir)

    if architecture_type.lower() == "densenet":
        model = densenet3d.DenseNet(
            growth_rate=32,
            block_config=(6, 12, 24, 16),  # original values
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
    from visualisations import visualise_per_class_accuracies

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

        return result


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
