import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import wandb

# todo add way of encoding of patches with offset bigger than patch (unrelated) - then just try to give a patch where
#  the offset is bigger than the patch - this should be tried in all 6 spatial directions until one is found that is not
#  out of bounds


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


def display_volume_slice(volume):
    """
    Displays a slice of a 3D volume in a matplotlib figure
    :param volume: the volume
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    ax.imshow(volume[volume.shape[0] // 2, :, :], cmap='gray')

    ax_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
    slice = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=volume.shape[0] // 2)

    def update(val):
        ax.clear()
        ax.imshow(volume[int(slice.val), :, :], cmap='gray')
        fig.canvas.draw_idle()

    slice.on_changed(update)

    plt.show()


def display_tensor_and_label(tensor, label):
    """
    Display a tensor and its label
    :param tensor: the tensor
    :param label: the label
    :return:
    """
    tensor = tensor.squeeze()
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    label = label.numpy()

    label = "dog" if all(label == [1.0, 0.0]) else "cat"

    plt.imshow(tensor)
    plt.title(label)
    plt.show()


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
        "learning_rate": params["learning_rate"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "training_data": params["train_and_val_dir"],
        "test_data": params["test_dir"],
        "architecture_type": params["architecture_type"],
        "device": params["device"],
    }
    wandb.config = config_dict
    wandb.log(config_dict)
    wandb.log({"Training size": len_train,
               "Validation size": len_val})


def get_architecture():

    if True:
        model = None
    else:
        raise NotImplementedError("Architecture not supported. Only ViTStandard and ViTForSmallDatasets are supported.")

    return model


def calculate_accuracy(output, label):
    """
    Calculate the accuracy between the output and the label
    :param output: output of the network
    :param label: label/target/ground truth
    :return: accuracy
    """

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

