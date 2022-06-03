import numpy as np
import warnings
import random
import glob
import os
import nibabel as nib
import ast
from itertools import permutations

from utils import crop_volume_borders


class Patcher:
    def __init__(self):
        self.unrelated_offset = "[7, 7, 7]"
        self.offsets = [
            "[0, 0, 0]",
            "[-16, 0, 0]",
            "[0, -16, 0]",
            "[0, 0, -16]",
            "[-8, 0, 0]",
            "[0, -8, 0]",
            "[0, 0, -8]",
            "[-4, 0, 0]",
            "[0, -4, 0]",
            "[0, 0, -4]",
            "[4, 0, 0]",
            "[0, 4, 0]",
            "[0, 0, 4]",
            "[8, 0, 0]",
            "[0, 8, 0]",
            "[0, 0, 8]",
            "[16, 0, 0]",
            "[0, 16, 0]",
            "[0, 0, 16]",
            self.unrelated_offset,
        ]
        self.offset_to_label_dict = {
            "[0, 0, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[-16, 0, 0]":  "[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, -16, 0]":  "[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, -16]":  "[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[-8, 0, 0]":   "[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, -8, 0]":   "[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, -8]":   "[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[-4, 0, 0]":   "[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, -4, 0]":   "[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, -4]":   "[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[4, 0, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 4, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, 4]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]",
            "[8, 0, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]",
            "[0, 8, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]",
            "[0, 0, 8]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]",
            "[16, 0, 0]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]",
            "[0, 16, 0]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]",
            "[0, 0, 16]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]",
            self.unrelated_offset: "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]",
        }
        self.label_to_offset_dict = {v: k for k, v in self.offset_to_label_dict.items()}

    @staticmethod
    def create_unrelated_offset(volume_shape, center, patch_size, side_dist=1.5):
        """
        Function that creates an offset that is bigger than the patch size in any of the six directions - this is the
        unrelated class. The offset will be only in one spatial direction
        :param volume_shape: shape of the volume
        :param center: coordinates of the centre of the fixed patch
        :param patch_size: size of the cubical patch
        :param side_dist: factor which determines how far the unrelated patch will be from the patch - if it's one, the
        patches will be touching each other. shouldn't be less than one, because then there is overlap
        :return: An array of length 3 with the offset
        """
        assert side_dist > 1, "side_dist should be greater than 1"
        dist = patch_size*side_dist

        # create a list of all possible unrelated offsets (positive offset e.g. has to be three times in the options,
        # because during permutations we want each to be treated as a separate value so that we can get offset at more
        # than one position)
        displacements = [dist, dist, dist, -dist, -dist, -dist, 0, 0]

        # get all possible permutations of length 3 of the displacements
        all_permutations = permutations(displacements, 3)

        # many duplicates, so remove them with set() and then change to list()
        all_permutations = list(set(all_permutations))

        # shuffle list order
        random.shuffle(all_permutations)

        # loop through all offsets and choose first one which is in the bounds
        for offset in all_permutations:
            x_max = int(center[0] + patch_size / 2 + offset[0])
            x_min = int(center[0] - patch_size / 2 + offset[0])
            y_min = int(center[1] - patch_size / 2 + offset[1])
            y_max = int(center[1] + patch_size / 2 + offset[1])
            z_min = int(center[2] - patch_size / 2 + offset[2])
            z_max = int(center[2] + patch_size / 2 + offset[2])

            # check if the patch is out of bounds
            if x_min < 0 or x_max >= volume_shape[0] or \
               y_min < 0 or y_max >= volume_shape[1] or \
               z_min < 0 or z_max >= volume_shape[2]:
                continue

            return offset

    def extract_cubical_patch_with_offset(self, image, center, size, offset=None):
        """
        Extract a cubical patch from the image.
        :param image: the volume as an nd array
        :param center: the center of the cubical patch
        :param size: the size of the cubical patch
        :param offset: the offset of the cubical patch
        :return: the cubical patch as an nd array
        """

        if offset is None:
            offset = [0, 0, 0]

        assert len(offset) == 3, "Offset must be a 3D vector"
        assert len(center) == 3, "Center must be a 3D vector"
        assert isinstance(size, int), "Size must be a scalar integer"

        # check if we want the unrelated offset - if yes, then create it
        if offset == ast.literal_eval(self.unrelated_offset):
            offset = self.create_unrelated_offset(image.shape, center, size, side_dist=1.5)

        x_max = int(center[0] + size / 2 + offset[0])
        x_min = int(center[0] - size / 2 + offset[0])
        y_min = int(center[1] - size / 2 + offset[1])
        y_max = int(center[1] + size / 2 + offset[1])
        z_min = int(center[2] - size / 2 + offset[2])
        z_max = int(center[2] + size / 2 + offset[2])

        # check if the patch is out of bounds
        if x_min < 0 or x_max >= image.shape[0] or \
           y_min < 0 or y_max >= image.shape[1] or \
           z_min < 0 or z_max >= image.shape[2]:
            warnings.warn("The patch is out of bounds.")
            return np.zeros((1, 1))

        return image[x_min:x_max, y_min:y_max, z_min:z_max]

    def extract_overlapping_patches(self, image_fixed, image_offset, centre, size, offset=None):
        """
        Extract overlapping patches from the two volumes. One of the volume patches will be offset by 'offset'
        :param image_fixed: the volume with the standard patch
        :param image_offset: the volume with the offset patch
        :param centre: centre of the patch
        :param size: size of the patch
        :param offset: offset of the image_offset patch
        :return: the fixed and offset patches
        """

        assert image_fixed.shape == image_offset.shape, "The two volumes must have the same shape"

        patch_fixed = self.extract_cubical_patch_with_offset(image_fixed, centre, size, offset=None)

        patch_offset = self.extract_cubical_patch_with_offset(image_offset, centre, size, offset=offset)

        return patch_fixed, patch_offset

    @staticmethod
    def generate_list_of_patch_centres(centres_per_dimension, volume, patch_size=32):
        """
        Returns a list of patch centres that follow a grid based on centres_per_dimension. The grid is scaled in each
        direction by the volume size, so that the centres are uniformly distributed. The patch has to contain at least 25%
        of non-black pixels
        :param centres_per_dimension: Amount of centres per x,y and z dimension (symmetric)
        :param volume: the entire volume
        :param patch_size: Size of the patch has to be provided so that there won't be any out of bounds problems (cubical)
        :return: a list of patch centres in randomised order
        """

        centres_list = []

        # crop the volume to remove black borders
        volume = crop_volume_borders(volume)
        volume_size = volume.shape

        # get maximum dimension
        max_dim = np.max(volume_size)
        dimension_factor = volume_size / max_dim  # so we have a uniform grid in all dimensions

        # loop through the amount of centres per dimension with three nesting loops (one for each dimension)
        for i in range(int(centres_per_dimension * dimension_factor[0])):
            for j in range(int(centres_per_dimension * dimension_factor[1])):
                for k in range(int(centres_per_dimension * dimension_factor[2])):

                    # calculate the factors, that is by how much we have to move the centre in each dimension
                    factor_x = volume_size[0] // centres_per_dimension
                    factor_y = volume_size[1] // centres_per_dimension
                    factor_z = volume_size[2] // centres_per_dimension

                    # check if the function argument centres_per_dimension is greater than image size
                    # if so, this would mean that we would try to extract 'subpixel' centres, so we set the factors to 1
                    if centres_per_dimension > volume_size[0]:
                        factor_x = 1
                    if centres_per_dimension > volume_size[1]:
                        factor_y = 1
                    if centres_per_dimension > volume_size[2]:
                        factor_z = 1

                    # multiply the factors with the indices to get the i,j,k centre
                    centre_x = i * factor_x
                    centre_y = j * factor_y
                    centre_z = k * factor_z

                    # check for out of bounds
                    if centre_x < patch_size // 2 or centre_x > volume_size[0] - patch_size // 2:
                        continue
                    if centre_y < patch_size // 2 or centre_y > volume_size[1] - patch_size // 2:
                        continue
                    if centre_z < patch_size // 2 or centre_z > volume_size[2] - patch_size // 2:
                        continue

                    # check if patch has in at least 25% of the volume non-black pixels
                    patch = volume[centre_x - patch_size // 2:centre_x + patch_size // 2,
                                   centre_y - patch_size // 2:centre_y + patch_size // 2,
                                   centre_z - patch_size // 2:centre_z + patch_size // 2]
                    if np.count_nonzero(patch) / (patch_size ** 3) < 0.25:
                        continue

                    centre = [centre_x, centre_y, centre_z]

                    centres_list.append(centre)

        # randomise the order of the centres
        random.shuffle(centres_list)

        return centres_list

    def get_patch_and_label(self, volume, patch_centre, offset, patch_size=32):
        """
        This function creates a patch and the corresponding label from a volume, a patch centre, patch size and offset
        :param volume: The full volume
        :param patch_centre: the centre of the patch
        :param patch_size: the size of the batch (int)
        :param offset: the offset (list len 3)
        :return: a dict containing the patch and the label
        """

        # extract both patches
        patch_fixed, patch_offset = self.extract_overlapping_patches(volume, volume, patch_centre, patch_size,
                                                                     ast.literal_eval(offset))

        # join patches along new 4th dimension
        combined_patch = np.zeros((2, patch_size, patch_size, patch_size))
        combined_patch[0, :, :, :] = patch_fixed
        combined_patch[1, :, :, :] = patch_offset

        # get the label from the dict
        label = ast.literal_eval(self.offset_to_label_dict[offset])

        # combine everything into a dict
        packet = {'patch': combined_patch,
                  'label': label}

        return packet

    def create_and_save_all_patches_and_labels(self, load_directory_path, save_directory_path):

        # get a list of all files
        file_list = glob.glob(os.path.join(load_directory_path, "*.nii.gz"))
        file_list.sort()

        # generate patches and labels
        patcher = Patcher()
        all_labels = []
        idx = 0

        for file in file_list:
            ds = nib.load(file)
            volume = ds.get_fdata()
            patch_centres = patcher.generate_list_of_patch_centres(10, volume, patch_size=32)

            for centre in patch_centres:
                for offset in self.offsets:
                    patch_and_label = self.get_patch_and_label(volume, centre, offset, patch_size=32)
                    patch = patch_and_label['patch'].astype(np.uint8)
                    all_labels.append(np.asarray(patch_and_label['label']).astype(np.uint8))

                    # save the patch and label
                    np.save(os.path.join(save_directory_path, str(idx).zfill(9) + "_fixed_and_moving" + ".npy"), patch)

                    idx += 1

        all_labels = np.asarray(all_labels)
        np.save(os.path.join(save_directory_path, "labels.npy"), all_labels)











