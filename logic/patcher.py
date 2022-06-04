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
    def __init__(self, load_directory, save_directory, file_type, centres_per_dimension, patch_size=32, scale_dist=1.5):
        """
        :param load_directory: Directory with the niftis
        :param save_directory: Directory where al the patches will be saved
        :param file_type: ".nii" or ".nii.gz"
        :param centres_per_dimension:
        :param patch_size: size of the cubical patch (side of the cube - an int)
        :param scale_dist: factor which determines how far the unrelated patch will be from the patch - if it's one, the
        patches will be touching each other. shouldn't be less than one, because then there is overlap
        """

        self.load_directory = load_directory
        self.save_directory = save_directory
        self.file_type = file_type
        self.centres_per_dimension = int(centres_per_dimension)
        self.patch_size = int(patch_size)
        self.scale_dist = float(scale_dist)

        self.unrelated_offset = "[7, 7, 7]"
        self.offsets = [
            self.unrelated_offset,
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
        ]
        self.offset_to_label_dict = {
            self.unrelated_offset:  "[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, 0]":    "[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[-16, 0, 0]":  "[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, -16, 0]":  "[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, -16]":  "[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[-8, 0, 0]":   "[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, -8, 0]":   "[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, -8]":   "[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[-4, 0, 0]":   "[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, -4, 0]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, -4]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[4, 0, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 4, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]",
            "[0, 0, 4]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]",
            "[8, 0, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]",
            "[0, 8, 0]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]",
            "[0, 0, 8]":    "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]",
            "[16, 0, 0]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]",
            "[0, 16, 0]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]",
            "[0, 0, 16]":   "[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]",
        }
        self.label_to_offset_dict = {v: k for k, v in self.offset_to_label_dict.items()}

    def get_bounds(self, centre, offset):
        """
        Get bounds of a patch
        :param centre: centre of the fixed patch
        :param offset: offset of the offset patch
        :return:
        """
        x_min = int(centre[0] - self.patch_size / 2 + offset[0])
        x_max = int(centre[0] + self.patch_size / 2 + offset[0])
        y_min = int(centre[1] - self.patch_size / 2 + offset[1])
        y_max = int(centre[1] + self.patch_size / 2 + offset[1])
        z_min = int(centre[2] - self.patch_size / 2 + offset[2])
        z_max = int(centre[2] + self.patch_size / 2 + offset[2])

        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def in_bounds(self, volume_shape, bounds):
        """
        Checks if a patch is in the bounds
        :param volume_shape:
        :param bounds: bounds of the patch
        :return: bounds if true, else False
        """

        # check if the patch is out of bounds
        if bounds[0] < 0 or bounds[1] >= volume_shape[0] or \
           bounds[2] < 0 or bounds[3] >= volume_shape[1] or \
           bounds[4] < 0 or bounds[5] >= volume_shape[2]:
            return False

        return True

    def create_unrelated_offset(self, volume_shape, center):
        """
        Function that creates an offset that is bigger than the patch size in any of the six directions - this is the
        unrelated class. The offset will be only in one spatial direction
        :param volume_shape: shape of the volume
        :param center: coordinates of the centre of the fixed patch
        :return: An array of length 3 with the offset
        """
        assert self.scale_dist > 1, "side_dist should be greater than 1"
        dist = self.patch_size*self.scale_dist

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
            bounds = self.get_bounds(center, offset)

            if self.in_bounds(volume_shape, bounds):
                return offset
            else:
                continue

    def extract_cubical_patch_with_offset(self, image, center, offset=None):
        """
        Extract a cubical patch from the image. It is assumed that the center and offset are correct
        :param image: the volume as an nd array
        :param center: the center of the cubical patch
        :param offset: the offset of the cubical patch
        :return: the cubical patch as an nd array
        """

        if offset is None:
            offset = [0, 0, 0]

        assert len(offset) == 3, "Offset must be a 3D vector"
        assert len(center) == 3, "Center must be a 3D vector"
        assert isinstance(self.patch_size, int), "Size must be a scalar integer"

        # check if we want the unrelated offset - if yes, then create it
        if offset == ast.literal_eval(self.unrelated_offset):
            offset = self.create_unrelated_offset(image.shape, center)

        bounds = self.get_bounds(center, offset)

        return image[bounds[0]:bounds[1], bounds[2]:bounds[3], bounds[4]:bounds[5]]

    def extract_overlapping_patches(self, image_fixed, image_offset, centre, offset=None):
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

        patch_fixed = self.extract_cubical_patch_with_offset(image_fixed, centre, offset=None)

        patch_offset = self.extract_cubical_patch_with_offset(image_offset, centre, offset=offset)

        return patch_fixed, patch_offset

    def generate_list_of_patch_centres(self, volume):
        """
        Returns a list of patch centres that follow a grid based on centres_per_dimension. The grid is scaled in each
        direction by the volume size, so that the centres are uniformly distributed. The patch has to contain at least 25%
        of non-black pixels
        :param volume: the entire volume
        :return: a list of patch centres in randomised order
        """

        cpd = self.centres_per_dimension

        centres_list = []

        # crop the volume to remove black borders
        volume = crop_volume_borders(volume)
        volume_size = volume.shape

        # get maximum dimension
        max_dim = np.max(volume_size)
        dimension_factor = volume_size / max_dim  # so we have a uniform grid in all dimensions

        # loop through the amount of centres per dimension with three nesting loops (one for each dimension)
        for i in range(int(cpd * dimension_factor[0])):
            for j in range(int(cpd * dimension_factor[1])):
                for k in range(int(cpd * dimension_factor[2])):

                    # calculate the factors, that is by how much we have to move the centre in each dimension
                    factor_x = volume_size[0] // cpd
                    factor_y = volume_size[1] // cpd
                    factor_z = volume_size[2] // cpd

                    # check if the function argument centres_per_dimension is greater than image size
                    # if so, this would mean that we would try to extract 'subpixel' centres, so we set the factors to 1
                    if cpd > volume_size[0]:
                        factor_x = 1
                    if cpd > volume_size[1]:
                        factor_y = 1
                    if cpd > volume_size[2]:
                        factor_z = 1

                    # multiply the factors with the indices to get the i,j,k centre
                    centre_x = i * factor_x
                    centre_y = j * factor_y
                    centre_z = k * factor_z

                    # check for out of bounds
                    if centre_x < self.patch_size // 2 or centre_x > volume_size[0] - self.patch_size // 2:
                        continue
                    if centre_y < self.patch_size // 2 or centre_y > volume_size[1] - self.patch_size // 2:
                        continue
                    if centre_z < self.patch_size // 2 or centre_z > volume_size[2] - self.patch_size // 2:
                        continue

                    # check if patch has in at least 25% of the volume non-black pixels
                    patch = volume[centre_x - self.patch_size // 2:centre_x + self.patch_size // 2,
                                   centre_y - self.patch_size // 2:centre_y + self.patch_size // 2,
                                   centre_z - self.patch_size // 2:centre_z + self.patch_size // 2]
                    if np.count_nonzero(patch) / (self.patch_size ** 3) < 0.25:
                        continue

                    centre = [centre_x, centre_y, centre_z]

                    centres_list.append(centre)

        # randomise the order of the centres
        random.shuffle(centres_list)

        return centres_list

    def get_patch_and_label(self, volume, patch_centre, offset):
        """
        This function creates a patch and the corresponding label from a volume, a patch centre, patch size and offset
        :param volume: The full volume
        :param patch_centre: the centre of the patch
        :param offset: the offset (list len 3)
        :return: a dict containing the patch and the label
        """

        # extract both patches (we give the same volume twice, because we want to extract the patches from the same one)
        patch_fixed, patch_offset = self.extract_overlapping_patches(volume, volume, patch_centre,
                                                                     ast.literal_eval(offset))

        # join patches along new 4th dimension
        combined_patch = np.zeros((2, self.patch_size, self.patch_size, self.patch_size))
        combined_patch[0, :, :, :] = patch_fixed
        combined_patch[1, :, :, :] = patch_offset

        # get the label from the dict
        label = ast.literal_eval(self.offset_to_label_dict[offset])

        # combine everything into a dict
        packet = {'patch': combined_patch,
                  'label': label}

        return packet

    def create_and_save_all_patches_and_labels(self):

        assert self.file_type == 'nii' or self.file_type, "Only nii files are supported"

        # get a list of all files
        file_list = glob.glob(os.path.join(self.load_directory, f"*.{self.file_type}"))
        file_list.sort()

        # generate patches and labels
        all_labels = []
        idx = 0

        for file in file_list:
            ds = nib.load(file)
            volume = ds.get_fdata()
            patch_centres = self.generate_list_of_patch_centres(volume)

            for centre in patch_centres:
                random.shuffle(self.offsets)
                for offset in self.offsets:

                    # check if patch is in bounds
                    bounds = self.get_bounds(centre, ast.literal_eval(offset))
                    if self.in_bounds(volume.shape, bounds) is False:
                        continue

                    patch_and_label = self.get_patch_and_label(volume, centre, offset)
                    patch = patch_and_label['patch'].astype(np.uint8)
                    all_labels.append(np.asarray(patch_and_label['label']).astype(np.uint8))

                    # save the patch and label
                    np.save(os.path.join(self.save_directory, str(idx).zfill(9) + "_fixed_and_moving" + ".npy"), patch)

                    idx += 1

        all_labels = np.asarray(all_labels)
        np.save(os.path.join(self.save_directory, "labels.npy"), all_labels)
