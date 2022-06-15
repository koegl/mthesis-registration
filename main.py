import elasticdeform
import matplotlib.pyplot as plt
import numpy as np
import argparse
import nibabel as nib

from utils import create_deformation_grid, display_volume_slice


def main(params):

    # volume = nib.load(params.nii_volume_path).get_fdata()
    # _ = display_volume_slice(volume, title='Original volume')

    # create 2D checkerboard
    x = np.ones((10, 10), dtype=float)
    x[::2] = 0
    x[:, ::2] = 1 - x[:, ::2]
    checkerboard_2d = x.repeat(10, axis=0).repeat(10, axis=1)

    # create 3D checkerboard
    checkerboard_2d_r = np.rot90(checkerboard_2d)

    checkerboard_2d_3 = np.tile(checkerboard_2d, (10, 1, 1))
    checkerboard_2d_r_3 = np.tile(checkerboard_2d_r, (10, 1, 1))

    packet = np.concatenate((checkerboard_2d_3, checkerboard_2d_r_3))
    checkerboard_3d = np.tile(packet, (5, 1, 1))

    # plt.imshow(x, cmap='gray')
    # plt.title('Original')

    # deformation = create_deformation_grid([[0, -16], [0, 0],
    #                                        [0, 0],   [0, 0],
    #
    #                                        [0, -16], [0, 0],
    #                                        [0, 0],   [0, 0],
    #                                        ],
    #                                       [2, 2, 2], dim=3)

    deformation = [
                   [[0, 0], [0, 0]],
                   [[0, 0], [0, 0]],
                   [[0, 0], [0, 0]]
                   ]

    deformation = np.zeros((3, 2, 2, 2))

    deformation_vectors = [
        [-32, 0, 0], [-32, 0, 32],
        [-32, 0, 0], [-32, 0, 32],

        [32, 0, 0], [32, 0, 32],
        [32, 0, 0], [32, 0, 32],
    ]

    """
    deformation_vectors = [
        [x1, y1, z1], [x3, y3, z3],
        [x2, y2, z2], [x4, y4, z4],

        [x5, y5, z5], [x7, y7, z7],
        [x6, y6, z6], [x8, y8, z8],
    ]
    """

    deformation = create_deformation_grid(deformation_vectors, [2, 2, 2], dim=3)

    y = elasticdeform.deform_grid(X=checkerboard_3d, displacement=deformation, order=0)

    _ = display_volume_slice(y)

    # plt.figure()
    # plt.imshow(y)

    print(4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-nvp", "--nii_volume_path",
                        default="/Users/fryderykkogl/Data"
                                "/IXI_nii_gz_cleaned_preprocessed_2019_07_18_12_17_38_1mm_registered_BEST_InterSUbj"
                                "/case012_T1.nii.gz")
    # parser.add_argument("-v", "--validate", default=True, type=bool, help="Choose whether to validate or not")
    # parser.add_argument("-lg", "--logging", default="wandb", choices=["print", "wandb"])
    args = parser.parse_args()

    main(args)

