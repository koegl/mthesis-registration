import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import scipy.ndimage as ndimage
import nibabel as nib
import helpers.volumes as volumes
import SimpleITK as sitk
import numpy as np
import scipy.ndimage as nd

from helpers.volumes import create_checkerboard
import helpers.visualisations as vis


def transform_with_ndimage_affine(volume, transform):

    return ndimage.affine_transform(volume, transform)


def transform_with_ndimage_map(volume, transform):
    i, j, k = volume.shape
    i_vals, j_vals, k_vals = np.meshgrid(range(j), range(j), range(k), indexing='ij')
    coords = np.array([i_vals, j_vals, k_vals]).transpose((1, 2, 3, 0))

    new_coords = nib.affines.apply_affine(transform, coords)
    new_coords = new_coords.transpose(3, 0, 1, 2)

    return ndimage.map_coordinates(volume, new_coords)


def transform_with_sitk(volume):

    transform = sitk.AffineTransform(3)
    transform.SetTranslation((70, 70, 70))

    sitk_volume = sitk.GetImageFromArray(volume, isVector=False)

    # create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_volume)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)

    # deform the volume
    deformed_volume = resampler.Execute(sitk_volume)
    deformed_volume = sitk.GetArrayFromImage(deformed_volume)
    deformed_volume = deformed_volume.astype(dtype=np.float32)

    return deformed_volume


def main(params):
    # volume = np.load(params.offset_volume_path).astype(np.float32)

    volume = create_checkerboard(3, [400, 400, 400])

    transform = volumes.create_transform_matrix(10, 2, 5, 0, 0, 0)

    t = perf_counter()
    volume_sitk = volumes.apply_affine_transform_sitk(volume, transform)
    t_sitk = perf_counter() - t

    t = perf_counter()
    volume_normal_affine = transform_with_ndimage_affine(volume, transform)
    t_normal_affine = perf_counter() - t

    t = perf_counter()
    volume_map_coordinates = transform_with_ndimage_map(volume, transform)
    t_map_coordinates = perf_counter() - t

    print(5)


# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# import numpy as np
#
# def plot_3d_quiver(x, y, z, u, v, w):
#     c = np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2 + np.abs(w) ** 2)
#     c = (c.ravel() - c.min()) / c.ptp()
#     # Repeat for each body line and two head lines
#     c = np.concatenate((c, np.repeat(c, 2)))
#     # Colormap
#     c = plt.cm.jet(c)
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.quiver(x, y, z, u, v, w, colors=c, length=0.1)
#     plt.show()
#
# field = np.random.rand(3, 3, 3, 3)
#
# x, y, z = np.meshgrid(np.linspace(60, 0, 3),
#                       np.linspace(60, 0, 3),
#                       np.linspace(60, 0, 3))
#
# color = np.sqrt(field[:, :, :, 0] ** 2 + field[:, :, :, 1] ** 2 + field[:, :, :, 2] ** 2)
#
# # fig = plt.figure(figsize=(5, 5))
# # ax = fig.gca(projection='3d')
# # ax.quiver(x, y, z, field[:, :, :, 0], field[:, :, :, 1], field[:, :, :, 2], color)
# # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-mp", "--model_path", default="/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/Models/"
                                                       "39-dense-515k-mr-vocal-sweep3-10/model_epoch7_valacc0.910.pt",
                        help="Path to the trained .pt model file")
    parser.add_argument("-fvp", "--fixed_volume_path", default="/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/"
                                                               "Experiments/mr patch convergence/data/49.npy",
                        help="Path to the fixed .npy volume")
    parser.add_argument("-ovp", "--offset_volume_path", default="/Users/fryderykkogl/Dropbox (Partners HealthCare)/DL/"
                                                                "Experiments/mr patch convergence/data/50_49.npy",
                        help="Path to the offset .npy volume")

    args = parser.parse_args()

    main(args)
