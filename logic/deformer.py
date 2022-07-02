import numpy as np
import SimpleITK as sitk


class Deformer:

    @staticmethod
    def generate_bspline_deformation(displacements, shape):
        """
        Creates a b-spline deformation. Its inputs are the shape of the volume and an array of displacement vectors
        following the elasticdeform package convention.
        https://elasticdeform.readthedocs.io/en/latest/
        https://www.programcreek.com/python/example/96384/SimpleITK.BSplineTransformInitializer
        :param displacements: nd array of displacements
        :param shape: shape of the volume
        :return: the bspline deformation
        """
        assert isinstance(displacements, np.ndarray), 'displacements must be a numpy array'
        assert displacements.shape[0] == len(shape),\
            "The dimension of the displacement array must match the dimension of the volume"

        squares_per_d = [x - 1 for x in displacements.shape[1:]]

        # Initialize bspline transform
        args = shape + (sitk.sitkFloat32,)
        ref_volume = sitk.Image(*args)

        bst = sitk.BSplineTransformInitializer(ref_volume, squares_per_d)

        # pad the displacement array with zeros
        padded_displacements = np.pad(displacements, ((0, 0), (1, 1), (1, 1)))

        # Transform displacements so that they can be used by the bspline transform
        p = padded_displacements.flatten('A')

        # Set bspline transform parameters to the above shifts
        bst.SetParameters(p)

        return bst

    @staticmethod
    def transform_volume(volume, bspline_deformation):
        """
        Transforms a volume using a sitk bspline deformation field.
        :param volume: the volume to transform
        :param bspline_deformation: the deformation field
        """
        # check if the volume is a numpy array
        assert isinstance(volume, np.ndarray), 'volume must be a numpy array'
        assert isinstance(bspline_deformation, sitk.BSplineTransform), 'bspline_deformation must be a bspline transform'

        # create sitk volume from numpy array
        sitk_volume = sitk.GetImageFromArray(volume, isVector=False)

        # create resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_volume)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(bspline_deformation)
        resampler.SetDefaultPixelValue(0)

        # deform the volume
        deformed_volume = resampler.Execute(sitk_volume)
        deformed_volume = sitk.GetArrayFromImage(deformed_volume)
        deformed_volume = deformed_volume.astype(dtype=np.float32)

        return deformed_volume

    @staticmethod
    def generate_deformation_field(bspline_deformation, shape):
        """
        Generates a deformation field from a bspline deformation.
        :param bspline_deformation: the deformation
        :param shape: the shape of the volume
        :return:
        """
        assert isinstance(bspline_deformation, sitk.BSplineTransform), 'bspline_deformation must be a bspline transform'

        args = shape + (sitk.sitkFloat32,)
        ref_volume = sitk.Image(*args)

        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(ref_volume)
        displacement_field = displacement_filter.Execute(bspline_deformation)

        field = np.asarray(displacement_field).reshape(shape + (len(shape),))

        return field

    @staticmethod
    def generate_grid_coordinates(grid_shape, volume_shape):
        """
        This function generates a grid (that is uniform along each axis separately). The 0th grid point lies in the corner
        of the volume (not in the center of the corresponding patch)
        :param grid_shape: shape of the grid
        :param volume_shape: shape of the volume
        :return: the grid coordinates
        """

        assert len(grid_shape) == len(volume_shape), 'grid_shape and volume_shape must be of same length'
        assert len(grid_shape) in [2, 3], 'grid_shape must be of length 2 or 3'

        if len(volume_shape) == 2:
            x, y = np.mgrid[0:volume_shape[0]:complex(0, grid_shape[0]),
                            0:volume_shape[1]:complex(0, grid_shape[1])]

            coordinates = np.stack((x, y))

        else:
            x, y, z = np.mgrid[0:volume_shape[0]:complex(0, grid_shape[0]),
                               0:volume_shape[1]:complex(0, grid_shape[1]),
                               0:volume_shape[2]:complex(0, grid_shape[2])]

            coordinates = np.stack((x, y, z))

        return coordinates
