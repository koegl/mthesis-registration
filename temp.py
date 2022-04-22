import os
import numpy as np

# def extract_pixel_to_world_matrix(node_vtk_id):
#     """
#     Function to extract the ijktoras matrix of a node
#     :param node_vtk_id: The vtk id of the node
#     :return: numpy matrix containing the ijk to ras matrix
#     """
#
#     # get the node
#     x = slicer.util.getNode(node_vtk_id)
#
#     # create empty vtk matrix and store the ijktoras there
#     ijktoras_vtk = vtk.vtkMatrix4x4()  # can be inverted with ".Invert()"
#     x.GetIJKToRASMatrix(ijktoras_vtk)  # same for RAStoIJK
#     ijktoras = np.eye(4)
#     ijktoras_vtk.DeepCopy(ijktoras.ravel(), ijktoras_vtk)
#
#     # Matrix vector product: point_pix @ ijktoras
#
#     return ijktoras
#
# # list all files in subdirectories
# def list_files(dir_path):
#     """
#     Function to list all files in subdirectories
#     :param dir_path: The path to the directory
#     :return: List of all files in the directory
#     """
#     # create empty list
#     files = []
#
#     # get all files in subdirectories
#     for root, dirs, file_names in os.walk(dir_path):
#         for file_name in file_names:
#             files.append(os.path.join(root, file_name))
#
#     return files

