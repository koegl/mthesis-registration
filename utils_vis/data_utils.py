"""
Contains utility functions for interacting with the data.
"""

import os
import numpy as np
import nibabel as nib

from nibabel.affines import apply_affine


def load_nifti_image(filepath):
    """
    Loads NIFTI Image
    """
    img = nib.load(filepath)
    canonical_img = nib.as_closest_canonical(img)

    return canonical_img 

def load_tag_file(path):
    """
    Loads point coordinates in .tag file.
    """
    # read contents of file
    with open(path, encoding = 'utf-8') as f:
        lines = f.readlines()
    
    if lines[0] != 'MNI Tag Point File\n':
        raise ValueError('This is not an MNI Tag Point File')
    
    # extract classes
    class1, class2 = lines[2].split(' ')[3].split('-')
    # define output dictionary
    out_dict = {class1: [], class2: []}

    coordinate_line = False
    for line in lines:
        if line == 'Points =\n':
            coordinate_line = True
            continue
        if coordinate_line == True:
            coord = line.split(' ')[1:-1]
            out_dict[class1].append(tuple([float(i) for i in coord[:3]]))
            out_dict[class2].append(tuple([float(i) for i in coord[3:]]))

    return out_dict

def landmark_world2vox(landmarks, transform, shape):
    """
    Transforms landmarks in list form world space to voxel space using specified transform.
    """
    transformed_landmarks = []
    for landmark in landmarks:
        # convert to [1,3] ndarray for the transformation
        transformed_landmark = tuple(apply_affine(transform, np.asarray(landmark)[None, ...])[0,...])
        transformed_landmarks.append(transformed_landmark)
    
    return transformed_landmarks