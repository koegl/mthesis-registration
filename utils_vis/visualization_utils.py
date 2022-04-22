"""
Contains utility functions for visualizing the data.
"""

import os 
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Index:
    """
    Simple class to track index.
    """
    def __init__(self, initial: int, minimum: int, maximum: int) -> None:
        """
        Args:
            initial:  initial index
            min:  minimum index
            max:  maximum index
        """
        self.idx = initial
        self.minimum = minimum
        self.maximum = maximum

        # check if the current index is not outside of the range
        if self.idx < self.minimum or self.idx > self.maximum:
            raise ValueError('Initialized index outside of specified range.')
    
    def current(self) -> int:
        return self.idx
    
    def add(self, step: int = 1) -> None:
        """
        Args:
            step:  number to add to idx.
        """
        self.idx = min(self.idx+step, self.maximum)

    def subtract(self, step: int = 1) -> None:
        """
        Args:
            step:  number to subtract from idx.
        """
        self.idx = max(self.idx-step, self.minimum)


class ImageViewer:
    """
    Tool for visualization of 3D images.
    Use the scroll wheel to scroll through the image slices.
    Press 'SHIFT' to scroll at a higher scrolling speed.
    Press '1', '2', '3' to switch between dimensions.
    Press '-' and '=' to decrease and increase the intensity, respectively.
    """
    # define class attributes
    scroll_speed = 4
    intensity_factor = 1.2
    init_slice = 'mid'
    colors = [
        'orangered', 
        'mediumseagreen', 
        'lightskyblue',
        'crimson',
        'greenyellow',
        'slateblue'
    ]

    def __init__(
        self, 
        ax: np.ndarray, 
        image_array: np.ndarray, 
        voxel_size: list = [1,1,1],
        landmarks: dict = {},
        cmap: str = 'gray'
    ) -> None:
        """ 
        Args:
            ax:  matplotlib.pyplot figure axis.
            image_array:  image series with the following expected shape: (frame, channel, height, width).
            cmap:  colormap recognized by matplotlib.pyplot module.
        """
        # configure canvas
        self.ax = ax
        self.ax.set_axis_off()
        canvas = self.ax.figure.canvas
        # connect events to canvas
        canvas.mpl_connect('scroll_event', self.onscroll)
        canvas.mpl_connect('key_press_event', self.keypress)
        canvas.mpl_connect('key_release_event', self.keyrelease)
        self.canvas = canvas
        self.cmap = cmap

        # define image attributes
        self.image_array = image_array
        self.shape = self.image_array.shape
        self.voxel_size = voxel_size
        self.landmarks = landmarks
        self.plotted = []

        # add additional dimension if only three are present
        if len(self.shape) != 3:
            raise ValueError('Incorrect number of dimensions')

        # initialize objects to track class and frame
        self.speed = 1
        self.factor_power = 0
        self.dimension = 0
        self.idxs = [Index(length//2 if self.init_slice == 'mid' else self.init_slice, 0, length-1) for length in self.shape]
        self.image = self.ax.imshow(self.change_image(), origin='lower', vmin=np.min(self.image_array), vmax=np.max(self.image_array), cmap=self.cmap)

        # set the first image
        self.update(dim_change=True)

    def onscroll(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Add or subtract 'self.speed' from the selected index (either frame or class) when scrolling up or down respectively.
        Update the image frame afterwards. Do not update the index when scolling up for the last or down for the first image.

        Args:
            event:  mouse event (up or down scroll).
        """
        if event.button == 'up':
            # update index after scolling
            self.idxs[self.dimension].add(self.speed)
            self.update()

        elif event.button == 'down': 
            # update index after scolling
            self.idxs[self.dimension].subtract(self.speed)
            self.update()

    def keypress(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (press) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        for key in event.key.split('+'):# handles multiple keys pressed at once
            # increases scroll speed
            if key == 'shift' and self.speed == 1:
                self.speed = self.scroll_speed
            # decrease factor power by 1
            if key == '-':
                self.factor_power -= 1
                self.update()
            # increase factor power by 1
            if key == '=':
                self.factor_power += 1
                self.update()
            if key == '1' and self.dimension != 0:
                self.dimension = 0
                self.update(dim_change=True)
            if key == '2' and self.dimension != 1:
                self.dimension = 1
                self.update(dim_change=True)
            if key == '3' and self.dimension != 2:
                self.dimension = 2
                self.update(dim_change=True)

    def keyrelease(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Args:
            event:  key event (release) that is checked for a connected action.
                    If there is one, the implemented action is performed.
        """
        for key in event.key.split('+'): # handles multiple keys pressed at once
            # decreases scroll speed
            if key == 'shift' and self.speed == self.scroll_speed:
                self.speed = 1

    def change_image(self):
        """
        Change image based on the selected dimension and slice.
        """
        if self.dimension == 0:
            image = np.swapaxes(self.image_array[self.idxs[self.dimension].current(), :, :], 0, 1)
        elif self.dimension == 1:
            image = np.swapaxes(self.image_array[:, self.idxs[self.dimension].current(), :], 0, 1)
        elif self.dimension == 2:
            image = np.swapaxes(self.image_array[:, :, self.idxs[self.dimension].current()], 0, 1)
        else:
            raise ValueError('Dimension was not recognized.')

        return image * self.intensity_factor**self.factor_power

    def change_landmarks(self):
        """
        Change landmarks that are plotted based on the selected dimension and slice.
        """
        # remove all currently plotted landmarks
        for landmark in self.plotted:
            landmark.remove()
        self.plotted = []

        # define a variable with the dimension indices and remove the currently selected one
        dims = [0,1,2]
        offset = 2
        dims.remove(self.dimension)
        # loop over all landmark sets
        for j, key in enumerate(self.landmarks):
            for i, landmark in enumerate(self.landmarks[key]):
                # check if the landmark is in the current image
                if round(landmark[self.dimension]) == self.idxs[self.dimension].current():
                    # plot the landmark, the landmark number, and add the objects to a list to keep track of them
                    plotted_landmark = plt.scatter(landmark[dims[0]], landmark[dims[1]], edgecolors=self.colors[j], marker='o', facecolors='none', lw=1.25)
                    plotted_text = plt.annotate(str(i), (landmark[dims[0]]+offset, landmark[dims[1]]+offset), color=self.colors[j], fontsize=11)
                    self.plotted.append(plotted_landmark)
                    self.plotted.append(plotted_text)

    def update(self, dim_change=False) -> None:
        """ 
        Update the image and corresponding labels.
        """
        # load the new image
        image = self.change_image()
        title = f'Dimension: {self.dimension+1}, Frame: {self.idxs[self.dimension].current()}/{self.idxs[self.dimension].maximum}'

        if dim_change:
            self.ax.cla()
            self.image = self.ax.imshow(self.change_image(), origin='lower', vmin=np.min(self.image_array), vmax=np.max(self.image_array), cmap=self.cmap)
            self.ax.set_axis_off()
            dims = [0,1,2]
            dims.remove(self.dimension)
            self.ax.set_aspect(self.voxel_size[dims[1]]/self.voxel_size[dims[0]])
        else:
            self.image.set_data(image)
        self.change_landmarks()
        self.ax.set_title(title)

        # update the canvas
        self.ax.figure.canvas.draw()


def image_viewer(
    image_array: np.ndarray, 
    voxel_size: list = [1,1,1],
    landmarks: dict = {},
    cmap: str = 'gray',
) -> None:
    """ 
    Args:
        image_array:  image series with the following expected shape: (frame, channel, height, width).
        cmap:  colormap recognized by matplotlib.pyplot module.
    """
    fig, ax = plt.subplots(figsize=(7,6))
    viewer = ImageViewer(ax, image_array, voxel_size=voxel_size, landmarks=landmarks, cmap=cmap)
    plt.tight_layout()
    plt.show()
