import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from helpers.visualisations import plot_offsets


offset = [16, 0, 2]
predicted_offsets = [
    ([4, 0, 0], 0.3),
    ([8, 0, 0], 0.5),
    ([16, 0, 0], 0.2),
    ([0, 4, 7], 0.3),
]

plot_offsets(offset, predicted_offsets)
