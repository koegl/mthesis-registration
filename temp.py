import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from helpers.visualisations import plot_offsets


offset = [4, -8, 0]
predicted_offsets = [
    ([4, 0, 0], 0.3),
    ([0, 16, 0], 0.5),
    ([16, 0, 0], 0.2),
    ([0, 4, 0], 0.3),
]

plot_offsets(offset, [3, -10, 0], predicted_offsets)
