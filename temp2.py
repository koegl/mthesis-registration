import numpy as np
import scipy.ndimage as ndimage

transform = np.array([
       [0.34917696, 0.88594443, 0.30525056, 0.        ],
       [0.22639266, 0.23634492, 0.        , 0.        ],
       [0.        , 0.3990533 , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        ]
])

volume = np.random.rand(400, 400, 400)

transformed_volume = ndimage.affine_transform(volume, transform, order=1)

print(np.min(transformed_volume))
print(np.max(transformed_volume))

print(5)
