import torch
from vit_pytorch import ViT
import matplotlib.image as mpimg

v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=2,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

# load an image from the data folder
img_cat = mpimg.imread("/Users/fryderykkogl/Documents/university/master/thesis/code/mthesis-registration/data/cat.jpg")
img_cat = img_cat.transpose()
img_dog = mpimg.imread("/Users/fryderykkogl/Documents/university/master/thesis/code/mthesis-registration/data/dog.jpg")
img_dog = img_dog.transpose()

# convert to tensor
img_cat = torch.from_numpy(img_cat).float().unsqueeze(0)
img_dog = torch.from_numpy(img_dog).float().unsqueeze(0)

# img = torch.randn(1, 3, 256, 256)
