#%%
# IMPORTS
import random
import os
import numpy as np
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import torch

#%%
print(f"Torch: {torch.__version__}")

#%%
# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

#%%
train_dir = 'data_overfit/train'
test_dir = 'data_overfit/test'

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

labels = [path.split('/')[-1].split('.')[0] for path in train_list]