
# -- vision --
from torchvision.utils import save_image as th_save_image

# -- linalg --
import torch as th
import numpy as np

def save_image(img,fn):
    if not th.is_tensor(img):
        img = th.from_numpy(img)
    th_save_image(img,fn)
