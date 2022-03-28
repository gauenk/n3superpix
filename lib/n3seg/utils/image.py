
# -- vision --
from torchvision.utils import save_image as th_save_image

# -- linalg --
import torch as th
import numpy as np

# -- pathlib --
from pathlib import Path


def save_image(img,fn):
    if not th.is_tensor(img):
        img = th.from_numpy(img)
    if img.dtype == th.uint8:
        img = img/255.
    th_save_image(img,fn)

def save_burst(burst,root,name):
    root = Path(root)
    if not root.exists(): root.mkdir()
    nframes = len(burst)
    for t in range(nframes):
        img_t = burst[t]
        fn_t = "%s_%d.png" % (name,t)
        path_t = root / fn_t
        save_image(img_t,path_t)

