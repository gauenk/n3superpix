
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat


def run_superpix_warp(src,tgt,neighs):
    """
    Warp the src image into the tgt image
    """
    return src.img/255.
