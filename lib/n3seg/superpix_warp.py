
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange


def run_superpix_warp(src,tgt,topk):
    """
    Warp the src image into the tgt image
    """
    c,h,w = src.img.shape
    fill_img = np.zeros((c,h,w),dtype=np.float32)
    fill_numba(fill_img,src.img,tgt.img,
               src.labels2pix,tgt.labels2pix,topk.inds)
    return fill_img

@njit(inplace=True)
def bounds(val,lim):
    if val < 0: val = (-val-1)
    if val >= lim: val = (2*lim - val - 1)
    return val

@njit(debug=False)
def fill_numba(fill_img,src_img,tgt_img,src_labels2pix,tgt_labels2pix,topk_inds):

    c,h,w = fill_img.shape
    src_nlabels,npix,two = src_labels2pix.shape
    src_nlabels = topk_inds.shape[0]
    for src_label in prange(src_nlabels):
        tgt_label = topk_inds[src_label,0]
        # new_label_pix = magic()
        for pidx in range(npix):
            src_h = src_labels2pix[src_label,pidx,0]
            src_w = src_labels2pix[src_label,pidx,1]
            if (src_h == -1) and (src_w == -1): break
            for ci in range(c):
                fill_img[ci,src_h,src_w] = 255.
