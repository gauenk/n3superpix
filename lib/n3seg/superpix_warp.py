
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
               src.labels2pix,tgt.labels2pix,
               src.labels2pix_ave,tgt.labels2pix_ave,
               topk.inds)
    fill_img = fill_img.clip(0,255.).astype(np.uint8)
    return fill_img

@njit(inplace=True)
def bounds(val,lim):
    if val < 0: val = (-val-1)
    if val >= lim: val = (2*lim - val - 1)
    return val

@njit(debug=False)
def fill_numba(fill_img,src_img,tgt_img,
               src_labels2pix,tgt_labels2pix,
               src_labels2pix_ave,tgt_labels2pix_ave,
               topk_labels):

    c,h,w = fill_img.shape
    src_nlabels,src_npix,two = src_labels2pix.shape
    tgt_nlabels,tgt_npix,two = tgt_labels2pix.shape
    src_nlabels = topk_labels.shape[0]
    for src_label in prange(src_nlabels):
        tgt_label = topk_labels[src_label,0]

        for pidx in prange(src_npix):
            src_h = src_labels2pix[src_label,pidx,0]
            src_w = src_labels2pix[src_label,pidx,1]
            if (src_h == -1) and (src_w == -1): break
            src_ave_h = src_labels2pix_ave[src_label,0]
            src_ave_w = src_labels2pix_ave[src_label,1]

            src_rel_h = src_h - src_ave_h
            src_rel_w = src_w - src_ave_w
            Z = 0

            for pidx in range(tgt_npix):
                tgt_h = tgt_labels2pix[tgt_label,pidx,0]
                tgt_w = tgt_labels2pix[tgt_label,pidx,1]

                tgt_ave_h = tgt_labels2pix_ave[tgt_label,0]
                tgt_ave_w = tgt_labels2pix_ave[tgt_label,1]

                tgt_rel_h = tgt_h - tgt_ave_h
                tgt_rel_w = tgt_w - tgt_ave_w

                dist = (src_rel_h - tgt_rel_h)**2 + (src_rel_w - tgt_rel_w)**2
                weight = np.exp(-dist*10.)

                for ci in range(c):
                    pix_val = tgt_img[ci,tgt_h,tgt_w]
                    fill_img[ci,src_h,src_w] += weight * pix_val
                Z += weight

            for ci in range(c):
                fill_img[ci,src_h,src_w] /= Z
