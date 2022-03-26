
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,prange

# -- org --
from easydict import EasyDict as edict
from .utils import optional


def run_superpix_nn(src,tgt,**kwargs):
    """

    Compute nearest neighbors between two frame
    using the superpixels

    """

    # -- unpack args --
    ws = optional(kwargs,'ws',10)

    # -- compute superpix differences --
    nlabels = np.max(src.labels).item()+1
    fbuf = np.zeros((nlabels,nlabels),dtype=np.float32)
    cbuf = np.zeros((nlabels,nlabels),dtype=np.int16)
    superpix_nn_numba(src.img,tgt.img,src.labels,tgt.labels,fbuf,cbuf,ws)

    # -- renormalize errors --
    buf = fbuf / (cbuf+1e-8)
    buf[np.where(cbuf==0)] = float("inf")

    # -- compute topk difference per src label --
    k = 2
    topk = edict()
    topk.inds = np.argpartition(buf,k,axis=1)[:,:k]
    topk.vals = np.take_along_axis(buf,topk.inds,1)

    return topk

@jit
def superpix_nn_numba(img_a,img_b,label_a,label_b,fbuf,cbuf,ws):
    c,h,w = img_a.shape
    for hi in prange(h):
        for wi in prange(w):
            for hoff in prange(2*ws):
                hj = (hi + hoff - ws)
                hj = hj if hj >= 0 else -hj
                hj = hj if hj < h else 2*h - hj - 1
                for woff in prange(2*ws):
                    wj = (wi + woff - ws)
                    wj = wj if wj >= 0 else -wj
                    wj = wj if wj < w else 2*w - wj - 1

                    label_i = label_a[hi,wi]
                    label_j = label_b[hj,wj]

                    for ci in prange(c):
                        pix_i = img_a[ci,hi,wi]/255.
                        pix_j = img_b[ci,hj,wj]/255.
                        fbuf[label_i,label_j] += (pix_i - pix_j)**2
                    cbuf[label_i,label_j] += 1
                    # buf[label_i,label_j] = acc
