
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
    c,h,w = src.img.shape
    ws = optional(kwargs,'ws',20)

    # -- counts of labels --
    tgt_nlabels = np.max(tgt.labels).item()+1
    src_nlabels = np.max(src.labels).item()+1
    src_cnts = np.bincount(src.labels.ravel())
    src_cmax = src_cnts.max().item()
    tgt_cnts = np.bincount(tgt.labels.ravel())
    tgt_cmax = tgt_cnts.max().item()

    # -- What labels are associated with this pixel? create labels 2 pixels --
    """
    e.g. labels2pix[label] = [list of pix]
    """
    tgt_labels2pix = -np.ones((tgt_nlabels,tgt_cmax,2),dtype=np.int16)
    counts = np.zeros((tgt_nlabels),dtype=np.int16)
    labels2pix_numba(tgt_labels2pix,counts,tgt.labels)
    tgt_labels2pix_ave = compute_labels2pix_ave(tgt_labels2pix)

    src_labels2pix = -np.ones((src_nlabels,src_cmax,2),dtype=np.int16)
    counts = np.zeros((src_nlabels),dtype=np.int16)
    labels2pix_numba(src_labels2pix,counts,src.labels)
    src_labels2pix_ave = compute_labels2pix_ave(src_labels2pix)

    # -- Compute SuperPixel Spatial Weights (distance from center) --
    tgt_weights = compute_spatial_weights(tgt_labels2pix,tgt_labels2pix_ave,tgt.labels)
    src_weights = compute_spatial_weights(src_labels2pix,src_labels2pix_ave,src.labels)

    # -- Which labels are neighbors of my pixel? create pixel 2 window-labels --
    """
    e.g. pix2windowLabels[hi,wi] = [list of label values]
    """
    smax = 40 # just set randomly
    tgt_pix2windowLabels = -np.ones((h,w,smax),dtype=np.int16)
    counts = np.zeros((h,w),dtype=np.int16)
    pix2windowLabels_numba(tgt_pix2windowLabels,counts,tgt.labels,ws)

    src_pix2windowLabels = -np.ones((h,w,smax),dtype=np.int16)
    counts[...] = 0
    pix2windowLabels_numba(src_pix2windowLabels,counts,src.labels,ws)

    # print(pix2windowLabels[0,0])
    # print(pix2windowLabels[16,16])

    # -- L2 Norm between super-pixels? --
    norm_sp = float("inf") * np.ones((h,w,smax),dtype=np.float32)
    superpixel_norm_numba(norm_sp,
                          src.img,src.labels,src_pix2windowLabels,#src_labe2pix,
                          src_labels2pix_ave,src_weights,
                          tgt.img,tgt_pix2windowLabels,tgt_labels2pix,
                          tgt_labels2pix_ave,tgt_weights)

    norms = np.zeros((src_nlabels,tgt_nlabels),dtype=np.float32)
    counts = np.zeros((src_nlabels,tgt_nlabels),dtype=np.float32)
    superpixel_reduce_numba(norms,counts,norm_sp,
                            src_labels2pix,tgt_pix2windowLabels)
    norms[np.where(counts == 0)] = float("inf")
    print(norms)

    k = 2
    topk = edict()
    topk.inds = np.argpartition(norms,k,axis=1)[:,:k]
    topk.vals = np.take_along_axis(norms,topk.inds,1)
    print(topk.vals)
    print(topk.inds)

    return

@jit
def superpixel_reduce_numba(norm_labels,counts,norm_sp,
                            src_labels2pix,tgt_pix2windowLabels):
    nlabels,npix,two = src_labels2pix.shape
    h,w,nsearch = tgt_pix2windowLabels.shape
    for src_label in prange(nlabels):
        for index in range(npix):
            src_h = src_labels2pix[src_label,index,0]
            src_w = src_labels2pix[src_label,index,1]
            for sidx in range(nsearch):
                tgt_label = tgt_pix2windowLabels[src_h,src_w,sidx]
                sp = norm_sp[src_h,src_w,sidx]
                if sp > 10000.: break
                norm_labels[src_label,tgt_label] += sp
                counts[src_label,tgt_label] += 1
    return norm_labels

def compute_spatial_weights(labels2pix,labels2pix_ave,labels):
    labels2pix = labels2pix.astype(np.float32)
    h,w = labels.shape
    weights = np.zeros((h,w),dtype=np.float32)
    fill_weights(weights,labels2pix_ave,labels)
    return weights

@jit
def fill_weights(weights,label_aves,labels):
    h,w = weights.shape
    for hi in prange(h):
        for wi in prange(w):
            label = labels[hi,wi]
            h_ave = label_aves[label,0]
            w_ave = label_aves[label,1]
            dist = (h_ave - hi)**2 + (w_ave - wi)**2
            weights[hi,wi] = np.exp(-dist)

def compute_labels2pix_ave(labels2pix):
    nlabels,pmax,two = labels2pix.shape
    ave = np.zeros((nlabels,2),dtype=np.float32)
    counts = np.zeros((nlabels),dtype=np.int)
    compute_labels2pix_ave_numba(labels2pix,ave,counts)
    assert np.all(counts > 0).item()
    ave = ave / counts[:,None]
    return ave

@jit
def compute_labels2pix_ave_numba(labels2pix,ave,counts):
    nlabels,pmax,two = labels2pix.shape
    for label in range(nlabels):
        for pi in range(pmax):
            h = labels2pix[label,pi,0]
            w = labels2pix[label,pi,1]
            if h == -1 or w == -1: break
            ave[label,0] += h
            ave[label,1] += w
            counts[label] += 1

@jit
def labels2pix_numba(labels2pix,counts,labels):
    h,w = labels.shape
    for hi in prange(h):
        for wi in prange(w):
            label = labels[hi,wi]
            index = counts[label]
            labels2pix[label,index,0] = hi
            labels2pix[label,index,1] = wi
            counts[label] += 1

@jit
def superpixel_norm_numba(norm_sp,
                          src_img,src_labels,src_pix2windowLabels,
                          src_labels2pix_ave,src_weights,
                          tgt_img,tgt_pix2windowLabels,tgt_labels2pix,
                          tgt_labels2pix_ave,tgt_weights):
    # -- unpack --
    h,w,smax = norm_sp.shape
    c,h,w = src_img.shape
    nlabels,pmax = tgt_labels2pix.shape[:2]

    # -- iterate over pixels --
    for hk in prange(h):
        for wk in prange(w):
            # -- source info --
            src_label = src_labels[hk,wk]
            src_w = src_weights[hk,wk]

            # -- search space of superpixels --
            for sk in prange(smax):

                # -- get candidate label --
                tgt_label = tgt_pix2windowLabels[hk,wk,sk]
                if tgt_label == -1: break

                # -- init srch value --
                dist = 0

                # -- get pixels of superpixel --
                for pk in range(pmax):
                    hj = tgt_labels2pix[tgt_label,pk,0]
                    wj = tgt_labels2pix[tgt_label,pk,1]
                    if hj == -1 or wj == -1: break

                    # -- target weight --
                    tgt_w = tgt_weights[hj,wj]

                    # -- joint weight --
                    src_ave_h = src_labels2pix_ave[src_label,0]
                    src_ave_w = src_labels2pix_ave[src_label,1]
                    tgt_ave_h = tgt_labels2pix_ave[tgt_label,0]
                    tgt_ave_w = tgt_labels2pix_ave[tgt_label,1]
                    joint_d = (src_ave_h - tgt_ave_h)**2
                    joint_d +=(src_ave_w - tgt_ave_w)**2
                    joint_d += (hk - hj)**2 + (wk - wj)**2
                    joint_w = 1.#np.exp(-joint_d)

                    # -- compte deltas --
                    pk_dist = 0
                    for ci in range(c):
                        src_pix = src_img[ci,hk,wk]/255.
                        tgt_pix = tgt_img[ci,hj,wj]/255.
                        pk_dist += (src_pix - tgt_pix)**2/c
                    pk_dist = pk_dist #* tgt_w * src_w * joint_w

                    # -- accumulate across superpixel --
                    dist += pk_dist

                # -- add to distances --
                norm_sp[hk,wk,sk] = dist

                #
                #           --> think <--
                #
                # "norm_sp[src_label,tgt_label] += ..."
                # but we aggregate after to avoid race cond.

@jit
def pix2windowLabels_numba(pix2windowLabels,counts,labels,ws):
    h,w = labels.shape
    wmax = int(pix2windowLabels.shape[-1])

    # -- iterate over pixels --
    for hi in prange(h):
        for wi in prange(w):

            # -- iterate over offsets --
            for hoff in range(2*ws):
                hj = ( hi + hoff - ws )
                hj = hj if hj >= 0 else -hj
                hj = hj if hj < h else 2*h - hj - 1
                for woff in range(2*ws):
                    wj = ( wi + woff - ws )
                    wj = wj if wj >= 0 else -wj
                    wj = wj if wj < w else 2*w - wj - 1

                    # -- get counts --
                    count_i = counts[hi,wi]

                    # -- get label at "j" --
                    label = labels[hj,wj]
                    if count_i >= wmax: continue

                    # -- do we add the label as a new one? --
                    do_add = True
                    for di in range(count_i):
                        li = pix2windowLabels[hi,wi,di]
                        if li == label:
                            do_add = False
                            break
                        # if li != label:
                        #     do_add = True
                        #     break
                    # if count_i == 0: do_add = True

                    # -- add the new one if necessary --
                    if do_add:
                        pix2windowLabels[hi,wi,count_i] = label
                        counts[hi,wi] += 1

                # -- continue break condition --
                if count_i >= wmax: continue


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
                        fbuf[label_i,label_j] += (pix_i - pix_j)**2/c
                    cbuf[hi,wi,label_j] += 1
                    # buf[label_i,label_j] = acc
