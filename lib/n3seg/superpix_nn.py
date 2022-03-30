
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,prange

# -- org --
from easydict import EasyDict as edict
from .utils import optional


def run_superpix_nn(src,tgt,flows,**kwargs):
    """

    Compute nearest neighbors between two frame
    using the superpixels

    """

    # -- unpack args --
    c,h,w = src.img.shape
    ws = optional(kwargs,'ws',30)
    flow,zflow = get_flows(src,tgt,flows)
    flow = zflow

    # -- relevant superpix shapes  --
    delta = np.mean(((src.labels - tgt.labels)**2).astype(np.float))
    print("delta: ",delta)
    superpix_shapes(tgt)
    superpix_shapes(src)

    # -- associate labels with set of pixels --
    tgt.labels2pix = compute_labels2pix(tgt.labels,cmax=tgt.cmax)
    tgt.labels2pix_ave = compute_labels2pix_ave(tgt.labels2pix)
    src.labels2pix = compute_labels2pix(src.labels,cmax=src.cmax)
    src.labels2pix_ave = compute_labels2pix_ave(src.labels2pix)

    # -- spatial weights (distance from center) --
    tgt.weights = compute_spatial_weights(tgt.labels2pix,tgt.labels2pix_ave,tgt.labels)
    src.weights = compute_spatial_weights(src.labels2pix,src.labels2pix_ave,src.labels)

    # -- labels in search window around each pix --
    smax = 10*ws # just set randomly
    compute_search_labels(tgt,flow,smax,ws)
    compute_search_labels(src,zflow,smax,ws)

    # -- L2 Norm between super-pixels? --
    norm_sp = float("inf") * np.ones((h,w,smax),dtype=np.float32)
    weight_sp = float("inf") * np.ones((h,w,smax),dtype=np.float32)
    # superpixel_norm_numba(norm_sp,weight_sp,flow,
    #                       src.img,src.labels,src.pix2windowLabels,#src.labe2pix,
    #                       src.labels2pix_ave,src.weights,
    #                       tgt.img,tgt.pix2windowLabels,tgt.labels2pix,
    #                       tgt.labels2pix_ave,tgt.weights)

    # norms = np.zeros((src.nlabels,tgt.nlabels),dtype=np.float32)
    # norms_w = np.zeros((src.nlabels,tgt.nlabels),dtype=np.float32)
    # counts = np.zeros((src.nlabels,tgt.nlabels),dtype=np.float32)

    # superpixel_reduce_numba(norms,norms_w,counts,norm_sp,weight_sp,
    #                         src.labels2pix,tgt.pix2windowLabels)
    # norms[np.where(counts == 0)] = float("inf")
    # norms = norms/norms_w

    # -- from (h,w,list of tgt labels) -> (source label,list of tgt labels)
    labelSearchWindow = create_search_window(src.labels2pix,tgt.pix2windowLabels)

    # -- compute topk values --
    norms = float("inf") * np.ones((src.nlabels,tgt.nlabels),dtype=np.float32)
    superpixel_norm_labels_numba(norms,labelSearchWindow,
                                 src.img,src.labels,src.pix2windowLabels,
                                 src.labels2pix,src.labels2pix_ave,src.weights,
                                 tgt.img,tgt.pix2windowLabels,tgt.labels2pix,
                                 tgt.labels2pix_ave,tgt.weights)

    print(labelSearchWindow[50])
    print(norms[50])
    print("norms[50,40]: ",norms[40,50])
    print("norms[50,50]: ",norms[50,50])

    #
    # -- take topk norms --
    #
    # norms = np.random.rand(src.nlabels,tgt.nlabels)

    # -- compute topk --
    k = 5
    topk = edict()
    topk.inds = np.argpartition(norms,k,axis=1)[:,:k]
    topk.vals = np.take_along_axis(norms,topk.inds,1)

    # -- reorder to decr. values --
    order = np.argsort(topk.vals,1)
    topk.vals = np.take_along_axis(topk.vals,order,1)
    topk.inds = np.take_along_axis(topk.inds,order,1)
    print(topk.vals[[0,10,30,40,50,80,10]])
    print(topk.inds[[0,10,30,40,50,80,10]])
    print("-"*30)
    print(topk.vals[[127,37,149,19,104,179]])
    print(topk.inds[[127,37,149,19,104,179]])


    return topk


def get_flows(src,tgt,flows):
    # -- get flow --
    if tgt.index - src.index > 0:
        flow = flows.fflow[src.index].copy()
        for t in range(src.index,tgt.index):
            flow += flows.fflow[t]
    elif tgt.index - src.index < 0:
        flow = flows.bflow[src.index].copy()
        for t in range(src.index,tgt.index,-1):
            flow += flows.bflow[t]
    else:
        flow = np.zeros_like(flows.fflow[0])
    zflow = np.zeros_like(flow)
    return flow,zflow

def superpix_shapes(spix):
    spix.nlabels = np.max(spix.labels).item()+1
    spix.cnts = np.bincount(spix.labels.ravel())
    spix.cmax = spix.cnts.max().item()

@jit
def superpixel_reduce_numba(norm_labels,norm_weights,counts,norm_sp,weight_sp,
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
                w = weight_sp[src_h,src_w,sidx]
                if sp > 1000.: break
                norm_labels[src_label,tgt_label] += sp
                norm_weights[src_label,tgt_label] += w
                counts[src_label,tgt_label] += 1

def create_search_window(src_labels2pix,tgt_pix2windowLabels):
    src_nlabels = src_labels2pix.shape[0]
    h,w,nsearch = tgt_pix2windowLabels.shape[:3]
    searchLabelWindow = np.zeros((src_nlabels,nsearch),dtype=np.int16)
    create_search_window_numba(searchLabelWindow,src_labels2pix,tgt_pix2windowLabels)
    return searchLabelWindow

@jit
def create_search_window_numba(searchLabelWindow,src_labels2pix,tgt_pix2windowLabels):
    src_nlabels,npix = src_labels2pix.shape[:2]
    h,w,nsearch = tgt_pix2windowLabels.shape
    for src_label in prange(src_nlabels):
        search_index = 0
        slw_index = 0
        for pix_index in range(npix):
            tgt_h = src_labels2pix[src_label,pix_index,0]
            tgt_w = src_labels2pix[src_label,pix_index,1]
            if tgt_h == -1: break
            tgt_label = tgt_pix2windowLabels[tgt_h,tgt_w,pix_index]
            do_add = True
            for pix_j in range(npix):
                if pix_j >= slw_index: break
                curr_label = searchLabelWindow[src_label,pix_j]
                if curr_label == tgt_label:
                    do_add = False
                    break
            if do_add:
                searchLabelWindow[src_label,slw_index] = tgt_label
                slw_index += 1

def compute_search_labels(spix,flow,smax,ws):
    # -- Which labels are neighbors of my pixel? create pixel 2 window-labels --
    """
    e.g. pix2windowLabels[hi,wi] = [list of label values]
    """
    c,h,w = spix.img.shape
    spix.pix2windowLabels = -np.ones((h,w,smax),dtype=np.int16)
    counts = np.zeros((h,w),dtype=np.int16)
    pix2windowLabels_numba(spix.pix2windowLabels,counts,spix.labels,flow,ws)

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
            weights[hi,wi] = np.exp(-dist/10.)

def compute_labels2pix_ave(labels2pix):
    # -- What labels are associated with this pixel? create labels 2 pixels --
    """
    e.g. labels2pix[label] = [list of pix]
    """
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

def compute_labels2pix(labels,cmax=0):
    nlabels = labels.max().item()+1
    if cmax == 0: cmax = np.bincount(labels.ravel()).max().item()
    labels2pix = -np.ones((nlabels,cmax,2),dtype=np.int16)
    counts = np.zeros((nlabels),dtype=np.int16)
    labels2pix_numba(labels2pix,counts,labels)
    return labels2pix

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
def superpixel_norm_labels_numba(norms,labelSearchWindow,
                                 src_img,src_labels,src_pix2windowLabels,
                                 src_labels2pix,src_labels2pix_ave,src_weights,
                                 tgt_img,tgt_pix2windowLabels,tgt_labels2pix,
                                 tgt_labels2pix_ave,tgt_weights):

    # -- unpack --
    src_nlabels,tgt_nlabels = norms.shape
    c,h,w = src_img.shape
    tgt_nlabels,tgt_npix = tgt_labels2pix.shape[:2]
    src_nlabels,src_npix = src_labels2pix.shape[:2]
    nchnls = c
    src_nlabels,nsearch = labelSearchWindow.shape

    # -- compute each coordinate in parallel --
    for src_label in prange(src_nlabels):
        for search_index in prange(nsearch):
            tgt_label = labelSearchWindow[src_label,search_index]

            # -- init srch value --
            dist = 0
            Z = 0

            # -- source pixels --
            for src_pix in range(src_npix):
                src_h = src_labels2pix[src_label,src_pix,0]
                src_w = src_labels2pix[src_label,src_pix,1]
                if src_h == -1 or src_w == -1: break
                src_weight = src_weights[src_h,src_w]

                # -- target pixels --
                for tgt_pix in range(tgt_npix):
                    tgt_h = tgt_labels2pix[tgt_label,tgt_pix,0]
                    tgt_w = tgt_labels2pix[tgt_label,tgt_pix,1]
                    if tgt_h == -1 or tgt_w == -1: break
                    tgt_weight = tgt_weights[tgt_h,tgt_w]

                    # -- joint weight --
                    src_ave_h = src_labels2pix_ave[src_label,0]
                    src_ave_w = src_labels2pix_ave[src_label,1]
                    tgt_ave_h = tgt_labels2pix_ave[tgt_label,0]
                    tgt_ave_w = tgt_labels2pix_ave[tgt_label,1]
                    joint_d = ((src_h - src_ave_h) - (tgt_h - tgt_ave_h) )**2
                    joint_d += ((src_w - src_ave_w) - (tgt_w - tgt_ave_w))**2
                    joint_w = np.exp(-joint_d/(1e-3)) # how shapely the loss is

                    # -- compte deltas --
                    pk_dist = 0
                    for ci in range(nchnls):
                        src_pix = src_img[ci,src_h,src_w]/255.
                        tgt_pix = tgt_img[ci,tgt_h,tgt_w]/255.
                        pk_dist += (src_pix - tgt_pix)**2/nchnls
                    weight = joint_w * tgt_weight * src_weight
                    pk_dist = pk_dist * weight

                    # -- update agg --
                    dist += pk_dist
                    Z += weight

            # -- add to distances --
            norms[src_label,tgt_label] = dist/Z

@jit
def superpixel_norm_numba(norm_sp,weight_sp,flow,
                          src_img,src_labels,src_pix2windowLabels,
                          src_labels2pix_ave,src_weights,
                          tgt_img,tgt_pix2windowLabels,tgt_labels2pix,
                          tgt_labels2pix_ave,tgt_weights):
    # -- unpack --
    h,w,smax = norm_sp.shape
    c,h,w = src_img.shape
    nlabels,pmax = tgt_labels2pix.shape[:2]
    nchnls = c

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
                Z = 0

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
                    # joint_d=(src_ave_h - tgt_ave_h)**2 + (src_ave_w - tgt_ave_w)**2
                    # joint_d += (hk - hj)**2 + (wk - wj)**2
                    # joint_d = ((hj - (hk + flow[0,hk,wk]) - src_ave_h-tgt_ave_h) )**2
                    # joint_d += ((wj - (wk + flow[1,hk,wk]) - src_ave_w-tgt_ave_w) )**2
                    joint_d = ((hk - src_ave_h) - (hj - tgt_ave_h) )**2
                    joint_d += ((wk - src_ave_w) - (wj - tgt_ave_w))**2
                    joint_w = np.exp(-joint_d/(20.))

                    # -- compte deltas --
                    pk_dist = 0
                    for ci in range(nchnls):
                        src_pix = src_img[ci,hk,wk]/255.
                        tgt_pix = tgt_img[ci,hj,wj]/255.
                        pk_dist += (src_pix - tgt_pix)**2/nchnls
                    weight = joint_w# * tgt_w * src_w
                    pk_dist = pk_dist * weight
                    Z += weight

                    # -- accumulate across superpixel --
                    dist += pk_dist

                # -- add to distances --
                norm_sp[hk,wk,sk] = dist
                weight_sp[hk,wk,sk] = Z

                #
                #           --> think <--
                #
                # "norm_sp[src_label,tgt_label] += ..."
                # but we aggregate after to avoid race cond.

@jit
def pix2windowLabels_numba(pix2windowLabels,counts,labels,flow,ws):

    # -- unpacking --
    h,w = labels.shape
    h,w,wmax = pix2windowLabels.shape

    # -- iterate over pixels --
    for hi in prange(h):
        for wi in prange(w):

            # -- iterate over offsets --
            h_left,h_right = -1,1
            for hindex in range(2*ws):
                if hindex == 0:
                    h_shift = 0
                elif hindex % 2 == 0:
                    h_shift = h_left
                    h_left -= 1
                else:
                    h_shift = h_right
                    h_right += 1

                # -- offsets --
                hoff = h_shift
                h0 = round( hi + hoff + flow[0,hi,wi])
                h0 = h0 if h0 >= 0 else -h0
                h0 = h0 if h0 < h else 2*h - h0 - 1

                w_left,w_right = -1,1
                for windex in range(2*ws):
                    if windex == 0:
                        w_shift = 0
                    elif windex % 2 == 0:
                        w_shift = w_left
                        w_left -= 1
                    else:
                        w_shift = w_right
                        w_right += 1

                    # -- offsets --
                    woff = w_shift
                    w0 = round( wi + woff + flow[1,hi,wi])
                    w0 = w0 if w0 >= 0 else -w0
                    w0 = w0 if w0 < w else 2*w - w0 - 1

                    # -- location --
                    hj = h0
                    wj = w0

                    # -- get flow location --
                    # hj = round(h0 + flow[0,h0,w0])
                    # wj = round(w0 + flow[1,h0,w0])

                    # # -- bounds --
                    # hj = hj if hj >= 0 else -hj
                    # hj = hj if hj < h else 2*h - hj -1

                    # wj = wj if wj >= 0 else -wj
                    # wj = wj if wj < w else 2*w - wj -1

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
