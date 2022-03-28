
import numpy as np
import torch as th


from .image import save_image

def viz_superpix(src_sp,tgt_sp,label):
    superpix_box = get_boxes(src_sp.img,src_sp.labels,label)
    fn = "superpix_box.png"
    save_image(superpix_box,fn)

def get_boxes(img,labels,label):
    hargs,wargs = np.where(labels == label)
    hstart,hstop = hargs.min(),hargs.max()
    wstart,wstop = wargs.min(),wargs.max()
    hslice = slice(hstart,hstop)
    wslice = slice(wstart,wstop)
    hslice = hstart,hstop
    superpix_box = img[:,hslice,wslice]
    c,sh,sw = superpix_box.shape
    mask = np.ones((sh,ws),dtype=np.bool)
    superpix_box = mask * superpix_box
    return superpix_box
