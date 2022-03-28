
# -- viz utils --
import torchvision.utils as tv_utils
import torch.nn.functional as nnf
from skimage import segmentation, color
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- pathlib --
from pathlib import Path

# -- org --
from easydict import EasyDict as edict

# ------------------------------------
#
#        Data Structs
#
# ------------------------------------

def superpix_dict(burst,labels,slics,index):
    return edict({'img':burst[index],
                  'labels':labels[index],
                  'slic':slics[index],
                  'index':index})


# ------------------------------------
#
#    Indiviaul Superpixels Mangling
#
# ------------------------------------

def get_superpix_box_nn(src,tgt,neighs,viz_list=None):
    if viz_list is None:
        viz_list = np.arange(120,140)
    boxes_list = []
    for viz_label in viz_list:

        # -- get boxes --
        label_nn = neighs.inds[viz_label]
        print("label_nn: ",viz_label,label_nn)
        src_box = get_superpix_box(src.img,src.labels,viz_label)
        boxes = get_superpix_list(tgt.img,tgt.labels,label_nn)
        boxes = update_boxes(boxes,src_box)
        boxes = [boxes[-1]] + [x for x in boxes[:-1]]

        # -- create grid --
        boxes = np.stack(boxes)
        boxes = th.from_numpy(boxes)
        boxes = tv_utils.make_grid(boxes,nrow=len(boxes))
        boxes = boxes.numpy()

        # -- create grid --
        boxes_list = update_boxes(boxes_list,boxes)

    # -- make a grid --
    boxes_list = np.stack(boxes_list)
    nviz = boxes_list.shape[0]
    boxes_list = th.from_numpy(boxes_list)
    boxes_list = tv_utils.make_grid(boxes_list,nrow=1)

    # -- single frame --
    boxes_list = boxes_list[None,:]

    return boxes_list

def get_superpix_list(img,labels,query_labels):
    boxes = []
    for query_label in query_labels:
        box_q = get_superpix_box(img,labels,query_label)
        # -- update boundaries --
        boxes = update_boxes(boxes,box_q)
    return boxes

def update_boxes(boxes,prop_box):
    boxes.append(prop_box)
    h_max,w_max = 0,0
    for box in boxes:
        c,h_p,w_p = box.shape
        if h_max < h_p: h_max = h_p
        if w_max < w_p: w_max = w_p
    nboxes = len(boxes)
    for i in range(nboxes):
        th_box_i = th.from_numpy(boxes[i])[None,:].type(th.float)
        th_box_i = nnf.interpolate(th_box_i,size=(h_max,w_max),mode="bicubic")
        boxes[i] = th_box_i.cpu().clamp(0,255.).numpy()[0].astype(np.uint8)
    return boxes

def get_superpix_box(img,labels,query_label):

    # -- indices for enclosing square --
    hargs,wargs = np.where(labels == query_label)
    hstart,hstop = hargs.min(),hargs.max()+1
    wstart,wstop = wargs.min(),wargs.max()+1
    hmod = hargs - hstart
    wmod = wargs - wstart
    hslice = hstart,hstop

    # -- get enclosing square --
    superpix_box = img[:,hstart:hstop,wstart:wstop]

    # -- mask non-included boundaries --
    c,sh,sw = superpix_box.shape
    mask = np.zeros((sh,sw),dtype=np.bool)
    mask[hmod,wmod] = 1
    superpix_box = mask * superpix_box

    return superpix_box


# ------------------------------------
#
#       Vizualize Superpixels
#
# ------------------------------------


def viz_burst_superpix(burst,slics,labels,root,name):

    # -- root path -
    root = Path(root)
    if not root.exists(): root.mkdir(parents=True)

    # -- save for each frame --
    nframes = len(burst)
    for t in range(nframes):
        name_t  = "%s_%d.png" % (name,t)
        fn_t = str(root / name_t)
        viz_frame_superpix(burst[t],slics[t],labels[t],fn_t)


def viz_frame_superpix(frame,slic,labels,fn):

    # -- frame mod --
    if th.is_tensor(frame):
        frame = frame.cpu().numpy()
    if th.is_tensor(labels):
        labels = labels.cpu().numpy()

    # -- format frame --
    frame = rearrange(frame,'c h w -> h w c')
    frame = frame.astype(np.float32)/255.
    frame = np.ascontiguousarray(frame)

    # -- create viz --
    fig, ax_arr = plt.subplots(1, 3)
    ax0, ax1, ax2 = ax_arr.ravel()
    marked_frame = segmentation.mark_boundaries(frame, labels,background_label=-1)
    ax0.imshow(frame)
    ax1.imshow(marked_frame)
    ax2.imshow(labels)

    # -- save -
    plt.savefig(fn,dpi=300)
    plt.cla()
    plt.close("all")
