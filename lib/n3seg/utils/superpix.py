
# -- viz utils --
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
    return edict({'img':burst[index],'labels':labels[index],'slic':slics[index]})

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
    frame = frame.astype(np.int16)

    # -- create viz --
    fig, ax_arr = plt.subplots(1, 2)
    ax1, ax2 = ax_arr.ravel()
    ax1.imshow(segmentation.mark_boundaries(frame, labels))
    ax2.imshow(labels)

    # -- save -
    plt.savefig(fn,dpi=300)
    plt.close("all")
