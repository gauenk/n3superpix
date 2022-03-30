
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- slic for super-pixels --
from fast_slic import Slic


def run_slic(burst,ncomps=200,comp=30):
    nframes = burst.shape[0]
    slics,labels = [],[]
    for t in range(nframes):

        # -- prepare --
        frame_t = burst[t]
        frame_t = rearrange(frame_t,'c h w -> h w c')
        frame_t = frame_t.cpu().numpy() if th.is_tensor(frame_t) else frame_t
        frame_t = np.ascontiguousarray(frame_t.astype(np.uint8))

        # -- exec --
        slic_t = Slic(num_components=ncomps, compactness=comp)
        labels_t = slic_t.iterate(frame_t)

        # -- append --
        slics.append(slic_t)
        labels.append(labels_t)

    # -- formatting --
    # labels = th.from_numpy(np.stack(labels))

    return slics,labels
