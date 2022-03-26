
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- this lib --
import n3seg

def main():

    # -- params --
    viz = True

    # -- load image burst --
    burst = n3seg.testing.load_data("motorbike")[:3]
    burst = burst[:,:32,:32]

    # -- superpixels --
    slics,labels = n3seg.run_slic(burst)

    # -- [optional] viz superpix --
    if viz:
        n3seg.utils.viz_burst_superpix(burst,slics,labels,"./output/example/","ex")

    # -- get superpix data struct --
    src = n3seg.utils.superpix_dict(burst,labels,slics,0)
    tgt = n3seg.utils.superpix_dict(burst,labels,slics,1)

    # -- compute nearest neighbors --
    neighs = n3seg.run_superpix_nn(src,tgt)
    warped = n3seg.run_superpix_warp(src,tgt,neighs)

    # -- [optional] viz warped frame --
    if viz:
        n3seg.utils.save_image(warped,"./output/warped.png")

    # -- compute sim quality --
    delta = np.mean((warped - burst[0])**2).item()
    print("[Sim Image (SSD)]: ",delta)


if __name__ == "__main__":
    main()
