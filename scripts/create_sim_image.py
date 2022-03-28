
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- this lib --
import n3seg

def main():

    # -- params --
    viz = True
    seed = 123
    np.random.seed(seed)

    # -- load image burst --
    burst = n3seg.testing.load_data("motorbike")[:4]
    burst = burst[:,:,:128,:128]

    # -- compute flow --
    flows = n3seg.flows.compute_flows(burst)

    # -- superpixels --
    slics,labels = n3seg.run_slic(burst)

    # -- [optional] viz superpix --
    if viz:
        n3seg.utils.viz_burst_superpix(burst,slics,labels,"./output/example/","ex")

    # -- get superpix data struct --
    src = n3seg.utils.superpix_dict(burst,labels,slics,0)
    tgt = n3seg.utils.superpix_dict(burst,labels,slics,1)

    # -- show a single superpixel --
    if viz:
        sp_box = n3seg.utils.get_superpix_box(src.img,src.labels,0)
        print("sp_box.shape: ",sp_box.shape)
        n3seg.utils.save_image(sp_box,"./output/sp_box.png")

    # -- compute nearest neighbors --
    neighs = n3seg.run_superpix_nn(src,tgt,flows)
    if viz:
        viz_list = np.random.permutation(200)[:20]
        sp_boxes = n3seg.utils.get_superpix_box_nn(src,tgt,neighs,viz_list)
        n3seg.utils.save_burst(sp_boxes,"./output/boxes/","a")
        viz_list = np.random.permutation(200)[:20]
        sp_boxes = n3seg.utils.get_superpix_box_nn(src,tgt,neighs,viz_list)
        n3seg.utils.save_burst(sp_boxes,"./output/boxes/","b")
        viz_list = np.random.permutation(200)[:20]
        sp_boxes = n3seg.utils.get_superpix_box_nn(src,tgt,neighs,viz_list)
        n3seg.utils.save_burst(sp_boxes,"./output/boxes/","c")

    # -- warp frame --
    warped = n3seg.run_superpix_warp(src,tgt,neighs)

    # -- [optional] viz warped frame --
    if viz:
        n3seg.utils.save_image(warped,"./output/warped.png")

    # -- compute sim quality --
    delta = np.mean((warped - burst[0])**2).item()
    print("[Sim Image (SSD)]: ",delta)


if __name__ == "__main__":
    main()
