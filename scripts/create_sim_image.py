
# -- org --
from easydict import EasyDict as edict

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- vision --

# -- competitor --
import vpss

# -- this lib --
import n3seg

def mse(a,b):
    return th.mean(((a - b)/255.)**2).item()

def viz_patch_warping(burst,flows):
    warped_ref = burst[0]
    warped_db = burst[1]
    vflows = edict({k:v[:2] for k,v in flows.items()})
    warped_3 = vpss.sim_img.get_sim_from_pair(burst[0],burst[1],flows=vflows,ps=3)
    warped_7 = vpss.sim_img.get_sim_from_pair(burst[0],burst[1],flows=vflows,ps=7)
    error_w3 = mse(warped_3.cpu(),warped_ref)
    error_w7 = mse(warped_7.cpu(),warped_ref)
    n3seg.utils.save_image(warped_ref,"./output/warped_ref.png")
    n3seg.utils.save_image(warped_db,"./output/warped_db.png")
    n3seg.utils.save_image(warped_3,"./output/warped_3.png")
    n3seg.utils.save_image(warped_7,"./output/warped_7.png")
    print("Error[ps=3]: ",error_w3)
    print("Error[ps=7]: ",error_w7)

def main():

    # -- params --
    viz = True
    seed = 123
    np.random.seed(seed)

    # -- load image burst --
    burst = n3seg.testing.load_data("motorbike")[:3]
    burst = n3seg.utils.interpolate(burst,size=(256,256),mode='bicubic')
    burst = burst[:,:,64:128,64:128]
    burst[1] = burst[0]

    # -- compute flow --
    flows = n3seg.flows.compute_flows(burst)
    if viz:
        viz_patch_warping(burst,flows)

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
    print(warped)

    # -- compute sim quality --
    delta = np.mean((warped - burst[0])**2).item()
    print("[Sim Image (SSD)]: ",delta)


if __name__ == "__main__":
    main()
