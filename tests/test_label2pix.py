
# -- python --
import cv2,tqdm,copy
import numpy as np
import unittest
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- linalg --
import torch as th
import numpy as np

# -- package  --
import n3seg

#
#
# -- Primary Testing Class --
#
#
PYTEST_OUTPUT = Path("./pytests/output/")


class TestLabel2Pix(unittest.TestCase):


    #
    # -- [Primary Logi] --
    #

    def run_comparison(self,burst,args):

        # -- fixed testing params --
        K = 10
        BSIZE = 50
        NBATCHES = 3
        shape = burst.shape

        # -- get labels --
        slics,labels = n3seg.run_slic(burst)
        src = n3seg.utils.superpix_dict(burst,labels,slics,0)

        # -- execute label2pix --
        labels2pix = n3seg.superpix_nn.compute_labels2pix(src.labels)

        # -- check each label --
        nlabels = labels2pix.shape[0]
        for label in range(nlabels):
            label_pixels = labels2pix[label]
            invalid = np.where(label_pixels[:,0] == -1)[0]
            if len(invalid) > 0: stop = np.min(invalid[0])
            else: stop = -1
            hargs,wargs = np.where(src.labels == label)
            gt_label_pixels = np.c_[hargs,wargs]
            label_pixels = label_pixels[:stop]
            deltas = label_pixels[None,:,:] - gt_label_pixels[:,None,:]
            deltas = np.mean(deltas**2,-1)
            assert np.all(np.any(deltas,0))

    #
    # -- Load Data --
    #

    def do_load_data(self,dname,device="cuda:0"):
        #  -- Read Data (Image & VNLB-C++ Results) --
        burst = n3seg.testing.load_data(dname)[:4]
        burst = burst[:,:,:128,:128]
        return burst

    def run_single_test(self,dname,args):
        burst = self.do_load_data(dname)
        self.run_comparison(burst,args)

    def test_sim_search(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = PYTEST_OUTPUT
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- test 1 --
        dname = "motorbike"
        args = edict()#{'ps':7,'pt':1,'c':3})
        self.run_single_test(dname,args)
