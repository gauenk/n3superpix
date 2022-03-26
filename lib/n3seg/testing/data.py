
# -- vision --
from PIL import Image

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- paths --
from pathlib import Path


DATA_ROOT = Path("./data/")
MAX_FRAMES = 30

def load_data(name,ext="png"):
    path = DATA_ROOT/name
    burst = []
    for t in range(MAX_FRAMES):
        fn_t = path / ("%05d.%s" % (t,ext))
        img = np.array(Image.open(fn_t))
        img = th.from_numpy(img)
        img = rearrange(img,'h w c -> c h w')
        burst.append(img)
    burst = th.stack(burst)
    return burst
