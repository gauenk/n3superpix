"""
Optical Flow with OpenCV
"""

import cv2
import torch as th
import numpy as np
from easydict import EasyDict as edict

def compute_flows(burst):
    fflows = []
    bflows = []
    nframes = burst.shape[0]
    burst = burst * 255.
    for t in range(nframes):
        fflow_t = get_flow(burst,t,t+1)
        fflows.append(fflow_t)
        bflow_t = get_flow(burst,t-1,t)
        bflows.append(bflow_t)
    fflows = th.stack(fflows).numpy()
    bflows = th.stack(bflows).numpy()

    flows = edict()
    flows.fflows = fflows
    flows.bflows = bflows

    return flows

def get_flow(burst,tc,tn):

    # -- unpack --
    tf32 = th.float32
    t,c,h,w = burst.shape
    if th.is_tensor(burst):
        device = burst.device
    else:
        device = 'cpu'
        burst = th.from_numpy(burst).to(device)

    # -- handle error case --
    invalid = (t <= tn) or (tc < 0)
    if invalid:
        zflow = th.zeros((2,h,w),device=device,dtype=tf32)
        return zflow

    # -- compute flows --
    frame_a = burst[tc].cpu().numpy()
    frame_b = burst[tn].cpu().numpy()

    # -- fixup types --
    frame_a = frame_a.astype(np.uint8)
    frame_b = frame_b.astype(np.uint8)

    # -- correct outputs --
    frame_a = frame_a.transpose(1,2,0)
    frame_b = frame_b.transpose(1,2,0)

    # -- frames to gray --
    frame_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    frame_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

    # -- create opencv-gpu frames --
    gpu_frame_a = cv2.cuda_GpuMat()
    gpu_frame_b = cv2.cuda_GpuMat()
    gpu_frame_a.upload(frame_a)
    gpu_frame_b.upload(frame_b)

    # -- create flow object --
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False,
                                                    15, 3, 5, 1.2, 0)

    # -- exec flow --
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_frame_a,
                                                  gpu_frame_b, None)
    gpu_flow = gpu_flow.download()
    gpu_flow = gpu_flow.transpose(2,0,1)
    gpu_flow = th.from_numpy(gpu_flow).half()

    return gpu_flow
