

import os
import sys
import cv2
import glob 
import time
import torch
import Imath
import lietorch
import argparse
import numpy as np
import OpenEXR as exr
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

sys.path.append('droid_slam')
from torch.multiprocessing import Process
from droid import Droid
from droid_slam.pcl import save_pcl


def read_depth_exr_file(filepath):
    exrfile = exr.InputFile(filepath)
    raw_bytes = exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    import matplotlib
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(imagedir, calib, stride, maskdir=None, depthdir=None):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    mask_list = sorted(os.listdir(maskdir))[::stride] if maskdir is not None else None
    
    if depthdir is not None:
        depth_list = []
        depth_list += glob.glob(os.path.join(depthdir, "*.exr"))[::stride]
        depth_list += glob.glob(os.path.join(depthdir, "*.npy"))[::stride]
        depth_list = sorted(depth_list)
        assert len(depth_list) == len(image_list), "Number of depth files does not match number of images"
        assert len(depth_list) > 0, "No depth files found"
    else:
        depth_list = None

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        if mask_list is not None:
            mask = cv2.imread(os.path.join(maskdir, mask_list[t]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (w1, h1), interpolation=cv2.INTER_CUBIC)
            mask = mask[:h1-h1%8, :w1-w1%8]
            mask = 1 - (torch.from_numpy(mask) / 255)
            mask.unsqueeze_(0)
        else:
            mask = torch.ones(1, h1, w1)
            mask = mask[:, :h1-h1%8, :w1-w1%8]
            
        if depth_list is not None:
            if depth_list[t].endswith('.exr'):
                depth = read_depth_exr_file(depth_list[t])
                depth = cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_CUBIC)
                depth = depth[:h1-h1%8, :w1-w1%8]
                depth = torch.from_numpy(depth).float()
            elif depth_list[t].endswith('.npy'):
                depth = np.load(depth_list[t])
                depth = cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_CUBIC)
                depth = depth[:h1-h1%8, :w1-w1%8]
                depth = torch.from_numpy(depth).float()
        else:
            depth = None
        
        # TODO: mask the mask based on the depth map
        masked_image = image * mask
        
        # resize the mask to 1/8th of the original size
        mask = F.interpolate(mask[None], (h1//8, w1//8), mode='nearest')
        
        if depth is None:
            print("Warning: No depth map provided")
        
        yield t, masked_image[None], intrinsics, mask, depth


def save_reconstruction(droid, reconstruction_path, traj_est=None):

    from pathlib import Path
    import random
    import string

    # pcl, clr = get_pcl(droid.video)
    if traj_est is not None:
        print("Debug")
        print('poses:', droid.video.poses.shape)
        print('traj_est:', traj_est.shape)
        print('t:', droid.video.counter.value)
        # import ipdb; ipdb.set_trace()
    else:
        print("Debug")
        print('poses:', droid.video.poses.shape)
        print('t:', droid.video.counter.value)
        
    
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()
    
    os.makedirs(reconstruction_path, exist_ok=True)
    print(f"Saving reconstruction to {reconstruction_path}")
    
    np.save(f"{reconstruction_path}/tstamps.npy", tstamps)
    np.save(f"{reconstruction_path}/images.npy", images)
    np.save(f"{reconstruction_path}/disps.npy", disps)
    np.save(f"{reconstruction_path}/poses.npy", poses)
    np.save(f"{reconstruction_path}/intrinsics.npy", intrinsics)
    
    print(f"Number of keyframes: {tstamps.shape}")
    print(f"Poses shape: {poses.shape}")
    
    if traj_est is not None:
        Ps = lietorch.SE3(torch.from_numpy(traj_est)).matrix().cpu().numpy()
        np.save(f"{reconstruction_path}/poses_full.npy", Ps)
        print(f"Poses_full shape: {Ps.shape}")
    
    # save disparity maps as EXR format
    depths = 1.0 / disps
    os.makedirs(f"{reconstruction_path}/depthmaps", exist_ok=True)
    for depth, tstamp in zip(depths, tstamps):
        cdepth = colorize(depth, vmin=0.01, vmax=10.0)
        Image.fromarray(cdepth).save(f"{reconstruction_path}/depthmaps/{int(tstamp):06d}.png")

    print("Reconstruction saved")
    files = os.listdir(reconstruction_path)
    print([f"{reconstruction_path}/{f}" for f in files])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--maskdir", type=str, help="path to mask directory", default=None)
    parser.add_argument("--depthdir", type=str, help="path to depth directory", default=None)
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="ckpt/droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction", default='./outputs')
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    os.makedirs(args.reconstruction_path, exist_ok=True)
    
    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    stream = image_stream(args.imagedir, args.calib, args.stride, args.maskdir, args.depthdir)
    for (t, image, intrinsics, mask, depth) in tqdm(stream):
        if t < args.t0:
            continue

        # if not args.disable_vis:
        #     show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, depth=depth, intrinsics=intrinsics, mask=mask)

    # before bundle adjustment
    
    if args.reconstruction_path is not None:
        out_dir = args.reconstruction_path + '/before_ba'
        os.makedirs(out_dir, exist_ok=True)
        
        save_pcl(droid.video, out_dir, filter_thresh=0.005, filter_dirty=True)
        save_pcl(droid.video, out_dir, filter_thresh=0.005, filter_dirty=False)
        
        save_pcl(droid.video, out_dir, filter_thresh=0.01, filter_dirty=True)
        save_pcl(droid.video, out_dir, filter_thresh=0.01, filter_dirty=False)
        
        save_pcl(droid.video, out_dir, filter_thresh=0.0025, filter_dirty=True)
        save_pcl(droid.video, out_dir, filter_thresh=0.0025, filter_dirty=False)
        
        save_reconstruction(droid, out_dir)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride, args.maskdir, args.depthdir))
    
    if args.reconstruction_path is not None:
        out_dir = args.reconstruction_path + '/after_ba'
        os.makedirs(out_dir, exist_ok=True)
        
        save_pcl(droid.video, out_dir, filter_thresh=0.005, filter_dirty=False)
        save_pcl(droid.video, out_dir, filter_thresh=0.005, filter_dirty=True)
        
        save_pcl(droid.video, out_dir, filter_thresh=0.01, filter_dirty=False)
        save_pcl(droid.video, out_dir, filter_thresh=0.01, filter_dirty=True)
        
        save_pcl(droid.video, out_dir, filter_thresh=0.0025, filter_dirty=False)
        save_pcl(droid.video, out_dir, filter_thresh=0.0025, filter_dirty=True)
        
        save_reconstruction(droid, out_dir, traj_est)
