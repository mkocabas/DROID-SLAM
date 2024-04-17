import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import droid_backends
import cv2
import os
import glob 
import time
import argparse

from lietorch import SE3
from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

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

        yield t, image[None], intrinsics


def save_reconstruction(droid, traj_est, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    # Save 3D point cloud
    points = droid_backends.iproj(
        SE3(droid.video.poses[:t]).inv().data, 
        droid.video.disps[:t], droid.video.intrinsics[0]
    ).cpu().numpy()
    colors = torch.from_numpy(images)[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0

    thresh = 0.005 * torch.ones_like(droid.video.disps[:t].mean(dim=[1,2]))
    count = droid_backends.depth_filter(
                droid.video.poses[:t], 
                droid.video.disps[:t], 
                droid.video.intrinsics[0], 
                torch.arange(0, t).to('cuda'), 
                thresh
    )

    count = count.cpu()
    disps_vis = droid.video.disps[:t].cpu()
    masks = ((count >= 2) & (disps_vis > .5*disps_vis.mean(dim=[1,2], keepdim=True)))

    mask = masks.reshape(-1)
    pcl = np.concatenate([points.reshape(-1, 3)[mask], colors.reshape(-1, 3)[mask]], axis=-1)

    ## 4x4 camera matrix
    Ps = SE3(droid.video.poses[:t]).inv().matrix().cpu().numpy()
    full_Ps = SE3(torch.from_numpy(traj_est)).matrix().cpu().numpy()

    # Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    # np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    # np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    # np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    # np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    # np.save("reconstructions/{}/cam_poses.npy".format(reconstruction_path), Ps)
    # np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    # Path("{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    # np.save("{}/tstamps.npy".format(reconstruction_path), tstamps)
    # np.save("{}/images.npy".format(reconstruction_path), images)
    # np.save("{}/disps.npy".format(reconstruction_path), disps)
    # np.save("{}/poses.npy".format(reconstruction_path), poses)
    # np.save("{}/cam_poses.npy".format(reconstruction_path), Ps)
    # np.save("{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    # np.save("{}/points.npy".format(reconstruction_path), points)
    # np.save("{}/colors.npy".format(reconstruction_path), colors)
    # np.save("{}/masks.npy".format(reconstruction_path), masks)
    # np.save("{}/point_cloud.npy".format(reconstruction_path), pcl)
    # np.save("{}/cam_poses_full.npy".format(args.reconstruction_path), full_Ps)
    
    # breakpoint()
    np.save("{}/droid_cam_poses.npy".format(reconstruction_path), full_Ps)
    torch.save(torch.from_numpy(full_Ps), "{}/droid_cam_poses.pt".format(reconstruction_path))
    np.save("{}/droid_point_cloud.npy".format(reconstruction_path), pcl)
    print(f'Saved results in {reconstruction_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
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
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    print(args)
    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    
    if args.reconstruction_path is not None:
        print('Saving reconstruction to {}'.format(args.reconstruction_path))
        save_reconstruction(droid, traj_est, args.reconstruction_path)
        