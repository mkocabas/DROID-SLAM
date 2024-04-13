import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops


CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])


def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


@torch.no_grad()
def save_pcl(video, save_path, device="cuda:0", filter_thresh=0.005, filter_dirty=True):
    
    torch.cuda.set_device(device)
    
    points_dict = {}
    cameras = {}

    with video.get_lock():
        t = video.counter.value 
        if filter_dirty:
            dirty_index, = torch.where(video.dirty.clone())
            dirty_index = dirty_index
        else:
            dirty_index = torch.arange(video.dirty.shape[0]).to(video.dirty.device)

    if len(dirty_index) == 0:
        return

    # video.dirty[dirty_index] = False

    # convert poses to 4x4 matrix
    poses = torch.index_select(video.poses, 0, dirty_index)
    disps = torch.index_select(video.disps, 0, dirty_index)
    Ps = SE3(poses).inv().matrix().cpu().numpy()

    images = torch.index_select(video.images, 0, dirty_index)
    images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
    points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
    
    count = droid_backends.depth_filter(
        video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
    
    count = count.cpu()
    disps = disps.cpu()
    masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))     

    for i in range(len(dirty_index)):
        pose = Ps[i]
        ix = dirty_index[i].item()

        ### add camera actor ###
        cam_actor = create_camera_actor(True)
        cam_actor.transform(pose)

        cameras[ix] = cam_actor
        
        
        mask = masks[i].reshape(-1)
        pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
        clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
        
        ## add point actor ###
        point_actor = create_point_actor(pts, clr)
        points_dict[ix] = point_actor

    ### Hack to save Point Cloud Data and Camnera results ###
    
    # Save points
    pcd_points = o3d.geometry.PointCloud()
    for p in points_dict.items():
        pcd_points += p[1]
    
    o3d.io.write_point_cloud(f"{save_path}/points_ft={filter_thresh}_fdirty={filter_dirty}.ply", pcd_points, write_ascii=False)
        
    # Save pose
    pcd_camera = create_camera_actor(True)
    for c in cameras.items():
        pcd_camera += c[1]

    o3d.io.write_line_set(f"{save_path}/camera.ply", pcd_camera, write_ascii=False)
    