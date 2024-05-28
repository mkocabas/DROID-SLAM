import cv2
import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops

import matplotlib

def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
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


class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, filter_inp_depth=False, device="cuda:0"):
        
        self.filter_inp_depth = filter_inp_depth
        
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.masks = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        
    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]
            
        if len(item) > 9:
            self.masks[index] = item[9]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
                self.masks[index],
            )

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, 
           motion_only=False, robustify=False, debug=True, iter_n=0):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            with torch.no_grad():
                weight_mask = self.masks[ii] * self.masks[jj]
                weight_mask.unsqueeze_(1)
                weight = weight * weight_mask
        
            if self.filter_inp_depth:
                idxs = torch.cat([ii, jj]).unique()
                with torch.no_grad():
                    disps_sens = self.disps_sens.clone()
                    disps_est = self.disps.detach().clone()
                    # filter based on mask
                    # disps_sens = torch.where(self.masks.bool(), disps_sens, disps_est)
                    # filter based on far threshold
                    disps_sens[idxs] = torch.where(disps_sens[idxs] < (1/10.), disps_est[idxs], disps_sens[idxs])
                    disps_sens[idxs] = torch.where(disps_est[idxs] < (1/10.), disps_est[idxs], disps_sens[idxs])
                    
                    if robustify:
                        
                        # robust loss as a quadratic equation
                        # ax^2 + bx + c = 0, 
                        # a = 1, b = -2*disps_est, c = disps_est^2 - sigma^2 * res / (res + sigma^2)
                        depth_est = 1.0 / disps_est[idxs]
                        depth_sens = 1.0 / disps_sens[idxs]
                        b = -2.0 * depth_est
                        sigma = 100.0
                        res_sq = (depth_est - depth_sens).pow(2)
                        c = depth_est.pow(2) - (sigma**2 * res_sq / (res_sq + sigma**2))
                        d_prime_1 = (-b + torch.sqrt(b.pow(2) - 4.*c)) / (2.) 
                        d_prime_2 = (-b - torch.sqrt(b.pow(2) - 4.*c)) * 0.5
                        new_depth_sens = d_prime_2
                        
                        print(f"Depth Sens: {depth_sens.min().item()}, {depth_sens.max().item()}")
                        print(f"Depth Est: {depth_est.min().item()}, {depth_est.max().item()}")
                        print(f"New depth sens: {new_depth_sens.min().item()}, {new_depth_sens.max().item()}")
                        
                        rres_1 = (depth_est - d_prime_1).pow(2)
                        rres_2 = (depth_est - d_prime_2).pow(2)
                        rres_gt = sigma**2 * res_sq / (res_sq + sigma**2)
                        print(f"rres_1: {rres_1.mean().item()}, rres_2: {rres_2.mean().item()}, rres_gt: {rres_gt.mean().item()}")
                        
                        disps_sens[idxs] = 1.0 / new_depth_sens
                        
                        # Outlier detection
                        '''
                        # Compute residuals
                        res = torch.abs(disps_est[idxs] - disps_sens[idxs]).reshape(idxs.shape[0], -1)

                        # Compute z-scores
                        mean_residual = torch.mean(res, dim=1, keepdim=True)
                        std_residual = torch.std(res, dim=1, keepdim=True)
                        z_scores = (res - mean_residual) / std_residual

                        # Identify outliers (e.g., z-score > 2)
                        outlier_threshold = 2
                        outliers = z_scores > outlier_threshold
                        outliers = outliers.reshape(*disps_est[idxs].shape)
                        print(f"ration of outliers: {outliers.sum().item() / outliers.numel()}")
                        
                        disps_sens[idxs][outliers] = disps_est[idxs][outliers]
                        '''
                        
                    # import ipdb; ipdb.set_trace()
                    print(f"counter: ", self.counter.value)
                    if debug:
                        for idx in torch.cat([ii, jj]).unique().cpu().numpy().tolist():
                            inp_depth = 1./disps_sens[idx].cpu().numpy()
                            slam_depth = 1./self.disps[idx].cpu().numpy()
                            depth_arel = np.abs(inp_depth - slam_depth) / inp_depth
                            
                            inp_depth_col = colorize(inp_depth, 0.1, 10.0)
                            slam_depth_col = colorize(slam_depth, 0.1, 10.0)
                            depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")
                            
                            img = np.concatenate([inp_depth_col, slam_depth_col, depth_error_col], axis=1)
                            cv2.imwrite(f"../../results/.depth_cache/inp_depth_{idx:06d}.jpg", img[..., ::-1])
                        
                        total_filtered = (1 - self.masks[ii]).bool().sum() + (disps_sens[ii] < (1/10.)).sum()
                        total = self.masks[ii].numel()
                        print(f"Filtered %{(total_filtered/total)*100:.2f} points")
            else:
                disps_sens = self.disps_sens

            droid_backends.ba(self.poses, self.disps, self.intrinsics[0], disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)
