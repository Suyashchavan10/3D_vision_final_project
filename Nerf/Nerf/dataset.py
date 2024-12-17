import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import json
from scipy.spatial.transform import Rotation as R


def get_rays(datapath, mode='train'):
    # pose_file_name = os.listdir(f"{datapath}/{mode}/pose")
    # intrinsics_file_name = os.listdir(f"{datapath}/{mode}/intrinsics")
    img_file_names = [f for f in os.listdir(f"{datapath}/images/")]

    # assert len(pose_file_name) == len(intrinsics_file_name)
    # assert len(img_file_names) == len(pose_file_name)


    json_file = json.load(open(f"{datapath}/reconstruction.json"))

    N = len(json_file[0]["shots"])
    poses = np.zeros((N, 4, 4))
    intrinsics = np.zeros((N, 4, 4))
    i = 0

    for keys, values in json_file[0]["shots"].items():
        rot = values["rotation"]
        trans = values["translation"]
        rotation = R.from_rotvec(rot).as_matrix()

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation

        pose_matrix[:3, 3] = trans

        focal = json_file[0]["cameras"]["v2 unknown unknown 1280 720 perspective 0"]["focal"]
        cx = json_file[0]["cameras"]["v2 unknown unknown 1280 720 perspective 0"]["k1"]
        cy = json_file[0]["cameras"]["v2 unknown unknown 1280 720 perspective 0"]["k2"]

        K = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1]
        ])

        # Extend to a 4x4 matrix
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K

        poses[i] = pose_matrix
        intrinsics[i] = intrinsic
        i += 1

    images = []

    for i in range(N):
        # name = pose_file_name[i]
        # pose = open(f"{datapath}/{mode}/pose/{name}").read().split()
        # poses[i] = np.array(pose, dtype=float).reshape(4, 4)

        # intrin = open(f"{datapath}/{mode}/intrinsics/{name}").read().split()
        # intrinsics[i] = np.array(intrin, dtype=float).reshape(4, 4)

        img = imageio.imread(f"{datapath}/images/{img_file_names[i].replace('txt', 'jpg')}") / 255.

        images.append(img[None, ...])
    images = np.concatenate(images)

    H = images.shape[1]
    W = images.shape[2]

    if images.shape[3] == 4:
        images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])



    rays_d = np.zeros((N, H*W, 3))
    rays_o = np.zeros((N, H*W, 3))
    target_px_values = images.reshape((N, H*W, 3))


    for i in range(N):
        c2w = poses[i]
        f = intrinsics[i, 0, 0]

        u = np.arange(W)
        v = np.arange(H)

        u, v = np.meshgrid(u, v)
        dirs = np.stack((u - W / 2,
                 -(v - H / 2),
                 - np.ones_like(u) * f), axis=-1)

        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)

        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        rays_d[i] = dirs.reshape(-1, 3)
        rays_o[i] = c2w[:3, 3]

    return rays_o, rays_d, target_px_values