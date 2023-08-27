import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_multiscale_data(basedir, white_bkgd):
    print("loading")
    splits = ['train', 'val', 'test']
    metas = {}
    for split in splits:
        with open(os.path.join(basedir, 'metadata.json'), 'r') as fp:
            metas[split] = json.load(fp)[split]

    # print(metas["train"].keys())
    # dict_keys(['file_path', 'cam2world', 'width', 'height', 'focal', 'label', 'near', 'far', 'lossmult', 'pix2cam'])
    all_imgs = []
    all_poses = []
    counts = [0]
    H = []
    W = []
    focal = []
    lossmult = []

    for split in splits:
        print(split)
        meta = metas[split]
        imgs = []

        for i, fbase in enumerate(meta['file_path']):
            fname = os.path.join(basedir, fbase)
            image=imageio.imread(fname)
            if white_bkgd:
                image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])
            else:
                image = image[..., :3]
            imgs.append(image)
            all_poses.append(np.array(meta['cam2world'][i]))
            H.append(meta['height'][i])
            W.append(meta['width'][i])
            focal.append(meta['focal'][i])
            lossmult.append([meta['lossmult'][i]])

        imgs = (np.array(imgs, dtype=object) / 255.) # keep all 4 channels (RGBA)
        for i, img in enumerate(imgs):
            imgs[i] = img.astype(np.float32)
        # poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        # all_poses.append(poses)

    all_poses = np.array(all_poses).astype(np.float32)
    H = np.array(H)
    W = np.array(W)
    focal = np.array(focal)
    lossmult = np.array(lossmult)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    # all_poses=np.array(all_poses).astype(np.float32)
    imgs = np.concatenate(all_imgs, 0)
    # poses = np.concatenate(all_poses, 0)
    return imgs, all_poses, render_poses, [H, W, focal], i_split, lossmult

