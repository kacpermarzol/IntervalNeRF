import os, sys
import torch.distributed as dist

import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import torch.utils.tensorboard as tb

import matplotlib.pyplot as plt

from run_nerf_helpers_interval import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from torch.nn.parallel import DistributedDataParallel as DDP
from load_multiscale import load_multiscale_data

# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """

    if chunk is None:
        return fn

    def ret(inputs, epsilon):
        mu_list = []
        eps_list = []
        for i in range(0, inputs.shape[0], chunk):
            mu_, eps_ = fn(inputs[i:i + chunk], epsilon[i : i +chunk])
            mu_list.append(mu_)
            eps_list.append(eps_)
        mu = torch.cat(mu_list, 0)
        eps = torch.cat(eps_list, 0)
        return mu, eps

    return ret


def run_network(inputs, viewdirs, fn, eps, embed_fn, embeddirs_fn, netchunk=1024 * 32):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)


    outputs_flat = batchify(fn, netchunk)(embedded, eps)
    mu_flat = outputs_flat[0]
    eps_flat = outputs_flat[1]
    mu = torch.reshape(mu_flat, list(inputs.shape[:-1]) + [mu_flat.shape[-1]])
    eps = torch.reshape(eps_flat, list(inputs.shape[:-1]) + [eps_flat.shape[-1]])

    return mu, eps


def batchify_rays(rays_flat, eps, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], epsilon=eps, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, eps, chunk=1024 * 32, rays=None, radii=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        distances = torch.norm(viewdirs, p=2, dim=-1, keepdim=True)

        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / distances
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()



    # if H_train is not None:
    #     pixel = 1 / H_train
    #     if np.shape(H_train) == ():
    #         pixel = torch.ones_like(distances) * pixel



    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)



    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        # if c2w is not None:
        #     rays = torch.cat([rays, viewdirs], -1)
        # else:
        rays = torch.cat([rays, viewdirs, torch.tensor(distances), torch.tensor(radii)], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, eps, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'rgb_map_left', 'rgb_map_right']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, eps, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, rgb_map_left, rgb_map_right, extras = render(H, W, K, eps=eps, chunk=chunk, H_train=H, c2w=c2w[:3, :4],
                                                                     **render_kwargs)

        rgb = rgb.view(H, W, 3)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # port = np.random.randint(12355, 12399)
    # os.environ['MASTER_PORT'] = '{}'.format(port)
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def create_nerf(args, gpu, rank):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    model = model.to(gpu)
    ddp_model = DDP(model, device_ids=[gpu])
    model= ddp_model.module
    grad_vars = list(ddp_model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        model_fine.to(gpu)
        ddp_fine_model = DDP(model_fine, device_ids=[gpu])
        model_fine = ddp_fine_model.module
        grad_vars += list(ddp_fine_model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn, eps: run_network(inputs, viewdirs, network_fn, eps,
                                                                             embed_fn=embed_fn,
                                                                             embeddirs_fn=embeddirs_fn,
                                                                             netchunk=args.netchunk)  ##Aded epsilon

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr_init, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

        # dist.barrier()
        # # configure map_location properly
        # ddp_model.load_state_dict(
        #     torch.load(CHECKPOINT_PATH, map_location=map_location))

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=map_location)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': ddp_fine_model,
        'N_samples': args.N_samples,
        'network_fn': ddp_model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    t1e10 = torch.Tensor([1e10]).to(dists.device)
    dists = torch.cat([dists, t1e10.expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    t1 = torch.ones((alpha.shape[0], 1)).to(alpha.device)
    weights = alpha * torch.cumprod(torch.cat([t1, 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    t1_2 = torch.ones_like(depth_map).to(alpha.device)
    disp_map = 1. / torch.max(1e-10 * t1_2, depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def raw2outputs_eps(raw_left, raw_right, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw_left: [num_rays, num_samples along ray, 4] left end of the calculated interval
        raw_right: [num_rays, num_samples along ray, 4] right end of the calculated interval
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map_left: [num_rays, 3]. Left end of the interval for estimated RGB color of a ray
        rgb_map_right: ^^^
        disp_map_left: [num_rays]. Left end of the inrercal for disparity map. Inverse of depth map
        disp_map_right: ^^^
        acc_map_left: [num_rays]. Left end of the interval for the sum of weights along each ray.
        acc_map_right: ^^^
        weights_left: [num_rays, num_samples]. Left end of the interval for the weights assigned to each sampled color.
        weights_right: ^^^
        depth_map_left: [num_rays]. Left end of the interval for estimated distance to object.
        depth_map_tight: ^^^
    """

    raw2alpha = lambda raw, dists, act_fn=F.softplus: 1. - torch.exp(
        -act_fn(raw) * dists)  # the result of this function is always non-negative

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # the distances between adjacent samples
    # dists = torch.cat([dists, torch.full(dists[..., :1].shape, 1e10)], -1)  # changed second arg in torch cat
    t1e10 = torch.Tensor([1e10]).to(dists.device)
    dists = torch.cat([dists, t1e10.expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # sigmoid is a monotonic function of one variable, so calculating rgb left and right is as follows:
    # (https://en.wikipedia.org/wiki/Interval_arithmetic     - section "Elementary functions")
    rgb_left = torch.sigmoid(raw_left[..., :3]) + 1e-20  # [N_rays, N_samples, 3]
    rgb_right = torch.sigmoid(raw_right[..., :3]) + 1e-20  # [N_rays, N_samples, 3]

    noise = 0.
    # if raw_noise_std > 0.:
    #     noise = torch.randn(raw_left[..., 3].shape) * raw_noise_std
    #
    #     # Overwrite randomly sampled data if pytest
    #     if pytest:
    #         np.random.seed(0)
    #         noise = np.random.rand(*list(raw_left[..., 3].shape)) * raw_noise_std
    #         noise = torch.Tensor(noise)

    # because dist is not an interval
    alpha_left = raw2alpha(raw_left[..., 3] + noise, dists)   # [N_rays, N_samples]
    alpha_right = raw2alpha(raw_right[..., 3] + noise, dists)   # [N_rays, N_samples]

    # assume sigma is denoted with s, delta with d, alpha with a

    # w_i = T_i * a_i
    # T_i = exp(- sum_{j=1}^{i-1}(s_j * d_j)  )
    # 1-a_i = 1 - 1 + exp(- s_i * d_i) = exp(- s_i * d_i)

    # T_i = exp(- sum_{j=1}^{i-1}(s_j * d_j)) = exp(-(s_1*d_1 + s_2*d_2 * ... * s_{i-1} * d_{i-1})) =
    # = exp(-s_1*d_1) * exp(-s_2*d_2) * ... * exp(-s_{i-1} * d_{i-1})) =
    # = (1-a_1) * (1-a_2) * ... * (1-a_{i-1})

    t1 = torch.ones((alpha_left.shape[0], 1)).to(dists.device)
    t1_2 = torch.ones((alpha_right.shape[0], 1)).to(dists.device)
    T_left = torch.cumprod(torch.cat([t1, 1. - alpha_left + 1e-10], -1), -1)[:, :-1]
    T_right = torch.cumprod(torch.cat([t1_2, 1. - alpha_right + 1e-10], -1), -1)[:,:-1]

    assert assert_all_nonnegative(alpha_left), "Not all elements are nonnegative (alpha_left)"
    assert assert_all_nonnegative(alpha_right), "Not all elements are nonnegative (alpha_right)"
    assert assert_all_nonnegative(T_left), "Not all elements are nonnegative (T_left)"
    assert assert_all_nonnegative(T_right), f"Not all elements are nonnegative (T_right)"

    # After we have asserted that all of the elements in tensors are non-negative, we can use the simplified
    # version of interval multiplication, i.e [a,b] * [x,y] == [a*x, b*y]

    weights_left = alpha_left * T_left + 1e-20
    weights_right = alpha_right * T_right + 1e-20

    assert assert_all_nonnegative(weights_left), "Not all elements are nonnegative (weights_left)"
    assert assert_all_nonnegative(weights_right), "Not all elements are nonnegative (weights_right)"
    assert assert_all_nonnegative(rgb_left), "Not all elements are nonnegative (rgb_left)"
    assert assert_all_nonnegative(rgb_right), "Not all elements are nonnegative (rgb_right)"

    # Again we've checked that all elements of the tensors needed for calculating the rgb map are non-negative, hence we
    # use simplified multiplication formula

    rgb_map_left = torch.sum(weights_left[..., None] * rgb_left, -2)  # [N_rays, 3]
    rgb_map_right = torch.sum(weights_right[..., None] * rgb_right, -2)  # [N_rays, 3]

    # depth_map_left = torch.sum(weights_left * z_vals, -1)
    # depth_map_right = torch.sum(weights_right * z_vals, -1)
    #
    # disp_map_left = 1. / torch.max(1e-10 * torch.ones_like(depth_map_left), depth_map_left / torch.sum(weights_left, -1))
    # disp_map_right = 1. / torch.max(1e-10 * torch.ones_like(depth_map_left), depth_map_right / torch.sum(weights_right, -1))
    #
    acc_map_left = torch.sum(weights_left, -1)
    acc_map_right = torch.sum(weights_right, -1)

    if white_bkgd:
        rgb_map_left = rgb_map_left + (1. - acc_map_left[..., None])
        rgb_map_right = rgb_map_right + (1. - acc_map_right[..., None])

    return rgb_map_left, rgb_map_right
    # , disp_map_left, disp_map_right, acc_map_left, acc_map_right, weights_left, weights_right, depth_map_left, depth_map_right


def render_rays(ray_batch,
                epsilon,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, 8 : 11] if ray_batch.shape[-1] > 8 else None
    distances = ray_batch[:, -2] if ray_batch.shape[-1] > 10 else None
    radii = ray_batch[:, -1] if ray_batch.shape[-1]>10 else None

    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(near.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(near.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    if distances is not None:
        # distances1 = distances.unsqueeze(1).repeat(1, N_samples)
        # rays_origins = rays_o.unsqueeze(1).repeat(1, N_samples, 1)

        distances1 = distances.unsqueeze(1).unsqueeze(2)
        rays_origins = rays_o.unsqueeze(1).broadcast_to(N_rays, N_samples, 3)
        radii_ = radii.unsqueeze(1).unsqueeze(2)

        distances2 = torch.norm(pts - rays_origins, p=2, dim=-1).unsqueeze(2)
        # radii_ = radii.unsqueeze(1).repeat(1, N_samples)

        pre_eps = ((radii_ * distances2) / distances1)
        eps = (epsilon * pre_eps).to(torch.float32)
        eps = eps.reshape(eps.shape[0] * eps.shape[1], 1)

        # eps = (pre_eps * epsilon).to(torch.float32)
        # eps = eps.unsqueeze(-1).expand(N_rays, N_samples, 3)

    else:
        print("Oh no")



    raw_mu, raw_eps = network_query_fn(pts, viewdirs, network_fn, eps)

    # raw_eps[:, :, -1] = 0
    raw_left, raw_right = raw_mu - raw_eps, raw_mu + raw_eps

    # get outputs from the basic nerf
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_mu, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest)

    # get left and right ends of interval for rgb map
    rgb_map_left, rgb_map_right = raw2outputs_eps(raw_left,
                                                  raw_right,
                                                  z_vals,
                                                  rays_d, raw_noise_std, white_bkgd,
                                                  pytest=pytest)

    # rgb_map = rgb_map.to('cpu')
    # disp_map = disp_map.to('cpu')
    # acc_map = acc_map.to('cpu')
    # rgb_map_left = rgb_map_left.to('cpu')
    # rgb_map_right = rgb_map_right.to('cpu')

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, rgb_map_left_0, rgb_map_right_0 = rgb_map, disp_map, acc_map, rgb_map_left, rgb_map_right

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        del weights
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        if distances is not None:
            distances1 = distances.unsqueeze(1).unsqueeze(2)
            rays_origins = rays_o.unsqueeze(1).broadcast_to(N_rays, N_importance + N_samples, 3)
            radii_ = radii.unsqueeze(1).unsqueeze(2)
            distances2 = torch.norm(pts - rays_origins, p=2, dim=-1).unsqueeze(2)

            pre_eps = (radii_ * distances2) / distances1
            eps = (epsilon * pre_eps).to(torch.float32)
            eps = eps.reshape(eps.shape[0] * eps.shape[1], 1)

        else:
            print("oh no")


        run_fn = network_fn if network_fine is None else network_fine

        raw_mu, raw_eps = network_query_fn(pts, viewdirs, run_fn, eps)

        raw_left, raw_right = raw_mu - raw_eps, raw_mu + raw_eps

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_mu,
                                                                     z_vals,
                                                                     rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

        del weights

        rgb_map_left, rgb_map_right = raw2outputs_eps(raw_left,
                                                      raw_right,
                                                      z_vals,
                                                      rays_d, raw_noise_std, white_bkgd,
                                                      pytest=pytest)


    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'rgb_map_left': rgb_map_left,
           'rgb_map_right': rgb_map_right}
    if retraw:
        ret['raw'] = raw_mu

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['rgb_map_left0'] = rgb_map_left_0
        ret['rgb_map_right0'] = rgb_map_right_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    # parser.add_argument("--lrate", type=float, default=5e-4,
    #                     help='learning rate')
    # parser.add_argument("--lrate_decay", type=int, default=500,  ## to make lrate go from 5e-4 to 5e-6 as in papers
    #                     help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 16,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 32,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options f
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # distributed training options
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default="", type=str,
                        help='number of gpus of each node')
    parser.add_argument('-i', '--id', default=0, type=int,
                        help='the id of the node which is determined by the correponding index in the gpu list')
    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1',
                        help='number of processes')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=1000000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=5000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=1000000,  ###!!!
                        help='frequency of render_poses video saving')

    parser.add_argument("--eps", type=float, default=0.0,
                        help=' todo ')

    parser.add_argument("--save_every", type=int, default=1000, help="The number of steps to run eval on testset")
    parser.add_argument("--metrics_only", type=bool, default=False)
    parser.add_argument("--log_every", type=int, default=10, help="The number of steps to log into tensorboard")


    parser.add_argument("--lr_init", type=float, default=5e-4)
    parser.add_argument("--lr_final", type=float, default=5e-6)
    parser.add_argument("--lr_delay_steps", type=int, default=2500)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    return parser

def ddp_train_nerf(gpu, args):
    ###### set up multi-processing
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
    rank = sum(gpu_list[:args.id]) + gpu
    dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    ###### set up logger
    logger = None
    if rank == 0:
        logger = tb.SummaryWriter(os.path.join(args.basedir, 'summaries', args.expname))
        basedir = args.basedir
        expname = args.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

    ###### decide chunk size according to gpu memory
    #logger.info('gpu_mem: {}'.format(torch.cuda.get_device_properties(rank).total_memory))
    # if torch.cuda.get_device_properties(gpu).total_memory / 1e9 > 14:
    #     #logger.info('setting batch size according to 24G gpu')
    #     args.N_rand = 1024
    #     args.chunk_size = 8192
    # else:
    #     #logger.info('setting batch size according to 12G gpu')
    #     args.N_rand = 512
    #     args.chunk_size = 4096

    # Load data
    K = None

    if args.dataset_type == 'multiscale':
        print("multiscale")
        images, poses, render_poses, hwf, i_split, lossmult = load_multiscale_data(args.datadir, args.white_bkgd)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

    elif args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                    args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # for generating videos with different resoultions
    H, W, focal = hwf
    H_test, W_test, focal_test = H[0], W[0], focal[0]
    hwf_800 = (H_test, W_test, focal_test)
    K_800 = np.array([
        [focal_test, 0, 0.5 * W_test],
        [0, focal_test, 0.5 * H_test],
        [0, 0, 1]])

    H_test, W_test, focal_test = H[1], W[1], focal[1]
    hwf_400 = (H_test, W_test, focal_test)
    K_400 = np.array([
        [focal_test, 0, 0.5 * W_test],
        [0, focal_test, 0.5 * H_test],
        [0, 0, 1]])

    H_test, W_test, focal_test = H[2], W[2], focal[2]
    hwf_200 = (H_test, W_test, focal_test)
    K_200 = np.array([
        [focal_test, 0, 0.5 * W_test],
        [0, focal_test, 0.5 * H_test],
        [0, 0, 1]])

    H_test, W_test, focal_test = H[3], W[3], focal[3]
    hwf_100 = (H_test, W_test, focal_test)
    K_100 = np.array([
        [focal_test, 0, 0.5 * W_test],
        [0, focal_test, 0.5 * H_test],
        [0, 0, 1]])

    # if args.render_test:
    #     render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, gpu, rank)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    # render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')

        rays = [get_rays_np(H[i], W[i], np.array([
            [focal[i], 0, 0.5 * W[i]],
            [0, focal[i], 0.5 * H[i]],
            [0, 0, 1]]), poses[i, :3, :4]) for i in i_train]  # [N, ro+rd, H, W, 3]

        rays_test = [get_rays_np(H[i], W[i], np.array([
            [focal[i], 0, 0.5 * W[i]],
            [0, focal[i], 0.5 * H[i]],
            [0, 0, 1]]), poses[i, :3, :4]) for i in i_test]  # [N, ro+rd, H, W, 3]

        lossmult2 = np.concatenate(
            [np.array(np.full((H[i] * W[i], 1), lossmult[i])) for i in i_train]).flatten().reshape(-1, 1)

        # H_train = np.concatenate(
        #     [np.array(np.full((H[i] * W[i], 1), H[i])) for i in i_train]).flatten().reshape(-1, 1)
        #
        # H_test = np.concatenate(
        #     [np.array(np.full((H[i] * W[i], 1), H[i])) for i in i_test]).flatten().reshape(-1, 1)


        dirs = np.array(rays, dtype=object)
        dirs = dirs[:, 1]
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in dirs
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii_train = [v[..., None] * 2 / np.sqrt(12) for v in dx]
        radii_train = (np.concatenate([subarray.flatten() for subarray in radii_train])).reshape(-1,1)


        dirs = np.array(rays_test, dtype=object)
        dirs = dirs[:, 1]
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in dirs
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii_test = [v[..., None] * 2 / np.sqrt(12) for v in dx]
        radii_test = (np.concatenate([subarray.flatten() for subarray in radii_test])).reshape(-1,1)



        print('done, concats')

        rays_rgb = []
        rays_rgb_test = []
        for i, ray in enumerate(rays):
            img = images[i_train[i]]
            concatenated_array = np.concatenate((ray, img[None, :]))
            rays_rgb.append(concatenated_array)

        for i, ray in enumerate(rays_test):
            img = images[i_test[i]]
            concatenated_array = np.concatenate((ray, img[None, :]))
            rays_rgb_test.append(concatenated_array)

        for i, ray in enumerate(rays_rgb):
            rays_rgb[i] = np.transpose(ray, [1, 2, 0, 3])

        for i, ray in enumerate(rays_rgb_test):
            rays_rgb_test[i] = np.transpose(ray, [1, 2, 0, 3])

        for i, ray in enumerate(rays_rgb):
            rays_rgb[i] = ray.reshape(-1, 3, 3)

        for i, ray in enumerate(rays_rgb_test):
            rays_rgb_test[i] = ray.reshape(-1, 3, 3)

        rays_rgb = np.concatenate(rays_rgb, 0)
        # rays_rgb = np.array(rays_rgb, dtype=np.float32)

        rays_rgb_test = np.concatenate(rays_rgb_test, 0)
        # rays_rgb_test = np.array(rays_rgb_test, dtype=np.float32)
        print('shuffle rays')


        rand_idx = np.random.permutation(rays_rgb.shape[0])

        rays_rgb = rays_rgb[rand_idx]
        lossmult2 = lossmult2[rand_idx]
        radii_train = radii_train[rand_idx]
        # H_train = H_train[rand_idx]

        rand_idx_test = np.random.permutation(
            5000000)  # assuming 1 milion iterations with batch size = 4096, this is enough
        rays_rgb_test = rays_rgb_test[rand_idx_test]
        radii_test = radii_test[rand_idx_test]
        # H_test = H_test[rand_idx_test]

        i_batch = 0
        i_batch_test = 0


    # Move training data to GPU
    # if use_batching:
    poses = torch.from_numpy(poses)
    # images = torch.Tensor(images).to(device)

    if use_batching:
        # H_test = torch.from_numpy(H_test)
        # H_train = torch.from_numpy(H_train)
        rays_rgb_test = torch.from_numpy(rays_rgb_test)
        rays_rgb = torch.from_numpy(rays_rgb)
        lossmult2 = torch.from_numpy(lossmult2)
        radii_train = torch.from_numpy(radii_train)
        radii_test = torch.from_numpy(radii_test)


    # Short circuit if only calculating metrics
    if args.metrics_only:
        psnr800 = []
        psnr800eps0 = []
        psnr400 = []
        psnr400eps0 = []
        psnr200 = []
        psnr200eps0 = []
        psnr100 = []
        psnr100eps0 = []

        print('METRICS ONLY')


        with torch.no_grad():
            tensor10 = torch.log(torch.Tensor([10.])).to(gpu)

            for it in trange(200):
                i = i_test[it]
                hh, ww, ff = H[i], W[i], focal[i]
                # HH = (torch.ones(hh ** 2, 1) * hh).to(gpu)

                kk = np.array([[ff, 0, 0.5 * ww],
                               [0, ff, 0.5 * hh],
                               [0, 0, 1]])
                pose = poses[i, :3, :4]


                rays_o, rays_d = get_rays(hh, ww, kk, pose)


                dirs = np.array(rays_d)

                # dirs = dirs[:, 1]
                dx = np.sqrt(np.sum((dirs[:-1, :, :] - dirs[1:, :, :]) ** 2, -1))
                dx = np.concatenate([dx, dx[-2:-1, :]], axis=0)

                radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
                radii = (np.concatenate([subarray.flatten() for subarray in radii])).reshape(-1, 1)

                radii = torch.from_numpy(radii).to(gpu)
                rays_o, rays_d = rays_o.to(gpu), rays_d.to(gpu)
                rays = (rays_o, rays_d)


                rgb, _, _, _, _, _ = render(hh, ww, kk, eps=args.eps, rays=rays, radii = radii , chunk=args.chunk, **render_kwargs_test)
                rgb = rgb.view(hh, ww, 3).to('cpu')

                loss = img2mse(rgb, images[i])
                psnr = -10. * torch.log(loss) / tensor10

                # psnr = mse2psnr(img2mse(rgb, images[i]))

                # rgb, _, _, _, _, _ = render(hh, ww, kk, eps=0, rays=rays, radii=radii, chunk=args.chunk,
                #                             **render_kwargs_test)
                # rgb = rgb.view(hh, ww, 3).to('cpu')
                #
                # loss = img2mse(rgb, images[i])
                # psnr0 = -10. * torch.log(loss) / tensor10

                if lossmult[i] == 1:
                    psnr800.append(psnr)
                    # psnr800eps0.append(psnr0)
                elif lossmult[i] == 4:
                    psnr400.append(psnr)
                    # psnr400eps0.append(psnr0)
                elif lossmult[i] == 16:
                    psnr200.append(psnr)
                    # psnr200eps0.append(psnr0)
                elif lossmult[i] == 64:
                    psnr100.append(psnr)
                    # psnr100eps0.append(psnr0)
                else:
                    print("something went wrong")

            print("_-_-_-_-_-_-_-_")
            print("PSNR")
            print("_-_-_-_-_-_-_-_")

            print(f'EPS {args.eps}')
            print(f'Full res {torch.mean(torch.tensor(psnr800))}')
            print(f'1/2 res {torch.mean(torch.tensor(psnr400))}')
            print(f'1/4 res {torch.mean(torch.tensor(psnr200))}')
            print(f'1/8 res {torch.mean(torch.tensor(psnr100))}')

            # print('\n _-_-_-_-_-_-_-_ \n')
            # print("EPS 0")
            # print(f'Full res {torch.mean(torch.tensor(psnr800eps0))}')
            # print(f'1/2 res {torch.mean(torch.tensor(psnr400eps0))}')
            # print(f'1/4 res {torch.mean(torch.tensor(psnr200eps0))}')
            # print(f'1/8 res {torch.mean(torch.tensor(psnr100eps0))}')

            return

    N_iters = 500001 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # make sure different processes sample different rays
    np.random.seed((gpu + 1) * 777)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((gpu + 1) * 777)

    epsilon = args.eps
    start = start + 1
    new_lrate = args.lr_init
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            # use to partition data

            batch = rays_rgb[i_batch:i_batch + N_rand].to(gpu)  # [B, 2+1, 3*?]
            mask = lossmult2[i_batch:i_batch + N_rand].to(gpu)
            # HH = H_train[i_batch: i_batch + N_rand].to(gpu)
            radii = radii_train[i_batch : i_batch + N_rand].to(gpu)
            # for making the loss like in MipNeRF:
            # "loss of each pixel by the area
            # of that pixel’s footprint in the original image (the loss for pixels f12rom the 1/4 images is scaled by 16, etc)
            # so that the few low-resolution pixels have comparable influence to the many high-resolution pixels. "
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = np.random.permutation(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                lossmult2 = lossmult2[rand_idx]
                radii_train = radii_train[rand_idx]
                # H_train = H_train[rand_idx]
                i_batch = 0

            partitions = list(range(0, batch_rays.shape[1], int(batch_rays.shape[1] / (args.world_size))))
            partitions.append(batch_rays.shape[1])

            batch_rays_ddp = batch_rays[:, partitions[rank]: partitions[rank + 1]]
            # HH_ddp = HH[partitions[rank]: partitions[rank + 1]]
            mask_ddp = mask[partitions[rank]: partitions[rank + 1]]
            target_s_ddp = target_s[partitions[rank]: partitions[rank + 1]]

        else:
            print("not implemented")

        #####  Core optimization loop  #####

        # kappa is a hyperparameter that governs the relative weight of satisfying the interval loss versus fit loss
        # with warmup
        #
        # if i < 200000:

        #     eps = 0
        # elif i < 250000:
        #     eps = ((i - 199999) / 50000) * epsilon
        # else:
        #     eps = epsilon
        #
        # if i < 200000:
        #     kappa = 1
        # elif i < 300000:
        #     kappa = max(1 - 0.000005 * (i - 199999), 0.5)
        # else:
        #     kappa = 0.5

        # if i < 100000:
        #     eps = 0
        # elif i < 250000:
        #     eps = ((i - 199999) / 50000) * epsilon
        # else:
        #     eps = epsilon
        #
        # if i < 200000:
        #     kappa = 1
        # elif i < 300000:
        #     kappa = max(1 - 0.000005 * (i - 199999), 0.5)
        # else:
        #     kappa = 0.5

        if i < 100000:
            eps = 0
        elif i < 150000:
            eps = ((i - 99999) / 50000) * epsilon
        else:
            eps = epsilon

        if i < 100000:
            kappa = 1
        elif i < 200000:
            kappa = max(1 - 0.000005 * (i - 99999), 0.5)
        else:
            kappa = 0.5

        # without warmup
        #
        # if i < 50000:
        #     eps = (i / 50000) * epsilon

        #batch4096
        # if i < 60000:
        #     eps = 0
        # elif i < 160000:
        #     eps = ((i - 59999) / 100000) * epsilon
# >>>>>>> ddp
        # else:
        #     eps = epsilon
        #
        # if i < 60000:
        #     kappa = 1
        # elif i < 160000:
        #     kappa = max(1 - 0.000005 * (i - 59999), 0.5)
        # else:
        #     kappa = 0.5

        # without warmup

        # if i < 50000:
        #     eps = (i / 50000) * epsilon
        # else:
        #     eps = epsilon
        #
        # if i < 100000:
        #     kappa = max(1 - 0.000005 * i, 0.5)
        # else:
        #     kappa = 0.5

        rgb, disp, acc, rgb_map_left, rgb_map_right, extras = render(1, 1, 1, eps, chunk=args.chunk, rays=batch_rays_ddp,
                                                                     verbose=i < 10, retraw=True, radii=radii,
                                                                     **render_kwargs_train)
        # rgb = rgb.to('cpu')
        # rgb_map_left, rgb_map_right = rgb_map_left.to('cpu'), rgb_map_right.to('cpu')

        if i % 5000 == 0:
            print("target : ", target_s_ddp[0:3])
            print("rgb : ", rgb[0:3])
            print("rgb_left : ", rgb_map_left[0:3])
            print("rgb_right : ", rgb_map_right[0:3], '\n')

        # loss_fit = img2mse(rgb, target_s)
        loss_fit = img2mse2(rgb, target_s_ddp, mask_ddp)
        # loss_spec = interval_loss(target_s, rgb_map_left, rgb_map_right)
        loss_spec = interval_loss2(target_s_ddp, rgb_map_left, rgb_map_right, mask_ddp)


        # psnr = mse2psnr(loss_fit)

        tensor10 = torch.log(torch.Tensor([10.])).to(gpu)
        psnr = -10. * torch.log(loss_fit) / tensor10


        # if logger is not None:
        #     logger.add_scalar('train/fine_psnr', psnr, global_step=i)

        if 'rgb0' in extras:
            # img_loss0 = img2mse(extras['rgb0'], target_s)
            img_loss0 = img2mse2(extras['rgb0'], target_s_ddp, mask_ddp)
            # loss_spec0 = interval_loss(target_s, extras['rgb_map_left0'], extras['rgb_map_right0'])
        loss_spec0 = interval_loss2(target_s_ddp, extras['rgb_map_left0'], extras['rgb_map_right0'], mask_ddp)
        psnr0 = -10. * torch.log(img_loss0) / tensor10
        loss_fit_final = loss_fit + img_loss0
        loss_spec_final = loss_spec + loss_spec0

        loss = kappa * loss_fit_final + (1 - kappa) * loss_spec_final
        logpsnr = (psnr.item() + psnr0.item()) / 2

        loss.backward()
        optimizer.step()

        if logger is not None and i % args.log_every == 0:
            logger.add_scalar('losses/loss_fit', loss_fit.item(), global_step=i)
            logger.add_scalar('losses/loss_spec', loss_spec.item(), global_step=i)
            logger.add_scalar('train/fine_psnr', psnr, global_step=i)
            logger.add_scalar('losses/loss_fit0', img_loss0.item(), global_step=i)
            logger.add_scalar('losses/loss_spec0', loss_spec0.item(), global_step=i)
            logger.add_scalar('train/coarse_psnr', psnr0, global_step=i)
            logger.add_scalar('train/loss', float(loss.detach().cpu().numpy()), global_step=i)
            logger.add_scalar('train/avg_psnr', logpsnr, global_step=i)
            logger.add_scalar('train/lr', new_lrate, global_step=i)


        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= args.world_size
        optimizer.zero_grad()

        # if logger is not None:
        #     logger.add_scalar('train/loss', float(loss.detach().cpu().numpy()), global_step=i)
        #     logger.add_scalar('train/avg_psnr', logpsnr, global_step=i)
        #     logger.add_scalar('train/lr', new_lrate, global_step=i)


        # NOTE: IMPORTANT!
        ###   update learning rate   ###


        step = i
        if args.lr_delay_steps > 0:
            delay_rate = args.lr_delay_mult + (1 - args.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step/ args.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / N_iters, 0, 1)
        log_lerp = np.exp(np.log(args.lr_init) * (1 - t) + np.log(args.lr_final) * t)
        new_lrate = delay_rate * log_lerp
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # decay_rate = 0.1
        # decay_steps = args.lrate_decay * 1000
        # if i < 5000:
        #     new_lrate = 1e-4 * np.exp((np.log(5) - np.log(1)) / 5000 * global_step)
        # else:
        #     new_lrate = 5e-4 * (decay_rate ** ((global_step - 5000) / decay_steps))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lrate
        ################################

        dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        if i % args.save_every == 0 and rank == 0:
            batch_test = rays_rgb_test[i_batch_test:i_batch_test + N_rand].to(gpu)  # [B, 2+1, 3*?]
            # HH = H_test[i_batch_test: i_batch_test + N_rand].to(gpu)
            radii_t = radii_test[i_batch_test : i_batch_test+ N_rand].to(gpu)
            i_batch_test += N_rand
            batch_test = torch.transpose(batch_test, 0, 1)
            batch_rays_test, target_s_test = batch_test[:2], batch_test[2]
            with torch.no_grad():
                rgb, _, _, _, _, extras = render(1, 1, 1, eps, chunk=args.chunk, rays=batch_rays_test,
                                                 verbose=i < 10, retraw=True, radii=radii_t, **render_kwargs_test)

                loss_fit = img2mse(rgb, target_s_test)
                # psnr = mse2psnr(loss_fit)
                psnr = -10. * torch.log(loss_fit) / tensor10

                logger.add_scalar('eval/fine_psnr', psnr, global_step=i)

                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s_test)
                    # psnr0 = mse2psnr(img_loss0)
                    psnr0 = -10. * torch.log(img_loss0) / tensor10
                    logger.add_scalar('eval/coarse_psnr', psnr0, global_step=i)

                    avg_psnr = (psnr + psnr0) / 2
                    logger.add_scalar('eval/avg_psnr', avg_psnr, global_step=i)

                rgb, _, _, _, _, extras = render(1, 1, 1, 0, chunk=args.chunk, rays=batch_rays_test,
                                                 verbose=i < 10, retraw=True, radii=radii_t, **render_kwargs_test)

                loss_fit = img2mse(rgb, target_s_test)
                # psnr = mse2psnr(loss_fit)
                psnr = -10. * torch.log(loss_fit) / tensor10

                logger.add_scalar('eval/fine_psnr_eps0', psnr, global_step=i)

                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s_test)
                    # psnr0 = mse2psnr(img_loss0)
                    psnr0 = -10. * torch.log(img_loss0) / tensor10
                    logger.add_scalar('eval/coarse_psnr_eps0', psnr0, global_step=i)

                    avg_psnr = (psnr + psnr0) / 2
                    logger.add_scalar('eval/avg_psnr_eps0', avg_psnr, global_step=i)

        # Rest is logging
        if i % args.i_weights == 0 and rank == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            network_fn_save = render_kwargs_train['network_fn'].module
            network_fine_save = render_kwargs_train['network_fine'].module
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': network_fn_save.state_dict(),
                'network_fine_state_dict': network_fine_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0 and rank == 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs800, disps800 = render_path(render_poses, hwf_800, K_800, args.chunk, render_kwargs_test, epsilon)
                print('Done, saving', rgbs800.shape, disps800.shape)
                rgbs400, disps400 = render_path(render_poses, hwf_400, K_400, args.chunk, render_kwargs_test, epsilon)
                print('Done, saving', rgbs400.shape, disps400.shape)
                rgbs200, disps200 = render_path(render_poses, hwf_200, K_200, args.chunk, render_kwargs_test, epsilon)
                print('Done, saving', rgbs200.shape, disps200.shape)
                rgbs100, disps100 = render_path(render_poses, hwf_100, K_100, args.chunk, render_kwargs_test, epsilon)
                print('Done, saving', rgbs100.shape, disps100.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb800.mp4', to8b(rgbs800), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'rgb400.mp4', to8b(rgbs400), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'rgb200.mp4', to8b(rgbs200), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'rgb100.mp4', to8b(rgbs100), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # if i % args.i_testset == 0 and i > 0:
        #     testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        #     os.makedirs(testsavedir, exist_ok=True)
        #     print('test poses shape', poses[i_test].shape)
        #     with torch.no_grad():
        #         render_path(torch.Tensor(poses[i_test]).to(device), hwf_test, K_test, args.chunk, render_kwargs_test, epsilon,
        #                     gt_imgs=images[i_test], savedir=testsavedir)
        #     print('Saved test set')

        if i % args.i_img == 0 and rank == 0:
            img_i = i_val[45]
            hh, ww, ff = H[img_i], W[img_i], focal[img_i]
            kk = np.array([[ff, 0, 0.5 * ww],
                           [0, ff, 0.5 * hh],
                           [0, 0, 1]])

            # HH = (torch.ones(hh ** 2, 1) * hh).to(gpu)

            pose = poses[img_i, :3, :4]
            rays_o, rays_d = get_rays(hh, ww, kk, pose)

            dirs = np.array(rays_d)

            # dirs = dirs[:, 1]
            dx = np.sqrt(np.sum((dirs[:-1, :, :] - dirs[1:, :, :]) ** 2, -1))
            dx = np.concatenate([dx, dx[-2:-1, :]], axis=0)

            # Cut the distance in half, and then round it out so that it's
            # halfway between inscribed by / circumscribed about the pixel.
            radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]
            radii = (np.concatenate([subarray.flatten() for subarray in radii])).reshape(-1, 1)

            radii = torch.from_numpy(radii).to(gpu)
            rays_o, rays_d =rays_o.to(gpu), rays_d.to(gpu)
            rays = (rays_o, rays_d)

            with torch.no_grad():
                rgb, _, _, _, _, _ = \
                    render(hh, ww, kk, eps=eps, chunk=args.chunk, rays=rays, radii=radii,
                           **render_kwargs_test)

                rgb = rgb.view(hh, ww, 3)
                logger.add_image('image', rgb, dataformats='HWC', global_step=i)

                # rgb, _, _, _, _, _ = render(hh, ww, kk, eps=0.0, chunk=args.chunk, c2w=pose, H_train = HH,
                #                             **render_kwargs_test)
                # rgb = rgb.view(hh, ww, 3)
                #
                # logger.add_image('image_eps0', rgb, dataformats='HWC', global_step=i)

        # if i % args.i_print == 0:
                # rgb, _, _, _, _, _ = render(hh, ww, kk, eps=0.0, chunk=args.chunk, c2w=pose, H_train=HH,
                #                             **render_kwargs_test)
                # rgb = rgb.view(hh, ww, 3)
                #
                # #logger.add_image('image_eps0', rgb, dataformats='HWC', global_step=i)

        if i % args.i_print == 0 and rank == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {logpsnr}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1
    logger.close()


def train():
    parser = config_parser()
    args = parser.parse_args()
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
    args.world_size = sum(gpu_list)
    print('world size')
    print(args.world_size)
    print("This code is running. Check your master IP address if you see nothing after a while.")
    # This is the master node's IP
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29610"
        #logger.log('Using # gpus: {}'.format(args.world_size))

    torch.multiprocessing.spawn(ddp_train_nerf,
                                args=(args,),
                                nprocs=gpu_list[args.id],
                                )


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
