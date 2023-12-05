#!/usr/bin/env python3

from typing import Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_gif(file_dir, gif_name=None):
    import imageio
    import os
    filenames = os.listdir(file_dir)
    filenames.sort(key=natural_keys)
    gif_name = gif_name if gif_name is not None else filenames[0]
    images = []
    for filename in filenames:
        jpg = os.path.join(file_dir, filename)
        if jpg.endswith('.jpg'):
            images.append(imageio.imread(jpg))
    imageio.mimsave('{}/{}.gif'.format(file_dir, gif_name), images)

def score_function_heat_map(
        score_function: Callable,
        version: int,
        t_eps: float,
        device,
        mu: float = 0.,
        sigma: float = 1.,
        limit: int = 99,
        make_gif: bool = False
):
    lwr = -5 * sigma + mu
    upr = 5 * sigma + mu
    xs = torch.linspace(lwr, upr, 100, device=device)
    ts = torch.linspace(t_eps, 1, 100, device=device)
    # If meshgrid is given n input tensor of shape s1, s2, \dots, sn, respectively
    # Then there are n corresponding output tensors, all of which have shape
    # s1 \times s2 \times \dots \times sn.
    # The ith input tensor determines the shape of the ith output dimension,
    # and the ith output tensor is the ith input tensor copied along all other
    # dimensions to have the output shape above.
    grid_t, grid_x = torch.meshgrid(ts, xs, indexing='ij')
    input_x = grid_x.reshape(-1, 1, 1)
    input_t = grid_t.reshape(-1)
    scores = score_function(x=input_x, time=input_t)

    fig, ax = plt.subplots()
    mesh_x = xs.cpu().numpy()
    mesh_t = ts.cpu().numpy()
    # reshape and flip scores so that increasing row index of mesh_scores corresponds
    # to decreasing time t. We need this flip for pcolormesh to visualize correctly.
    mesh_scores = scores.reshape(100, 100).flip(dims=(0,)).detach().cpu().numpy()
    visual_mesh_scores = mesh_scores[:limit]
    score_max = visual_mesh_scores.max()
    score_min = visual_mesh_scores.min()
    # score_max = np.max(np.abs([visual_mesh_scores.min(), visual_mesh_scores.max()]))
    # score_min = -score_max
    # score_max = mesh_scores.mean() + 3 * mesh_scores.std()
    # score_min = mesh_scores.mean() - 3 * mesh_scores.std()

    ax.pcolormesh(
        mesh_x,
        mesh_t[:limit],
        visual_mesh_scores,
        shading='auto',
        cmap='RdBu',
        vmin=score_min,
        vmax=score_max
    )
    fig.savefig('figs/heat_maps/nn_score_v{}.jpg'.format(version))

    if make_gif:
        create_gif('figs/heat_maps', 'training_scores')
