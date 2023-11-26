#!/usr/bin/env python3

from typing import Callable
import numpy as np
import torch
import matplotlib.pyplot as plt


def score_function_heat_map(
        score_function: Callable,
        version: int,
        t_eps: float,
        device,
        mu: float = 0.,
        sigma: float = 1.
):
    lwr = -5 * sigma + mu
    upr = 5 * sigma + mu
    xs = torch.linspace(lwr, upr, 100, device=device)
    ts = torch.linspace(t_eps, 1, 100, device=device)
    grid_t, grid_x = torch.meshgrid(ts, xs, indexing='ij')
    input_x = grid_x.reshape(-1, 1, 1)
    input_t = grid_t.reshape(-1)
    scores = score_function(x=input_x, time=input_t)

    fig, ax = plt.subplots()
    mesh_x = xs.cpu().numpy()
    mesh_t = ts.cpu().numpy()
    mesh_scores = scores.reshape(100, 100).detach().cpu().numpy()
    # score_max = np.max(np.abs([mesh_scores.min(), mesh_scores.max()]))
    # score_min = -score_max
    score_max = mesh_scores.mean() + 3 * mesh_scores.std()
    score_min = mesh_scores.mean() - 3 * mesh_scores.std()

    ax.pcolormesh(
        mesh_x,
        mesh_t,
        mesh_scores,
        cmap='RdBu',
        vmin=score_min,
        vmax=score_max
    )
    fig.savefig('figs/heat_maps/nn_score_v{}.jpg'.format(version))
