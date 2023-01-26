from __future__ import annotations

import typing as t
from itertools import cycle

import matplotlib.pyplot as plt
from torch import Tensor


def visualize_rays(rays: t.Sequence[Tensor] | Tensor) -> None:
    markers = cycle(['o', '^'])
    if not isinstance(rays, t.Sequence):
        rays = [rays]
    rays = [x.clone() for x in rays]
    rays = [x.cpu().numpy() for x in rays]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for r, m in zip(rays, markers):
        ax.scatter(*r.transpose(-1, -2), marker=m)
    plt.show()
