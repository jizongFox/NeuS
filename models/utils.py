from __future__ import annotations

import contextlib
import sys
import typing as t
from itertools import cycle

import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm


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


class DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        tqdm.write(x, file=self.file)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(save_stdout)
    yield
    sys.stdout = save_stdout
