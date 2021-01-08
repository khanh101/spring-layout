from typing import Iterator

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def draw(data_iter: Iterator[tuple[
    np.ndarray,
    list[list[int]],
    tuple[tuple[int, int], tuple[int, int]],
]], s: float = 10, lw: float = 1, c: str = "C0", interval: int=0):
    '''
    data_iter: an iterator of data
    example:
        np.array([
            [0, 1], # point 0
            [2, 3], # point 1
            [4, 5]  # point 2
        ]),
        [
            [0, 1, 2], # path 1 2 3
            [2, 0], # path 2 1
        ],
        (
            (xmin, xmax),
            (ymin, ymax),
        ),
    '''
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    scat: PathCollection = ax.scatter([], [], s=s, c=c)
    line_list: list[Line2D] = []

    def draw(data: tuple[
        np.ndarray,
        list[list[int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]):
        coord, line_data_list, xylim = data
        (xmin, xmax), (ymin, ymax) = xylim
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        scat.set_offsets(coord)
        if len(line_data_list) > len(line_list):
            for _ in range(len(line_data_list) - len(line_list)):
                line_list.append(ax.plot([], [], lw=lw, c=c)[0])
        for i, line_data in enumerate(line_data_list):
            x = coord[line_data, 0]
            y = coord[line_data, 1]
            line_list[i].set_data(x, y)

    ani = animation.FuncAnimation(
        fig=fig,
        func=draw,
        frames=data_iter,
        blit=False,
        interval=interval,
        repeat=False,
    )
    plt.show()


if __name__ == "__main__":
    def data_iter() -> Iterator[tuple[
        np.ndarray,
        list[list[int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]]:
        point = np.random.normal(loc=0, scale=1, size=(100, 2))
        path = [list(range(-1, 100))]
        xylim = ((-3, +3), (-3, +3))
        while True:
            yield point, path, xylim
            point += np.random.normal(loc=0, scale=0.01, size=point.shape)


    draw(data_iter())
