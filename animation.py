from typing import Iterator, Union

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


class State:
    title: str  # title
    xylim: tuple[tuple[float, float], tuple[float, float]]  # xy lim
    vertex: np.ndarray  # (N, 2) vertex matrix
    vertex_size: Union[float, list[float]]  # vertex size or list of vertex size list
    vertex_color: Union[str, list[str]]  # vertex color or list of vertex color list
    line: list[list[int]]  # line list
    line_width: Union[float, list[float]]  # line width of list of line width list
    line_color: Union[str, list[str]]  # line color or list of line color list
    polygon: list[list[int]]  # polygon list
    polygon_color: Union[str, list[str]]  # polygon color or polygon color list

    def __init__(self):
        self.title = ""
        self.xylim = ((-1, +1), (-1, +1))
        self.vertex = np.zeros(shape=(0, 2))
        self.vertex_size = 0
        self.vertex_color = "C0"
        self.line = []
        self.line_width = 0
        self.line_color = "C1"
        self.polygon = []
        self.polygon_color = "C2"


def draw(state_iter: Iterator[State], interval: int = 0, save: bool = False):
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    scat: PathCollection = ax.scatter([], [])
    line_list: list[Line2D] = []
    polygon_list: list[Polygon] = []

    def draw(state: State):
        fig.suptitle(state.title)
        (xmin, xmax), (ymin, ymax) = state.xylim
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        scat.set_offsets(state.vertex)
        if isinstance(state.vertex_size, float) or isinstance(state.vertex_size, int):
            scat.set_sizes(np.array([state.vertex_size for _ in range(state.vertex.shape[0])]))
        else:
            scat.set_sizes(state.vertex_size)
        scat.set_color(state.vertex_color)
        if len(state.line) > len(line_list):
            for _ in range(len(state.line) - len(line_list)):
                line_list.append(ax.plot([], [])[0])
        for i, line_data in enumerate(state.line):
            line_list[i].set_data(state.vertex[line_data, 0], state.vertex[line_data, 1])
            line_list[i].set_linewidth(state.line_width)
            line_list[i].set_color(state.line_color)
        if len(state.polygon) > len(polygon_list):
            for _ in range(len(state.polygon) - len(polygon_list)):
                polygon_list.append(plt.Polygon(np.zeros((0, 2))))
                fig.gca().add_patch(polygon_list[-1])
        for i, polygon_data in enumerate(state.polygon):
            polygon_list[i].set_xy(state.vertex[polygon_data])
            polygon_list[i].set_color(state.polygon_color)

        if save:
            fig.savefig(state.title + ".png")

    ani = animation.FuncAnimation(
        fig=fig,
        func=draw,
        frames=state_iter,
        blit=False,
        interval=interval,
        repeat=False,
    )
    plt.show()


if __name__ == "__main__":
    def state_iter() -> Iterator[State]:
        state = State()
        state.vertex_size = 10
        state.line_width = 3
        state.vertex = np.random.normal(loc=0, scale=1, size=(100, 2))
        state.line = [list(range(0, 10))]
        state.polygon = [list(range(10, 20))]
        state.xylim = ((-3, +3), (-3, +3))
        count = 0
        while True:
            yield state
            count += 1
            state.title = f"{count}"
            state.vertex += np.random.normal(loc=0, scale=0.01, size=state.vertex.shape)


    draw(state_iter())
