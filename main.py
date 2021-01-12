import itertools
import multiprocessing as mp
from typing import Iterator

import numpy as np
import scipy as sp
import scipy.optimize

import animation


def spring_layout(adj: np.ndarray, coord: np.ndarray) -> Iterator[np.ndarray]:
    n, d = coord.shape

    def _optimizer(adj: np.ndarray, d: int, coord: np.ndarray, child_conn: mp.Pipe):
        x0 = coord.flatten()

        def objective(x: np.ndarray) -> float:
            coord = x.reshape((n, d))
            obj = 0
            for u in range(n):
                for v in range(u + 1, n):
                    dist2 = ((coord[u] - coord[v]) ** 2).sum()
                    obj += adj[u, v] * dist2 + dist2 ** (-0.5)
            return obj

        def callback(x: np.ndarray):
            coord = x.reshape((n, d))
            child_conn.send(coord)

        result = sp.optimize.minimize(
            fun=objective,
            x0=x0,
            method="L-BFGS-B",
            callback=callback,
        )
        if result.success:
            callback(result.x)
        child_conn.send(StopIteration)

    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=_optimizer, args=(adj, d, coord, child_conn))
    process.start()
    while True:
        coord = parent_conn.recv()
        if coord is StopIteration:
            break
        yield coord
    process.join()


np.random.seed(1234)


def grid_network(shape: list[int]) -> np.ndarray:
    coord_list = [list(coord) for coord in itertools.product(*[range(s) for s in shape])]
    n = len(coord_list)

    c2i = {}
    for i, c in enumerate(coord_list):
        c2i[tuple(c)] = i

    def within_range(coord: list[int]) -> bool:
        for i, c in enumerate(coord):
            if c < 0 or c >= shape[i]:
                return False
        return True

    adj = np.zeros(shape=(n, n), dtype=float)
    for c in coord_list:
        for i in range(len(shape)):
            for dc in [-1, +1]:
                new_c = [*c]
                new_c[i] += dc
                if within_range(new_c):
                    i0 = c2i[tuple(c)]
                    i1 = c2i[tuple(new_c)]
                    adj[i0, i1] = 1
    return adj


adj = grid_network([10, 2])
n = adj.shape[0]


state = animation.State()
state.vertex_size = 5
state.line_width = 1
state.line = []
for u in range(n):
    for v in range(u + 1, n):
        if adj[u, v] > 0:
            state.line.append([u, v])

def helper(d: int=3) -> Iterator[animation.State]:
    last_vertex = None
    iteration = 1
    print("Running...")
    while d >= 2:
        print(f"dim {d}")
        if last_vertex is None:
            vertex = np.random.normal(loc=0, scale=0.01, size=(n, d))
        else:
            vertex = last_vertex[:, 0:d]
        sl = spring_layout(adj, coord=vertex)
        for vertex in sl:
            state.title = str(iteration)
            iteration += 1
            last_vertex = vertex
            vertex = vertex[:, 0:2]
            minxy = vertex.min(initial=+np.inf) - 1
            maxxy = vertex.max(initial=-np.inf) + 1
            state.vertex = vertex
            state.xylim = ((minxy, maxxy), (minxy, maxxy))
            yield state
        d -= 1
    print("done")

d = 2
animation.draw(helper(d), save=True)
