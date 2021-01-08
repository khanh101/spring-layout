import itertools
import multiprocessing as mp
from typing import Iterator

import numpy as np
import scipy as sp
import scipy.optimize

from animation import draw


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
        print("done", flush=True)
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


# adj = grid_network([4, 4, 4])
adj = grid_network([2, 2, 2])
n = adj.shape[0]
edge_list = []
for u in range(n):
    for v in range(u + 1, n):
        if adj[u, v] > 0:
            edge_list.append([u, v])


def helper():
    coord_list = []
    d = 4
    while d >= 2:
        print(d)
        if len(coord_list) == 0:
            coord = np.random.normal(loc=0, scale=0.01, size=(n, d))
        else:
            coord = coord_list[-1][:, 0:d]
        sl = spring_layout(adj, coord=coord)
        for coord in sl:
            coord_list.append(coord)
            coord = coord[:, 0:2]
            minxy = coord.min(initial=+np.inf) - 1
            maxxy = coord.max(initial=-np.inf) + 1
            yield coord, edge_list, ((minxy, maxxy), (minxy, maxxy))
        d -= 1

    while True:
        for coord in coord_list:
            coord = coord[:, 0:2]
            minxy = coord.min(initial=+np.inf) - 1
            maxxy = coord.max(initial=-np.inf) + 1
            yield coord, edge_list, ((minxy, maxxy), (minxy, maxxy))


draw(helper(), s=20)
