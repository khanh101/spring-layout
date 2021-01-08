from typing import Callable, Any, Iterator, Union, Optional
import numpy as np
import scipy as sp
import scipy.optimize
import multiprocessing as mp

from sklearn.decomposition import PCA

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

height = 4
width = 4
n = height * width

hw2i = np.arange(0, n).reshape((height, width))

def within_range(h: int, w: int) -> bool:
    if h < 0 or h >= height:
        return False
    if w < 0 or w >= width:
        return False
    return True

adj = np.zeros(shape=(n, n), dtype=float)
for h0 in range(height):
    for w0 in range(width):
        for dh, dw in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                h1 = h0+dh
                w1 = w0+dw
                if within_range(h1, w1):
                    i0 = hw2i[h0, w0]
                    i1 = hw2i[h1, w1]
                    adj[i0, i1] = 1

edge_list = []
for u in range(n):
    for v in range(u + 1, n):
        if adj[u, v] > 0:
            edge_list.append([u, v])
def helper():
    coord_list = []
    d = 3
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
            minxy = coord.min(initial=+np.inf)-1
            maxxy = coord.max(initial=-np.inf)+1
            yield coord, edge_list, ((minxy, maxxy), (minxy, maxxy))
        d -= 1


draw(helper(), s=20)
