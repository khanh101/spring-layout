from typing import Callable, Any, Iterator, Union, Optional
import numpy as np
import scipy as sp
import scipy.optimize
import multiprocessing as mp
from animation import draw

class SpringLayout:
    def __init__(self, adj: np.ndarray, d: int=2, coord: Optional[np.ndarray] = None):
        self.adj = 0.5 * (adj + adj.T)
        self.n = self.adj.shape[0]
        self.d = d
        for u in range(self.n):
            self.adj[u, u] = 0
        self.edge_list = []
        for u in range(self.n):
            for v in range(u+1, self.n):
                if adj[u, v] > 0:
                    self.edge_list.append([u, v])
        if coord is None:
            coord = np.random.normal(loc=0, scale=0.01, size=(self.n, self.d))
        self.coord = coord

    def data_iter(self) -> Iterator[tuple[
        np.ndarray,
        list[list[int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]]:
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=SpringLayout._optimizer, args=(self, child_conn))
        process.start()
        while True:
            coord = parent_conn.recv()
            if coord is StopIteration:
                break
            yield coord, self.edge_list, ((coord[:, 0].min()-1, coord[:, 0].max()+1), (coord[:, 1].min()-1, coord[:, 1].max()+1))
        process.join()

    def _optimizer(self, child_conn: mp.Pipe):
        x0 = self.coord.flatten()
        def objective(x: np.ndarray) -> float:
            coord = x.reshape((self.n, self.d))
            obj = 0
            for u in range(self.n):
                for v in range(u + 1, self.n):
                    dist2 = ((coord[u] - coord[v]) ** 2).sum()
                    obj += self.adj[u, v] * dist2 + dist2 ** (-0.5)
            return obj

        def callback(x: np.ndarray):
            print(objective(x))
            coord = x.reshape((self.n, self.d))
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

np.random.seed(1234)

height = 5
width = 5
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

draw(SpringLayout(adj).data_iter(), s=20)
