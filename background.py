import multiprocessing as mp
from typing import Iterator, Any


def bg(iterator: Iterator[Any]) -> Iterator[Any]:
    def _bg_fetch(iterator: Iterator[Any], child_conn: mp.Pipe):
        while child_conn.recv():
            try:
                child_conn.send(next(iterator))
            except StopIteration:
                child_conn.send(StopIteration)
                break

    parent_conn, child_conn = mp.Pipe()
    process = mp.Process(target=_bg_fetch, args=(iterator, child_conn))
    process.start()

    parent_conn.send(True)  # signal first value
    while True:
        value = parent_conn.recv()
        if value is StopIteration:
            break
        parent_conn.send(True)  # signal next value
        yield value

    process.join()


if __name__ == "__main__":
    import time


    def my_iter() -> Iterator[int]:
        for i in range(5):
            yield i
            time.sleep(1)


    bg_iter = bg(my_iter())
    t0 = time.time()
    for item in bg_iter:
        print(item)
        time.sleep(1)  # do something else
    t1 = time.time()
    print(f"elapsed time: {t1 - t0}")
