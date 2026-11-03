import os
from multiprocessing import Pool
import numpy as np


def _run(job):
    fn, r, c, x, y = job
    return r, c, fn(x, y)


def _save(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)   # atomic: killing the run mid-save can't corrupt the cache


def grid_sweep(fn, xs, ys, cache, fields):
    """Evaluate fn(x, y) -> (fate, *values) at every grid point, on all cores.

    Returns {field: array} indexed [row, col] = point (xs[col], ys[row]).
    fields[0] is the int fate (-1 = not computed yet); the rest are floats.
    Saves to `cache` every len(xs) pixels and skips finished ones, so an interrupted run resumes where it stopped.
    fn must be picklable: a module-level function, or a functools.partial of one.
    """
    shape = (len(ys), len(xs))
    if os.path.exists(cache):
        d = np.load(cache)
        out = {k: d[k] for k in fields}
    else:
        out = {k: np.full(shape, -1 if k == fields[0] else np.nan) for k in fields}
    todo = [(fn, r, c, x, y) for r, y in enumerate(ys) for c, x in enumerate(xs) if out[fields[0]][r, c] < 0]
    if todo:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with Pool() as pool:
            for n, (r, c, res) in enumerate(pool.imap_unordered(_run, todo, chunksize=4), 1):
                for k, v in zip(fields, res):
                    out[k][r, c] = v
                if n % len(xs) == 0 or n == len(todo):
                    _save(cache, xs=xs, ys=ys, **out)
                    print(f"{n}/{len(todo)} pixels saved", flush=True)
    return out
