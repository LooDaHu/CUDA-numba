from numba import guvectorize
import math
import numpy as np


@guvectorize(['(float32[:], float32[:])'],  # have to include the output array in the type signature
             '(i)->()',  # map a 1D array to a scalar output
             target='cuda')
def l2_norm(vec, out):
    acc = 0.0
    for value in vec:
        acc += value ** 2
    out[0] = math.sqrt(acc)


angles = np.random.uniform(-np.pi, np.pi, 10)
coords = np.stack([np.cos(angles), np.sin(angles)], axis=1)
print(coords)

l2_norm(coords)
