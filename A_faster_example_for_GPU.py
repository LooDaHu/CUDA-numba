import numpy as np
import timeit
from numba import vectorize
import math  # Note that for the CUDA target, we need to use the scalar functions from the math module, not NumPy
import scipy.stats  # for definition of gaussian distribution, so we can compare CPU to GPU time

SQRT_2PI = np.float32(
    (2 * math.pi) ** 0.5)  # Precompute this constant as a float32.  Numba will inline it at compile time.


@vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian_pdf(x, mean, sigma):
    """
    Compute the value of a Gaussian probability density function at x with given mean and sigma.
    """
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * SQRT_2PI)


@vectorize
def cpu_gaussian_pdf(x, mean, sigma):
    """Compute the value of a Gaussian probability density function at x with given mean and sigma."""
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * SQRT_2PI)


# Evaluate the Gaussian a million times!
x = np.random.uniform(-3, 3, size=1000000).astype(np.float32)
mean = np.float32(0.0)
sigma = np.float32(1.0)

# Quick test on a single element just to make sure it works
gaussian_pdf(x[0], 0.0, 1.0)

norm_pdf = scipy.stats.norm


def test1():
    return norm_pdf.pdf(x, loc=mean, scale=sigma)


def test2():
    return gaussian_pdf(x, mean, sigma)


def test3():
    return cpu_gaussian_pdf(x, mean, sigma)


time1 = timeit.repeat(stmt="test1()", setup="from __main__ import test1", number=2, repeat=1)
print("Time with normal pdf:" + str(sum(time1) / len(time1)))

time1 = timeit.repeat(stmt="test2()", setup="from __main__ import test2", number=2, repeat=1)
print("Time with Gaussian pdf with GPU:" + str(sum(time1) / len(time1)))

time1 = timeit.repeat(stmt="test3()", setup="from __main__ import test3", number=2, repeat=1)
print("Time with Gaussian pdf with CPU:" + str(sum(time1) / len(time1)))
