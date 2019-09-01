from numba import jit
import math
import timeit


# This is the function decorator syntax and is equivalent to `hypot = jit(hypot)`.
# The Numba compiler is just a function you can call whenever you want!
@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    t = t / x
    return x * math.sqrt(1 + t * t)


time = timeit.repeat(stmt="hypot(3.0,4.0)", setup="from __main__ import hypot", number=100000, repeat=100)
print("Time with Numba compiler:" + str(sum(time) / len(time)))

time1 = timeit.repeat(stmt="hypot.py_func(3.0,4.0)", setup="from __main__ import hypot", number=100000, repeat=100)
print("Time without Numba compiler:" + str(sum(time1) / len(time1)))

# We can see the result of type inference by using the .inspect_types() method,
# which prints an annotated version of the source code:
print(hypot.inspect_types())
