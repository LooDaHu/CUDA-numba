from numba import vectorize
import numpy
import timeit

a = numpy.array([10, 20, 30, 40])
b = numpy.arange(16).reshape(4, 4)


# This is vectorize for CPU
@vectorize
def add_ten(num):
    return num + 10


nums = numpy.arange(10)
print(add_ten(nums))


# This is vectorize for GPU
@vectorize(['int64(int64, int64)'], target='cuda')  # Type signature and target are required for the GPU
def add_ufunc(x, y):
    return x + y


def test1():
    return add_ufunc(a, b)


def test2():
    return numpy.add(a, b)


# For such a simple function call, a lot of things just happened! Numba just automatically:
#
# Compiled a CUDA kernel to execute the ufunc operation in parallel over all the input elements.
# Allocated GPU memory for the inputs and the output.
# Copied the input data to the GPU.
# Executed the CUDA kernel (GPU function) with the correct kernel dimensions given the input sizes.
# Copied the result back from the GPU to the CPU.
# Returned the result as a NumPy array on the host.
# Compared to an implementation in C, the above is remarkably more concise.

time = timeit.repeat(stmt="test1()", setup="from __main__ import test1", number=100000, repeat=1)
print("Time with GPU:" + str(sum(time) / len(time)))
# Time with GPU:115.03884158399978

time1 = timeit.repeat(stmt="test2()", setup="from __main__ import test2", number=100000, repeat=1)
print("Time with CPU:" + str(sum(time1) / len(time1)))
# Time without CPU:0.09147244100040552

'''
# Wow, the GPU is a lot slower than the CPU?? For the time being this is to be expected because we have (deliberately) misused the GPU in several ways in this example. How we have misused the GPU will help clarify what kinds of problems are well-suited for GPU computing, and which are best left to be performed on the CPU:

# Our inputs are too small: the GPU achieves performance through parallelism, operating on thousands of values at once. Our test inputs have only 4 and 16 integers, respectively. We need a much larger array to even keep the GPU busy.
# Our calculation is too simple: Sending a calculation to the GPU involves quite a bit of overhead compared to calling a function on the CPU. If our calculation does not involve enough math operations (often called "arithmetic intensity"), then the GPU will spend most of its time waiting for data to move around.
# We copy the data to and from the GPU: While in some scenarios, paying the cost of copying data to and from the GPU can be worth it for a single function, often it will be preferred to to run several GPU operations in sequence. In those cases, it makes sense to send data to the GPU and keep it there until all of our processing is complete.
# Our data types are larger than necessary: Our example uses int64 when we probably don't need it.
# Scalar code using data types that are 32 and 64-bit run basically the same speed on the CPU,
# and for integer types the difference may not be drastic,
# but 64-bit floating point data types have a significant performance cost on the GPU.
# Basic arithmetic on 64-bit floats can be anywhere from 2x (Pascal-architecture Tesla) to 24x (Maxwell-architecture GeForce) slower than 32-bit floats.
# NumPy defaults to 64-bit data types when creating arrays, so it is important to set the dtype attribute or use the ndarray.astype() method to pick 32-bit types when you need them.

'''
