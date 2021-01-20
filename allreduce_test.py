import sys
from time import time
import numpy as np
from my_naive_allreduce import *
from my_ring_allreduce import *
from mpi4py import MPI


la_comm = MPI.COMM_WORLD
ma_rank = la_comm.Get_rank()
la_size = la_comm.Get_size()


def _op(x, y):
    return x + y


for size in [2**12, 2**13, 2**14]:
    print("array size:", size)
    data = np.random.rand(size)
    data.resize((2, int(size//2)))
    # data = np.random.randint(0, 100, size)
    # data = np.full(size, ma_rank)
    # data = np.arange(size)
    res1 = np.zeros_like(data)
    res2 = np.zeros_like(data)
    res = np.zeros_like(data)
    la_comm.Allreduce(data, res, op=MPI.SUM)
    start1 = time()
    allreduce(data, res1, la_comm, _op)
    end1 = time()
    # print("naive impl output:")
    # print(res1)
    print("naive impl time:", end1-start1)
    start1 = time()
    ringallreduce(data, res2, la_comm, _op)
    end1 = time()
    # print("ring impl output:")
    # print(res2)
    print("ring impl time:", end1-start1)
    # print(np.allclose(res1, res2))
    # print("******", np.allclose(res1, res), "******")
    # print("******", np.allclose(res2, res), "******")
    assert np.allclose(res1, res2)

# print("*****************************************")

