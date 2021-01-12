import numpy as np
from mpi4py import MPI


def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(0, size):
        if i != rank:
            comm.Isend([send, len(send), MPI.INT], dest=i, tag=1)

    for i in range(0, size):
        if i != rank:
            tmp = np.empty_like(send)
            comm.Recv(tmp, source=i, tag=1)
            recv = op(recv, tmp)

    return recv
