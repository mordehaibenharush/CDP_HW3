import numpy as np
from mpi4py import MPI


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

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
    block_size = int(len(send)/size)
    prv = (rank - 1)
    if rank == 0:
        prv = size-1
    nxt = (rank + 1) % size
    recv = send
    tmp = np.empty(block_size)

    for i in range(size):
        src = (rank-i) if (rank-i) >= 0 else (size-1)
        comm.Isend([recv[src*block_size:(src+1)*block_size], block_size, MPI.INT], dest=nxt, tag=1)
        dst = (src-1) if (src-1) >= 0 else (size-1)
        comm.Recv(tmp, source=prv, tag=1)
        t = recv[dst*block_size:(dst+1)*block_size]
        recv[dst*block_size:(dst+1)*block_size] = op(t, tmp)

    for j in range(size):
        src = ((rank+1)-j) if ((rank+1)-j) >= 0 else (size-1)
        comm.Isend([recv[src * block_size:(src + 1) * block_size], block_size, MPI.INT], dest=nxt, tag=1)
        dst = (src-1) if (src-1) >= 0 else (size-1)
        comm.Recv(tmp, source=prv, tag=1)
        recv[dst * block_size:(dst + 1) * block_size] = tmp

    return recv
