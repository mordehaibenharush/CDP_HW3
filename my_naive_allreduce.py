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
            #print("Sending: ", send, " to process ", i)
            # comm.Isend([send, len(send), MPI.INT], dest=i, tag=1)
            comm.Isend(send, dest=i, tag=1)

    trecv = np.copy(send)
    for i in range(0, size):
        if i != rank:
            tmp = np.zeros_like(send)
            stats = MPI.Status()
            comm.Probe(source=i, tag=1, status=stats)
            # tmp = np.zeros(stats.Get_elements(MPI.INT))
            # tmp = np.empty_like(send)
            comm.Recv(tmp, source=i, tag=1)
            #print("Receive: ", tmp, " from process ", i)
            # t = recv
            trecv = op(trecv, tmp)

    for i in range(len(send)):
        recv[i] = trecv[i]

