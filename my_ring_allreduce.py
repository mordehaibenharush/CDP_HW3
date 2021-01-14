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
    #recv = recvv
    rank = comm.Get_rank()
    #print("process num ", rank)
    size = comm.Get_size()
    block_size = int(len(send)/size)
    prv = (rank - 1)
    if rank == 0:
        prv = size-1
    nxt = (rank + 1) % size
    trecv = np.copy(send)
    #tmp = np.zeros(block_size)
    src = rank + 1
    for i in range(size-1):
        src = (src-1) if (src-1) >= 0 else (size-1)
        s = trecv[src*block_size:(src+1)*block_size]
        #print(rank, "Sending  from block: ", src, " to process ", nxt)
        #print(rank, "Sending: ", s, " to process ", nxt)
        comm.Isend(trecv[src*block_size:(src+1)*block_size], dest=nxt, tag=1)
        #comm.Send(trecv[(src * block_size):((src + 1) * block_size)], dest=nxt, tag=1)

        dst = (src-1) if (src-1) >= 0 else (size-1)
        tmp = np.zeros(block_size, dtype=float)
        #print(rank, "Receiving to block: ", dst, " from process ", prv)
        comm.Recv(tmp, source=prv, tag=1)
        #print(rank, "Receiving: ", tmp, " from process ", prv)
        t = trecv[dst*block_size:(dst+1)*block_size]
        trecv[dst*block_size:(dst+1)*block_size] = op(t, tmp)

    src = (rank + 2) % size
    for j in range(size-1):
        src = (src-1) if (src-1) >= 0 else (size-1)
        #print(rank, "Sending  from block: ", src, " to process ", nxt)
        s = trecv[(src * block_size):((src + 1) * block_size)]
        #print(rank, "Sending: ", s, " from block ", src, " to process ", nxt)
        comm.Isend(trecv[(src * block_size):((src + 1) * block_size)], dest=nxt, tag=1)
        dst = (src-1) if (src-1) >= 0 else (size-1)
        #print(rank, "Receiving to block: ", dst, " from process ", prv)
        tmp = np.zeros(block_size, dtype=float)
        comm.Recv(tmp, source=prv, tag=1)
        #print(rank, "Receiving: ", tmp, " to block ", dst, " from process ", prv)
        trecv[dst * block_size:(dst + 1) * block_size] = tmp

    for i in range(len(send)):
        recv[i] = trecv[i]
