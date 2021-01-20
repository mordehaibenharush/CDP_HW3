import numpy as np
# from mpi4py import MPI


def ringallreduce(send, recv, comm, op):

    rank = comm.Get_rank()
    size = comm.Get_size()

    len_send = len(send)
    block_size = int(len_send/size) if (len_send % size == 0) else int(len_send//size + 1)

    tsend = np.copy(send)
    send_shape = send.shape
    new_shape = (block_size * size,) if len(send_shape) == 1 else (block_size * size, send_shape[1])
    tsend.resize(new_shape, refcheck=False)

    trecv = np.copy(tsend)

    prv = (rank - 1)
    if rank == 0:
        prv = size-1
    nxt = (rank + 1) % size

    src = rank + 1

    for i in range(size-1):
        src = (src-1) if (src-1) >= 0 else (size-1)
        s = trecv[src*block_size:(src+1)*block_size]
        # print(rank, "Sending  from block: ", src, " to process ", nxt)
        # print(rank, "Sending: ", s, " to process ", nxt)
        comm.Isend(trecv[src*block_size:(src+1)*block_size], dest=nxt, tag=1)
        # comm.Send(trecv[(src * block_size):((src + 1) * block_size)], dest=nxt, tag=1)

        dst = (src-1) if (src-1) >= 0 else (size-1)
        # tmp = np.zeros(block_size, dtype=float)
        tmp = np.zeros_like(tsend)
        # print(rank, "Receiving to block: ", dst, " from process ", prv)
        comm.Recv(tmp, source=prv, tag=1)
        v = tmp[:block_size]
        # print(rank, "Receiving: ", v, " from process ", prv)
        t = trecv[dst*block_size:(dst+1)*block_size]
        trecv[dst*block_size:(dst+1)*block_size] = op(t, v)

    src = (rank + 2) % size
    for j in range(size-1):
        src = (src-1) if (src-1) >= 0 else (size-1)
        # print(rank, "Sending  from block: ", src, " to process ", nxt)
        s = trecv[(src * block_size):((src + 1) * block_size)]
        # print(rank, "Sending: ", s, " from block ", src, " to process ", nxt)
        comm.Isend(trecv[(src * block_size):((src + 1) * block_size)], dest=nxt, tag=1)
        dst = (src-1) if (src-1) >= 0 else (size-1)
        # print(rank, "Receiving to block: ", dst, " from process ", prv)
        # tmp = np.zeros(block_size, dtype=float)
        tmp = np.zeros_like(tsend)
        comm.Recv(tmp, source=prv, tag=1)
        # print(rank, "Receiving: ", tmp[:block_size], " to block ", dst, " from process ", prv)
        trecv[dst * block_size:(dst + 1) * block_size] = tmp[:block_size]

    comm.barrier()

    for i in range(len_send):
        recv[i] = trecv[i]
