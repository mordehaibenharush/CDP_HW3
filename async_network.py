from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters
        
        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        """
        worker functionality
        :param training_data: a tuple of data and labels to train the NN with
        """
        # setting up the number of batches the worker should do every epoch
        # TODO: add your code
        self.number_of_batches = sum([1 for ii in range(self.rank - self.num_masters, self.number_of_batches, self.num_workers)])    ##TODO: check this line
        print("num batches for worker", self.rank, "is ", self.number_of_batches)
        for epoch in range(self.epochs):
            # creating batches for epoch
            #file = open('worker.txt', 'r')
            #file.write('1')
            
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

            # send nabla_b, nabla_w to masters 
            # TODO: add your code
            #file.write('2')
            
            for i in range(0, len(nabla_w)):  ##masters- 0 to num_masters - 1
                dst = i % self.num_masters
                ind = int(i / self.num_masters)
                self.comm.Isend(nabla_w[i], dst, ind)
                self.comm.Isend(nabla_b[i], dst, ind)
            #file.write('3')
            
            # recieve new self.weight and self.biases values from masters
            # TODO: add your code
            wreqs = []
            breqs = []
            for i in range(0, len(self.weights)):  ##masters- 0 to num_masters - 1 (including)
                dst = i % self.num_masters
                ind = int(i / self.num_masters)
                # s = MPI.Status()
                req = self.comm.Irecv(self.weights[i], dst, ind)
                wreqs.append(req)
                req = self.comm.Irecv(self.biases[i], dst, ind)
                breqs.append(req)

            for wr, br in zip(wreqs, breqs):
                MPI.Request.Wait(wr)
                MPI.Request.Wait(br)

            #file.write('4')
            #file.close()

    def do_master(self, validation_data):
        """
        master functionality
        :param validation_data: a tuple of data and labels to train the NN with
        """
        # setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))
        print("master number of batches ", self.number_of_batches)
        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):
                print(batch)
                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                # TODO: add your code
                s = MPI.Status()
                self.comm.Probe(MPI.ANY_SOURCE, MPI.ANY_TAG, status=s)
                # req = self.comm.Irecv(nabla_w[0], MPI.ANY_SOURCE, 0)
                # MPI.Request.Wait(req, s)
                
                # print(2)
                
                src = s.Get_source()       
                # req = self.comm.Irecv(nabla_b[0], src, 0)
                # MPI.Request.Wait(req)
                wreqs = []
                breqs = []
                for i in range(0, len(nabla_w)):
                    # print(3)
                    
                    req = self.comm.Irecv(nabla_w[i], src, i)
                    wreqs.append(req)
                    req = self.comm.Irecv(nabla_b[i], src, i)
                    breqs.append(req)

                for wr, br in zip(wreqs, breqs):
                    MPI.Request.Wait(wr)
                    MPI.Request.Wait(br)
                
                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                # TODO: add your code
                for i in range(self.rank, self.num_layers, self.num_masters):
                    self.comm.Isend(self.weights[i], src, i)
                    self.comm.Isend(self.biases[i], src, i)
                
            self.print_progress(validation_data, epoch)

        self.comm.barrier()

        # gather relevant weight and biases to process 0
        for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
            self.comm.Bcast(self.weights[i], source=self.rank)
            self.comm.Bcast(self.biases[i], source=self.rank)
        # TODO: add your code
