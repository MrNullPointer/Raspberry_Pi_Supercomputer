'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Rpi_MatrixMul.py - Matrix Multiplication using MPI for parallel programming
ver 1.0
Language    : Python
Platform    : Raspberry Pi 4B
Author      : Arva Kagdi,      Parikshit Dubey
              (315) 664-1817,  (315) 395-7575
              Syracuse University 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from mpi4py import MPI
import sys
import numpy as np
import time

# Getting number of rows and column from user
Rows = int(sys.argv[1])
Columns = int(sys.argv[2])
TaskMaster = 0

assert Rows == Columns            # asserting that number of rows and columns are equal

start_time = time.time()          # Records start time
t_start0 = MPI.Wtime()

print("Initialising variables.\n")
mat1 = np.zeros(shape=(Rows, Columns))
mat2 = np.zeros(shape=(Rows, Columns))
c = np.zeros(shape=(Rows, Columns))

# populating matrix
print ("Populating Matrix.\n")
def populateMatrix(p):
    for i in range(0, Rows):
        for j in range(0, Columns):
            p[i][j] = i + j

# MPI command for getting the size, name, and rank of cluster
comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()
processorName = MPI.Get_processor_name()

print("Starting Process %d.\n" % (rank))
print("Running from processor %s, rank %d out of %d processors.\n" % (processorName, rank, worldSize))

# Calculating slices for each worker
if (worldSize == 1):
    slice = Rows
else:
    slice = Rows / (worldSize - 1)  # checking if it is divisible

assert slice >= 1

populateMatrix(mat2)

comm.Barrier()

if rank == TaskMaster:
    print("Initialising Matrices with rows and column (%d,%d).\n" % (Rows, Columns))
    print("Begin")
    populateMatrix(mat1)

    for size in range(1, worldSize):
        offset = (size - 1) * slice  # 0, 10, 20
        row = mat1[offset, :]
        comm.send(offset, dest=size, tag=size)
        comm.send(row, dest=size, tag=size)
        for j in range(0, slice):
            comm.send(mat1[j + offset, :], dest=size, tag=j + offset)
    print("Duty assigned and sent to workers.\n")

comm.Barrier()

# if not a master node receive data and process
if rank != TaskMaster:

    print("Data Received from process %d.\n" % (rank))
    offset = comm.recv(source=0, tag=rank)
    recv_data = comm.recv(source=0, tag=rank)
    for j in range(1, slice):
        c = comm.recv(source=0, tag=j + offset)
        recv_data = np.vstack((recv_data, c))

    print("Start Calculation from process %d.\n" % (rank))

    # Looping through rows
    t_start = MPI.Wtime()
    for size in range(0, slice):
        res = np.zeros(shape=(Columns))
        if (slice == 1):
            r = recv_data
        else:
            r = recv_data[size, :]
        ai = 0
        for j in range(0, Columns):
            q = mat2[:, j]  # get the column we want
            for x in range(0, Columns):
                res[j] = res[j] + (r[x] * q[x])
            ai = ai + 1
        if (size > 0):
            send = np.vstack((send, res))
        else:
            send = res
    t_diff = MPI.Wtime() - t_start

    print("Process %d finished in %5.4fs.\n" % (rank, t_diff))
    # Send large data
    print("Sending results to Master %d bytes.\n" % (send.nbytes))
    comm.Send([send, MPI.FLOAT], dest=0, tag=rank)  # 1, 12, 23

comm.Barrier()

# checking response from workers
if rank == TaskMaster:
    print("Checking response from Workers.\n")
    res1 = np.zeros(shape=(slice, Columns))
    comm.Recv([res1, MPI.FLOAT], source=1, tag=1)
    print("Received response from 1.\n")
    kl = np.vstack((res1))
    for size in range(2, worldSize):
        resx = np.zeros(shape=(slice, Columns))
        comm.Recv([resx, MPI.FLOAT], source=size, tag=size)
        print("Received response from %d.\n" % (size))
        kl = np.vstack((kl, resx))
    print("End")
    print("Result mat1 X mat2.\n")
    print(kl)

comm.Barrier()