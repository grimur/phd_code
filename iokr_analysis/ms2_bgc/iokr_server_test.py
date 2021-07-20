#!/usr/bin/env python
# coding: utf-8

import numpy


class Node(object):
    def __init__(self, int_index=None, pos=None):
        self.int_index = int_index
        self.ext_index = None
        self.label = "Node {}".format(self.int_index)
        self.pos = pos
        #self.fp = None
        
    def __repr__(self):
        return self.label

    @property    
    def fp(self):
        fp_vec = [0]*7
        if self.int_index in (1, 2, 3):
            fp_vec[self.int_index] = 1
        else:
            fp_vec[0] = 1
            fp_vec[self.int_index] = 1
        return numpy.array(fp_vec)

# int. indices
# 1\   /5
# | 2-4 |
# 3/  \6


# This isn't really needed, as we end up embedding it in R^2 and using dot products for kernels...
# Good for visualisation, though.
import networkx as nx
import matplotlib.pyplot as plt

# Pos is coordinates in R^2
n1 = Node(1, pos=(1, 2))
n2 = Node(2, pos=(1, 1))
n3 = Node(3, pos=(2, 1))
n4 = Node(4, pos=(-1, -1))
n5 = Node(5, pos=(-2, -1))
n6 = Node(6, pos=(-1, -2))

edge_list = [
    (n1, n2),
    (n2, n3),
    (n3, n1),
    (n2, n4),
    (n4, n5),
    (n5, n6),
    (n6, n4)
]

G = nx.Graph()
G.add_edges_from(edge_list)

def kernel(node1, node2):
    # path distance in the graph space (x and z) is not a kernel
    #shortest_path = nx.shortest_path(G, node1, node2)
    #return len(shortest_path) - 1
    
    # The graph is embedded in R^2 - use the gaussian for the pos. vectors
    l2 = numpy.sqrt(numpy.sum([(x - y)**2 for x, y in zip(node1.pos, node2.pos)]))
    gaussian = numpy.exp(-0.1 * l2)
    return gaussian


def fp_kernel(fp1, fp2):
    l2 = numpy.sqrt(numpy.sum([(x - y)**2 for x, y in zip(fp1, fp2)]))
    gaussian = numpy.exp(-0.1 * l2)
    return gaussian


# Start with varying lengths to force the dimensions to line up
x_input = [n1, n2, n3, n4, n5, n6]
z_input = [n1, n2, n3, n4, n5, n6]
numpy.random.shuffle(x_input)
numpy.random.shuffle(z_input)


# x = ms2
# z = bgc
# s(x, z) = (K_x + \lambda I_n_x)^{-1} k_x(x)^T K_y(I_x, I_z) (K_z + \lambda I_n_z)^{-1} k_z(z)
# K_x: kernel matrix in X space (dim: no. of data points in training set X)
# I_n_x: Identity matrix in the number of data points in training set X
# k_x(x): Vector of the kernel products of x with the points in the training set
# K_y(I_x, I_z): Kernel matrix with value K_y_(i, j) = k(x_i, y_j) -- i.e. the kernel products between 
#    the data points in the latent space from the two training sets

# Precomputed stuff: 
#    (K_x + \lambda I_n_x)^{-1} and similar for z
#    K_y(I_x, I_z)
# Input stuff:
#    Kernel vectors k_x(x) and k_y(y)

# Implement like this for the initial version and then see about optimising with lin.alg.

from iokr_server import IOKRServer
        

# build x kernel matrix
K_x = numpy.zeros((len(x_input), len(x_input)))
for i in range(len(x_input)):
    for j in range(i+1):
        x_i = x_input[i]
        x_j = x_input[j]
        K_x_i_x_j = kernel(x_i, x_j)
        K_x[i, j] = K_x_i_x_j
        K_x[j, i] = K_x_i_x_j
        
# build z kernel matrix
K_z = numpy.zeros((len(z_input), len(z_input)))
for i in range(len(z_input)):
    for j in range(len(z_input)):
        z_i = z_input[i]
        z_j = z_input[j]
        K_z_i_z_j = kernel(z_i, z_j)
        K_z[i, j] = K_z_i_z_j
        K_z[j, i] = K_z_i_z_j
        
# build y kernel matrix
# Need to extract the y vectors before!
# The x_input_y vector is a vector of y-space elements corresponding to the x-space training set elements
# ditto z_input_y

x_input_y = [x.fp for x in x_input]
z_input_y = [x.fp for x in z_input]
K_y = numpy.zeros((len(x_input), len(z_input)))
for i in range(len(x_input)):
    for j in range(len(z_input)):
        y_i = x_input_y[i]
        y_j = z_input_y[j]
        K_y[i, j] = fp_kernel(y_i, y_j)


# Make sure they're all valid kernel functions
# (unnecessary for the Gaussian, but spoilers - the inital ones weren't!)
K_y_full = numpy.zeros((len(z_input), len(z_input)))
for i in range(len(z_input)):
    for j in range(len(z_input)):
        z_i = z_input_y[i]
        z_j = z_input_y[j]
        K_z_i_z_j = fp_kernel(z_i, z_j)
        K_y_full[i, j] = K_z_i_z_j
        K_y_full[j, i] = K_z_i_z_j
print(numpy.linalg.eig(K_y_full)[0])
print(numpy.linalg.eig(K_x)[0])
print(numpy.linalg.eig(K_z)[0])


# Initialise the IOKR server
i = IOKRServer()
i._lambda = 0.1
i.K_y = K_y
i.K_x = K_x
i.K_z = K_z
i.training_data_x = x_input
i.training_data_z = z_input
i.kernel_function_x = kernel
i.kernel_function_z = kernel

i.init()

# Smoke test
print('Pushing one sample through')
print(i.score(x_input[0], z_input[0]))


print('n-by-n test on the training samples')
# Test n-by-n
for x in x_input:
    print(x, ['{}, {:.4f}'.format(z, i.score(x, z)[0,0]) for z in z_input])


