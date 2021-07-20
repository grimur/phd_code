#!/usr/bin/env python
# coding: utf-8

import numpy
from numba import jit


class IOKRServer(object):
    def __init__(self):
        self.training_data_x = None  # only need to manage the x- and z-space training examples after the initialisation
        self.training_data_z = None

        self._lambda = None  # regularisation parameter

        self.K_x = None  # This is the kernel matrix for the training examples in x-space
        self.K_y = None  # This is the linking kernel matrix for the projections 
                         # of the training examples in x-space and z-space
        self.K_z = None  # This is the kernel matrix for the training examples in z-space

        self.kernel_function_x = None
        self.kernel_function_z = None

    def init(self):
        self.K_x_inv = numpy.linalg.inv(self.K_x + self._lambda * numpy.eye(len(self.training_data_x)))
        self.K_z_inv = numpy.linalg.inv(self.K_z + self._lambda * numpy.eye(len(self.training_data_z)))

    def score(self, x, z):
        # future optimisation: pass precomputed kernel values
        k_x = numpy.array([self.kernel_function_x(x, x_i) for x_i in self.training_data_x])
        k_z = numpy.array([self.kernel_function_z(z, z_i) for z_i in self.training_data_z])

        k_x = numpy.reshape(k_x, newshape=(1, -1))
        k_z = numpy.reshape(k_z, newshape=(1, -1))

        score = (self.K_x_inv @ k_x.T).T @ self.K_y @ self.K_z_inv @ k_z.T

        return score

    def score_vector(self, k_x, k_z):
        score = (self.K_x_inv @ k_x.T).T @ self.K_y @ self.K_z_inv @ k_z.T
        return score


@jit(nopython=True)
def score_vector(k_x, k_z, K_x_inv, K_y, K_z_inv):
    return (K_x_inv @ k_x.T).T @ K_y @ K_z_inv @ k_z.T
