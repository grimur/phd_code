#!/usr/bin/env python
# coding: utf-8


import numpy
import molecular_fingerprints as fingerprints


class Metabolite(object):
    def __init__(self, smiles):
        self.smiles = smiles
        self.fp = fingerprints.fingerprint_from_smiles(smiles)


class IOKRDataServer(object):
    def __init__(self):
        # data set objects. May need corresponding y objects for both?
        self.data_set_x_ids = []
        self.data_set_x = []
        self.data_set_z_ids = []
        self.data_set_z = []

        self.kernel_function_x = None
        self.kernel_function_z = None

        self.K_x = None
        self.K_y = None
        self.K_z = None

        self.kernel_vector_cache = {}

        self.data_type_z = None
        self.data_type_x = None

    def find_kernel(self, a):
        if isinstance(a, self.data_type_x):
            kernel = self.kernel_function_x
            samples = self.data_set_x
        elif isinstance(a, self.data_type_z):
            kernel = self.kernel_function_z
            samples = self.data_set_z
        else:
            raise TypeError('Unknown kernel function for type {}'.format(type(a)))

        return kernel, samples

    def kernel(self, a, b):
        # Return the kernel function btw a and b
        # recognises if they are training examples and extracts from matrix if appropriate
        kernel, samples = self.find_kernel(a)

        if not isinstance(a, type(b)):
            raise TypeError('Mismatched vector types ({} / {})'.format(type(a), type(b)))

        return kernel(a, b)

    def kernel_vector(self, a, v_id=None):
        # returns the appropriate kernel vector
        # see comments on kernel
        vector_id = str(type(a)) + v_id

        if v_id is None or vector_id not in self.kernel_vector_cache:
            kernel, samples = self.find_kernel(a)
            kernel_vector = numpy.array([kernel(a, x) for x in samples])
            kernel_vector = numpy.reshape(kernel_vector, newshape=(1, -1))

            if v_id is not None:
                self.kernel_vector_cache[vector_id] = kernel_vector
        else:
            kernel_vector = self.kernel_vector_cache[vector_id]

        return kernel_vector

