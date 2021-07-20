import numpy


def l2_gaussian(v1, v2, gamma=0.01):
    d_sq = numpy.sum(numpy.power(v1 - v2, 2))
    return gaussian(d_sq, gamma)


def gaussian(x, gamma=0.01):
    return numpy.exp(-(gamma * x))


