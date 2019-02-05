#cython: boundscheck=False, wraparound=False, nonecheck=False

# cython functions_knn.pyx -a


cimport cython
import numpy as np
cimport numpy as np


from libc.math cimport sqrt

def distance(np.ndarray[np.double_t, ndim=1] Y1, double[:] Y2, int numDim):
    cdef double result = 0
    cdef int i = 0
    cdef float t = 0
    cdef double out = 0.0

    while i < numDim:
        t = (Y1[i] - Y2[i])
        result += t*t
        i+=1

    out = sqrt(result)
    return out


def dist2all( np.ndarray[np.double_t, ndim=2] dataTrain,
              double[:, :] dataTest,
              int numDim, int K, semi_labels, model):

    cdef int sizeTest = dataTest.shape[0]
    cdef int sizeTrain = dataTrain.shape[0]
    cdef int i_test, i_train
    cdef double[:] semi_dist
    cdef long[:] inds

    for i_test in range(sizeTest):
      semi_dist = np.empty(sizeTrain)
      for i_train in range(sizeTrain):
        semi_dist[i_train] =  distance(dataTrain[i_train], dataTest[i_test], numDim) #np.linalg.norm(dataTrain[i_train]-dataTest[i_test])
      idx = np.argpartition(semi_dist, K)
      semi_labels[i_test] =  model[idx[:K]]

    return semi_labels
