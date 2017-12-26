#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport cython
import numpy as np
cimport numpy as np



cdef extern from "math.h":
  double pow(double x,double y)

cdef extern from "math.h":
  double sqrt(double x)

def powC(double x, double y):
  return pow(x,y)

def sqrtC(double x):
  return sqrt(x)

@cython.boundscheck(False)
def distance(np.ndarray[np.double_t, ndim=1] Y1, np.ndarray[np.double_t, ndim=1] Y2, int numDim):
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

@cython.boundscheck(False)
@cython.wraparound(False)
def dist2all( np.ndarray[np.double_t, ndim=2] dataTrain,
              np.ndarray[np.double_t, ndim=2] dataTest,
              int numDim, int K, semi_labels, model):

    cdef int sizeTest = dataTest.shape[0]
    cdef int sizeTrain = dataTrain.shape[0]

    cdef double[:] semi_dist
    cdef long[:] inds
    for i_test in range(sizeTest):
      semi_dist = np.empty(sizeTrain)
      for i_train in range(sizeTrain):
        semi_dist[i_train] =  np.linalg.norm(dataTrain[i_train]-dataTest[i_test])
      inds = np.argsort(semi_dist)
      semi_labels[i_test] =  model[inds][0:K]

    return semi_labels
