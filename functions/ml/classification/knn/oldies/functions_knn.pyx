

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
