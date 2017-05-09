cimport cython
import numpy as np
cimport numpy as np


cdef extern from "math.h":
  double pow(double x,double y)

cdef extern from "math.h":
  double sqrt(double x)

cdef extern from "math.h":
  double exp(double x)

def powC(double x, double y):
  return pow(x,y)

def sqrtC(double x):
  return sqrt(x)



@cython.boundscheck(False)
def calculateProbability(double x, double mean, double stdev):
    cdef double pi =  3.14159265358979323846

    cdef double result = 0
    cdef double exponent =  exp(-(pow(x-mean,2)/(2*pow(stdev,2))))

    if stdev == 0:
        stdev = 0.0000001

    result = (1.0 / (sqrt(2*pi) * stdev)) * exponent
    return result
