#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter    import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data   import chunks
import numpy as np
import math
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


@task(returns=list)
def merge_accuracy(list1,list2):
    list1[0]+=list2[0]
    list1[1]+=list2[1]
    return list1

@task(returns=list)
def accuracy(y_predicted, y_real):
    num_c = 0
    for p,r in zip(y_predicted,y_real):
        if p == r:
            num_c+=1

    print y_predicted
    print y_real

    return [num_c, len(y_predicted)]

def get_accuracy(y_pred, y_real, numFrag):
        """   """

        size = len(y_real)
        if len(y_pred) != size:
            print "[ERROR] - Accuracy"
            return []

        from pycompss.api.api import compss_wait_on

        n = size/numFrag
        print n
        y_real   = [ d for d in chunks(y_real, n)]
        y_pred   = [ d for d in chunks(y_pred, n)]

        print y_real


        result_p = [ accuracy(y_real[f], y_pred[f]) for f in range(numFrag)]
        v_accuracy   = [ mergeReduce(merge_accuracy, result_p) ]
        v_accuracy   = compss_wait_on(v_accuracy)

        value = float(v_accuracy[0][0])/float(v_accuracy[0][1])
        return value





def precision_recall(y_predicted, y_real, num_labels):
    """Compute the precision, recall and f1-mesure """
    pass

@task(returns=list)
def confusion_matrix(y_true, y_pred, matrix):
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]]+=1
    return matrix

@task(returns=list)
def merge_matrix(matrix1,matrix2):
    #print matrix1
    #print matrix2
    return np.matrix(matrix1) + np.matrix(matrix2)

def Confusion_Matrix(y_true, y_pred,numFrag):
    """ By definition, entry i, j in a confusion matrix is the number of
        observations actually in group i, but predicted to be in group j. """

    size = len(set(y_true))
    if len(set(y_pred)) > size:
        print "[ERROR] - Confusion_Matrix"
        return []

    matrix = [ [0 for i in range(size)] for j in range(size)]

    from pycompss.api.api import compss_wait_on
    n = int(math.ceil(float(len(y_true))/numFrag))
    y_true   = [ d for d in chunks(y_true, n)]
    y_pred   = [ d for d in chunks(y_pred, n)]

    result_p = [ confusion_matrix(y_true[f], y_pred[f], matrix) for f in range(numFrag)]
    matrix   = [ mergeReduce(merge_matrix, result_p) ]
    matrix   = compss_wait_on(matrix)
    return matrix



if __name__ == "__main__":
    numFrag = 4
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    #matrix = Confusion_Matrix(y_true, y_pred, numFrag)
    #for m in matrix:
    #    print m

    print get_accuracy(y_true, y_pred, numFrag)
