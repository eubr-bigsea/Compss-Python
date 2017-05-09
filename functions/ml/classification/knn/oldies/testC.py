#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import time
import math
import numpy as np

#============ To Profile ===================
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap
#=========================================

# def distance(Y1,Y2,numDim):
#     result = 0
#     import functions_knn
#     i = 0
#     while i < numDim:
#         t = (Y1[i]-Y2[i])
#         result += t*t
#         i +=1
#
#     result = functions_knn.sqrtC(result)
#     return result


def getPopularElement(labels,K): # ok
    u, indices = np.unique(labels[0:K], return_inverse=True)
    label = u[np.argmax(np.bincount(indices))]
    return label

def getKNN(neighborhood,K):  # ok
    start = time.time()

    result = [0 for i in range(len(neighborhood))]
    for i in range(len(neighborhood)):
        result[i] = getPopularElement(neighborhood[i],K)

    end = time.time()
    print "\n[INFO] - getKNN -> Time elapsed: %.2f seconds\n" % (end-start)
    return result

@timing
def classifyBlock(test_data,train_labels,train_features,numDim,K):
    start=time.time()

    sizeTest = len(test_data)
    semi_labels = np.zeros((sizeTest, K+1))                #np.array([[0 for i in range(K+1) ] for i in range(len(test_data))])
    semi_dist   = np.full( (sizeTest, K+1), float("inf"))  #np.array([[float("inf") for i in range(K+1) ] for i in range(len(test_data))])

    tmp_dist  = 0;
    tmp_label = 0;

    import functions_knn

    #calculate part
    for i_test in range(sizeTest):

        for i_train in range(len(train_labels)):
            semi_dist  [i_test][K] = functions_knn.distance(train_features[i_train], test_data[i_test], numDim)
            semi_labels[i_test][K] = train_labels[i_train]

            j=K
            while(j>0):
                if(semi_dist[i_test][j] < semi_dist[i_test][j-1]):
                    tmp_label = semi_labels[i_test][j]
                    semi_labels[i_test][j]   = semi_labels[i_test][j-1]
                    semi_labels[i_test][j-1] = tmp_label

                    tmp_dist = semi_dist[i_test][j];
                    semi_dist[i_test][j]    = semi_dist[i_test][j-1];
                    semi_dist[i_test][j-1]  = tmp_dist;
                j-=1

    #Choose part
    result_partial= getKNN(semi_labels, K)

    end = time.time()
    print "\n[INFO] - ClassifierBlock -> Time elapsed: %.2f seconds\n" % (end-start)

    return result_partial

# @task(returns=list)
# def merge_lists(list1,list2):
#     list1 = list1+list2
#     return list1

def knn(train_labels,train_features,test_data, K, numFrag, output_file):
    """ Knn:

    :param data: data
    :param K: num of K nearest neighborhood to take in count
    :param numFrag: num fragments, if -1 data is considered chunked
    :return: list of labels predicted
    """

    if numFrag == -1:
        numFrag = len(test_data)
    else:
        test_data = np.array_split(test_data, numFrag)
        # size = int(math.ceil(float(len(test_data))/numFrag))
        # test_data = [d for d in chunks(test_data, size )]

    numDim = 28


    partialResult = [ classifyBlock(test_data[i],train_labels,train_features,numDim,K)  for i in range(numFrag) ]
    #result = mergeReduce(merge_lists, partialResult)
    #result = compss_wait_on(result)
    if len(output_file) > 1:
        for p in range(len(partialResult)):
            filename= output_file+"_part"+str(p)
            f = open(filename,"w")
            f.close()
            savePredictionToFile(partialResult[p], filename)


    #return partialResult #result

def loadData(name,separator,istrainSet):
    row = []

    for line in open(name,"r"):
        col = line.split(separator)
        label = -1
        features = []

        for i in xrange(0,len(col)):
            if i is 0: # change
                label = float(col[i])
            else:
                features.append( float(col[i]))
        if istrainSet:
            row.append([label,features])
        else:
            row.append(features)

    return row


def savePredictionToFile(result, filename):
    np.savetxt(filename,result,delimiter=',',fmt="%s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='KNN PyCOMPSs')
    parser.add_argument('-t', '--TrainSet', required=True, help='path to the train file')
    parser.add_argument('-v', '--TestSet',  required=True, help='path to the test file')
    parser.add_argument('-f', '--Nodes',    type=int,  default=2, required=False, help='Number of nodes')
    parser.add_argument('-k', '--K',        type=int,  default=1, required=False, help='Number of nearest neighborhood')
    parser.add_argument('-o', '--output',   required=False, help='path to the output file')
    arg = vars(parser.parse_args())

    fileTrain = arg['TrainSet']
    fileTest  = arg['TestSet']
    k         = arg['K']
    numFrag   = arg['Nodes']
    if arg['output']:
        output_file= arg['output']
    else:
        output_file = ""
    separator = ","

    print """Running KNN with the following parameters:
    - K: {}
    - Nodes: {}
    - Train Set: {}
    - Test Set: {}\n
    """.format(k,numFrag,fileTrain,fileTest)


    start= time.time()
    train_features = np.loadtxt(fileTrain,delimiter=separator)
    train_labels = train_features[:,0]
    train_features  = np.delete(train_features, [0], axis=1)
    end= time.time()
    print "[INFO] - Time to Load Dataset Train -> %.02f" % (end-start)
    #print train_data[0]
    #test_data  = loadData(fileTest,separator,False)

    start= time.time()
    test_data  = np.loadtxt(fileTest  ,delimiter=separator) #,usecols =(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28)
    test_data  = np.delete(test_data, [0], axis=1)
    end= time.time()
    print "[INFO] - Time to Load Dataset Train -> %.02f" % (end-start)

    start =time.time()
    result_labels = knn(train_labels,train_features,test_data, k, numFrag, output_file)
    end  =time.time()
    print "[INFO] - Time to classifyBlock -> %.02f" % (end-start)
    #savePredictionToFile(result_labels, output_file)
