#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"



import sys

sys.path.insert(0, '/home/lucasmsp/workspace/BigSea/Compss-Python/functions/data')
from data_functions import *


from knn import *
import time
import numpy as np



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

    train_data = ReadFromFile(fileTrain,separator,[0,1,2])
    train_data = VectorAssemble(train_data, 0)
    test_data = ReadFromFile(fileTest, separator,[1,2])
    test_data = Partitionize(test_data,numFrag)

    start = time.time()

    knn   = KNN()
    model = knn.fit(train_data, k)
    result_labels = knn.transform(test_data,model, numFrag, output_file)

    end   = time.time()
    print "[INFO] - Time to Load Datasets -> %.02f" % (end-start)


    #print result_labels
