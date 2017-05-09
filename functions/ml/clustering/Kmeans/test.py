# -*- coding: utf-8 -*-
#!/usr/bin/env python


from Kmeans import Kmeans

import sys

sys.path.insert(0, '/home/lucasmsp/workspace/BigSea/Compss-Python/functions/data')
from data_functions import *

if __name__ == "__main__":

    filename = "/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.0012m.csv"
    k=2
    numFrag=4
    maxIterations=5
    epsilon=1e-4
    initMode = "kmeans++"

    print """Running Kmeans with the following parameters:
    - Clusters: {}
    - Nodes: {}
    - DataSet: {}
    - Max Iterations: {}
    - epsilon: {}
    - initMode: {}\n
    """.format(k,numFrag,filename,maxIterations,epsilon,initMode)

    data = ReadFromFile("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.0012m.csv",",",[1,2])
    print "Len data:{}".format(len(data))

    data = Partitionize(data,numFrag)
    print "Len data:{}".format(len(data))

    KM = Kmeans()
    model = KM.fit(k, maxIterations, epsilon, initMode)
    mu = KM.transform(data,model,numFrag)

    print mu
