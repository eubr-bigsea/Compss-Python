#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


from naivebayes import *

import sys

sys.path.insert(0, '/home/lucasmsp/workspace/BigSea/Compss-Python/functions/data')
from data_functions import *

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Gaussian Naive Bayes PyCOMPSs')
    parser.add_argument('-t', '--TrainSet', required=True, help='path to the train file')
    parser.add_argument('-v', '--TestSet',  required=True, help='path to the test file')
    parser.add_argument('-f', '--Nodes',    type=int,  default=2, required=False, help='Number of nodes')
    parser.add_argument('-o', '--output',   required=False, help='path to the output file')
    arg = vars(parser.parse_args())

    fileTrain = arg['TrainSet']
    fileTest  = arg['TestSet']
    numFrag   = arg['Nodes']
    separator = ","
    if arg['output']:
        output_file= arg['output']
    else:
        output_file = ""

    print """Running Gaussian Naive Bayes with the following parameters:\n\t- Nodes: {}\n\t- Train Set: {}\n\t- Test Set: {}\n
          """.format(numFrag,fileTrain,fileTest)


    #trainingSet = loadCsv(fileTrain)
    #testSet     = loadCsv(fileTest)

    #trainingSet = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.012m.csv")
    #testSet     = loadCsv("/home/lucasmsp/workspace/BigSea/Benchmark-COMPSs_SPARK/Java_COMPSs/Datasets/higgs/train_0.012m.csv")
    #dataset  = loadCsv(filename)
    #trainingSet, testSet = splitDataset(dataset, splitRatio)
    #print "Train={0} and Test={1} rows".format(len(trainingSet), len(testSet))


    train_data = ReadFromFile(fileTrain,separator,[0,1,2])
    train_data = VectorAssemble(train_data, 0)
    test_data  = ReadFromFile(fileTest, separator,[1,2])

    train_data = Partitionize(train_data,numFrag)
    test_data  = Partitionize(test_data,numFrag)


    nb = GaussianNB()
    model = nb.fit(train_data,numFrag)

    predictions = nb.transform(test_data,model, numFrag)

    print predictions
    #accuracy = getAccuracy(test_y, predictions)
    #print('Accuracy: {0}%').format(accuracy)
