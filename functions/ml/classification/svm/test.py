#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# Developed by Lucas Miguel Ponce
# Mestrando em Ciências da Computação - UFMG
# <lucasmsp@gmail.com>
#



import sys
sys.path.insert(0, '/home/lucasmsp/workspace/BigSea/Compss-Python/functions/data')


from data_functions import *
from svm import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SVM - PyCOMPSs')
    parser.add_argument('-t', '--TrainSet', required=True, help='path to the train file')
    parser.add_argument('-v', '--TestSet',  required=True, help='path to the test file')
    parser.add_argument('-f', '--Nodes',        type=int,    default=2, required=False, help='Number of nodes')
    parser.add_argument('-C', '--lambda',       type=float,  default=0.001, required=False, help='Regularization parameter')
    parser.add_argument('-it', '--MaxIters',    type=int,    default=10, required=False, help='Number max of iterations')
    parser.add_argument('-lr', '--lr',          type=float,  default=0.01, required=False, help='Learning rate parameter')
    parser.add_argument('-thr', '--threshold',  type=float,  default=0.01, required=False, help='Tolerance for stopping criterion')
    arg = vars(parser.parse_args())


    fileTrain   = arg['TrainSet']
    fileTest    = arg['TestSet']
    numFrag     = arg['Nodes']

    settings = {}
    settings['coef_lambda']     = arg['lambda']
    settings['coef_lr']         = arg['lr']
    settings['coef_threshold']  = arg['threshold']
    settings['coef_maxIters']   = arg['MaxIters']


    separator = ","
    train_data = ReadFromFile(fileTrain,",",[0,1,2])
    test_data  = ReadFromFile(fileTest,",", [1,2]) #if the label is removed, else use "Drop"

    train_data = Partitionize(train_data,numFrag)
    test_data = Partitionize(test_data,numFrag)

    svm = SVM()
    model   = svm.fit(train_data,settings,numFrag)
    result  = svm.transform(test_data,model,numFrag)
    print result
