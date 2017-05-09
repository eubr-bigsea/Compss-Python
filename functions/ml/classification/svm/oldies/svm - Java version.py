#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks



def updateWeight(coef_lr,grad_p,w):
    for i in xrange(len(w)):
        w[i] -=coef_lr*grad_p[i]

    return w

@task(returns=list)
def calc_yp(features,labels,w,numDim):
    ypp  = [0 for i in range(len(labels))]

    for i in range(len(labels)):
        ypp[i]=0
        for d in range(numDim):
            ypp[i]+=features[i][d]*w[d]

    return ypp

@task(returns=list)
def calc_cost(labels,ypp):
    cost = [0,0]
    for i in range(len(labels)):
        if (labels[i]*ypp[i]-1) < 0:
            cost[0]+=(1 - labels[i]*ypp[i])

    return cost

@task(returns=list)
def partial_grad(ypp,features,labels,f,numDim,coef_lambda,w):
    grad  = [0 for i in range(numDim)]

    for d in range(numDim):
        grad[d]=0
        if f is 0:
            grad[d]+=abs(coef_lambda * w[d])

        for i in range(len(labels)):
            if (labels[i]*ypp[i]-1) < 0:
                grad[d] -= labels[i] *features[i][d]

    return grad

@task(returns=list)
def accumulate_cost(cost_p1,cost_p2):
    for i in range(len(cost_p1)):
        cost_p1[i]+=cost_p2[i]

    return cost_p1

@task(returns=list)
def accumulate_grad(grad_p1,grad_p2):

    for d in range(len(grad_p1)):
        grad_p1[d]+=grad_p2[d]

    return grad_p1


def predict(test_xi, w): #
    pre = 0
    for i in range(len(test_xi)):
        pre+=test_xi[i]*w[i]

    if pre >= 0:
        return 1.0
    return -1.0

def predict_test(test_x,test_label,w):
    error =0
    values = []
    for i in range(len(test_x)):
        values.append(predict(test_x[i],w))
        if values[i] != test_label[i]:
            error+=1

    print values
    return error

def svm(train_data,test_data, k, numFrag):
    coef_lambda = 0.0001
    coef_lr = 0.01
    coef_threshold = 0.0001
    coef_maxIters = 3
    numDim = 2

    w       = [0 for i in range(numDim)]
    #cost    = [[0,0] for i in range(numFrag)]
    #grad_p  = [[0 for x in range(numDim)] for y in range(numFrag)]
    #yp      = [[0,0] for i in range(numFrag)]
    #cost_final = 0

    features = [i[1] for i in train_data]
    labels   = [i[0] for i in train_data]

    features = [d for d in chunks(features, len(features)/numFrag)]
    labels   = [d for d in chunks(labels,   len(labels)/numFrag)]

    for it in range(coef_maxIters):
        from pycompss.api.api import compss_wait_on

        yp      = [ calc_yp(features[f],labels[f],w,numDim)             for f in range(numFrag) ]
        cost_p  = [ calc_cost(labels[f],yp[f])                          for f in range(numFrag) ]
        grad_p  = [ partial_grad(yp[f],features[f],labels[f],f,numDim,coef_lambda,w)  for f in range(numFrag) ]
        grad    = [ mergeReduce(accumulate_grad, grad_p) ]
        cost    = [ mergeReduce(accumulate_cost, cost_p) ]

        grad  =  compss_wait_on(grad)
        grad = grad[0]

        cost  =  compss_wait_on(cost)
        cost = cost[0][0]
        print "[INFO] - Current Cost %.4f" % (cost)

        if cost < coef_threshold:
            break

        print "grad: %s" % grad
        ###update the weights
        w = updateWeight(coef_lr,grad,w)
        print "w: %s" %w



    return w




def read_file_vector(name,separator):
    row = []

    for line in open(name,"r"):
        col = line.split(separator)
        label = -1
        features =[]

        for i in xrange(0,len(col)):
            if i is 0: # change
                label = float(col[i])
            else:
                features.append( float(col[i]))
        row.append([label,features])

    return row


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SVM - PyCOMPSs')
    parser.add_argument('-t', '--TrainSet', required=True, help='path to the train file')
    parser.add_argument('-v', '--TestSet',  required=True, help='path to the test file')
    parser.add_argument('-f', '--Nodes',    type=int,  default=2, required=False, help='Number of nodes')
    parser.add_argument('-k', '--K',        type=int,  default=1, required=False, help='Number of nearest neighborhood')
    arg = vars(parser.parse_args())
	#parser.add_argument('-o', '--output', required=True, help='path to the output file')

    fileTrain = arg['TrainSet']
    fileTest = arg['TestSet']
    k = arg['K']
    numFrag = arg['Nodes']

    separator = ","
    train_data = read_file_vector(fileTrain,separator)
    test_data = read_file_vector(fileTest,separator)


    model = svm(train_data,test_data, k, numFrag)

    features = [i[1] for i in test_data]
    labels   = [i[0] for i in test_data]

    error = predict_test(features,labels,model)
    print "error {}".format(error)
    #print "Acurracy: {}".format(float(result_labels[0])/result_labels[1])

    #print partialResult
