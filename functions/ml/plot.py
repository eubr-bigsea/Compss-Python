#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_file_vector(name,separator):
    row = []

    for line in open(name,"r"):
        col = line.split(separator)
        label = -1
        features =[]

        for i in xrange(0,len(col)):
            if i is 0:
                label = float(col[i])
            else:
                features.append( float(col[i]))
        row.append([label,features])

    return row


def read_file_iris(name,separator):
    row = []

    for line in open(name,"r"):
        col = line.split(separator)
        label = -1
        features =[]

        for i in xrange(0,len(col)):
            if i is 0:
                label = float(col[i])
            else:
                features.append( float(col[i]))
        row.append([label,features])

    return row


def sintetizador():
    test_data = np.zeros((40000, 4))
    test_data[0:10000, :] = 30.0
    test_data[10000:20000, :] = 60.0
    test_data[20000:30000, :] = 90.0
    test_data[30000:, :] = 120.0

    np.savetxt('test.out', test_data, delimiter=',')


if __name__ == "__main__":

    separator = ","
    train_data = read_file_iris("Iris2d.data",   separator)
    #train_data = read_file_iris("test.out",   separator)
    #train_data = read_file_iris("irisTrain.data",separator)
    #train_data = read_file_vector("knn/higgs-train-0.0001m.csv",separator)
    xy     = [t[1] for t in train_data]
    labels = [t[0] for t in train_data]
    #pca = PCA(n_components=2)
    #X = pca.fit(xy).transform(xy)


    labels = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]

    labes = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
#    f_out = open("Iris2d.data","w")
#    for i,l in zip(X,labels):
#        f_out.write(str(l)+","+str(i[0])+","+str(i[1])+"\n")

    # colors =[]
    # for t in labels:
    #     if t > 0.5:
    #         colors.append('r')
    #     else:
    #         colors.append('b')

    #for t,x in zip(train_data, X):

    #colors=[t[0] for t in train_data]

    #x = X[:, 0]
    x = [t[0] for t in xy]
    #y = X[:, 1]
    y = [t[1] for t in xy]


    centroids = [ [1.6435643, 0.040626846], [-1.9294015, -0.047692403]]

    centroids_x=[t[0] for t in centroids]
    centroids_y=[t[1] for t in centroids]
    centroids_c=['black','black']

    plt.title("Synthetic Data - Kmeans")
    plt.scatter(x,y)
    plt.scatter(centroids_x,centroids_y,c='black',s=100)
    #plt.axis([-4, 5, -1.5, 2.0])



    # centroids = [[-2.53737542,  0.12775509],  [ 1.38640101, -0.06980433]]
    #
    # centroids_x=[t[0] for t in centroids]
    # centroids_y=[t[1] for t in centroids]
    # plt.scatter(centroids_x,centroids_y,c='white',s=100)

    plt.show()


    #sintetizador()
