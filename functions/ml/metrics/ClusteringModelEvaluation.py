#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.api.api import compss_wait_on

import numpy as np
import pandas as pd
import sys

#
#   Como só existe o WSSSE, ele ja vai ser calculado antes
#
#
#
#



def ClusteringModelEvaluation(data,settings,numFrag):

    col_test = settings['test_col']
    col_predicted = settings['pred_col']


    """
        Within Set Sum of Squared Errors (WSSSE)

    """

    stage1 = [CME_stage1(data[f],col_test,col_predicted) for f in range(numFrag)]
    merged_stage1 = mergeReduce(mergeStage1,stage1)
    if settings['binary']:
        true_label = settings['pos_label'] if "pos_label" in settings else 1
        result = CME_stage2_binary(merged_stage1,true_label)
    else:
        result = CME_stage2(merged_stage1)
    return result



def CME_stage1(data,col_test,col_predicted):
    pos_negs = dict()
    labels = data[col_test].unique().astype(float)
    N_labels = len(labels)
    matrix = np.zeros((N_labels,N_labels))

    df = pd.DataFrame(matrix,columns=labels,index=labels)

    for real, pred in zip(data[col_test].values.astype(float), data[col_predicted].values.astype(float)):
        idx = np.where(labels == real)[0][0]
        df.loc[idx, pred]+=1

    return df

def mergeStage1(p1,p2):
    p1 =  p1.add(p2, fill_value=0)
    return p1

def CME_stage2(confusion_matrix):

    N = confusion_matrix.sum().sum()
    labels = confusion_matrix.index
    acertos = 0
    Precisions = []     #  TPR
    Recalls = []        #  FPR
    for i in labels:
        acertos += confusion_matrix[i].ix[i]
        TP = confusion_matrix[i].ix[i]
        Precisions.append(   float(TP) / confusion_matrix.ix[i].sum())
        Recalls.append( float(TP) / confusion_matrix[i].sum())

    Accuracy = float(acertos) / N

    F1s = []
    for p,r in zip(Precisions,Recalls):
        F1s.append(2 * (p * r) / (p + r))


    precision_recall = pd.DataFrame(np.array([Precisions,Recalls,F1s]).T,
                                    columns=['Precision','Recall',"F-Mesure"])

    Precision = np.mean(Precisions)
    Recall = np.mean(Recalls)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    table =  pd.DataFrame([
                            ["Accuracy",Accuracy],
                            ["Precision",Precision],
                            ["Recall",Recall],
                            ["F-measure (F1)",F1]
                        ],columns=["Metric","Value"])


    return [confusion_matrix, table, precision_recall]