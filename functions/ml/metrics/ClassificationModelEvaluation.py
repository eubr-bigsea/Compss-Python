#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import pandas as pd

class ClassificationModelEvaluation(object):
    """
            * True Positive (TP) - label is positive and prediction is also positive
            * True Negative (TN) - label is negative and prediction is also negative
            * False Positive (FP) - label is negative but prediction is positive
            * False Negative (FN) - label is positive but prediction is negative

            Metrics:

            * Accuracy
            * Precision (Positive Predictive Value) = tp / (tp + fp)
            * Recall (True Positive Rate) = tp / (tp + fn)
            * F-measure = F1 = 2 * (precision * recall) / (precision + recall)
    """

    def calculate(self,data,settings,numFrag):

        col_test = settings['test_col']
        col_predicted = settings['pred_col']

        stage1 = [self.CME_stage1(data[f],col_test,col_predicted) for f in range(numFrag)]
        merged_stage1 = mergeReduce(self.mergeStage1,stage1)
        op = settings.get('binary', False)
        if op:
            true_label = settings['pos_label'] if "pos_label" in settings else 1
            result = self.CME_stage2_binary(merged_stage1,true_label)
        else:
            result = self.CME_stage2(merged_stage1)
        return result


    @task(returns=list, isModifier = False)
    def CME_stage1(self,data, col_test, col_predicted):

        labels = data[col_test].unique().astype(float)

        N_labels = len(labels)
        matrix = np.zeros((N_labels,N_labels))

        df = pd.DataFrame(matrix,columns=labels,index=labels)

        Reals = data[col_test].values.astype(float)
        Preds = data[col_predicted].values.astype(float)
        for real, pred in zip(Reals, Preds):
             #idx = np.where(labels == real)[0][0]
            df.loc[real, pred]+=1

        return df

    @task(returns=list, isModifier = False)
    def mergeStage1(self,p1,p2):
        p1 =  p1.add(p2, fill_value=0)
        return p1

    @task(returns=list, isModifier = False)
    def CME_stage2(self,confusion_matrix):

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


    @task(returns=list, isModifier = False)
    def CME_stage2_binary(self,confusion_matrix,true_label):

        N = confusion_matrix.sum().sum()
        labels = confusion_matrix.index
        acertos = 0
        Precisions = []     #  TPR
        Recalls = []        #  FPR

        for i in labels:
            acertos += confusion_matrix[i].ix[i]

        TP = confusion_matrix[true_label].ix[true_label]
        Precision =   float(TP) / confusion_matrix.ix[true_label].sum()
        Recall    =   float(TP) / confusion_matrix[true_label].sum()
        Accuracy = float(acertos) / N
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        table =  pd.DataFrame([
                                ["Accuracy",Accuracy],
                                ["Precision",Precision],
                                ["Recall",Recall],
                                ["F-measure (F1)",F1]
                            ],columns=["Metric","Value"])


        return [confusion_matrix, table]
