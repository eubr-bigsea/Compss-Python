# K-Nearest Neighbor:

K-Nearest Neighbor is a algorithm used that can be used for both classification and regression predictive problems. However, it is more widely used in classification problems. Is a non parametric lazy learning algorithm. Meaning that it does not use the training data points to do any generalization.  In other words, there is no explicit training phase. More precisely, all the training data is needed during the testing phase.

To do a classification, the algorithm computes from a simple majority vote of the K nearest neighbors of each point present in the training set. The choice of the parameter K is very crucial in this algorithm, and depends on the dataset. However, values of one or tree is more commom.



## Instructions:

To use this implementation, you can the method `fit()` to create a model based in the training data and then, use the method `transform()` to predict the test data. If you want to predict the same input data of the `fit()` method, you can use the method `fit_transform()` to do the both steps.


All parameters are explained below:


**fit():**

- :param data:        A list with numFrag pandas's dataframe used to training the model.
- :param settings:    A dictionary that contains:
 - K:  			 	 Number of K nearest neighborhood to take in count.
 - features: 		 Column name of the features in the training data;
 - label:          	 Column name of the labels   in the training data;
- :param numFrag:     A number of fragments;
- :return:            The model created (which is a pandas dataframe).

**transform():**

- :param data:        A list with numFrag pandas's dataframe that will be predicted.
- :param model:		 The KNN model created;
- :param settings:    A dictionary that contains:
 - K:     	 		 Number of K nearest neighborhood to take in count.
 - features: 		 Column name of the features in the test data;
 - predlabel:    	 Alias to the new column with the labels predicted;
- :param numFrag:     A number of fragments;
- :return:            The prediction (in the same input format).


**fit_transform():**

- :param data:        A list with numFrag pandas's dataframe used to training the model and to classify it.
- :param settings:    A dictionary that contains:
 - K:  			 	 Number of K nearest neighborhood to take in count.
 - features: 		 Column name of the features in the training/test data;
 - label:          	 Column name of the labels   in the training/test data;
 - predlabel:    	 Alias to the new column with the labels predicted;
- :param numFrag:     A number of fragments;
- :return:            The prediction (in the same input format) and the model (which is a pandas dataframe).


## Example:


```sh
from functions.ml.classification.Knn.knn import *

numFrag = 4
knn = KNN()
settings = dict()
settings['K']         = 1
setting['label']      = 'column1'
settings['features']  = 'column2'
settings['predlabel'] = 'result_data1'

model, output1 = knn.fit_tranform(data1,settings,numFrag)

settings = dict()
settings['K']         = 1
settings['features']  = 'col1'
settings['predlabel'] = 'result_data2'
output2 	= knn.tranform(data2,model,settings,numFrag)


```