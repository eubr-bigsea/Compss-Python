# Support vector machines (SVM):
 SVM is a supervised learning model used for binary classification. Given a set of training examples, each marked as belonging to one or the other of two categories, a SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. This algorithm is effective in high dimensional spaces and it is still effective in cases where number of dimensions is greater than the number of samples.


## Instructions:

The algorithm reads a dataset composed by labels (-1.0 or 1.0) and features (numeric fields).

To use this implementation, you can the method `fit()` to create a model based in the training data and then, use the method `transform()` to predict the test data.

All parameters are explained below:

**fit():**

- :param data:        A list with numFrag pandas's dataframe used to training the model.
- :param settings:    A dictionary that contains:
 - coef_lambda:   Regularization parameter (float);
 - coef_lr: Learning rate parameter (float);
 - coef_threshold: Tolerance for stopping criterion (float);
 - coef_maxIters: Number max of iterations (integer);
 - features: 		   Column name of the features in the training data;
 - label:          	 Column name of the labels   in the training data;
- :param numFrag:     A number of fragments;
- :return:            The model created (which is a pandas dataframe).

**transform():**

- :param data: A list with numFrag pandas's dataframe that will be predicted.
- :param model: A model already trained (np.array);
- :param settings: A dictionary that contains:
 - features: Column name of the features in the test data;
 - predlabel: Alias to the new column with the labels predicted;
- :param numFrag: A number of fragments;
- :return: The prediction (in the same input format).


## Example:


```sh
from functions.ml.classification.Svm.svm import *

numFrag = 4
svm = SVM()
settings = dict()
settings['coef_lambda'] = 0.001
settings['coef_lr']     = 0.01
settings['coef_threshold'] = 0.01
settings['coef_maxIters'] = 100
settings['label']      = 'column1'
settings['features']  = 'column2'


model = svm.fit(data1,settings,numFrag)
settings = dict()
settings['features']  = 'col1'
settings['predlabel'] = 'result_data2'
output 	= knn.tranform(data2,model,settings,numFrag)

```
