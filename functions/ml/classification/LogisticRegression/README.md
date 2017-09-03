# Logistic Regression:


Logistic regression is named for the function used at the core of the method, the logistic function. It is the go-to method for binary classification problems (problems with two class values).

The logistic function, also called the sigmoid function was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

This implementation uses a Stochastic Gradient Ascent (a variant of the Stochastic gradient descent). It is called stochastic because the derivative based on a randomly chosen single example is a random approximation to the true derivative based on all the training data.


## Instructions:

The algorithm reads a dataset composed by labels (numeric or categorical data) and features (numeric fields).

To use this implementation, you can the method `fit()` to create a model based in the training data and then, use the method `transform()` to predict the test data. 


All parameters are explained below:


**fit():**

```
  :param data:        A list with numFrag pandas's dataframe used to training the model.
  :param settings:    A dictionary that contains:
      - iters:            Maximum number of iterations (integer);
      - threshold:        Tolerance for stopping criterion (float);
      - regularization:   Regularization parameter (float);
      - alpha:            The Learning rate. How large of steps to take on our cost curve (float);
      - features:         Column name of the features in the training data;
      - label:            Column name of the labels   in the training data;
  :param numFrag:     A number of fragments;
  :return:            The model created (which is a pandas dataframe).
```

**transform():**

```
  :param data:       A list with numFrag pandas's dataframe that will be predicted.
  :param model:      The Logistic Regression model created;
  :param settings:   A dictionary that contains:
 	- features:      Column name of the features in the test data;
 	- predCol:       Alias to the new column with the labels predicted;
  :param numFrag:    A number of fragments;
  :return:           The prediction (in the same input format).
```


## Example:


```sh
from logisticRegression import *

lr = logisticRegression()
settings = dict()
settings['alpha'] = 0.01
settings['threshold'] = .003
settings['regularization'] = .001
settings['iters'] = 100
settings['label'] = 'label_col'
settings['features'] = 'feature_col'

model = lr.fit(data, settings, numFrag)

settings['predCol'] = 'Y-predicted'
output = lr.transform(data, model, settings, numFrag)


```
