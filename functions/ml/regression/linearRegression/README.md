# Linear Regression:

Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables and the single output variable. More specifically, that y can be calculated from a linear combination of the input variables (x).

When there is a single input variable (x), the method is referred to as simple linear regression. When there are multiple input variables, literature from statistics often refers to the method as multiple linear regression.

Datasets with more than two dimensions, the method of Stochastic Gradient Descent to learn the best options to the regression's coefficients.


## Instructions:

The algorithm reads a dataset composed by a column of features (array of numeric fields) in case of more than one dimension.

To use this implementation, you can the method `fit()` to create a model based in the training data and then, use the method `transform()` to predict the data.

Note: Best results with a normalizated data.


All parameters are explained below:


**fit():**



- :param data:      A list with numFrag pandas's dataframe
                                used to create the model.
- :param settings:  A dictionary that contains:
   - features: 	Field of the features in the dataset;
   - label: 	    Field of the label in the dataset;
   - mode:
     * 'simple': Best option if is a 2D regression;
     * 'SDG':    Uses a Stochastic gradient descent to perform  the regression. Can be used to data of all  dimensions.
   - max_iter:     Maximum number of iterations, only using 'SDG' (integer, default: 100);
   - alpha:        Learning rate parameter, only using 'SDG' (float, default 0.01)
   - :param numFrag:   A number of fragments;
   - :return:          Returns a model (which is a pandas dataframe).

            
**transform():**

- :param data:        A list with numFrag pandas's dataframe that will be predicted.
- :param model:		    The Linear Regression's model created;
- :param settings:    A dictionary that contains:
 	- features: 		    Field of the features in the test data;
 	- predCol:    	    Alias to the new column with the labels predicted;
- :param numFrag:     A number of fragments;
- :return:            The prediction (in the same input format).


## Example:


```sh
from functions.ml.regression.linearRegression.linearRegression import linearRegression

lr = linearRegression()

settings = dict()
settings['mode']     = "SDG" 
settings['max_iter'] = 100
settings['alpha']    = 0.001
settings['label']    = 'y_norm'
settings['features'] = 'x_norm'

model = lr.fit(input,settings,numFrag)

output = lr.transform(input,model,settings,numFrag)

```
