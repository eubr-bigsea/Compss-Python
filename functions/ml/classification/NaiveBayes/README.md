# Gaussian Naive Bayes

The Naive Bayes algorithm is an intuitive method that uses the probabilities of each attribute belonged to each class to make a prediction. It is a supervised learning approach that you would  come up with if you wanted to model a predictive probabilistically modeling problem.

Naive bayes simplifies the calculation of probabilities by assuming that the probability of each attribute belonging to a given class value is independent of all other attributes. The probability of a class value given a value of an attribute is called the conditional probability. By multiplying the conditional probabilities together for each attribute for a given class value, we have the probability of a data instance belonging to that class.

To make a prediction we can calculate probabilities of the instance belonged to each class and select the class value with the highest probability.

## Instructions:

To use this implementation, you can the method `fit()` to create a model based in the training data and then, use the method `transform()` to predict the test data.

All parameters are explained below:

**fit():**

- :param data:        A list with numFrag pandas's dataframe used to training the model.
- :param settings:    A dictionary that contains:
 	- features: 		   Column name of the features in the training data;
 	- label:          	 Column name of the labels   in the training data;
- :param numFrag:     A number of fragments;
- :return:            The model created (which is a pandas dataframe).

**transform():**

- :param data: A list with numFrag pandas's dataframe that will be predicted.
- :param model: A model already trained (np.array);
- :param settings: A dictionary that contains:
 	- features: Column name of the features in the test data;
 	- predCol: Alias to the new column with the labels predicted;
- :param numFrag: A number of fragments;
- :return: The prediction (in the same input format).


## Example:


```sh
from functions.ml.classification.NaiveBayes.naivebayes import *

numFrag = 4
nb = GaussianNB()
settings = dict()
settings['label']     = 'column1'
settings['features']  = 'column2'


model = nb.fit(data1,settings,numFrag)
settings = dict()
settings['features']  = 'col1'
settings['predCol'] = 'result_data2'
output 	= nb.tranform(data2,model,settings,numFrag)

```
