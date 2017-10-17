# K-Means:

K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.

Two of the most well-known forms of initialization of the set of clusters are: "random" and "k-means||" (Bahmani et al., Scalable K-Means++, VLDB 2012):

* random: Starting with a set of randomly chosen initial centers;
* k-means||: This is a variant of k-means++ that tries to find dissimilar cluster centers by starting with a random center and then doing passes where more centers are chosen with probability proportional to their squared distance to the current cluster set. It results in a provable approximation to an optimal clustering.



## Instructions:

The algorithm reads a dataset composed by a column of features (array of numeric fields).

To use this implementation, you can the method `fit()` to create a model based in the training data and then, use the method `transform()` to predict the data.


All parameters are explained below:


**fit():**

- :param data:        A list with numFrag pandas's dataframe used to create the model.
- :param settings:    A dictionary that contains:
  - k:  			  Number of wanted clusters (default, 2).
  - features: 	  Field of the features in the dataset;
  - maxIterations:  Maximum number of iterations (default, 100);
  - epsilon:        Threshold to stop the iterations (default, 0.001);
  - initMode:       "random" or "k-means||" (default, 'k-means||')
- :param numFrag:     A number of fragments;
- :return:            The model created (which is a pandas dataframe).

**transform():**

- :param data:        A list with numFrag pandas's dataframe that will be predicted.
- :param model:		    The Kmeans model created;
- :param settings:    A dictionary that contains:
 	- features: 		    Field of the features in the test data;
 	- predCol:    	    Alias to the new column with the labels predicted;
- :param numFrag:     A number of fragments;
- :return:            The prediction (in the same input format).


## Example:


```sh
from functions.ml.clustering.Kmeans.Kmeans import *

kmeans = Kmeans()
numFrag = 4
settings = dict()
settings['k'] = 2
settings['maxIterations'] = 1000
settings['epsilon'] = 0.0001
settings['initMode'] = 'k-means||'

settings['features'] = 'Features_Col'
settings['predCol'] = 'Clusters'
model  = kmeansl.fit(data1, settings, numFrag)
output = kmeans.transform(data1, model, settings, numFrag)


```
