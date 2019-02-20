##################
API Reference
##################

.. automodule:: ddf



This page gives an overview of all public DDF objects, functions and methods. All classes and functions exposed in ddf namespace are public.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents

   install.rst
   guide.rst
   ddf.rst
   ddf.functions.ml.classification.rst
   ddf.functions.ml.clustering.rst
   ddf.functions.ml.evaluation.rst
   ddf.functions.ml.feature.rst
   ddf.functions.ml.fpm.rst
   ddf.functions.ml.regression.rst
   ddf.functions.graph.rst



*********
Contents
*********


*  :ref:`etl-anchor`

* :ref:`ml-anchor`

 * :ref:`classification-anchor`

 * :ref:`clustering-anchor`

 * :ref:`feature-anchor`

 * :ref:`fpm-anchor`

 * :ref:`evaluation-anchor`

 * :ref:`regression-anchor`

* :ref:`geo-anchor`

* :ref:`graph-anchor`


.. _etl-anchor:

ETL
====

:func:`DDF.add_column <ddf.DDF.add_column>` - Merges two dataframes, column-wise.

:func:`DDF.aggreation <ddf.DDF.aggregation>` - Computes aggregates and returns the result as a DDF.

:func:`DDF.cache <ddf.DDF.cache>` - Forces the computation of all tasks in the current stack.

:func:`DDF.clean_missing <ddf.DDF.clean_missing>` - Cleans missing rows or columns fields.

:func:`DDF.count <ddf.DDF.count>` - Returns a number of rows in this DDF.

:func:`DDF.difference <ddf.DDF.difference>` - Returns a new set with containing rows in the first frame but not in the second one.

:func:`DDF.distinct <ddf.DDF.distinct>` - Returns a new DDF with non duplicated rows.

:func:`DDF.drop <ddf.DDF.drop>` - Remove some columns from DDF.

:func:`DDF.filter <ddf.DDF.filter>` - Filters elements based on a condition.

:func:`DDF.intersect <ddf.DDF.intersect>` - Returns a new DDF containing rows in both DDF.

:func:`DDF.join <ddf.DDF.join>` - Joins two DDF using the given join expression.

:func:`DDF.load_text <ddf.DDF.load_text>` - Create a DDF from a commom file system or from HDFS.

:func:`DDF.parallelize <ddf.DDF.parallelize>` - Distributes a DataFrame into DDF.

:func:`DDF.replace <ddf.DDF.replace>` - Replaces one or more values to new ones.

:func:`DDF.sample <ddf.DDF.sample>` - Returns a sampled subset.

:func:`DDF.save <ddf.DDF.save>` - Saves the data in the storage.

:func:`DDF.select <ddf.DDF.select>` - Performs a projection of selected columns.

:func:`DDF.show <ddf.DDF.show>` - Collect the current DDF into a single DataFrame.

:func:`DDF.sort <ddf.DDF.sort>` - Returns a sorted DDF by the specified column(s).

:func:`DDF.split <ddf.DDF.split>` - Randomly splits a DDF into two DDF.

:func:`DDF.take <ddf.DDF.take>` - Returns the first num rows.

:func:`DDF.transform <ddf.DDF.transform>` - Applies a function to each row of this data set.

:func:`DDF.union <ddf.DDF.union>` - Combines two DDF (concatenate).

:func:`DDF.with_column <ddf.DDF.with_column>` - Renames or changes the data's type of some columns.


.. _ml-anchor:

ML
=========

Machine learning algorithms is divided in: classifiers, clusterings, feature extractor operations, frequent pattern mining algorithms, evaluators and regressors.

.. _classification-anchor:

ML.Classification
------------------

The **ml.classification** module includes some supervised classifiers algorithms:
        
:class:`ml.classification.KNearestNeighbors <ddf.functions.ml.classification.KNearestNeighbors>` - K-Nearest Neighbor is a algorithm used that can be used for both classification and regression predictive problems. However, it is more widely used in classification problems. To do a classification, the algorithm computes from a simple majority vote of the K nearest neighbors of each point present in the training set. The choice of the parameter K is very crucial in this algorithm, and depends on the dataset. However, values of one or tree is more commom.

:class:`ml.classification.GaussianNB <ddf.functions.ml.classification.GaussianNB>` - The Naive Bayes algorithm is an intuitive method that uses the probabilities of each attribute belonged to each class to make a prediction. It is a supervised learning approach that you would  come up with if you wanted to model a predictive probabilistically modeling problem.

:class:`ml.classification.LogisticRegression <ddf.functions.ml.classification.LogisticRegression>` - Logistic regression is named for the function used at the core of the method, the logistic function. It is the go-to method for binary classification problems (problems with two class values).

:class:`ml.classification.SVM <ddf.functions.ml.classification.SVM>` - Support vector machines (SVM) is a supervised learning model used for binary classification. Given a set of training examples, each marked as belonging to one or the other of two categories, a SVM training algorithm builds a model that assigns new points to one category or the other, making it a non-probabilistic binary linear classifier.


.. _clustering-anchor:

ML.Clustering
-------------

The **ml.clustering** module gathers popular unsupervised clustering algorithms:

:class:`ml.clustering.Kmeans <ddf.functions.ml.clustering.Kmeans>` - The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. 

:class:`ml.clustering.DBSCAN <ddf.functions.ml.clustering.DBSCAN>` - A density-based clustering algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

.. _evaluation-anchor:

ML.Evaluation
-------------

The **ml.evaluation** module includes score functions and performance metrics:

:class:`ml.evaluation.BinaryClassificationMetrics <ddf.functions.ml.evaluation.BinaryClassificationMetrics>` - Evaluator for binary classification. 

:class:`ml.evaluation.MultilabelMetrics <ddf.functions.ml.evaluation.MultilabelMetrics>` - Evaluator for multilabel classification.

:class:`ml.evaluation.RegressionMetrics <ddf.functions.ml.evaluation.RegressionMetrics>` - Evaluator for regression.

.. _feature-anchor:

ML.Feature
-----------

The **ml.feature** covers algorithms for working with features to extracting features from “raw” data, scaling, converting or modifying features:
Selection: Selecting a subset from a larger set of features:


:class:`ml.feature.VectorAssembler <ddf.functions.ml.feature.VectorAssembler>` - Vector Assembler is a transformer that combines a given list of columns into a single vector column.

:class:`ml.feature.Tokenizer <ddf.functions.ml.feature.Tokenizer>` - Tokenization is the process of taking text (such as a sentence) and breaking it into individual terms (usually words). A simple Tokenizer class provides this functionality.

:class:`ml.feature.RegexTokenizer <ddf.functions.ml.feature.RegexTokenizer>` - A regex based tokenizer that extracts tokens either by using the provided regex pattern (in Java dialect) to split the text.

:class:`ml.feature.RemoveStopWords <ddf.functions.ml.feature.RemoveStopWords>` - Remove stop-words is a operation to remove words which should be excluded from the input, typically because the words appear frequently and don’t carry as much meaning.

:class:`ml.feature.CountVectorizer <ddf.functions.ml.feature.CountVectorizer>` - Converts a collection of text documents to a matrix of token counts.

:class:`ml.feature.TfidfVectorizer <ddf.functions.ml.feature.TfidfVectorizer>` - Term frequency-inverse document frequency (TF-IDF) is a numerical statistic transformation that is intended to reflect how important a word is to a document in a collection or corpus.

:class:`ml.feature.StringIndexer <ddf.functions.ml.feature.StringIndexer>` - StringIndexer indexes a feature by encoding a string column as a column containing indexes.

:class:`ml.feature.IndexToString <ddf.functions.ml.feature.IndexToString>` - Symmetrically to StringIndexer, IndexToString maps a column of label indices back to a column containing the original labels as strings.

:class:`ml.feature.MaxAbsScaler <ddf.functions.ml.feature.MaxAbsScaler>` - MaxAbsScaler transforms a dataset of features rows, rescaling each feature to range [-1, 1] by dividing through the maximum absolute value in each feature. 

:class:`ml.feature.MinMaxScaler <ddf.functions.ml.feature.MinMaxScaler>` - MinMaxScaler transforms a dataset of features rows, rescaling each feature to a specific range (often [0, 1]).

:class:`ml.feature.StandardScaler <ddf.functions.ml.feature.StandardScaler>` - StandardScaler transforms a dataset of features rows, reascaling each feature by the standard score.

:class:`ml.feature.PCA <ddf.functions.ml.feature.PCA>` - Principal component analysis (PCA) is used widely in dimensionality reduction.


.. _fpm-anchor:

ML.Frequent Pattern Mining
----------------------------

:class:`ml.fpm.Apriori <ddf.functions.ml.fpm.Apriori>` - Apriori is a algorithm to find frequent item sets.

:class:`ml.feature.AssociationRules <ddf.functions.ml.feature.AssociationRules>` - AssociationRules implements a parallel rule generation algorithm for constructing rules that have a single item as the consequent.

.. _regression-anchor:

ML.Regression
--------------

:class:`ml.regression.LinearRegression <ddf.functions.ml.regression.LinearRegression>` - Linear Regression using method of least squares (works only for 2-D data) or using Stochastic Gradient Descent.


.. _geo-anchor:

Geographic Operations
=====================

:func:`DDF.load_shapefile <ddf.DDF.load_shapefile>` - Create a DDF from a shapefile.

:func:`DDF.geo_within <ddf.DDF.geo_within>` - Returns the sectors that the each point belongs.


.. _graph-anchor:

Graph Algorithms
================

:class:`graph.PageRank <ddf.functions.graph.PageRank>` - Perform PageRank.

