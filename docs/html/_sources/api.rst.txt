##################
API Reference
##################

.. automodule:: ddf_library



This page gives an overview of all public DDF objects, functions and methods. All classes and functions exposed in ddf namespace are public.

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Contents

   install
   guide
   ddf_library
   ddf_library.functions.ml.classification
   ddf_library.functions.ml.clustering
   ddf_library.functions.ml.evaluation
   ddf_library.functions.ml.feature
   ddf_library.functions.ml.fpm
   ddf_library.functions.ml.regression
   ddf_library.functions.graph


*********
Contents
*********


*  :ref:`etl-anchor`

*  :ref:`statistics-anchor`

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

:func:`DDF.add_column <ddf.DDF.add_column>` - Merges two dataFrames, column-wise.

:func:`DDF.cache <ddf.DDF.cache>` - Forces the computation of all tasks in the current stack.

:func:`DDF.cast <ddf.DDF.cast>` - Change the data's type of some columns.

:func:`DDF.distinct <ddf.DDF.distinct>` - Returns a new DDF with non duplicated rows.

:func:`DDF.drop <ddf.DDF.drop>` - Remove some columns from DDF.

:func:`DDF.dropna <ddf.DDF.dropna>` - Drop rows or columns with NaN elements.

:func:`DDF.except_all <ddf.DDF.except_all>` - Returns a new set with containing rows in the first frame but not in the second one while preserving duplicates.

:func:`DDF.fillna <ddf.DDF.fillna>` - Replace NaN elements by value or by median, mean or mode.

:func:`DDF.filter <ddf.DDF.filter>` - Filters elements based on a condition.

:func:`DDF.group_by <ddf.DDF.group_by>` - Returns a GroupedDFF with a set of methods for aggregations on a DDF.

:func:`DDF.intersect <ddf.DDF.intersect>` - Returns a new DDF containing rows in both DDF.

:func:`DDF.intersect_all <ddf.DDF.intersect_all>` - Returns a new DDF containing rows in both DDF while preserving duplicates.

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

:func:`DDF.subtract <ddf.DDF.subtract>` - Returns a new set with containing rows in the first frame but not in the second one.

:func:`DDF.take <ddf.DDF.take>` - Returns the first num rows.

:func:`DDF.map <ddf.DDF.map>` - Applies a function to each row of this data set.

:func:`DDF.union <ddf.DDF.union>` - Combines two DDF (concatenate) by column position.

:func:`DDF.union_by_name <ddf.DDF.union_by_name>` - Combines two DDF (concatenate) by column name.

:func:`DDF.with_column_renamed <ddf.DDF.with_column_renamed>` - Renames some columns.


.. _statistics-anchor:

Statistics
============

:func:`DDF.count <ddf.DDF.count>` - Returns a number of rows in this DDF.

:func:`DDF.correlation <ddf.DDF.correlation>` - Calculates the Pearson Correlation Coefficient.

:func:`DDF.covariance <ddf.DDF.covariance>` - Calculates the sample covariance for the given columns.

:func:`DDF.cross_tab <ddf.DDF.cross_tab>` - Computes a pair-wise frequency table of the given columns.

:func:`DDF.describe <ddf.DDF.describe>` - Computes basic statistics for numeric and string columns.

:func:`DDF.freq_items <ddf.DDF.freq_items>` - Finds frequent items for columns.



.. _ml-anchor:

ML
=========

Machine learning algorithms is divided in: classifiers, clusterings, feature extractor operations, frequent pattern mining algorithms, evaluators and regressors.

.. _classification-anchor:

ML.Classification
------------------

The **ml.classification** module includes some supervised classifiers algorithms:
        
:class:`ml.classification.KNearestNeighbors <ddf_library.functions.ml.classification.KNearestNeighbors>` - K-Nearest Neighbor is a algorithm used that can be used for both classification and regression predictive problems. However, it is more widely used in classification problems. To do a classification, the algorithm computes from a simple majority vote of the K nearest neighbors of each point present in the training set. The choice of the parameter K is very crucial in this algorithm, and depends on the dataset. However, values of one or tree is more commom.

:class:`ml.classification.GaussianNB <ddf_library.functions.ml.classification.GaussianNB>` - The Naive Bayes algorithm is an intuitive method that uses the probabilities of each attribute belonged to each class to make a prediction. It is a supervised learning approach that you would  come up with if you wanted to model a predictive probabilistically modeling problem.

:class:`ml.classification.LogisticRegression <ddf_library.functions.ml.classification.LogisticRegression>` - Logistic regression is named for the function used at the core of the method, the logistic function. It is the go-to method for binary classification problems (problems with two class values).

:class:`ml.classification.SVM <ddf_library.functions.ml.classification.SVM>` - Support vector machines (SVM) is a supervised learning model used for binary classification. Given a set of training examples, each marked as belonging to one or the other of two categories, a SVM training algorithm builds a model that assigns new points to one category or the other, making it a non-probabilistic binary linear classifier.


.. _clustering-anchor:

ML.Clustering
-------------

The **ml.clustering** module gathers popular unsupervised clustering algorithms:

:class:`ml.clustering.Kmeans <ddf_library.functions.ml.clustering.Kmeans>` - The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. 

:class:`ml.clustering.DBSCAN <ddf_library.functions.ml.clustering.DBSCAN>` - A density-based clustering algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

.. _evaluation-anchor:

ML.Evaluation
-------------

The **ml.evaluation** module includes score functions and performance metrics:

:class:`ml.evaluation.BinaryClassificationMetrics <ddf_library.functions.ml.evaluation.BinaryClassificationMetrics>` - Evaluator for binary classification. 

:class:`ml.evaluation.MultilabelMetrics <ddf_library.functions.ml.evaluation.MultilabelMetrics>` - Evaluator for multilabel classification.

:class:`ml.evaluation.RegressionMetrics <ddf_library.functions.ml.evaluation.RegressionMetrics>` - Evaluator for regression.

.. _feature-anchor:

ML.Feature
-----------

The **ml.feature** covers algorithms for working with features to extracting features from “raw” data, scaling, converting or modifying features:
Selection: Selecting a subset from a larger set of features:


:class:`ml.feature.VectorAssembler <ddf_library.functions.ml.feature.VectorAssembler>` - Vector Assembler is a transformer that combines a given list of columns into a single vector column.

:class:`ml.feature.VectorSlicer <ddf_library.functions.ml.feature.VectorSlicer>` - Vector Slicer create a new feature vector with a subarray of an original features.

:class:`ml.feature.Binarizer <ddf_library.functions.ml.feature.Binarizer>` - Binarize data (set feature values to 0 or 1) according to a threshold.

:class:`ml.feature.OneHotEncoder <ddf_library.functions.ml.feature.OneHotEncoder>` - Encode categorical integer features as a one-hot numeric array.

:class:`ml.feature.Tokenizer <ddf_library.functions.ml.feature.Tokenizer>` - Tokenization is the process of taking text (such as a sentence) and breaking it into individual terms (usually words). A simple Tokenizer class provides this functionality.

:class:`ml.feature.RegexTokenizer <ddf_library.functions.ml.feature.RegexTokenizer>` - A regex based tokenizer that extracts tokens either by using the provided regex pattern (in Java dialect) to split the text.

:class:`ml.feature.RemoveStopWords <ddf_library.functions.ml.feature.RemoveStopWords>` - Remove stop-words is a operation to remove words which should be excluded from the input, typically because the words appear frequently and don’t carry as much meaning.

:class:`ml.feature.NGram <ddf_library.functions.ml.feature.NGram>` - A feature transformer that converts the input array of strings into an array of n-grams.

:class:`ml.feature.CountVectorizer <ddf_library.functions.ml.feature.CountVectorizer>` - Converts a collection of text documents to a matrix of token counts.

:class:`ml.feature.TfidfVectorizer <ddf_library.functions.ml.feature.TfidfVectorizer>` - Term frequency-inverse document frequency (TF-IDF) is a numerical statistic transformation that is intended to reflect how important a word is to a document in a collection or corpus.

:class:`ml.feature.StringIndexer <ddf_library.functions.ml.feature.StringIndexer>` - StringIndexer indexes a feature by encoding a string column as a column containing indexes.

:class:`ml.feature.IndexToString <ddf_library.functions.ml.feature.IndexToString>` - Symmetrically to StringIndexer, IndexToString maps a column of label indices back to a column containing the original labels as strings.

:class:`ml.feature.MaxAbsScaler <ddf_library.functions.ml.feature.MaxAbsScaler>` - MaxAbsScaler transforms a dataset of features rows, rescaling each feature to range [-1, 1] by dividing through the maximum absolute value in each feature. 

:class:`ml.feature.MinMaxScaler <ddf_library.functions.ml.feature.MinMaxScaler>` - MinMaxScaler transforms a dataset of features rows, rescaling each feature to a specific range (often [0, 1]).

:class:`ml.feature.StandardScaler <ddf_library.functions.ml.feature.StandardScaler>` - StandardScaler transforms a dataset of features rows, reascaling each feature by the standard score.

:class:`ml.feature.PCA <ddf_library.functions.ml.feature.PCA>` - Principal component analysis (PCA) is used widely in dimensionality reduction.

:class:`ml.feature.PolynomialExpansion <ddf_library.functions.ml.feature.PolynomialExpansion>` - Perform feature expansion in a polynomial space..


.. _fpm-anchor:

ML.Frequent Pattern Mining
----------------------------

:class:`ml.fpm.Apriori <ddf.functions.ml.fpm.Apriori>` - Apriori is a algorithm to find frequent item sets.

:class:`ml.feature.AssociationRules <ddf_library.functions.ml.feature.AssociationRules>` - AssociationRules implements a parallel rule generation algorithm for constructing rules that have a single item as the consequent.

.. _regression-anchor:

ML.Regression
--------------

:class:`ml.regression.LinearRegression <ddf_library.functions.ml.regression.LinearRegression>` - Linear Regression using method of least squares (works only for 2-D data) or using Stochastic Gradient Descent.


.. _geo-anchor:

Geographic Operations
=====================

:func:`DDF.load_shapefile <ddf_library.DDF.load_shapefile>` - Create a DDF from a shapefile.

:func:`DDF.geo_within <ddf_library.DDF.geo_within>` - Returns the sectors that the each point belongs.


.. _graph-anchor:

Graph Algorithms
================

:class:`graph.PageRank <ddf_library.functions.graph.PageRank>` - Perform PageRank.

