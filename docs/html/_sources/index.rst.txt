

.. image:: ./ddf-logo.png
   :height: 90px
   :align: center

|
|
|

The Distributed DataFrame Library provides distributed algorithms and operations ready to use as a library implemented over PyCOMPSs programming model. Currently, it is highly focused on ETL (extract-transform-load) and Machine Learning algorithms to Data Science tasks. DDF is greatly inspired by Spark's DataFrame and its operators.

Currently, an operation can be of two types, transformations or actions. Action operations are those that produce a final result (whether to save to a file or to display on screen). Transformation operations are those that will transform an input DDF into another output DDF. Besides this classification, there are operations with one processing stage and those with two or more stages of processing (those that need to exchange information between the partitions).

When running DDF operation/algorithms, a context variable (COMPSs Context) will check the possibility of optimizations during the scheduling of COMPS tasks. These optimizations can be of the type: grouping one stage operations to a single task COMPSs and stacking operations until an action operation is found.

|
|

.. list-table:: **Summary of algorithms and operations**
   :widths: 25 75
   :header-rows: 1

   * - Category
     - Description
   * - ETL
     - Add Columns, Aggregation, Change attribute, Clean Missing Values, Difference, Distinct (Remove Duplicated Rows), Drop Columns, Set-Intersect, Join (Inner, Left or Right), Load Data, Replace Values, Sample Rows, Save data, Select Columns, Sort, Split, Transform (Map operations), Union  
   * - ML.Classification
     - K-Nearest Neighbors, Gaussian Naive Bayes, Logistic Regression, SVM
   * - ML.Clustering
     - K-means (using random or k-means|| initialization method), DBSCAN
   * - ML.Evaluator Models
     - Binary Classification Metrics, Multi-label Metrics and Regression Metrics
   * - ML.Feature Operations
     - Vector Assembler, Simple Tokenizer, Regex Tokenizer, Remove Stop-Words, Count Vectorizer, Tf-idf Vectorizer, String Indexer, Index To String, Max-Abs Scaler, Min-Max Scaler, Standard Scaler, PCA
   * - ML.Frequent Pattern Mining
     - Apriori and Association Rules
   * - ML.Regression
     - Linear Regression using method of least squares (works only for 2-D data) or using Stochastic Gradient Descent
   * - Geographic Operations
     - Load data from shapefile, Geo Within (Select rows that exists within a specified shape)
   * - Graph Operations
     - Initially, only Page Rank are present
         

Contents
---------

* :doc:`Installation <install>`
* :doc:`Example of use: Titanic <guide>`
* :doc:`DDF vs PySpark DataFrame <comparison>`
* :doc:`API Reference <api>`


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents

   install.rst
   guide.rst
   comparison.rst
   api.rst



Q&A Support
------------------

For further questions, please submit a `issue <https://github.com/eubr-bigsea/Compss-Python/issues>`_.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

