#########################
DDF vs PySpark DataFrame
#########################

Some DDF functions have interfaces similar to the PySpark DataFrame to help new users who want to migrate to the COMPSs.

The following tables show some of these correspondences.

ETL
====

+-------------------+-----------------+
| PySpark DataFrame | DDF             |
+===================+=================+
| parallelize       |  parallelize    |
+-------------------+-----------------+
| map               |  map            |
+-------------------+-----------------+
| cache             | cache           |
+-------------------+-----------------+
| count             | count           |
+-------------------+-----------------+
| distinct          | distinct        |
+-------------------+-----------------+
| drop              | drop            |
+-------------------+-----------------+
| dropDuplicates    | drop_duplicates |
+-------------------+-----------------+
| fillna            | clean_missing   |
+-------------------+-----------------+
| filter            | filter          |
+-------------------+-----------------+
| groupBy.agg       | aggregation     |
+-------------------+-----------------+
| intersect         | intersect       |
+-------------------+-----------------+
| join              | join            |
+-------------------+-----------------+
| randomSplit       | split           |
+-------------------+-----------------+
| replace           | replace         |
+-------------------+-----------------+
| sample            | sample          |
+-------------------+-----------------+
| select            | select          |
+-------------------+-----------------+
| show              | show            |
+-------------------+-----------------+
| sort              | sort            |
+-------------------+-----------------+
| take              | take            |
+-------------------+-----------------+
| union             | union           |
+-------------------+-----------------+
| withColumn        | with_column     |
+-------------------+-----------------+
| read.text         | load_text       |
+-------------------+-----------------+
| write             | save            |  
+-------------------+-----------------+

Machine Learning
=================

+----------------------------+-----------------------+
| PySpark DataFrame          | DDF                   |
+============================+=======================+
| VectorAssembler            | VectorAssembler       |
+----------------------------+-----------------------+
| TF-IDF                     |  TF-IDF               |
+----------------------------+-----------------------+
| CountVectorizer            | CountVectorizer       |
+----------------------------+-----------------------+
| Tokenizer                  | Tokenizer             |
+----------------------------+-----------------------+
| StopWordsRemover           | RemoveStopWords       |
+----------------------------+-----------------------+
| PCA                        | PCA                   |
+----------------------------+-----------------------+
| StringIndexer              | StringIndexer         |
+----------------------------+-----------------------+
| IndexToString              | IndexToString         |
+----------------------------+-----------------------+
| StandardScaler             | StandardScaler        |
+----------------------------+-----------------------+
| MaxAbsScaler               | MaxAbsScaler          |
+----------------------------+-----------------------+
| MinMaxScaler               | MinMaxScaler          |
+----------------------------+-----------------------+
| SVMWithSGD                 | SVM                   |
+----------------------------+-----------------------+
| LogisticRegressionWithSGD  | LogisticRegression    |
+----------------------------+-----------------------+
| NaiveBayes                 | Gaussian Naive Bayes  |
+----------------------------+-----------------------+
| LinearRegressionWithSGD    | LinearRegression      |
+----------------------------+-----------------------+
| K-means                    | K-Means               |
+----------------------------+-----------------------+
| AssociationRules           | AssociationRules      |
+----------------------------+-----------------------+



