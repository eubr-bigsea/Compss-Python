# COMPSs-Python

A list of all COMPS's operations supported by Lemonade.

## Operations:

* [Data operations](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/data):
 	- Data Reader
 	- Data Writer
 	- Workload Balancer
	- Attributes Changer

* [ETL Operations](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/etl):
 	- Add Columns
 	- Aggregation
 	- Clean Missing
 	- Difference
 	- Distinct (Remove Duplicated Rows)
 	- Drop
 	- Intersection
 	- Join
 	- Replace Values
 	- Sample
 	- Select
 	- Sort
 	- Split
 	- Transform
 	- Union

* [Machine Learning Operations](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/ml):
 	- K-Means Clustering
 	- DBSCAN Clustering
 	- K-NN Classifier
 	- Naive Bayes Classifier
 	- Svm Classifier
	- Logistic Regression
 	- Linear Regression
 	- Apriori
	- Save Model
	- Load Model
	- PCA
	- Feature Assembler
	- MinMax Scaler
	- MaxAbs Scaler
    - Standard Scaler


* [Text Operations](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/text):
 	- Tokenizer 
 	- Convert Words to Vector (BoW, TF-IDF)
 	- Remove Stopwords
 	- StringIndexer

* [Metrics](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/ml/metrics):
 	* Classification Model Evaluation:
 	- Accuracy
 	- Precision
 	- Recall
 	- F-mesure
 	* Regression Model Evaluation:
 	- MSE
 	- RMSE
 	- MAE
 	- R2

* [Geografic Operations](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/geo):
 	- Read Shapefile
 	- Geo Within
 	- ST-DBSCAN

* [Graph Operations](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/graph):
 	- PageRank



# Requirements

Some functions use third-party libraries, install all the dependencies below in order to use them, this can be done using the command `$ pip install -r requirements.txt`.



```
Cython == 0.25.2
Pyqtree == 0.24
matplotlib == 1.5.1
networkx == 1.11
numpy == 1.11.0
pandas == 0.20.3
pyshp == 1.2.11
python_dateutil == 2.6.1
```
