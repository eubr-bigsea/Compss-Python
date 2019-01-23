# PyCOMPSs Distributed DataFrame (DDF)

DDF is a lightweight library for [PyCOMPSs](https://pypi.org/project/pycompss/)
developers which contains ETL (extract-transform-load), machine learning, text and geographic operations. 
The main purpose of this library is to avoid implementations of simple 'task' functions by developers. 
DDF is trustful and it processes the data in the most adequate way in terms of parallelism.


### TO DO:

 - document all tasks
 - collect
 - COMPSsContext
 - add more functions
 - increase the optimizations guidelines
 - test, test and test it !
 
### Example of use:

```
from ddf import DDF, COMPSsContext

data1 = DDF().load_fs('/flights.csv', num_of_parts=4)\
             .transform(lambda col: col['Year']-2000, 'new_Year')\
             .drop(['Year'])\
             .filter('(CRSDepTime > 750)') \
             .split(0.5)

data2 = data1[0].sample(10).collect()
data3 = data1[1].sample(8).collect()

COMPSsContext().run()

print data2.toPandas()
print data3.toPandas()
 
```

### Operations: 

* [ETL Operations](https://github.com/eubr-bigsea/Compss-Python/tree/master/functions/etl):
 	- Add Columns
 	- Aggregation
 	- Attributes Changer
 	- Clean Missing
 	- Data Reader
 	- Data Writer
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



### Requirements

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