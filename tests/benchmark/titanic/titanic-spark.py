#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import sys

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import StorageLevel

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# Data generation
block = ''
with open('base_titanic.csv', 'r') as content_file:
    block = content_file.read()
size_block = float(sys.argv[1])*1024*1024 # in bytes
num_partitions = int(sys.argv[2]) 
print("Num partitions: ", num_partitions)
size_original = 61113
n_repeat  = int(math.ceil(size_block/size_original))
def splitter(x):
    x = pd.read_csv(StringIO(x), sep=',').values.tolist()
    return x
block = block * n_repeat
t1 = time.time()
rdd = sc.parallelize(range(num_partitions), numSlices=num_partitions).flatMap(lambda x: splitter(block))

mySchema = StructType([StructField("PassengerId", IntegerType(), True),
                       StructField("Survived", IntegerType(), True),
                       StructField("Pclass", IntegerType(), True),
                       StructField("Name", StringType(), True),
                       StructField("Sex", StringType(), True),
                       StructField("Age", DoubleType(), True),
                       StructField("SibSp", IntegerType(), True),
                       StructField("Parch", IntegerType(), True),
                       StructField("Ticket", StringType(), True),
                       StructField("Fare", DoubleType(), True),
                       StructField("Cabin", StringType(), True),
                       StructField("Embarked", StringType(), True)])

df = spark.createDataFrame(rdd, schema=mySchema)

titles = {"Mr.": 1, "Miss": 2, "Mrs.": 3, "Master": 4, "Rare": 5}

def title_checker(value):
    for title in titles:
        if title in value:
            return titles[title]
    return -1

def age_categorizer(value):
    category = 7

    if value <= 11:
        category = 0
    elif (value > 11) and (value <= 18):
        category = 1
    elif (value > 18) and (value <= 22):
        category = 2
    elif (value > 22) and (value <= 27):
        category = 3
    elif (value > 27) and (value <= 33):
        category = 4
    elif (value > 33) and (value <= 40):
        category = 5
    elif (value > 40) and (value <= 66):
        category = 6

    return category

def fare_categorizer(value):
    category = 5
    if value <= 7.91:
        category = 0
    elif (value > 7.91) and (value <= 14.454):
        category = 1
    elif (value > 14.454) and (value <= 31):
        category = 2
    elif (value > 31) and (value <= 99):
        category = 3
    elif (value > 99) and (value <= 250):
        category = 4
    return category


def sex_replacer(value):
    if 'male' == value:
        return 1
    else:
        return 0

cat1 =  udf (lambda x: sex_replacer(x), IntegerType())
cat2 =  udf (lambda x: title_checker(x), IntegerType())
cat3 =  udf (lambda x: age_categorizer(x), IntegerType())
cat4 =  udf (lambda x: fare_categorizer(x), IntegerType())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

clean_df = df\
            .drop('PassengerId', 'Cabin', 'Ticket')\
            .dropna(subset=features)\
            .withColumn('Sex', cat1(df.Sex))\
            .withColumn('Name', cat2(df.Name))\
            .withColumn('Age', cat3(df.Age))\
            .withColumn('Fare', cat4(df.Fare))

clean_df = StringIndexer(inputCol="Embarked", outputCol="EmbarkedC")\
    .fit(clean_df)\
    .transform(clean_df)\
    .drop('Embarked')

features[6] = 'EmbarkedC'
features +=['Survived', 'Name']

from pyspark.ml.feature import VectorAssembler

clean_df = VectorAssembler(inputCols=features, outputCol="features")\
    .transform(clean_df)\
    .drop('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'EmbarkedC', 'Survived', 'Name')

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)

clean_df = scaler.fit(clean_df).transform(clean_df).drop('features')
catf =  udf (lambda x: ', '.join([str(f) for f in x.toArray()]), StringType())
clean_df.withColumn('scaledFeatures', catf(col('scaledFeatures'))).write.save('/titanic', format='csv', mode='overwrite')

