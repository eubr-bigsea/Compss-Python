#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF
import pandas as pd


def use_case1():
    """
    In this problem statement, we will find the number of people who died
    or survived along with their gender and age.
    """
    df = pd.read_csv('tests/titanic.csv', sep='\t')

    ddf1 = DDF().parallelize(df, num_of_parts=4)\
        .select(['Sex', 'Age', 'Survived'])\
        .clean_missing(['Sex', 'Age'], mode='REMOVE_ROW')\
        .replace({0: 'No', 1: 'Yes'}, subset=['Survived']).cache()

    ddf_women = ddf1.filter('(Sex == "female") and (Age >= 18)').\
        aggregation(group_by=['Survived'],
                    exprs={'Survived': ['count']},
                    aliases={'Survived': ["Women"]})

    ddf_kids = ddf1.filter('Age < 18').\
        aggregation(group_by=['Survived'],
                    exprs={'Survived': ['count']},
                    aliases={'Survived': ["Kids"]})

    ddf_men = ddf1.filter('(Sex == "male") and (Age >= 18)').\
        aggregation(group_by=['Survived'],
                    exprs={'Survived': ['count']},
                    aliases={'Survived': ["Men"]})

    ddf_final = ddf_women\
        .join(ddf_men, key1=['Survived'], key2=['Survived'], mode='inner')\
        .join(ddf_kids, key1=['Survived'], key2=['Survived'], mode='inner')

    print ddf_final.show()


def use_case2():
    """
    In this challenge, we are asked to predict whether a passenger
    on the titanic would have been survived or not.

    Based in: https://towardsdatascience.com/predicting-
     the-survival-of-titanic-passengers-30870ccc7e8
    """
    df = pd.read_csv('tests/titanic.csv', sep='\t')

    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    def title_checker(row):
        for title in titles:
            if title in row['Name']:
                return titles[title]
        return -1

    def age_categorizer(row):
        category = 7

        if row['Age'] <= 11:
            category = 0
        elif (row['Age'] > 11) and (row['Age'] <= 18):
            category = 1
        elif (row['Age'] > 18) and (row['Age'] <= 22):
            category = 2
        elif (row['Age'] > 22) and (row['Age'] <= 27):
            category = 3
        elif (row['Age'] > 27) and (row['Age'] <= 33):
            category = 4
        elif (row['Age'] > 33) and (row['Age'] <= 40):
            category = 5
        elif (row['Age'] > 40) and (row['Age'] <= 66):
            category = 6

        return category

    def fare_categorizer(row):
        category = 5
        if row['Fare'] <= 7.91:
            category = 0
        elif (row['Fare'] > 7.91) and (row['Fare'] <= 14.454):
            category = 1
        elif (row['Fare'] > 14.454) and (row['Fare'] <= 31):
            category = 2
        elif (row['Fare'] > 31) and (row['Fare'] <= 99):
            category = 3
        elif (row['Fare'] > 99) and (row['Fare'] <= 250):
            category = 4
        return category

    """
    First of all, we need to remove some columns (Passenger id, Cabin number 
    and Ticket number) and remove rows with NaN values.
    After that, we want to convert the Sex column to numeric. In this case, 
    we know all possible values (male or female), so, we used a replace function
    to convert them. 
    Name, Age and Fare columns had their values categorized.
    And finally, we used a StringIndexer to convert Embarked to convert this 
    column to indexes.
    """
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    all_columns = features + ['Survived']

    ddf1 = DDF().parallelize(df, num_of_parts=4)\
        .drop(['PassengerId', 'Cabin', 'Ticket'])\
        .clean_missing(all_columns, mode='REMOVE_ROW')\
        .replace({'male': 1, 'female': 0}, subset=['Sex'])\
        .map(title_checker, 'Name')\
        .map(age_categorizer, 'Age')\
        .map(fare_categorizer, 'Fare')

    from ddf.functions.ml.feature import StringIndexer
    ddf1 = StringIndexer(input_col='Embarked',
                         output_col='Embarked').fit_transform(ddf1)

    """
    After that, we put together all columns (except Survived, which will be 
    the label) in a feature vector and remove these old column.
    
    We normalize the features using Standardscaler and them, and divide data 
    into one part with 70% and 30%.
    """

    # assembling a group of attributes as features and removing them after
    from ddf.functions.ml.feature import VectorAssembler
    assembler = VectorAssembler(input_col=features, output_col="features")
    ddf1 = assembler.transform(ddf1).drop(features)

    # scaling using StandardScaler
    from ddf.functions.ml.feature import StandardScaler
    ddf1 = StandardScaler(input_col='features', output_col='features')\
        .fit_transform(ddf1)

    # 70% to train the model and 30% to test
    ddf_train, ddf_test = ddf1.split(0.7)

    print "Number of rows to fit the model:", ddf_train.count()
    print "Number of rows to test the model:", ddf_test.count()

    """
    70% of data is used in the classifier (LogisticRegression) training stage.
    The others 30% is used to test the fitted model.
    """

    from ddf.functions.ml.classification import LogisticRegression
    logr = LogisticRegression(feature_col='features', label_col='Survived',
                              max_iters=10, pred_col='out_logr').fit(ddf_train)

    ddf_test = logr.transform(ddf_test).select(['Survived', 'out_logr'])

    """
    This model can be evaluated by some binary metrics
    """

    from ddf.functions.ml.evaluation import BinaryClassificationMetrics

    metrics_bin = BinaryClassificationMetrics(label_col='Survived',
                                              true_label=1.0,
                                              pred_col='out_logr',
                                              data=ddf_test)

    print ddf_train.show(10)
    print ddf_test.show(10)
    print "Number of rows to fit the model:", ddf_train.count()
    print "Number of rows to test the model:", ddf_test.count()

    print "Metrics:\n", metrics_bin.get_metrics()
    print "\nConfusion Matrix:\n", metrics_bin.confusion_matrix

    """
    Number of rows to fit the model: 87
    Number of rows to test the model: 38
    
    Metrics:
               Metric     Value
    0        Accuracy  0.850575
    1       Precision  0.653846
    2          Recall  0.809524
    3  F-measure (F1)  0.723404
    
    Confusion Matrix:
         0.0  1.0
    0.0   57    4
    1.0    9   17
    """


if __name__ == '__main__':
    print "_____Titanic's use case_____"
    # use_case1()
    use_case2()
