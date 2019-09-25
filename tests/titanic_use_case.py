#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def use_case1():
    """
    In this problem statement, we will find the number of people who died
    or survived along with their gender and age.
    """
    df = pd.read_csv('./titanic.csv', sep='\t')
    from ddf_library.context import COMPSsContext
    # COMPSsContext().set_log(True)
    ddf1 = DDF().parallelize(df, num_of_parts='*')\
        .select(['Sex', 'Age', 'Survived'])\
        .dropna(['Sex', 'Age'], mode='REMOVE_ROW')\
        .replace({0: 'No', 1: 'Yes'}, subset=['Survived']).persist()

    ddf_women = ddf1.filter('(Sex == "female") and (Age >= 18)').\
        group_by(['Survived']).count(['*'], alias=['Women'])

    ddf_kids = ddf1.filter('Age < 18'). \
        group_by(['Survived']).count(['*'], alias=['Kids'])

    ddf_men = ddf1.filter('(Sex == "male") and (Age >= 18)'). \
        group_by(['Survived']).count(['Survived'], alias=['Men'])

    ddf_final = ddf_women\
        .join(ddf_men, key1=['Survived'], key2=['Survived'], mode='inner')\
        .join(ddf_kids, key1=['Survived'], key2=['Survived'], mode='inner')

    ddf_final.show()
    # COMPSsContext().context_status()


def use_case2():
    from ddf_library.functions.ml.feature import StringIndexer, StandardScaler
    from ddf_library.functions.ml.classification import LogisticRegression
    from ddf_library.functions.ml.evaluation import BinaryClassificationMetrics
    """
    In this challenge, we are asked to predict whether a passenger
    on the titanic would have been survived or not.

    Based in: https://towardsdatascience.com/predicting-
     the-survival-of-titanic-passengers-30870ccc7e8
    """
    df = pd.read_csv('./titanic.csv', sep='\t')

    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    def title_checker(row):
        from ddf_library.utils import col
        titles = {"Mr.": 1, "Miss": 2, "Mrs.": 3, "Master": 4, "Rare": 5}
        for title in titles:
            if title in row[col('Name')]:
                return titles[title]
        return -1

    def age_categorizer(row):
        from ddf_library.utils import col
        category = 7

        if row[col('Age')] <= 11:
            category = 0
        elif (row[col('Age')] > 11) and (row[col('Age')] <= 18):
            category = 1
        elif (row[col('Age')] > 18) and (row[col('Age')] <= 22):
            category = 2
        elif (row[col('Age')] > 22) and (row[col('Age')] <= 27):
            category = 3
        elif (row[col('Age')] > 27) and (row[col('Age')] <= 33):
            category = 4
        elif (row[col('Age')] > 33) and (row[col('Age')] <= 40):
            category = 5
        elif (row[col('Age')] > 40) and (row[col('Age')] <= 66):
            category = 6

        return category

    def fare_categorizer(row):
        from ddf_library.utils import col
        category = 5
        if row[col('Fare')] <= 7.91:
            category = 0
        elif (row[col('Fare')] > 7.91) and (row[col('Fare')] <= 14.454):
            category = 1
        elif (row[col('Fare')] > 14.454) and (row[col('Fare')] <= 31):
            category = 2
        elif (row[col('Fare')] > 31) and (row[col('Fare')] <= 99):
            category = 3
        elif (row[col('Fare')] > 99) and (row[col('Fare')] <= 250):
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
        .dropna(all_columns, how='any')\
        .replace({'male': 1, 'female': 0}, subset=['Sex'])\
        .map(title_checker, 'Name')\
        .map(age_categorizer, 'Age')\
        .map(fare_categorizer, 'Fare')

    ddf1 = StringIndexer()\
        .fit_transform(ddf1, input_col='Embarked', output_col='Embarked')

    """
    After that, we put together all columns (except Survived, which will be
    the label) in a feature vector and remove these old column.

    We normalize the features using Standard scaler and them, and divide data
    into one part with 70% and 30%.
    """

    # scaling using StandardScaler
    ddf1 = StandardScaler()\
        .fit_transform(ddf1, output_col=features, input_col=features)

    # 70% to train the model and 30% to test
    ddf_train, ddf_test = ddf1.split(0.7)

    print("Number of rows to fit the model:", ddf_train.count_rows())
    print("Number of rows to test the model:", ddf_test.count_rows())

    """
    Number of rows to fit the model: 88
    Number of rows to test the model: 37
    
    70% of data is used in the classifier (LogisticRegression) training stage.
    The others 30% is used to test the fitted model.
    """

    logr = LogisticRegression(feature_col=features, label_col='Survived',
                              max_iter=10).fit(ddf_train)

    ddf_test = logr.transform(ddf_test, pred_col='out_logr')\
        .select(['Survived', 'out_logr'])

    """
    This model can be evaluated by some binary metrics
    """

    metrics_bin = BinaryClassificationMetrics(label_col='Survived',
                                              true_label=1,
                                              pred_col='out_logr',
                                              ddf_var=ddf_test)

    ddf_train.show(10)
    ddf_test.show(10)

    print("Number of rows to fit the model:", ddf_train.count_rows())
    print("Number of rows to test the model:", ddf_test.count_rows())

    print("Metrics:\n", metrics_bin.get_metrics())
    print("\nConfusion Matrix:\n", metrics_bin.confusion_matrix)

    """
    Number of rows to fit the model: 88
    Number of rows to test the model: 37

    Metrics:
               Metric     Value
    0        Accuracy  0.850575
    1       Precision  0.653846
    2          Recall  0.809524
    3  F-measure (F1)  0.723404

    Confusion Matrix:
         0    1
    0   18    6
    1   4     9
    """


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            description="Titanic's use case")
    parser.add_argument('-o', '--operation',
                        type=int,
                        required=True,
                        help="""
                            1. use case1: ratio between women, men and kids
                            2. use case2: prediction of survived
                            """)
    arg = vars(parser.parse_args())

    operation = arg['operation']
    list_operations = [use_case1,
                       use_case2]
    list_operations[operation - 1]()
