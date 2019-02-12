#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF
import pandas as pd
import numpy as np


def ml_feature_text_operations():

    from ddf.functions.ml.feature import Tokenizer, RemoveStopWords, \
        CountVectorizer, TfidfVectorizer

    df = pd.DataFrame.from_dict({'col': ['and', 'conan', 'What', 'rare']})
    stopswords = DDF().parallelize(df, num_of_parts=2).select(['col'])

    data1 = DDF()\
        .load_fs('/text_data.txt', num_of_parts=4, header=False, sep='\n')\
        .transform(lambda row: row['col_0'], 'col_1')

    tokenized = Tokenizer(input_col='col_0').transform(data1)

    remover = RemoveStopWords(input_col='col_0',
                              output_col='col_1',
                              stops_words_list=['rock', 'destined'])

    remover = remover.stopwords_from_ddf(stopswords, 'col')
    result = remover.transform(tokenized)

    counter = CountVectorizer(input_col='col_1',
                              output_col='col_2', min_tf=0).fit(result)

    result1 = counter.transform(result)
    df1 = result1.cache().toPandas()

    corpus = [
             'This is the first document',
             'This document is the second document',
             'And this is the third one',
             'Is this the first document',
        ]
    df = pd.DataFrame.from_dict({'col_0': corpus})

    test_data = DDF().parallelize(df, num_of_parts=2)

    tokenized = Tokenizer(input_col='col_0', min_token_length=1).transform(
    test_data)

    counter = CountVectorizer(input_col='col_0',
                              output_col='col_1', binary=False)\
        .fit(tokenized)

    counter = TfidfVectorizer(input_col='col_0', output_col='col_2')\
        .fit(tokenized)
    result = counter.transform(tokenized)
    df2 = result.cache().toPandas()

    from ddf.functions.ml.feature import StringIndexer, IndexToString
    data = pd.DataFrame([(0, "a"), (1, "b"), (2, "c"),
                         (3, "a"), (4, "a"), (5, "c")],
                        columns=["id", "category"])

    data = DDF().parallelize(data, 4).select(['id', 'category'])

    model = StringIndexer(input_col='category').fit(data)

    converted = model.transform(data)

    result = IndexToString(input_col='category_indexed', model=model) \
        .transform(converted).drop(['id']).cache().toPandas()

    print df1
    print df2
    print "RESULT :", result


def ml_classifiers():

    from ddf.functions.ml.feature import VectorAssembler
    from ddf.functions.ml.classification import KNearestNeighbors, \
        LogisticRegression, GaussianNB, SVM
    from ddf.functions.ml.evaluation import MultilabelMetrics, \
        BinaryClassificationMetrics

    dataset = DDF().load_fs('/iris_test.data', num_of_parts=4)
    assembler = VectorAssembler(input_col=["x", "y"], output_col="features")
    assembled = assembler.transform(dataset).drop(['x', 'y'])

    knn = KNearestNeighbors(k=1, feature_col='features', label_col='label')\
        .fit(assembled)
    classified_knn = knn.transform(assembled)

    svm = SVM(feature_col='features', label_col='label',
              max_iters=10).fit(classified_knn)
    classified_svm = svm.transform(classified_knn)

    logr = LogisticRegression(feature_col='features', label_col='label')\
        .fit(classified_svm)
    classified_logr = logr.transform(classified_svm)

    nb = GaussianNB(feature_col='features', label_col='label')\
        .fit(assembled)
    classified = nb.transform(classified_logr).cache().toPandas()

    metrics_bin = BinaryClassificationMetrics(label_col='label',
                                              true_label=1.0,
                                              pred_col='prediction_kNN',
                                              data=classified_knn)

    metrics_multi = MultilabelMetrics(label_col='label',
                                      pred_col='prediction_kNN',
                                      data=classified_knn)

    print classified[0:20]
    print metrics_bin.get_metrics()
    print metrics_multi.get_metrics()
    print metrics_multi.confusion_matrix
    print metrics_multi.precision_recall


def ml_clustering():
    from ddf.functions.ml.feature import VectorAssembler
    from ddf.functions.ml.clustering import Kmeans

    data = pd.DataFrame([[1.0, 2.0], [1.0, 4.0], [1.0, 0], [4.0, 2.0],
                         [4.0, 4.0], [4.0, 0.0]], columns=['x', 'y'])

    dataset = DDF().parallelize(data, 4)
    assembler = VectorAssembler(input_col=["x", "y"], output_col="features")
    assembled = assembler.transform(dataset).drop(['x', 'y'])

    kmeans = Kmeans(feature_col='features', n_clusters=2,
                    init_mode='random').fit(assembled)

    clustered = kmeans.transform(assembled)
    df = clustered.cache().toPandas()
    print "RESULT :", df[:20]


def ml_regression():
    from ddf.functions.ml.feature import VectorAssembler
    from ddf.functions.ml.regression import LinearRegression
    from ddf.functions.ml.evaluation import RegressionMetrics

    dataset = DDF().load_fs('/iris_test.data', num_of_parts=4)
    assembler = VectorAssembler(input_col=["x", "label"],
                                output_col="features")
    assembled = assembler.transform(dataset)

    linearreg = LinearRegression('features', 'y', max_iter=15)
    model = linearreg.fit(assembled)
    regressed = model.transform(assembled).select(['pred_LinearReg']).toPandas()

    data = pd.DataFrame([[14, 70, 2, 3.3490],
                         [16, 75, 5, 3.7180],
                         [27, 144, 7, 6.8472],
                         [42, 190, 9, 9.8400],
                         [39, 210, 10, 10.0151],
                         [50, 235, 13, 11.9783],
                         [83, 400, 20, 20.2529],
                         ], columns=['x', 'z', 'y', 'pred_LinearReg'])
    dataset = DDF().parallelize(data, 4)

    assembler = VectorAssembler(input_col=["x", "z"],
                                output_col="features")
    assembled = assembler.transform(dataset)

    metrics = RegressionMetrics(col_features='features', label_col='y',
                                pred_col='pred_LinearReg', data=assembled)

    print metrics.get_metrics()

    print "regressed:", regressed[:20]


def ml_fpm():
    from ddf.functions.ml.fpm import AssociationRules, Apriori
    dataset = DDF()\
        .load_fs('/transactions.txt', num_of_parts=4, header=False, sep='\n')\
        .transform(lambda row: row['col_0'].split(','), 'col_0')

    apriori = Apriori(column='col_0', min_support=0.10).run(dataset)

    itemset = apriori.get_frequent_itemsets()

    ar = AssociationRules(confidence=0.10)\
        .run(itemset)\
        .select(['Pre-Rule', 'Post-Rule', 'confidence']).cache()

    rules1 = apriori.generate_association_rules(confidence=0.1).cache()

    itemset = itemset.select(['items', 'support'])

    rules1 = rules1.toPandas()[:20]
    itemset = itemset.cache().toPandas()[:20]
    rules2 = ar.toPandas()[:20]
    print 'RESULT itemset:', itemset
    print "RESULT rules1:", rules1
    print "RESULT rules2:", rules2


def ml_feature_scalers():

    from ddf.functions.ml.feature import MinMaxScaler, MaxAbsScaler, \
        StandardScaler
    from ddf.functions.ml.feature import VectorAssembler

    # MinMaxScaler
    data = pd.DataFrame([[-1, 2], [-0.5, 6], [0, 10], [1, 18]],
                        columns=['x', 'y'])

    data = DDF().parallelize(data, 4)
    assembler = VectorAssembler(input_col=["x", "y"], output_col="features")
    assembled = assembler.transform(data)

    scaler = MinMaxScaler(input_col='features', output_col='output')\
        .fit(assembled)
    result1 = scaler.transform(assembled).select(['output']).cache()

    #  MaxAbsScaler
    data = pd.DataFrame([[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]],
                        columns=['x', 'y', 'z'])

    data = DDF().parallelize(data, 4)
    assembler = VectorAssembler(input_col=["x", "y", 'z'],
                                output_col="features")
    assembled = assembler.transform(data)

    scaler = MaxAbsScaler(input_col='features', output_col='features_norm')\
        .fit(assembled)
    result2 = scaler.transform(assembled).select(['features_norm']).cache()

    # Standard Scaler
    data = pd.DataFrame([[0, 0], [0, 0], [1, 1], [1, 1]],
                        columns=['x', 'y'])

    data = DDF().parallelize(data, 4)
    assembler = VectorAssembler(input_col=["x", "y"],
                                output_col="features")
    assembled = assembler.transform(data)

    scaler = StandardScaler(input_col='features', output_col='features_norm',
                            with_mean=True, with_std=True).fit(assembled)
    result3 = scaler.transform(assembled).select(['features_norm']).cache()

    print "MinMaxScaler :", result1.toPandas()
    # [[0.   0.]
    #  [0.25 0.25]
    # [0.5 0.5]
    # [1.   1.]]

    print "MaxAbsScaler :", result2.toPandas()
    # [[0.5, -1., 1.],
    #  [1., 0., 0.],
    #  [0., 1., -0.5]])

    print "StandardScaler :", result3.toPandas()
    # [[-1. - 1.]
    #  [-1. - 1.]
    #  [1.  1.]
    # [1.   1.]]


def ml_feature_dimensionality():

    from ddf.functions.ml.feature import PCA
    from ddf.functions.ml.feature import VectorAssembler

    data = pd.DataFrame([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                        columns=['x', 'y'])

    data = DDF().parallelize(data, 4).select(["x", "y"])
    assembler = VectorAssembler(input_col=["x", "y"],
                                output_col="features")
    assembled = assembler.transform(data)

    pca = PCA(input_col='features',
              output_col='features_pca', n_components=2).fit(assembled)
    assembled = pca.transform(assembled).select(['features_pca'])

    print "RESULT :", assembled.cache().toPandas()
    # [[1.38340578, 0.2935787],
    #  [2.22189802, -0.25133484],
    #  [3.6053038, 0.04224385],
    #  [-1.38340578, -0.2935787],
    #  [-2.22189802, 0.25133484],
    #  [-3.6053038, -0.04224385]]


def geographic():

    ddf1 = DDF()\
        .load_shapefile(shp_path='/41CURITI.shp', dbf_path='/41CURITI.dbf')\
        .select(['points', 'NOMEMESO'])

    data = pd.DataFrame([[-25.251240, -49.166195],
                         [-25.440731, -49.271526],
                         [-25.610885, -49.276478],
                         [-25.43774, -49.20549],
                         [-25.440731, -49.271526],
                         [25, 49],
                         ], columns=['LATITUDE', 'LONGITUDE'])

    ddf2 = DDF()\
        .parallelize(data, 4)\
        .select(['LATITUDE', 'LONGITUDE'])\
        .geo_within(ddf1, 'LATITUDE', 'LONGITUDE', 'points')\
        .select(['LATITUDE', 'LONGITUDE'])\
        .cache()
    print "> Print results: ", ddf2.toPandas()[:10]


def graph_pagerank():

    from ddf.functions.graph import PageRank
    data1 = DDF().load_fs('/edgelist_PageRank.csv', num_of_parts=4)\
        .select(['inlink', 'outlink'])

    result = PageRank(inlink_col='inlink', outlink_col='outlink')\
        .transform(data1).select(['Vertex', 'Rank'])

    print "RESULT :", result.cache().toPandas()[:20]


def simple_etl():
    from pandas.util.testing import assert_frame_equal

    data = pd.DataFrame([[i, i+5, 0] for i in range(10)],
                        columns=['a', 'b', 'c'])

    data1 = pd.DataFrame([[i, i + 5] for i in range(5)],
                         columns=['a', 'b'])

    data2 = pd.DataFrame([[i, i + 5, 0] for i in xrange(5, 15)],
                         columns=['a', 'b', 'c'])

    print "\n|-------- Aggregation --------|\n"
    express = {'a': ['count'], 'b': ['first', 'last']}
    aliases = {'a': ["COUNT"], 'b': ['col_First', 'col_Last']}
    ddf_1 = DDF().parallelize(data, 4)\
        .aggregation(['c'], exprs=express, aliases=aliases)
    df1 = ddf_1.cache().toPandas()

    res_agg = pd.DataFrame([[0, 10, 5, 14]],
                           columns=['c', 'COUNT', 'col_First', 'col_Last'])
    assert_frame_equal(df1, res_agg, check_index_type=False)
    print "etl_test - aggregation - OK",

    print "\n|-------- Difference --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    ddf_2 = ddf_1a.difference(ddf_1b)
    df1 = ddf_2.cache().toPandas()
    res_diff = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0],
                             [3, 8, 0], [4, 9, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_diff, check_index_type=False)
    print "etl_test - difference - OK"

    print "\n|-------- Distinct --------|\n"
    ddf_1 = DDF().parallelize(data, 4).distinct(['c'])
    df1 = ddf_1.cache().toPandas()

    res_dist = pd.DataFrame([[0, 5, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_dist, check_index_type=False)
    print "etl_test - distinct - OK"

    print "\n|-------- Drop --------|\n"
    ddf_1 = DDF().parallelize(data, 4).drop(['a'])
    df1 = ddf_1.cache().toPandas()
    res_drop = pd.DataFrame([[5, 0], [6, 0], [7, 0], [8, 0], [9, 0],
                             [10, 0], [11, 0], [12, 0],
                             [13, 0], [14, 0]], columns=['b', 'c'])
    assert_frame_equal(df1, res_drop, check_index_type=False)
    print "etl_test - drop - OK"

    print "\n|-------- Intersect --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data2, 4)
    ddf_2 = ddf_1a.intersect(ddf_1b)
    df1 = ddf_2.cache().toPandas()
    res_int = pd.DataFrame([[5, 10, 0], [6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_int, check_index_type=False)
    print "etl_test - intersect - OK"

    print "\n|-------- Filter --------|\n"
    ddf_1 = DDF().parallelize(data, 4).filter('a > 5')
    df1 = ddf_1.cache().toPandas()
    res_fil = pd.DataFrame([[6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_fil, check_index_type=False)
    print "etl_test - filter - OK"

    print "\n|-------- Join --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data, 4)
    ddf_2 = ddf_1a.join(ddf_1b, key1=['a'], key2=['a'])
    df1 = ddf_2.cache().toPandas()
    res_join = pd.DataFrame([[0, 5, 0, 5, 0], [1, 6, 0, 6, 0],
                             [2, 7, 0, 7, 0], [3, 8, 0, 8, 0],
                             [4, 9, 0, 9, 0], [5, 10, 0, 10, 0],
                             [6, 11, 0, 11, 0], [7, 12, 0, 12, 0],
                             [8, 13, 0, 13, 0], [9, 14, 0, 14, 0]],
                            columns=['a', 'b_l', 'c_l', 'b_r', 'c_r'])
    assert_frame_equal(df1, res_join, check_index_type=False)
    print "etl_test - join - OK"

    print "\n|-------- Replace Values --------|\n"
    ddf_1 = DDF().parallelize(data, 4).replace({'c': [[0], [42]]})
    df1 = ddf_1.cache().toPandas()
    res_rep = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42], [4, 8, 42],
                            [5, 9, 42], [6, 10, 42], [6, 11, 42], [7, 12, 42],
                            [8, 13, 42], [9, 14, 42]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print "etl_test - replace - OK"

    print "\n|-------- Sample --------|\n"
    ddf_1 = DDF().parallelize(data, 4).sample(7)
    df1 = ddf_1.cache().toPandas()
    if len(df1) != 7:
        raise Exception("Sample error")
    print "etl_test - sample - OK"

    print "\n|-------- Select --------|\n"
    ddf_1 = DDF().parallelize(data, 4).select(['a'])
    df1 = ddf_1.cache().toPandas()
    res_rep = pd.DataFrame([[0], [1], [2], [3], [4],  [5], [6], [7],
                            [8], [9]], columns=['a'])
    assert_frame_equal(df1, res_rep, check_index_type=False)
    print "etl_test - select - OK"

    print "\n|-------- Sort --------|\n"
    ddf_1 = DDF().parallelize(data, 4).sort(['a'], ascending=[False])
    df1 = ddf_1.cache().toPandas()
    res_sor = pd.DataFrame([[9, 14, 0], [8, 13, 0],  [7, 12, 0],
                            [6, 11, 0], [5, 10, 0], [4, 9, 0],
                            [3, 8, 0], [2, 7, 0], [1, 6, 0],
                            [0, 5, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_sor, check_index_type=False)
    print "etl_test - sort - OK"

    print "\n|-------- Split --------|\n"
    ddf_1a, ddf_1b = DDF().parallelize(data, 4).split(0.5)
    df1 = ddf_1a.cache().toPandas()
    df2 = ddf_1b.cache().toPandas()
    cond = any(pd.concat([df1, df2]).duplicated(['a', 'b', 'c']).values)
    if cond:
        raise Exception("Split")
    print "etl_test - split - OK"

    print "\n|-------- Take --------|\n"
    ddf_1 = DDF().parallelize(data, 4).take(3)
    df1 = ddf_1.cache().toPandas()
    res_tak = pd.DataFrame([[0, 5, 42], [1, 6, 42], [2, 7, 42]],
                           columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_tak, check_index_type=False)
    print "etl_test - take - OK"

    print "\n|-------- Transform --------|\n"
    f = lambda col: 7 if col['a'] > 5 else col['a']
    ddf_1 = DDF().parallelize(data, 4).transform(f, 'a')
    df1 = ddf_1.cache().toPandas()
    res_tra = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [7, 11, 0], [7, 12, 0],
                            [7, 13, 0], [7, 14, 0]], columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_tra, check_index_type=False)
    print "etl_test - transform - OK"

    print "\n|-------- Union --------|\n"
    ddf_1a = DDF().parallelize(data, 4)
    ddf_1b = DDF().parallelize(data1, 4)
    ddf_2 = ddf_1a.union(ddf_1b)
    df1 = ddf_2.cache().toPandas()
    res_uni = pd.DataFrame([[0, 5, 0.0], [1, 6, 0.0], [2, 7, 0.0],
                            [0, 5, None], [1, 6, None], [3, 8, 0.0],
                            [4, 9, 0.0], [5, 10, 0.0], [2, 7, None],
                            [3, 8, None], [6, 11, 0.0], [7, 12, 0.0],
                            [8, 13, 0.0], [4, 9, None], [9, 14, 0.0]],
                           columns=['a', 'b', 'c'])
    assert_frame_equal(df1, res_uni, check_index_type=False)
    print "etl_test - union - OK"

    print "\n|-------- With_column --------|\n"
    ddf_1 = DDF().parallelize(data, 4).with_column('a', 'A')
    df1 = ddf_1.cache().toPandas()
    res_with = pd.DataFrame([[0, 5, 0], [1, 6, 0], [2, 7, 0], [3, 8, 0],
                            [4, 9, 0], [5, 10, 0], [6, 11, 0], [7, 12, 0],
                            [8, 13, 0], [9, 14, 0]], columns=['A', 'b', 'c'])
    assert_frame_equal(df1, res_with, check_index_type=False)
    print "etl_test - with_column - OK"


def etl():

    url = ('https://archive.ics.uci.edu/ml/'
           'machine-learning-databases/abalone/abalone.data')
    cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
    data = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)[:20]

    f1 = lambda col: -42 if col['height'] > 0.090 else col['height']
    data1 = DDF().parallelize(data, 4)\
                 .transform(f1, 'height_nan').cache()

    data2 = data1.transform(lambda col: "."+col['sex'], 'sex')\
                 .cache()

    print """
    ----------------------------------------------------------------------------
        etl_test_1: Avaliar a capacidade de multiplos caches
        """

    df = data1.cache().toPandas()
    print "etl_test_1a:", df[0:5]

    df = data2.cache().toPandas()
    print "etl_test_1b:", df[0:5]

    df = data1.cache().toPandas()
    print "etl_test_1c:", df[0:5]

    print """
    ----------------------------------------------------------------------------
        etl_test_2: Avaliar a capacidade de branching:
        data2 e data3 são filhos de data1
        nesse caso: nenhum dos transforms das listas podem ser otimizadas
    """

    data3 = data1.drop(['length', 'diam']).cache()
    print "DATA1", data1.task_list
    print "DATA3", data3.task_list
    df1 = data1.toPandas()
    df2 = data2.toPandas()
    df3 = data3.toPandas()

    print "etl_test_2a:", df1[0:5]
    print "etl_test_2b:", df2[0:5]
    print "etl_test_2c:", df3[0:5]

    print """
    ----------------------------------------------------------------------------
          etl_test_3: Avaliar a capacidade de agrupar multiplas lazy tasks
          Nesse caso, drop, drop e replace vao ser agrupadas
    """
    data4 = data2.drop(['length']).drop(['diam'])\
        .replace({'rings': [[15], [42]]})

    df = data4.cache().toPandas()
    print "etl_test_3:", df[0:5]

    print """
    ----------------------------------------------------------------------------
          etl_test_4: Avaliar a capacidade de produção de dois resultados
    """

    data5a, data5b = data4.split(0.5)

    df = data5b.cache().toPandas()
    print "etl_test_4a:", df[0:5]

    df = data5a.cache().toPandas()
    print "etl_test_4b:", df[0:5]

    print """
    ----------------------------------------------------------------------------
        etl_test_5: Avaliar capacidade de esperar uma segunda entrada
    """

    data6 = data5b.join(data5a, ['rings'], ['rings'])\
                  .select(['sex_l', 'height_l', 'weight_l', 'rings'])\
                  .filter('(rings > 8)')

    df = data6.cache().toPandas()
    print "etl_test_5a len({}): {}".format(len(df), df[0:5])

    data7 = data6.sample(10).sort(['rings'], [True])
    data8 = data6.join(data7, ['rings'], ['rings'])

    print "data8", data8.task_list
    df = data8.cache().toPandas()
    print "etl_test_5b len({}): {}".format(len(df), df[0:5])

    print """
    ----------------------------------------------------------------------------
        etl_test_6: Avaliar capacidade de gerar resultado sem salvar varivel
    """
    df = data1.distinct(['rings']).cache().toPandas()
    df2 = data1.cache().toPandas()
    print "etl_test_6a:", df
    print "etl_test_6b:", df2

    print """
    ----------------------------------------------------------------------------
        etl_test_7: Avaliar capacidade 'count' and 'take'
    """

    v = data1.select(['rings']).count()
    df = data1.select(['rings']).take(10).cache().toPandas()

    print "etl_test_7a:", v
    print "etl_test_7b:", len(df)


def main():
    print "_____EXAMPLES_____"

    etl()
    # simple_etl()
    # ml_feature_text_operations()
    # ml_classifiers()
    # ml_clustering()
    # ml_regression()
    # ml_fpm()
    # ml_feature_scalers()
    # ml_feature_dimensionality()
    # geographic()
    # graph_pagerank()

if __name__ == '__main__':
    main()
