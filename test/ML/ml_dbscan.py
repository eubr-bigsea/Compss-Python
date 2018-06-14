#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.ml.clustering.DBSCAN.dbscan import DBSCAN
from functions.ml.feature_assembler import FeatureAssemblerOperation
import pandas as pd
import numpy as np
import time

pd.set_option('display.expand_frame_repr', False)


def plotter(result_df, eps, minPts):

    import matplotlib.pyplot as plt
    #from shapely.geometry.polygon import Polygon
    #from matplotlib.patches import Polygon

    table = dict()
    clusters = result_df['cluster'].tolist()
    Lat = result_df['LONGITUDE'].tolist()
    Long = result_df['LATITUDE'].tolist()
    LAT = []
    LONG = []
    COR = []
    LABELS = []
    indexes = []

    for i in range(len(clusters)):
        c = clusters[i]

        if c not in table:
            table[c] = len(table) +1

        if c > 0 :
            COR.append(table[c])
            LAT.append(Lat[i])
            LONG.append(Long[i])
            LABELS.append(i)
        else:
            indexes.append(i)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(LAT, LONG, c=COR, s=15)

    for i in indexes:
        x = Lat[i]
        y = Long[i]
        ax.annotate('*', xy=(x, y), xytext=(0, 0), textcoords='offset points')
        circ = plt.Circle((x, y), radius=eps, fill=False, facecolor='none')
        ax.add_patch(circ)

    plt.grid(True)
    plt.axvline(0.5, color='black')
    plt.axhline(0.5, color='black')

    plt.axvline(0.5+eps, color='blue')
    plt.axhline(0.5+eps, color='blue')
    plt.axis((0, 1, 0, 1))
    plt.title("DBSCAN")
    plt.show()


def debugger(df):
    print "\n[DEBUG-START] -------"
    c1 = df.cluster.unique()
    for c in c1:
        a = df.loc[df['cluster'] == c, 'ID_RECORD'].tolist()
        msg =  "c:{} --> {}".format(c, list(np.sort(a, axis=None)))
        print msg
    print "[DEBUG-END] -------"


if __name__ == '__main__':
    """Test DBSCAN function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/base100.csv'
    settings['header'] = True
    settings['separator'] = ','

    data = ReadOperationHDFS().transform(filename, settings, numFrag)

    settings = dict()
    settings['cols'] = ['LATITUDE', 'LONGITUDE']
    settings['alias'] = 'FEATURES'
    data0 = FeatureAssemblerOperation().transform(data, settings, numFrag)

    settings = dict()
    settings['feature'] = 'FEATURES'
    settings['predCol'] = 'cluster'
    settings['minPts'] = 5
    settings['eps'] = 0.05

    db = DBSCAN()
    data1 = db.fit_predict(data0, settings, numFrag)

    data1 = compss_wait_on(data1)
    data1 = pd.concat(data1, axis=0)

    print data1.to_string()

    # plotter(data, settings['eps'], settings['minPts'])
    # debugger(data)

