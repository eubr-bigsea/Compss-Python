#!/usr/bin/python
#

import sys
import time

from dps import DPS, COMPSsContext
import pandas as pd

def main_program():
    print("________RUNNING EXAMPLES_________")

    data1 = DPS().load_fs('/flights.csv', num_of_parts=4)
    # data2 = DPS().load_fs('/flights.csv', num_of_parts=4)

    data3 = data1.transform(lambda col: col['Year']-2000, 'oi')

    data4 = data3.with_column('DepTime', 'DepTime2', 'integer')

    data5 = data3.transform(lambda col: -9, 'oi2')\
        .drop(['oi'])\
        .filter('(CRSDepTime > 750)') \
        .split(0.5)

    data5_1 = data5[0].sample(10)
    data5_2 = data5[1].sample(8)

    #.aggregation(['Year'], {'Year': ['count']}, {'Year': ['COUNT']})
    #.clean_missing(['LateAircraftDelay'])\
    #.distinct(['Year']) \
    #.sample(10) \
    #.sort(['DepTime'], [True])\
    #.replace({'Year': [[2008], [42]]})
    #intersect(data1)

    # data6 = data5_2.join(data4, ['DepTime', 'CRSDepTime'],
    #                      ['DepTime2', 'CRSDepTime']).collect(keep_partitions=False)
    #
    # data6 = data5_1.difference(data5_2).collect(keep_partitions=False)

    # data6 = data5_1.union(data5_2).collect(keep_partitions=False)


    #GEO
    # data6 = DPS().load_shapefile(shp_path='/41CURITI.shp', dbf_path='/41CURITI.dbf')
    #
    # data = pd.DataFrame([[-25.251240, -49.166195],
    #                      [-25.440731, -49.271526],
    #                      [-25.610885, -49.276478],
    #                      [-25.43774, -49.20549],
    #                      [-25.440731, -49.271526],
    #                      [25, 49],
    #                      ], columns=['LATITUDE', 'LONGITUDE'])
    #
    # data7 = DPS().load_df(data, 4).geo_within(data6, 'LATITUDE', 'LONGITUDE', 'points').collect(keep_partitions=False)
    # print "> Print results: ", data7.toPandas()

    print COMPSsContext().run()

    print "> Print results: ", data7.toPandas()
    # .collect(keep_partitions=False)[0:10]

    # print data3\
    #     .select(['oi','oi2','Year'])\
    #     .drop(['Year'])\
    #     .collect(keep_partitions=False)
    #
    # # filter and distinct
    # print data3.filter("(FlightNum == 4)")\
    #
    #     .collect(keep_partitions=False)
    #
    # print data3.sample(value=10)\
    #     .collect(keep_partitions=False)
    #
    # print data3.sort(['DepTime'], [True]).collect(keep_partitions=False)
    #
    # print data3.replace({'oi2': [[-9], [1]]}).collect(keep_partitions=False)
    #
    # # join
    # print data3.select(['oi', 'oi2', 'Year', 'DepTime', 'CRSDepTime'])\
    #     .join(data2, ['DepTime', 'CRSDepTime'], ['DepTime', 'CRSDepTime'])\
    #     .collect(keep_partitions=False)
    #
    #
    # print data3.difference(data2)\
    #     .collect(keep_partitions=False)
    #
    # print data3.take(9)\
    #     .collect(keep_partitions=False)
    #
    # print data3.clean_missing(['LateAircraftDelay'])\
    #     .collect(keep_partitions=False)
    # data3.\
    #     aggregation(['Year'], {'Year': ['count']}, {'Year': ['COUNT']})\
    #     .collect(keep_partitions=False)
    # data3.show(10)
    # print "RESULT:", data3.count(reduce=False)


if __name__ == '__main__':
    main_program()
