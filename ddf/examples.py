#!/usr/bin/python
#

import pandas as pd
import time

from ddf import DDF


def main():

    data1 = DDF().load_fs('/flights.csv', num_of_parts=4)
    # data2 = DDF().load_fs('/flights.csv', num_of_parts=4)

    data3 = data1.transform(lambda col: col['Year']-2000, 'oi')

    data4 = data3.transform(lambda col: col['Year']-1000, 'oi2').collect()
    print "RESULT: 4", data4.toPandas()[:20]

    data5 = data3.transform(lambda col: -9, 'oi3')\
        .drop(['oi'])\
        .filter('(CRSDepTime > 750)')\
        .split(0.5)

    data5_1 = data5[0].sample(10)
    data5_2 = data5[1].sample(8)

    result = data5_1.collect()

    print "RESULT: 5", result.toPandas()[:20]

    print "RESULT: 1", data1.collect().toPandas()[:20]

    print "RESULT 3:", data3.collect().toPandas()[:20]


    # print "RESULT:", result.toPandas()[:20]

    # from functions.ml.feature.tokenizer import Tokenizer


    # data1 = DDF().load_fs('/text_data.txt', num_of_parts=4, header=False, sep='\n').collect()
    # tokenizer = Tokenizer(input_col='col_0').transform(data1).collect(keep_partitions=False)

    # .select(['oi','oi2','Year'])
    # .aggregation(['Year'], {'Year': ['count']}, {'Year': ['COUNT']})
    # .clean_missing(['LateAircraftDelay'])\
    # .distinct(['Year']) \
    # .sample(10) \
    # .sort(['DepTime'], [True])\
    # .replace({'Year': [[2008], [42]]})
    # .intersect(data1)
    # .difference(data2)
    # .count()
    # .take(9)
    # .join(data4, ['DepTime', 'CRSDepTime'], ['DepTime2', 'CRSDepTime'])


    # data6 = data5_1.union(data5_2).collect(keep_partitions=False)

    # GEO tests
    # data6 = DDF()\
    #     .load_shapefile(shp_path='/41CURITI.shp', dbf_path='/41CURITI.dbf')
    #
    # data = pd.DataFrame([[-25.251240, -49.166195],
    #                      [-25.440731, -49.271526],
    #                      [-25.610885, -49.276478],
    #                      [-25.43774, -49.20549],
    #                      [-25.440731, -49.271526],
    #                      [25, 49],
    #                      ], columns=['LATITUDE', 'LONGITUDE'])
    #
    # data7 = DDF()\
    #     .load_df(data, 4)\
    #     .geo_within(data6, 'LATITUDE', 'LONGITUDE', 'points')\
    #     .collect(keep_partitions=False)

    # COMPSsContext().run()

    #print "> Print results: ", data1.toPandas()[:20]


if __name__ == '__main__':
    main()
