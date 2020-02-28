#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.context import COMPSsContext

import pandas as pd
import numpy as np


def read_shapefile(cc):
    shp_path = 'hdfs://localhost:9000/pp/41CURITI.shp'
    dbf_path = 'hdfs://localhost:9000/pp/41CURITI.dbf'

    ddf1 = cc\
        .read.shapefile(shp_path=shp_path, dbf_path=dbf_path)\
        .select(['points', 'NOMEMESO', 'ID'])

    # ddf1.show()
    # print(ddf1.schema())
    return ddf1


def geo_within(shapefile_ddf):

    size = 50
    vec_lo = np.random.uniform(low=-49.4550, high=-48.9187, size=size)

    vec_la = np.random.uniform(low=-25.5870, high=-25.3139, size=size)
    data = pd.DataFrame()
    data['LATITUDE'] = vec_la
    data['LONGITUDE'] = vec_lo

    ddf2 = COMPSsContext()\
        .parallelize(data, 4)\
        .geo_within(shapefile_ddf, 'LATITUDE', 'LONGITUDE', polygon='points',
                    attributes=['ID'])

    print("> Print results: \n")
    ddf2.show()


if __name__ == '__main__':
    print("_____Geographic Operations_____")

    cc = COMPSsContext()
    geo_ddf = read_shapefile(cc)
    geo_within(geo_ddf)
    cc.stop()
