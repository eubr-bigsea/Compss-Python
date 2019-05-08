#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def load_shapefile():

    ddf1 = DDF()\
        .load_shapefile(shp_path='/41CURITI.shp', dbf_path='/41CURITI.dbf')\
        .select(['points', 'NOMEMESO'])

    ddf1.show()
    print(ddf1.schema())
    return ddf1


def geo_within(shapefile_ddf):

    data = pd.DataFrame([[-25.251240, -49.166195],
                         [-25.440731, -49.271526],
                         [-25.610885, -49.276478],
                         [-25.43774,  -49.20549],
                         [-25.440731, -49.271526],
                         [25, 49],
                         ], columns=['LATITUDE', 'LONGITUDE'])

    ddf2 = DDF()\
        .parallelize(data, 4)\
        .geo_within(shapefile_ddf, 'LATITUDE', 'LONGITUDE', polygon='points')

    print("> Print results: \n")
    ddf2.show()

    """
      -25.440731 -49.271526
      -25.610885 -49.276478
      -25.437740 -49.205490
      -25.440731 -49.271526
    """


if __name__ == '__main__':
    print("_____Geographic Operations_____")
    geo_ddf = load_shapefile()
    geo_within(geo_ddf)
