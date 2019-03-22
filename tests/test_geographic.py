#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def geographic():

    ddf1 = DDF()\
        .load_shapefile(shp_path='/41CURITI.shp', dbf_path='/41CURITI.dbf')\
        .select(['points', 'NOMEMESO'])

    data = pd.DataFrame([[-25.251240, -49.166195],
                         [-25.440731, -49.271526],
                         [-25.610885, -49.276478],
                         [-25.43774,  -49.20549],
                         [-25.440731, -49.271526],
                         [25, 49],
                         ], columns=['LATITUDE', 'LONGITUDE'])

    ddf2 = DDF()\
        .parallelize(data, 4)\
        .select(['LATITUDE', 'LONGITUDE'])\
        .geo_within(ddf1, 'LATITUDE', 'LONGITUDE', 'points')\
        .select(['LATITUDE', 'LONGITUDE'])

    print "> Print results: \n", ddf2.show()
    """
      -25.440731 -49.271526
      -25.610885 -49.276478
      -25.437740 -49.205490
      -25.440731 -49.271526
    """


if __name__ == '__main__':
    print "_____Geographic Operations_____"
    geographic()
