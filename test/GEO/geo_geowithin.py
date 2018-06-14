#!/usr/bin/python
# -*- coding: utf-8 -*-

from pycompss.api.api import compss_wait_on
from functions.geo.read_shapefile import ReadShapeFileOperation
from functions.geo.geo_within import GeoWithinOperation
import pandas as pd


def generate_data(numFrag):
    from functions.data.data_functions import Partitionize
    data = pd.DataFrame([[-25.251240, -49.166195],
                         [-25.440731, -49.271526],
                         [-25.610885, -49.276478],
                         [-25.43774, -49.20549],
                         [-25.440731, -49.271526],
                         [25, 49],
                         ], columns=['lat', 'lon'])

    data = Partitionize(data, numFrag)
    data[numFrag-1] = data[numFrag-1].head(0)
    return data


if __name__ == '__main__':
    """Test Geo Within function."""

    numFrag = 4
    settings = dict()
    settings['shp_path'] = '/41CURITI.shp'
    settings['dbf_path'] = '/41CURITI.dbf'
    settings['attributes'] = []  # all columns
    settings['polygon'] = 'POINTS'

    geo = ReadShapeFileOperation().transform(settings, numFrag)

    data = generate_data(numFrag)
    print data
    settings = dict()
    settings['lat_col'] = 'lat'
    settings['lon_col'] = 'lon'
    settings['attributes'] = ['CODBAIRR', 'CODDISTR']
    settings['polygon'] = 'POINTS'
    settings['alias'] = "_shp"

    data = GeoWithinOperation().transform(data, geo, settings, numFrag)

    data = compss_wait_on(data)

    data = pd.concat(data, sort=False, ignore_index=True)

    print len(data.columns) == 4 and len(data) == 4

    print data

