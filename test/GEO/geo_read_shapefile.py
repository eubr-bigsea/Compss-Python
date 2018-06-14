#!/usr/bin/python
# -*- coding: utf-8 -*-

from functions.geo.read_shapefile import ReadShapeFileOperation
import pandas as pd


if __name__ == '__main__':
    """Test ReadShapefile function."""

    numFrag = 4
    settings = dict()
    settings['shp_path'] = '/41CURITI.shp'
    settings['dbf_path'] = '/41CURITI.dbf'
    settings['attributes'] = []  # all columns
    settings['polygon'] = 'POINTS'

    data = ReadShapeFileOperation().transform(settings, numFrag)

    data = pd.concat(data)
    print len(data.columns) == 80 and len(data) == 2125

    settings['attributes'] = ['CODSETOR']
    data = ReadShapeFileOperation().transform(settings, numFrag)

    data = pd.concat(data)
    print len(data) == 2125 and len(data.columns) == 2  # CODSETOR and POINTS


