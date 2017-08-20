#!/usr/bin/python
# -*- coding: utf-8 -*-


from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks
from pycompss.api.api import compss_wait_on, barrier

import numpy as np
import pandas as pd
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#@task(returns=list)
def ReadShapeFile(settings):
    #pip install pyshp

    import shapefile
    from io import BytesIO


    shp_file = settings['url']
    dbf_file = re.sub('.shp$', '.dbf', shp_file)
    shp_content = open(shp_file, "rb")
    dbf_content = open(dbf_file, "rb")
    #shp_io = BytesIO(shp_content[0][1])
    #dbf_io = BytesIO(dbf_content[0][1])
    shp_object = shapefile.Reader(shp=shp_content, dbf=dbf_content)
    records = shp_object.records()
    sectors = shp_object.shapeRecords()

    fields = {}

    for i,f in enumerate(shp_object.fields):
        fields[f[0]] = i
    del fields['DeletionFlag']


    header = settings['attributes']
    if len(header) == 0:
        header= [f for f in fields]

    num_fields = [fields[f] for f in header]

    header.append('points')
    data = []
    for i, sector in enumerate(sectors):
        attributes = []
        r = records[i]
        for t in num_fields:
            attributes.append(r[t-1])

        points = []
        for point in sector.shape.points:
            if settings['lat_long']:
                points.append([point[1], point[0]])
            else:
                points.append([point[0], point[1]])
        attributes.append(points)
        data.append(attributes)


    geo_data = pd.DataFrame(data, columns=header)
    return geo_data


def GeoWithinOperation(data_input, shp_object, settings, numFrag):
    import pyqtree

    #olhar lat_long

    xmin = float('+inf')
    ymin = float('+inf')
    xmax = float('-inf')
    ymax = float('-inf')
    for i, sector in shp_object.iterrows():
        for point in sector['points']:
            xmin = min(xmin, point[1])
            ymin = min(ymin, point[0])
            xmax = max(xmax, point[1])
            ymax = max(ymax, point[0])

    spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])

    for inx, sector in shp_object.iterrows():
        points = []
        xmin = float('+inf')
        ymin = float('+inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for point in sector['points']:
            points.append((point[1], point[0]))
            xmin = min(xmin, point[1])
            ymin = min(ymin, point[0])
            xmax = max(xmax, point[1])
            ymax = max(ymax, point[0])
        spindex.insert(item=inx, bbox=[xmin, ymin, xmax, ymax])


    result =  [ get_sectors(data_input[f], spindex, shp_object,settings) for f in range(numFrag)]

    return result

@task(returns=list)
def get_sectors(data_input, spindex, shp_object,settings):
    from matplotlib.path import Path
    col_lat = settings['lat_col']
    col_long = settings['long_col']
    id_col = settings['id_col']

    def get_first_sector(lat, lng):
        x = float(lat)
        y = float(lng)

        matches = spindex.intersect([y, x, y, x]) # why reversed?

        for shp_inx in matches:
            row = shp_object.loc[shp_inx]
            polygon = Path(row['points'])
            if polygon.contains_point([x, y]):
                return [row[id_col]]#[col for col in row]

        return [None] # * len(shp_object.loc[0])


    sector_position = []

    for i,point in data_input.iterrows():
        sector_position.append( get_first_sector(point[col_lat], point[col_long]))

    print "%%%%%%%"
    print len(sector_position)
    print "%%%%%%%"
    data_input.loc[:,'sectors'] =  pd.Series(sector_position).values

    #shapefile_features_count_geo_data = len(shp_object.value[0])
    #{5} = within_{0}.select(within_{0}.columns + [within_{0}.sector_position[i]    for i in xrange(shapefile_features_count_{0})])
    #result =
    #print sector_position
    return data_input
