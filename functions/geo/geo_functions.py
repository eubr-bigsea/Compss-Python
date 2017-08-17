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


    df = pd.DataFrame(data, columns=header)
    return df
