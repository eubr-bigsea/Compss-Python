#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

#from pycompss.api.task import task
#from pycompss.api.parameter import *

import pandas as pd

#@task(returns=list)
def ReadShapeFileOperation(settings):
    """
        Reads a shapefile using the shp and dbf file.

        :param settings:    A dictionary that contains:
            - host:         The host of the Namenode HDFS; (default, default)
            - port:         Port of the Namenode HDFS; (default, 0)
            - shp_path:     Path to the shapefile (.shp)
            - dbf_path:     Path to the shapefile (.dbf)
            - polygon:      Alias to the new column to store the
                            polygon coordenates (default, 'points');
            - attributes:   List of attributes to keep in the dataframe,
                            empty to use all fields;
            - lat_long:     True  if the coordenates is (lat,log),
                            False if is (long,lat). Default is True;

    """
    #Note: pip install pyshp

    import shapefile
    from io import BytesIO
    import hdfs_pycompss.hdfsConnector as hdfs

    polygon = settings.get('polygon', 'points')
    lat_long = settings.get('lat_long', True)
    header = settings.get('attributes',[])

    settings['path'] = settings['shp_path']
    shp_path = hdfs.readBinary(settings)
    settings['path'] = settings['dbf_path']
    dbf_path = hdfs.readBinary(settings)

    shp_io = BytesIO(shp_path)
    dbf_io = BytesIO(dbf_path)

    shp_object = shapefile.Reader(shp=shp_io, dbf=dbf_io)
    records = shp_object.records()
    sectors = shp_object.shapeRecords()

    fields = {} #column: position
    for i,f in enumerate(shp_object.fields):
        fields[f[0]] = i
    del fields['DeletionFlag']

    if len(header) == 0:
        header = [f for f in fields]

    num_fields = [fields[f] for f in header] #positions of each selected field

    header.append(polygon)
    data = []
    for i, sector in enumerate(sectors):
        attributes = []
        r = records[i]
        for t in num_fields:
            attributes.append(r[t-1])

        points = []
        for point in sector.shape.points:
            if lat_long:
                points.append([point[1], point[0]])
            else:
                points.append([point[0], point[1]])
        attributes.append(points)
        data.append(attributes)


    geo_data = pd.DataFrame(data, columns=header)
    return geo_data
