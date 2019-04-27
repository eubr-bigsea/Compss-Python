#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.utils import generate_info

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce

import shapefile
from io import BytesIO, StringIO
import pandas as pd
import numpy as np


def read_shapefile(settings, nfrag):
    """
    Reads a shapefile using the shp and dbf file.

    :param settings: A dictionary that contains:
        - host: The host of the Namenode HDFS; (default, default)
        - port: Port of the Namenode HDFS; (default, 0)
        - shp_path: Path to the shapefile (.shp)
        - dbf_path: Path to the shapefile (.dbf)
        - polygon: Alias to the new column to store the
            polygon coordenates (default, 'points');
        - attributes: List of attributes to keep in the dataframe,
            empty to use all fields;
        - lat_long: True  if the coordenates is (lat,log),
            False if is (long,lat). Default is True;

    .. note:: pip install pyshp
    """

    from hdfspycompss.Block import Block
    from hdfspycompss.HDFS import HDFS
    host = settings.get('host', 'localhost')
    port = settings.get('port', 9000)
    dfs = HDFS(host=host, port=port)

    polygon = settings.get('polygon', 'points')
    lat_long = settings.get('lat_long', True)
    header = settings.get('attributes', [])

    # reading shapefile as a binary file in HDFS
    filename = settings['shp_path']
    blks = dfs.findNBlocks(filename, 1)
    shp_path = Block(blks[0]).readBinary()
    filename = settings['dbf_path']
    blks = dfs.findNBlocks(filename, 1)
    dbf_path = Block(blks[0]).readBinary()
    shp_io = BytesIO(shp_path)
    dbf_io = BytesIO(dbf_path)

    # importing to shapefile object
    shp_object = shapefile.Reader(shp=shp_io, dbf=dbf_io)
    records = shp_object.records()
    sectors = shp_object.shapeRecords()

    fields = {}  # column name: position
    for i, f in enumerate(shp_object.fields):
        fields[f[0]] = i
    del fields['DeletionFlag']

    if len(header) == 0:
        header = [f for f in fields]

    # position of each selected field
    num_fields = [fields[f] for f in header]

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

    # forcing pandas to infer the dtype
    tmp = geo_data.drop(polygon, axis=1)
    vector = geo_data[polygon]
    buff = tmp.to_csv(index=False)
    geo_data = pd.read_csv(StringIO(unicode(buff)))
    geo_data[polygon] = vector

    info = []
    geo_data = np.array_split(geo_data, nfrag)
    for f, df in enumerate(geo_data):
        info.append(generate_info(df, f))

    return geo_data, info
