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
        - host: The host of the NameNode HDFS; (default, default)
        - port: Port of the NameNode HDFS; (default, 0)
        - shp_path: Path to the shapefile (.shp)
        - dbf_path: Path to the shapefile (.dbf)
        - polygon: Alias to the new column to store the
            polygon coordinates (default, 'points');
        - attributes: List of attributes to keep in the DataFrame,
            empty to use all fields;
        - lat_long: True  if the coordinates is (lat,log),
            False if is (long,lat). Default is True;
    :param nfrag: The number of partitions to split this data set.

    .. note:: pip install pyshp
    """

    from hdfspycompss.block import Block
    from hdfspycompss.hdfs import HDFS
    host = settings.get('host', 'localhost')
    port = settings.get('port', 9000)
    dfs = HDFS(host=host, port=port)

    polygon = settings.get('polygon', 'points')
    lat_long = settings.get('lat_long', True)
    header = settings.get('attributes', [])

    # reading shapefile as a binary file in HDFS
    filename = settings['shp_path']
    blocks = dfs.find_n_blocks(filename, 1)
    shp_path = Block(blocks[0]).read_binary()
    filename = settings['dbf_path']
    blocks = dfs.find_n_blocks(filename, 1)
    dbf_path = Block(blocks[0]).read_binary()
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

    data = []
    data_points = []
    for i, sector in enumerate(sectors):
        attributes = []
        r = records[i]
        for t in num_fields:
            attributes.append(r[t-1])
        data.append(attributes)

        points = []
        for point in sector.shape.points:
            a, b = point[0], point[1]
            if lat_long:
                points.append([b, a])
            else:
                points.append([a, b])
        data_points.append(points)

    geo_data = pd.DataFrame(data, columns=header)

    # forcing pandas to infer the dtype (before pandas 0.23)
    geo_data = geo_data.infer_objects()
    geo_data[polygon] = data_points

    info = []
    geo_data = np.array_split(geo_data, nfrag)
    for f, df in enumerate(geo_data):
        info.append(generate_info(df, f))

    return geo_data, info
