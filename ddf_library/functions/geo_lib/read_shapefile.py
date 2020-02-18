#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.utils import generate_info, create_stage_files, \
    save_stage_file

import shapefile
from io import BytesIO
import pandas as pd
import numpy as np
import time


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
    t1 = time.time()
    storage = settings.get('storage', 'file')
    shp_path = settings['shp_path']
    dbf_path = settings['dbf_path']

    polygon = settings.get('polygon', 'points')
    lat_long = settings.get('lat_long', True)
    header = settings.get('attributes', [])

    if storage == 'hdfs':

        from hdfspycompss.block import Block
        from hdfspycompss.hdfs import HDFS
        host = settings.get('host', 'localhost')
        port = settings.get('port', 9000)
        dfs = HDFS(host=host, port=port)

        blocks = dfs.find_n_blocks(shp_path, 1)
        shp_path = Block(blocks[0]).read_binary()

        blocks = dfs.find_n_blocks(dbf_path, 1)
        dbf_path = Block(blocks[0]).read_binary()

        # reading shapefile as a binary file
        shp_io = BytesIO(shp_path)
        dbf_io = BytesIO(dbf_path)
    else:
        shp_io = shp_path
        dbf_io = dbf_path

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
    if polygon in header:
        header.remove(polygon)

    # position of each selected field
    fields_idx = [fields[f] for f in header]

    data = []
    data_points = []
    for i, sector in enumerate(sectors):
        attributes = []
        r = records[i]
        for t in fields_idx:
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
    geo_data = geo_data.infer_objects()
    # forcing pandas to infer dtype
    # geo_data = convert_int64_columns(geo_data)

    geo_data[polygon] = data_points

    info = []
    geo_data = np.array_split(geo_data, nfrag)
    output_files = create_stage_files(nfrag)

    for f, (out, df) in enumerate(zip(output_files, geo_data)):
        info.append(generate_info(df, f))
        save_stage_file(out, df)

    t2 = time.time()
    print('[INFO] - Time to process `read_shapefile`: {:.0f} seconds'
          .format(t2 - t1))
    return output_files, info

