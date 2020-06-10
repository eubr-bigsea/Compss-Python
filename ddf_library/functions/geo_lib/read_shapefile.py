#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.utils import generate_info, create_stage_files, \
    save_stage_file, read_stage_file

from shapefile import Reader
from io import BytesIO
import pandas as pd
import numpy as np
import time


def read_shapefile_all(settings, nfrag):
    # This method will be executed only if shapefile is in common file system.
    settings['nfrag'] = nfrag
    stage1_result, _, settings = read_shapefile_stage_1(settings)

    outputs = create_stage_files(nfrag)
    infos = []
    for f, out in enumerate(stage1_result):
        param = settings.copy()
        param['id_frag'] = f
        df = read_stage_file(out)
        geo_data, info = read_shapefile_stage_2(df, param)
        save_stage_file(outputs[f], geo_data)
        infos.append(info)

    from ddf_library.utils import delete_result
    delete_result(stage1_result)

    return outputs, infos


def read_shapefile_stage_1(settings):
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
        - nfrag: The number of partitions to split this data set.

    .. note:: pip install pyshp
    """
    # This method will be executed on master node because the input file is
    # a binary file, and cannot be splitted.
    t1 = time.time()
    polygon = settings.get('polygon', 'points')
    header = settings.get('attributes', [])
    lat_long = settings.get('lat_long', True)
    nfrag = settings['nfrag']
    # importing to shapefile object
    shp_object = _read(settings)

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

    size_shape = len(shp_object)
    parts_range, cum_range = _generate_distribution2(size_shape, nfrag)

    # the geometry do not have index to be used in the read stage, so, we canot
    # distribute this task to other workers.
    data_points = []
    if lat_long:
        x, y = 0, 1
    else:
        x, y = 1, 0

    for i, sector in enumerate(shp_object.iterShapes()):
        points = []
        for point in sector.points:
            a, b = point[y], point[x]
            points.append([a, b])
        data_points.append(points)

    geo_data = pd.DataFrame()
    geo_data[polygon] = data_points

    geo_data = np.array_split(geo_data, cum_range)
    output_files = create_stage_files(nfrag)
    for g, o in zip(geo_data, output_files):
        save_stage_file(o, g)

    settings['parts_range'] = parts_range
    settings['fields_idx'] = fields_idx
    settings['header'] = header

    t2 = time.time()
    print('[INFO] - Time to process `read_shapefile_stage_1`: {:.0f} seconds'
          .format(t2 - t1))

    return output_files, None, settings


def _generate_distribution2(n_rows, nfrag):
    """Data is divided among the partitions."""

    size = n_rows / nfrag
    size = int(np.ceil(size))
    sizes = [size for _ in range(nfrag)]

    i = 0
    while sum(sizes) > n_rows:
        i += 1
        sizes[i % nfrag] -= 1

    cum_range = np.array(sizes, dtype=np.intp).cumsum().tolist()
    divs = [0] + cum_range
    sizes = [0] * nfrag
    for f in range(nfrag):
        sizes[f] = [divs[f], divs[f+1]]
    return sizes, cum_range


def _read(settings):
    storage = settings.get('storage', 'file')
    shp_path = settings['shp_path']
    dbf_path = settings['dbf_path']

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
        shp_io = open(shp_path, "rb")
        dbf_io = open(dbf_path, "rb")

    shp_object = Reader(shp=shp_io, dbf=dbf_io)

    return shp_object


def read_shapefile_stage_2(df, settings):

    polygon = settings.get('polygon', 'points')
    header = settings.get('header')
    frag = settings['id_frag']
    fields_idx = settings['fields_idx']
    idx_range = settings['parts_range'][frag]
    t1 = time.time()

    # importing to shapefile object
    shp_object = _read(settings)

    records = _record_range(shp_object, idx_range[0], idx_range[1])

    data = []
    for r in records:
        attributes = []
        for t in fields_idx:
            attributes.append(r[t - 1])
        data.append(attributes)

    geo_data = pd.DataFrame(data, columns=header)
    geo_data = geo_data.infer_objects()
    # forcing pandas to infer dtype
    # geo_data = convert_int64_columns(geo_data)

    geo_data[polygon] = df[polygon]

    info = generate_info(geo_data, frag)

    t2 = time.time()
    print('[INFO] - Time to process `read_shapefile_stage_2`: {:.0f} seconds'
          .format(t2 - t1))
    return geo_data, info


def _record_range(shp_object, id_start, id_end):
    """Returns records in a dbf file."""
    if shp_object.numRecords is None:
        shp_object._Reader__dbfHeader()
    records = []
    f = shp_object._Reader__getFileObj(shp_object.dbf)
    f.seek(shp_object._Reader__dbfHdrLength)
    for i in range(id_start, id_end):
        r = shp_object._Reader__record(oid=i)
        if r:
            records.append(r)
    return records


def _shape_range(shp_object, id_start, id_end):
    """Returns a shape object for a shape in the the geometry
    record file."""
    shp = shp_object._Reader__getFileObj(shp_object.shp)
    id_start = shp_object._Reader__restrictIndex(id_start)

    shp.seek(0, 2)
    shpLength = shp.tell()
    shp.seek(100)
    shapes = []
    idx = 0
    while shp.tell() < shpLength:
        if id_start >= idx < id_end:
            shapes.append(shp.__shape())
        elif id_end == idx:
            break

    return shapes
