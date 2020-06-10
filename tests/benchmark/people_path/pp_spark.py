#!/usr/bin/env python
# coding: utf-8

import sys
import shapefile
from io import BytesIO
from matplotlib.path import Path
import pandas as pd
import numpy as np
import pyqtree
import time

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

size = sys.argv[1]
HOST = 'master'
PORT = 9000

TICKETING_FILE = 'hdfs://{}:{}/pp/doc1_p{}.csv'.format(HOST, PORT, size)
CENSUS_FILE = 'hdfs://{}:{}/pp/census.csv'.format(HOST, PORT, 'census.csv')
GPS_FILE = 'hdfs://{}:{}/pp/doc2_p{}.csv'.format(HOST, PORT, size)
print("Spark - {} and {}".format(TICKETING_FILE, GPS_FILE))

t1 = time.time()
spark = SparkSession.builder.getOrCreate()


def group_datetime(d, interval):
    """Group datetime in bins."""
    import datetime
    seconds = d.second + d.hour * 3600 + d.minute * 60 + d.microsecond / 1000
    k = d - datetime.timedelta(seconds=seconds % interval)
    return datetime.datetime(k.year, k.month, k.day, k.hour, k.minute, k.second)

udf1 = udf(lambda x: group_datetime(x, 300), TimestampType())

schema1 = StructType([StructField("CODLINHA", StringType(), True),
                      StructField("NOMELINHA", StringType(), True),
                      StructField("CODVEICULO", StringType(), True),
                      StructField("NUMEROCARTAO", StringType(), True),
                      StructField("SEXO", StringType(), True),
                      StructField("DATAUTILIZACAO", StringType(), True),
                      StructField("DATANASCIMENTO", StringType(), True)])

schema2 = StructType([StructField("COD_LINHA", StringType(), True),
                      StructField("VEIC", StringType(), True),
                      StructField("DTHR", StringType(), True),
                      StructField("LAT", StringType(), True),
                      StructField("LON", StringType(), True)])


bus_ticketing = spark\
    .read.json(TICKETING_FILE, schema=schema1, mode='DROPMALFORMED')\
    .select("CODLINHA", "CODVEICULO", "NUMEROCARTAO", "DATAUTILIZACAO")\
    .dropna(how='any', subset=["CODLINHA", "CODVEICULO", "NUMEROCARTAO",
                               "DATAUTILIZACAO"])\
    .withColumn('NUMEROCARTAO', col('NUMEROCARTAO').cast('integer'))\
    .withColumn('DATAUTILIZACAO', regexp_replace('DATAUTILIZACAO', ',', '.'))\
    .withColumn("DATAUTILIZACAO", to_timestamp("DATAUTILIZACAO",
                                               "dd/MM/yy HH:mm:ss.SSSSSS"))\
    .withColumn('BINS_5_MINS', udf1(col('DATAUTILIZACAO')))

print("bus_ticketing partitions:", bus_ticketing.rdd.partitions.size)

bus_gps = spark\
    .read.json(GPS_FILE, schema=schema2, mode='DROPMALFORMED')\
    .select('COD_LINHA', 'VEIC', 'DTHR', 'LAT', 'LON')\
    .dropna(how='any', subset=['COD_LINHA', 'VEIC', 'DTHR', 'LAT', 'LON'])\
    .withColumn('LAT', regexp_replace('LAT', ',', '.').cast('double'))\
    .withColumn('LON', regexp_replace('LON', ',', '.').cast('double'))\
    .withColumn('DTHR', to_timestamp("DTHR", "dd/MM/yyyy HH:mm:ss"))\
    .withColumn('BINS_5_MINS', udf1(col('DTHR')))\
    .groupby('COD_LINHA', 'VEIC', 'BINS_5_MINS').agg({'LAT': 'first',
                                                      'LON': 'first'})

print("bus_gps partitions:", bus_gps.rdd.partitions.size)

census = spark\
    .read.csv(CENSUS_FILE, sep=';', header=True)\
    .dropna(subset=['CODSETOR', 'BA_001', 'BA_002', 'BA_003',
                    'BA_005', 'BA_007', 'BA_009', 'BA_011',  'P1_001'])

df1_a = bus_gps.alias("df1_a")
df2_a = bus_ticketing.alias("df2_a")

user_trip = df1_a.join(df2_a,
                       (col('df2_a.BINS_5_MINS') == col('df1_a.BINS_5_MINS')) &
                       (col('df2_a.CODLINHA') == col('df1_a.COD_LINHA')) &
                       (col('df2_a.CODVEICULO') == col('df1_a.VEIC')))\
            .withColumn('DATE', col('df2_a.BINS_5_MINS').cast('date')).cache()

branch = user_trip.groupby('DATE', 'NUMEROCARTAO')\
    .agg({'NUMEROCARTAO': 'count'})\
    .filter('count(NUMEROCARTAO) >= 2')

user_trip = user_trip.join(branch, (user_trip['DATE'] == branch['DATE']) & 
                       (user_trip['NUMEROCARTAO'] == branch['NUMEROCARTAO']))\
    .select("CODLINHA", "CODVEICULO", col('df1_a.BINS_5_MINS'),
            user_trip.NUMEROCARTAO, "DATAUTILIZACAO",
            col("first(LAT)").alias('LAT'), col("first(LON)").alias('LON'),
            user_trip.DATE)\
    .distinct()


def read():
    settings = dict()
    settings['shp_path'] = '/pp/41CURITI.shp'
    settings['dbf_path'] = '/pp/41CURITI.dbf'
    settings['polygon'] = 'polygon'
    settings['attributes'] = []
    settings['host'] = HOST
    settings['port'] = PORT
    settings['storage'] = 'hdfs'

    storage = settings.get('storage', 'file')
    shp_path = settings['shp_path']
    dbf_path = settings['dbf_path']

    polygon = settings.get('polygon', 'points')
    lat_long = settings.get('lat_long', True)
    header = settings.get('attributes', [])

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

    # forcing pandas to infer dtype
    geo_data = geo_data.infer_objects()
    geo_data[polygon] = data_points
    out = '/tmp/geo_data'
    geo_data.to_parquet(out)
    return out


def _find_minmax(sector, lon_idx, lat_idx):

    tmp = pd.DataFrame(sector.tolist(), columns=[0, 1])

    mins = tmp.min(skipna=True, axis=0)
    maxs = tmp.max(skipna=True, axis=0)

    xmin, ymin = mins[lon_idx], mins[lat_idx]
    xmax, ymax = maxs[lon_idx], maxs[lat_idx]

    return [xmin, ymin, xmax, ymax]


def geo_within_stage_1(shp_object, settings):
    """
    :param data: A list of pandas DataFrame;
    :param shp_object: The DataFrame created by the function ReadShapeFile;
    :param settings: A dictionary that contains:
        - lat_col: Column which represents the Latitude field in the data;
        - lon_col: Column which represents the Longitude field in the data;
        - lat_long: True  if the coordinates is (lat, log),
                    False if is (long, lat). Default is True;
        - polygon: Field in shp_object where is store the
            coordinates of each sector;
        - attributes: Attributes to retrieve from shapefile, empty to all
                (default, empty);
        - alias: Alias for shapefile attributes
            (default, 'sector_position');
    :return: Returns a list of pandas DataFrame.
    """

    if not all(['lat_col' in settings, 'lon_col' in settings]):
        raise Exception("Please inform, at least, the fields: "
                        "`lat_col` and `lon_col`")

    polygon = settings.get('polygon', 'points')
    if settings.get('lat_long', True):
        lat_idx, lon_idx = 0, 1
    else:
        lat_idx, lon_idx = 1, 0

    # create the main bound box
    points_by_sector = shp_object[polygon].to_numpy()

    min_max = []
    for sector in points_by_sector:
        min_max.append(_find_minmax(sector, lon_idx, lat_idx))
    min_max = np.array(min_max)

    xmin, ymin = np.min(min_max[:, 0]), np.min(min_max[:, 1])
    xmax, ymax = np.max(min_max[:, 2]), np.max(min_max[:, 3])

    # Pyqtree is a pure Python spatial index for GIS or rendering usage.
    # It stores and quickly retrieves items from a 2x2 rectangular grid
    # area, and grows in depth and detail as more items are added.
    spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])

    # than, insert all sectors bbox
    for i, sector in enumerate(min_max):
        xmin, ymin, xmax, ymax = sector

        spindex.insert(item=i, bbox=[xmin, ymin, xmax, ymax])

    settings['spindex'] = spindex
    settings['shp_object'] = shp_object
    settings['intermediate_result'] = False
    return settings

geo_data = pd.read_parquet(read())
settings = {'lat_col': 'LAT', 'lon_col': "LON",
            'polygon': 'polygon', 'alias': '_shp'}
settings = geo_within_stage_1(geo_data, settings)

attributes_to_add = ['CODSETOR', 'CODBAIRR', 'NOMEBAIR']

sp_index = settings['spindex']
shp_object = geo_data[attributes_to_add + ['polygon']]

schema = attributes_to_add
polygon_col_idx = shp_object.columns.get_loc('polygon')
idxs = [shp_object.columns.get_loc(c) for c in attributes_to_add]


def get_first_polygon(lat, lng):
    x, y = float(lng), float(lat)

    matches = sp_index.intersect([x, y, x, y])

    for shp_inx in matches:
        row = shp_object.iat[shp_inx, polygon_col_idx].tolist()
        p_polygon = Path(row)
        # Here it uses longitude, latitude
        if p_polygon.contains_point([y, x]):
            row = shp_object.iloc[shp_inx, idxs]
            return [str(c) for c in row] # must return an array, no Row
    return [None] * len(attributes_to_add)

udf_get_first_polygon = udf(get_first_polygon, ArrayType(StringType()))
aliases = ['CODSETOR_shp', 'CODBAIRR_shp', 'NOMEBAIR_shp']


within = user_trip.withColumn('tmp_polygon_data',
                              udf_get_first_polygon(col('LAT'), col('LON')))
within2 = within.select(within.columns + [
    within.tmp_polygon_data[i].alias(aliases[i])
    for i, col in enumerate(attributes_to_add)])\
    .drop('tmp_polygon_data')\
    .dropna(how='any', subset=aliases)


final = census.join(within2, within2['CODSETOR_shp'] == census['CODSETOR'])\
              .groupby('NUMEROCARTAO', 'DATE')\
    .agg(first('LON').alias('o_lon'),
         last('LON').alias('d_lon'),
         first('LAT').alias('o_lat'),
         last('LAT').alias('d_lat'),
         first('BA_002').alias('o_BA_002'),
         last('BA_002').alias('d_BA_002'),
         first('BA_005').alias('o_BA_005'),
         last('BA_005').alias('d_BA_005'),
         first('P1_001').alias('o_P1_001'),
         last('P1_001').alias('d_P1_001'),
         first('NOMEBAIR_shp').alias('o_NOMEBAIR'),
         last('NOMEBAIR_shp').alias('d_NOMEBAIR'),
         first('DATAUTILIZACAO').alias('o_timestamp'),
         last('DATAUTILIZACAO').alias('d_timestamp'),
         first('CODVEICULO').alias('CODVEICULO'))\
    .sort(col("o_timestamp").desc())\
    .write.parquet("hdfs://{}:{}/pp/spark".format(HOST, PORT),
                   mode='overwrite')

t2 = time.time()
print("Time to preprocessing: ", t2-t1)

