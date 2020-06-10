#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.context import COMPSsContext
from ddf_library.columns import Column, udf
from ddf_library.types import DataType


def use_case(host, size, nfrag):

    HOST = host
    DOC1 = 'hdfs://{host}:9000/pp/doc1_p{size}.json'.format(host=HOST, size=size)
    DOC2 = 'hdfs://{host}:9000/pp/doc2_p{size}.json'.format(host=HOST, size=size)

    print("PP - nfrag {} - {} and {}".format(nfrag, DOC1, DOC2))
    cc = COMPSsContext()

    def group_datetime(d, interval):
        """Group datetime in bins."""
        import datetime
        seconds = d.second + d.hour * 3600 + d.minute * 60 + \
                  d.microsecond / 1000
        k = d - datetime.timedelta(seconds=seconds % interval)
        return datetime.datetime(k.year, k.month, k.day,
                                 k.hour, k.minute, k.second)

    convert1 = udf(group_datetime, DataType.TIMESTAMP,
                   Column('DATAUTILIZACAO'), 300)

    # stage 1
    bus_ticketing = cc\
        .read.json(DOC1, num_of_parts=nfrag, schema=DataType.STRING)\
        .select(["CODLINHA", "CODVEICULO", "NUMEROCARTAO", "DATAUTILIZACAO"])\
        .dropna(["CODLINHA", "CODVEICULO", "NUMEROCARTAO", "DATAUTILIZACAO"],
                mode='REMOVE_ROW')\
        .replace({',000000': '', '.000000': ''}, subset=['DATAUTILIZACAO'], 
                 regex=True)\
        .cast(['DATAUTILIZACAO', 'NUMEROCARTAO'],
              cast=[DataType.TIMESTAMP, DataType.INT])\
        .map(convert1, 'BINS_5_MINS')

    # bus_ticketing.show()
    # print(bus_ticketing.schema())

    convert2 = udf(group_datetime, 'date/time', Column('DTHR'), 300)

    bus_gps = cc\
        .read.json(DOC2, num_of_parts=nfrag, encoding="utf8")\
        .dropna(mode='REMOVE_ROW')\
        .replace({',': '.'}, subset=['LAT', 'LON'], regex=True)\
        .cast(['LAT', 'LON', 'DTHR'],
              cast=[DataType.DECIMAL, DataType.DECIMAL, DataType.TIMESTAMP])\
        .map(convert2, 'BINS_5_MINS')\
        .group_by(['COD_LINHA', 'VEIC', 'BINS_5_MINS'])\
        .agg(LAT=('LAT', 'first'), LON=('LON', 'first'))

    # bus_gps.show()
    # print(bus_gps.schema())

    user_trip = bus_ticketing\
        .join(bus_gps, key1=['BINS_5_MINS', 'CODLINHA', 'CODVEICULO'],
              key2=['BINS_5_MINS', 'COD_LINHA', 'VEIC'])\
        .map(Column('BINS_5_MINS').cast(DataType.DATE), 'DATE')\
        .cache()

    # user_trip.show()
    # print(user_trip.schema())

    branch = user_trip\
        .group_by(['DATE', 'NUMEROCARTAO'])\
        .agg(COUNT=('NUMEROCARTAO', 'count'))\
        .filter('(COUNT >= 2)')

    # branch.show()
    # print(branch.schema())

    shapefile = cc.read.shapefile(
            shp_path='hdfs://{host}:9000/pp/41CURITI.shp'.format(host=HOST),
            dbf_path='hdfs://{host}:9000/pp/41CURITI.dbf'.format(host=HOST),
            num_of_parts=nfrag,
            attributes=['CODSETOR', 'CODBAIRR', 'NOMEBAIR',
                        'NOMEMICR', 'points'])\
        .filter('NOMEMICR == "CURITIBA"')\
        .select(['CODSETOR', 'CODBAIRR', 'NOMEBAIR', 'points'])\
        .cast(['CODSETOR'], cast=[DataType.INT])

    # print('shapefile:')
    # shapefile.show()
    # print(shapefile.schema())

    filtered_user_trip = user_trip\
        .join(branch, key1=['DATE', 'NUMEROCARTAO'],
              key2=['DATE', 'NUMEROCARTAO'])\
        .select(["CODLINHA", "CODVEICULO", "BINS_5_MINS", "NUMEROCARTAO",
                 "DATAUTILIZACAO", "LAT", "LON", 'DATE'])\
        .distinct(["CODLINHA", "CODVEICULO", "BINS_5_MINS", "NUMEROCARTAO",
                   "DATAUTILIZACAO", "LAT", "LON"])\
        .geo_within(shapefile, lat_col='LAT', lon_col='LON', polygon='points',
                    attributes=['CODSETOR', 'CODBAIRR', 'NOMEBAIR']) \
        .dropna(['CODSETOR_shp'], mode='REMOVE_ROW')\
        .cache()

    # print('filtered_user_trip:')
    # filtered_user_trip.show()
    # print(filtered_user_trip.schema())

    census_data = cc\
        .read.csv('hdfs://{host}:9000/pp/census.csv'.format(host=HOST),
                  num_of_parts=nfrag, sep=';', schema=DataType.STRING)\
        .cast(['CODSETOR'], cast=[DataType.INT])\
        .dropna(['CODSETOR', 'BA_001', 'BA_002', 'BA_003',
                 'BA_005', 'BA_007', 'BA_009', 'BA_011',  'P1_001'],
                mode='REMOVE_ROW')\
        .join(filtered_user_trip,
              key1=['CODSETOR'],
              key2=['CODSETOR_shp'])\
        .group_by(['NUMEROCARTAO', 'DATE'])\
        .agg(o_lon=('LON', 'first'),
             d_lon=('LON', 'last'),
             o_lat=('LAT', 'first'),
             d_lat=('LAT', 'last'),
             o_num_pop=('BA_002', 'first'),
             d_num_pop=('BA_002', 'last'),
             o_renda=('BA_005', 'first'),
             d_renda=('BA_005', 'last'),
             o_num_alfa=('P1_001', 'first'),
             d_num_alfa=('P1_001', 'last'),
             o_neight_name=('NOMEBAIR_shp', 'first'),
             d_neight_name=('NOMEBAIR_shp', 'last'),
             o_timestamp=('DATAUTILIZACAO', 'first'),
             d_timestamp=('DATAUTILIZACAO', 'last'),
             codlinha=('CODVEICULO', 'first')
             )\
        .sort(['o_timestamp'])\
        .save.parquet('hdfs://{host}:9000/pp/ddf-pp'.format(host=HOST),
                      mode=True)

    # print('census:')
    # census_data.show()
    # print(census_data.count_rows())
    # cc.show_tasks()
    # cc.context_status()
    cc.stop()


if __name__ == '__main__':
    import sys
    size = sys.argv[1]
    nfrag = int(sys.argv[2])
    host = sys.argv[3]
    use_case(host, size, nfrag)
