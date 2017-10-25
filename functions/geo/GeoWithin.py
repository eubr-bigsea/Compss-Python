#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *

import numpy  as np
import pandas as pd
import pyqtree


def GeoWithinOperation(data, shp_object, settings, numFrag):
    """
    GeoWithinOperation():

    To each record in the data input, returns the sectors that the
    point belongs.

    :param data:       A list of pandas dataframe;
    :param shp_object: The dataframe created by the function ReadShapeFile;
    :param settings:   A dictionary that contains:
        - lat_col:     Column which represents the Latitute field in the data;
        - lon_col:     Column which represents the Longitude field in the data;
        - lat_long:    True  if the coordenates is (lat,log),
                       False if is (long,lat). Default is True;
        - polygon:     Field in shp_object where is store the
                       coordinates of each sector;
        - attributes:  Attributes to retrieve from shapefile, empty to all
                       (default, empty);
        - alias:       Alias for shapefile attributes
                       (default, 'sector_position');
    :param numFrag:    The number of fragments;
    :return:           Returns a list of pandas daraframe.

    """

    if not all(['lat_col' in settings,
                'lon_col' in settings]):
        raise Exception("Please inform, at least, the fields: "
                        "`attributes`,`lat_col` and `lon_col`")

    polygon = settings.get('polygon','points')
    if settings.get('lat_long', True):
        LAT_pos = 0
        LON_pos = 1
    else:
        LAT_pos = 1
        LON_pos = 0

    xmin = float('+inf')
    ymin = float('+inf')
    xmax = float('-inf')
    ymax = float('-inf')
    for i, sector in shp_object.iterrows():
        for point in sector[polygon]:
            xmin = min(xmin, point[LON_pos])
            ymin = min(ymin, point[LAT_pos])
            xmax = max(xmax, point[LON_pos])
            ymax = max(ymax, point[LAT_pos])

    #create the main bound box
    spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])

    #than, insert all sectors bbox
    for inx, sector in shp_object.iterrows():
        points = []
        xmin = float('+inf')
        ymin = float('+inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for point in sector[polygon]:
            points.append((point[LON_pos], point[LAT_pos]))
            xmin = min(xmin, point[LON_pos])
            ymin = min(ymin, point[LAT_pos])
            xmax = max(xmax, point[LON_pos])
            ymax = max(ymax, point[LAT_pos])
        spindex.insert(item=inx, bbox=[xmin, ymin, xmax, ymax])

    for f in range(numFrag):
        data[f] =  get_sectors(data[f], spindex, shp_object, settings)

    return data

@task(returns=list)
def get_sectors(data_input, spindex, shp_object, settings):
    from matplotlib.path import Path
    alias = settings.get('alias', 'sector_position')
    attributes = settings.get('attributes', [])

    sector_position = []

    if len(data_input)>0:

        col_lat  = settings['lat_col']
        col_long = settings['lon_col']
        if len(attributes) == 0:
            attributes = shp_object.columns
        polygon_col = settings.get('polygon','points')

        for i,point in data_input.iterrows():
            tmp = []
            y = float(point[col_lat])
            x = float(point[col_long])

            #(xmin,ymin,xmax,ymax)
            matches = spindex.intersect([x, y, x, y])

            for shp_inx in matches:
                row = shp_object.loc[shp_inx]
                polygon = Path(row[polygon_col])
                if polygon.contains_point([y, x]):
                    tmp.append(row[attributes])

            sector_position.append(tmp )

    data_input[alias] =  sector_position
    return data_input
