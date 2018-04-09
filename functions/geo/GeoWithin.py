#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Geo Within Operation: returns the sectors that the each point belongs."""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on
import pandas as pd
import pyqtree


class GeoWithinOperation(object):

    def __init__(self):
        pass

    def transform(self, data, shp_object, settings, numFrag):
        """
        GeoWithinOperation.

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

        dicty = self.prepare_spindex(settings, shp_object)
        dicty = compss_wait_on(dicty)
        spindex = dicty['spindex']
        shp_object = dicty['shp_object']
        result = [[] for f in range(numFrag)]
        for f in range(numFrag):
            result[f] = self.get_sectors(data[f], spindex, shp_object, settings)

        return result


    @task(returns=dict)
    def prepare_spindex(self, settings, shp_object):
        polygon = settings.get('polygon', 'points')
        if settings.get('lat_long', True):
            LAT_pos = 0
            LON_pos = 1
        else:
            LAT_pos = 1
            LON_pos = 0

        shp_object = pd.concat(shp_object)

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

        # create the main bound box
        spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])

        # than, insert all sectors bbox
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

        dicty = {}
        dicty['spindex'] = spindex
        dicty['shp_object'] = shp_object

        return dicty


    @task(returns=list)
    def get_sectors(self, data_input, spindex, shp_object, settings):
        """Retrieve the sectors of each fragment."""
        from matplotlib.path import Path
        alias = settings.get('alias', '_sector_position')
        attributes = settings.get('attributes', shp_object.columns)
        sector_position = []
        col_lat = settings['lat_col']
        col_long = settings['lon_col']
        polygon_col = settings.get('polygon', 'points')

        if len(data_input) > 0:
            for i, point in data_input.iterrows():
                y = float(point[col_lat])
                x = float(point[col_long])

                # (xmin,ymin,xmax,ymax)
                matches = spindex.intersect([x, y, x, y])

                for shp_inx in matches:
                    row = shp_object.loc[shp_inx]
                    polygon = Path(row[polygon_col])
                    if polygon.contains_point([y, x]):
                        content = [i] + row[attributes].tolist()
                        sector_position.append(content)

        if len(sector_position) > 0:
            tmp = pd.DataFrame(sector_position)
            attributes = ['index_geoWithin'] + attributes
            tmp.columns = ["{}{}".format(a, alias) for a in attributes]

            key = 'index_geoWithin'+alias
            data_input = pd.merge(data_input, tmp,
                                  left_index=True, right_on=key)
            data_input = data_input.drop([key], axis=1)
        else:
            import numpy as np
            for a in [a+alias for a in attributes]:
                data_input[a] = np.nan

        return data_input
