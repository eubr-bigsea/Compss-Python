#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.utils import generate_info, create_auxiliary_column

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

from matplotlib.path import Path
import pandas as pd
import numpy as np
import pyqtree


class GeoWithin(object):
    """Returns the sectors that the each point belongs."""

    @staticmethod
    def geo_within_stage_1(data, shp_object, settings):
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

        if not all(['lat_col' in settings,
                    'lon_col' in settings]):
            raise Exception("Please inform, at least, the fields: "
                            "`lat_col` and `lon_col`")

        shp_object = merge_reduce(_merge_shapefile, shp_object)

        shp_object = compss_wait_on(shp_object)
        polygon = settings.get('polygon', 'points')
        if settings.get('lat_long', True):
            lat_idx, lon_idx = 0, 1
        else:
            lat_idx, lon_idx = 1, 0

        # create the main bound box
        points_by_sector = shp_object[polygon].values.tolist()
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
        tmp = [[] for _ in data]
        return data, tmp, settings

    @staticmethod
    def geo_within_stage_2(data_input, _, settings):
        """Retrieve the sectors of each fragment."""
        spindex = settings['spindex']
        frag = settings['id_frag']
        shp_object = settings['shp_object']

        alias = settings.get('alias', '_shp')
        attributes = settings.get('attributes', list(shp_object.columns))
        col_lat, col_lon = settings['lat_col'], settings['lon_col']
        polygon_col = settings.get('polygon', 'points')
        polygon_col_idx = shp_object.columns.get_loc(polygon_col)

        data_input.reset_index(drop=True, inplace=True)
        sector_position = []

        if len(data_input) > 0:
            for i, point in enumerate(data_input[[col_lon, col_lat]].values):
                x, y = point

                # first, find the squares where point is inside (coarse-grained)
                # (xmin,ymin,xmax,ymax)
                matches = spindex.intersect([x, y, x, y])

                # then, to all selected squares, check if point is in polygon
                # (fine-grained)
                for shp_inx in matches:
                    row = shp_object.iat[shp_inx, polygon_col_idx]
                    polygon = Path(row)
                    if polygon.contains_point([y, x]):
                        sector_position.append([i, shp_inx])

        if len(sector_position) > 0:
            cols = data_input.columns.tolist()
            col_tmp1 = create_auxiliary_column(cols)
            col_tmp2 = create_auxiliary_column(cols + [col_tmp1])
            df = pd.DataFrame(sector_position, columns=[col_tmp1, col_tmp2])

            # filter rows in data_input and shp_object
            data_input = data_input[data_input.index.isin(df[col_tmp1].values)]

            shp_object = shp_object[attributes]
            shp_object.columns = ["{}{}".format(c, alias) for c in attributes]
            shp_object = shp_object[shp_object.index.isin(df[col_tmp2].values)]

            # merge with idx of each point
            data_input = data_input.merge(df, how='inner',
                                          left_index=True, right_on=col_tmp1,
                                          copy=False)
            del df

            # merge with each sector
            data_input = data_input.merge(shp_object, how='inner',
                                          left_on=col_tmp2, right_index=True,
                                          copy=False)
            del shp_object

            data_input = data_input.drop([col_tmp1, col_tmp2], axis=1)
        else:

            data_input = data_input[0:0]
            for a in [a + alias for a in attributes]:
                data_input[a] = np.nan

        data_input.reset_index(drop=True, inplace=True)
        info = generate_info(data_input, frag)
        return data_input, info


def _find_minmax(sector, lon_idx, lat_idx):
    sector = np.array(sector)
    mins = np.nanmin(sector, axis=0)
    maxs = np.nanmax(sector, axis=0)
    xmin, ymin = mins[lon_idx], mins[lat_idx]
    xmax, ymax = maxs[lon_idx], maxs[lat_idx]
    return [xmin, ymin, xmax, ymax]


@task(returns=1, priority=True)
def _merge_shapefile(shape1, shape2):
    shape1 = pd.concat([shape1, shape2], sort=False, ignore_index=True)
    shape1.reset_index(drop=True, inplace=True)
    return shape1
