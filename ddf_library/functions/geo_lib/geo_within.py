#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.utils import generate_info, create_auxiliary_column, \
    read_stage_file
from pycompss.api.task import task
from pycompss.api.parameter import COLLECTION_FILE_IN
from pycompss.api.api import compss_wait_on

from matplotlib.path import Path
import pandas as pd
import numpy as np
import pyqtree

"""Returns the sectors that the each point belongs."""


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

    if not all(['lat_col' in settings, 'lon_col' in settings]):
        raise Exception("Please inform, at least, the fields: "
                        "`lat_col` and `lon_col`")

    attributes = settings.get('attributes', None)
    polygon = settings.get('polygon', 'points')

    # merging the geo data, selecting only the important columns
    shp_object = _merge_shapefile(shp_object, attributes, polygon)
    shp_object = compss_wait_on(shp_object)

    if settings.get('lat_long', True):
        lat_idx, lon_idx = 0, 1
    else:
        lat_idx, lon_idx = 1, 0

    # creating the main bound box
    min_max = []
    for sector in shp_object[polygon].to_numpy():
        min_max.append(_find_minmax(sector, lon_idx, lat_idx))
    min_max = np.array(min_max)

    # lat = y and lon = x
    xmin, ymin = np.min(min_max[:, 0]), np.min(min_max[:, 1])
    xmax, ymax = np.max(min_max[:, 2]), np.max(min_max[:, 3])

    # Pyqtree is a pure Python spatial index for GIS or rendering usage.
    # It stores and quickly retrieves items from a 2x2 rectangular grid
    # area, and grows in depth and detail as more items are added.
    spindex = pyqtree.Index(bbox=[xmin, ymin, xmax, ymax])

    # than, insert all sectors bbox
    for i, sector in enumerate(min_max):
        spindex.insert(item=i, bbox=sector)

    settings['spindex'] = spindex
    settings['shp_object'] = shp_object
    settings['intermediate_result'] = False

    return data, None, settings


def geo_within_stage_2(data_input, settings):
    """Retrieve the sectors of each fragment."""
    spindex = settings['spindex']
    frag = settings['id_frag']
    shp_object = settings['shp_object']

    attributes = settings.get('attributes', None)
    if attributes is None:
        attributes = list(shp_object.columns)
    alias = settings.get('alias', '_shp')
    col_lon_lat = [settings['lon_col'], settings['lat_col']]
    polygon_col = settings.get('polygon', 'points')
    polygon_col_idx = shp_object.columns.get_loc(polygon_col)

    data_input.reset_index(drop=True, inplace=True)
    sector_position = [-1] * len(data_input)

    if len(data_input) > 0:
        points = data_input[col_lon_lat].to_numpy().tolist()

        def get_first_polygon(y, x):
            # first, find the squares where point is inside (coarse-grained)
            # (xmin,ymin,xmax,ymax)
            matches = spindex.intersect([x, y, x, y])

            # then, to all selected squares, check if point is in polygon
            # (fine-grained)
            for shp_inx in matches:
                row = shp_object.iat[shp_inx, polygon_col_idx].tolist()
                polygon = Path(row)
                if polygon.contains_point([y, x]):
                    return shp_inx
            return None

        for i, point in enumerate(points):
            x, y = point
            sector_position[i] = get_first_polygon(y, x)

    cols = data_input.columns.tolist()
    col_tmp1 = create_auxiliary_column(cols)
    data_input[col_tmp1] = sector_position

    if polygon_col not in attributes:
        shp_object = shp_object.drop([polygon_col], axis=1)

    shp_object.columns = ["{}{}".format(c, alias) for c in attributes]

    # merge with idx of each point
    data_input = data_input.merge(shp_object, how='left',
                                  left_on=col_tmp1, right_index=True,
                                  copy=False)

    del shp_object

    data_input = data_input.drop([col_tmp1], axis=1)\
        .reset_index(drop=True)
    info = generate_info(data_input, frag)
    return data_input, info


def _find_minmax(sector, lon_idx, lat_idx):

    tmp = pd.DataFrame(sector.tolist(), columns=[0, 1])

    mins = tmp.min(skipna=True, axis=0)
    maxs = tmp.max(skipna=True, axis=0)

    xmin, ymin = mins[lon_idx], mins[lat_idx]
    xmax, ymax = maxs[lon_idx], maxs[lat_idx]

    return [xmin, ymin, xmax, ymax]


@task(returns=1, priority=True, shapes=COLLECTION_FILE_IN)
def _merge_shapefile(shapes, attributes, polygon):
    if isinstance(attributes, list):
        attributes += [polygon]
    shapes = [read_stage_file(file, cols=attributes) for file in shapes]
    shapes = pd.concat(shapes, sort=False, ignore_index=True)
    shapes.reset_index(drop=True, inplace=True)
    return shapes
