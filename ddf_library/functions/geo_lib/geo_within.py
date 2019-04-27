#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.utils import generate_info

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.api import compss_wait_on

from matplotlib.path import Path
import pandas as pd
import numpy as np
import pyqtree


class GeoWithinOperation(object):
    """
    Returns the sectors that the each point belongs.
    """

    def transform(self, data, shp_object, settings):
        """
        :param data: A list of pandas dataframe;
        :param shp_object: The dataframe created by the function ReadShapeFile;
        :param settings: A dictionary that contains:
            - lat_col: Column which represents the Latitute field in the data;
            - lon_col: Column which represents the Longitude field in the data;
            - lat_long: True  if the coordenates is (lat,log),
                        False if is (long,lat). Default is True;
            - polygon: Field in shp_object where is store the
                coordinates of each sector;
            - attributes: Attributes to retrieve from shapefile, empty to all
                    (default, empty);
            - alias: Alias for shapefile attributes
                (default, 'sector_position');
        :return: Returns a list of pandas daraframe.
        """

        nfrag = len(data)
        settings = self.preprocessing(settings, shp_object)

        info = [[] for _ in range(nfrag)]
        result = [[] for _ in range(nfrag)]
        for f in range(nfrag):
            result[f], info[f] = _get_sectors(data[f], settings, f)

        output = {'key_data': ['data'], 'key_info': ['info'],
                  'data': result, 'info': info}
        return output

    def preprocessing(self, settings, shp_object):
        if not all(['lat_col' in settings,
                    'lon_col' in settings]):
            raise Exception("Please inform, at least, the fields: "
                            "`lat_col` and `lon_col`")

        shp_object = merge_reduce(_merge_shapefile, shp_object)

        settings = self.prepare_spindex(settings, shp_object)
        return settings

    def prepare_spindex(self, settings, shp_object):
        shp_object = compss_wait_on(shp_object)
        polygon = settings.get('polygon', 'points')
        if settings.get('lat_long', True):
            LAT_pos, LON_pos = 0, 1
        else:
            LAT_pos, LON_pos = 1, 0

        xmin, ymin = float('+inf'), float('+inf')
        xmax, ymax = float('-inf'), float('-inf')

        # TODO: Can be optimized
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

        settings['spindex'] = spindex
        settings['shp_object'] = shp_object

        return settings


@task(returns=1, priority=True)
def _merge_shapefile(shape1, shape2):
    return pd.concat([shape1, shape2], sort=False, ignore_index=True)


@task(returns=2)
def _get_sectors(data_input, settings, frag):
    """Retrieve the sectors of each fragment."""
    shp_object = settings['shp_object']
    spindex = settings['spindex']

    alias = settings.get('alias', '_sector_position')
    attributes = settings.get('attributes', shp_object.columns)
    sector_position = []
    col_lat, col_long = settings['lat_col'], settings['lon_col']
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
        attributes = ['index_geoWithin'] + attributes
        cols = ["{}{}".format(a, alias) for a in attributes]
        tmp = pd.DataFrame(sector_position, columns=cols)

        key = 'index_geoWithin'+alias
        data_input = pd.merge(data_input, tmp,
                              left_index=True, right_on=key)
        data_input = data_input.drop([key], axis=1)
    else:

        for a in [a + alias for a in attributes]:
            data_input[a] = np.nan

    data_input = data_input.reset_index(drop=True)

    info = generate_info(data_input, frag)
    return data_input, info
