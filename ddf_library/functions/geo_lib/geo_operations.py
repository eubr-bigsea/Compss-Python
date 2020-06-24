#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import generate_info

import pyproj


def crst_transform(data, settings):
    """
    Given a source EPSG code, and target EPSG code, convert the Spatial
    Reference System / Coordinate Reference System, e.g., OpenStreetMap is in
    a projected coordinate system that is based on the wgs84 datum. (EPSG 4326).

    :param data: DataFrame input
    :param settings: A dictionary with:
     * src_epsg: Geographic coordinate system used in the source points;
     * dst_epsg: UTM coordinate system to convert the input;
     * col_lat: Latitude column name;
     * col_lon:  Longitude column name;
     * alias_lon: Longitude column name (default, replace the input);
     * alias_lat: Latitude column name (default, replace the input);
    """

    src_epsg = settings['src_epsg']
    dst_epsg = settings['dst_epsg']
    lat_col = settings['lat_col']
    lon_col = settings['lon_col']
    alias_lon = settings.get('lon_alias', lon_col)
    alias_lat = settings.get('lat_col', lat_col)
    frag = settings['id_frag']

    old_proj = pyproj.Proj(src_epsg, preserve_units=True)
    new_proj = pyproj.Proj(dst_epsg, preserve_units=True)

    x, y = old_proj(data[lon_col].to_numpy(),
                    data[lat_col].to_numpy())

    x, y = pyproj.transform(old_proj, new_proj, x, y)

    if alias_lon is None:
        alias_lon = lat_col

    if alias_lat is None:
        alias_lat = lon_col

    data[alias_lon] = x
    data[alias_lat] = y

    info = generate_info(data, frag)
    return data, info


# TODO: WRT