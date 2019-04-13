#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from ddf_library.utils import generate_info
from pycompss.api.task import task
from pycompss.api.local import *
from pycompss.functions.reduce import merge_reduce

import pyqtree
import shapefile
from io import BytesIO, StringIO

import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta, datetime


def read_shapefile(settings, nfrag):
    """
    Reads a shapefile using the shp and dbf file.

    :param settings: A dictionary that contains:
        - host: The host of the Namenode HDFS; (default, default)
        - port: Port of the Namenode HDFS; (default, 0)
        - shp_path: Path to the shapefile (.shp)
        - dbf_path: Path to the shapefile (.dbf)
        - polygon: Alias to the new column to store the
            polygon coordenates (default, 'points');
        - attributes: List of attributes to keep in the dataframe,
            empty to use all fields;
        - lat_long: True  if the coordenates is (lat,log),
            False if is (long,lat). Default is True;

    .. note:: pip install pyshp
    """

    from hdfspycompss.Block import Block
    from hdfspycompss.HDFS import HDFS
    host = settings.get('host', 'localhost')
    port = settings.get('port', 9000)
    dfs = HDFS(host=host, port=port)

    polygon = settings.get('polygon', 'points')
    lat_long = settings.get('lat_long', True)
    header = settings.get('attributes', [])

    filename = settings['shp_path']
    blks = dfs.findNBlocks(filename, 1)
    shp_path = Block(blks[0]).readBinary()
    filename = settings['dbf_path']
    blks = dfs.findNBlocks(filename, 1)
    dbf_path = Block(blks[0]).readBinary()

    shp_io = BytesIO(shp_path)
    dbf_io = BytesIO(dbf_path)

    shp_object = shapefile.Reader(shp=shp_io, dbf=dbf_io)
    records = shp_object.records()
    sectors = shp_object.shapeRecords()

    fields = {}  # column: position
    for i, f in enumerate(shp_object.fields):
        fields[f[0]] = i
    del fields['DeletionFlag']

    if len(header) == 0:
        header = [f for f in fields]

    # position of each selected field
    num_fields = [fields[f] for f in header]

    header.append(polygon)
    data = []
    for i, sector in enumerate(sectors):
        attributes = []
        r = records[i]
        for t in num_fields:
            attributes.append(r[t-1])

        points = []
        for point in sector.shape.points:
            if lat_long:
                points.append([point[1], point[0]])
            else:
                points.append([point[0], point[1]])
        attributes.append(points)
        data.append(attributes)

    geo_data = pd.DataFrame(data, columns=header)

    # forcing pandas to infer the dtype
    tmp = geo_data.drop(polygon, axis=1)
    vector = geo_data[polygon]
    buff = tmp.to_csv(index=False)
    geo_data = pd.read_csv(StringIO(unicode(buff)))
    geo_data[polygon] = vector

    info = [geo_data.columns.tolist(), geo_data.dtypes.values, []]
    geo_data = np.array_split(geo_data, nfrag)
    for g in geo_data:
        info[2].append(len(g))

    return geo_data, info


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

    @local
    def prepare_spindex(self, settings, shp_object):
        polygon = settings.get('polygon', 'points')
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


class STDBSCAN(object):
    """STDBSCAN's methods.

    - fit_predict()
    """

    def fit_predict(self, df, settings, nfrag):
        """Fit_predict.

        :param df:       A list with nfrag pandas's dataframe.
        :param settings: A dictionary that contains:

          - lat_col:   Column name of the latitude  in the test data;
          - lon_col:   Column name of the longitude in the test data;
          - datetime:  Column name of the datetime  in the test data;

          - minPts:    The number of samples in a neighborhood for a point
                       to be considered as a core point;
                       This includes the point itself. (int, default: 15)

          - spatial_threshold:    The maximum distance (in meters) between two
                                  samples for them to be considered as in the
                                  same neighborhood. (float, default: 1000)
          - temporal_threshold:   The maximum distance temporal (in minutes)
                                  between two samples for them to be considered
                                  as in the same neighborhood.
                                  (float, default: 60)

        :param nfrag:  Number of fragments;
        :return: Returns the same list of dataframe with the cluster column.
        """
        if not all(['datetime' in settings,
                    'lat_col' in settings,
                    'lon_col' in settings]):
            raise Exception("Please inform, at least, the fields: "
                            "`lat_col`, `lon_col` and `datetime`")

        minPts = settings.get('minPts', 15)
        lat_col = settings['lat_col']
        lon_col = settings['lon_col']

        grids, divs = _fragment(df, nfrag, lat_col, lon_col)
        print ("[INFO] - Matrix: {}x{}".format(divs[0], divs[1]))
        nlat, nlon = divs

        # stage1 and stage2: partitionize and local dbscan
        t = 0
        partial = [[] for _ in range(nlat*nlon)]
        for l in range(nlat):
            for c in range(nlon):
                frag = []
                for f in range(nfrag):
                    frag = _partitionize(df[f], settings, grids[t], frag)

                partial[t] = _partial_stdbscan(frag, settings, "{}".format(t))
                t += 1

        # stage3: combining clusters
        n_iters_diagonal = (nlat-1)*(nlon-1)*2
        n_iters_horizontal = (nlat-1)*nlon
        n_iters_vertial = (nlon-1)*nlat
        n_iters_total = n_iters_vertial+n_iters_horizontal+n_iters_diagonal
        mapper = [[] for _ in range(n_iters_total)]

        m = 0
        for t in range(nlat*nlon):
            i = t % nlon  # which column
            j = t / nlon  # which line
            if i < (nlon-1):  # horizontal
                mapper[m] = _combine_clusters(partial[t], partial[t+1])
            if j < (nlat-1):
                mapper[m] = _combine_clusters(partial[t], partial[t+nlon])
            if i < (nlon-1) and j < (nlat-1):
                mapper[m] = _combine_clusters(partial[t], partial[t+nlon+1])
            if i > 0 and j < (nlat-1):
                mapper[m] = _combine_clusters(partial[t], partial[t+nlon-1])
            m += 1

        merged_mapper = merge_reduce(_merge_mapper, mapper)
        components = _find_components(merged_mapper)

        # stage4: _update_cluster
        result = [[] for _ in range(nlat*nlon)]
        for f in range(nlat*nlon):
            result[f] = _update_cluster(partial[f], components, grids[f])
        return result

# -----------------------------------------------------------------------------
#
#       stage1 and stage2: partitionize in 2dim and local dbscan
#
# -----------------------------------------------------------------------------


@task(returns=list)
def _get_bounds(df, lat_col, lon_col):
    """Get the maximum and minimum coordenates of each fragment."""
    mins = df[[lat_col, lon_col]].min(axis=0).values
    maxs = df[[lat_col, lon_col]].max(axis=0).values
    sums = df[[lat_col, lon_col]].sum(axis=0).values
    return [mins, maxs, sums, len(df)]


@task(returns=list)
def _merge_bounds(b1, b2):
    """Merge bounds coordenates."""
    mins1, maxs1, sums1, n1 = b1
    mins2, maxs2, sums2, n2 = b2

    if n1 > 0:
        if n2 > 0:
            min_lat = min([mins1[0], mins2[0]])
            min_lon = min([mins1[1], mins2[1]])
            max_lat = max([maxs1[0], maxs2[0]])
            max_lon = max([maxs1[1], maxs2[1]])
            sums = [sums1[0]+sums2[0], sums1[1]+sums2[1]]
            n = n1+n2
        else:
            min_lat = mins1[0]
            min_lon = mins1[1]
            max_lat = maxs1[0]
            max_lon = maxs1[1]
            sums = sums1
            n = n1
    else:
        min_lat = mins2[0]
        min_lon = mins2[1]
        max_lat = maxs2[0]
        max_lon = maxs2[1]
        sums = sums2
        n = n2

    mins = [min_lat, min_lon]
    maxs = [max_lat, max_lon]
    return [mins, maxs, sums, n]


@task(returns=list)
def _calc_variance(df, lat_col, lon_col, mean_lat, mean_lon):
    """Calculate the partial sum of latitude and longitude column."""
    if len(df) > 0:
        sum_lat = df.\
            apply(lambda row: (row[lat_col]-mean_lat)**2, axis=1).sum()
        sum_lon = df.\
            apply(lambda row: (row[lon_col]-mean_lon)**2, axis=1).sum()
    else:
        sum_lat = -1
        sum_lon = -1
    return [sum_lat, sum_lon]


@task(returns=list)
def _merge_variance(var1, var2):
    """Merge the variance."""
    if var1[0] > 0:
        if var2[0] > 0:
            var = [var1[0]+var2[0], var1[1]+var2[1]]
        else:
            var = var1
    else:
        var = var2
    return var


def _fragment(df, nfrag, lat_col, lon_col):
    """Create a list of grids."""
    from pycompss.api.api import compss_wait_on
    grids = []
    # retrieve the boundbox
    minmax = [_get_bounds(df[f], lat_col, lon_col) for f in range(nfrag)]
    minmax = merge_reduce(_merge_bounds, minmax)
    minmax = compss_wait_on(minmax)

    min_lat, min_lon = minmax[0]
    max_lat, max_lon = minmax[1]
    mean_lat = minmax[2][0]/minmax[3]
    mean_lon = minmax[2][1]/minmax[3]

    print """[INFO] - Boundbox:
     - South Latitude: {}
     - North Latitude: {}
     - West Longitude: {}
     - East Longitude: {}
    """.format(min_lat, max_lat, min_lon, max_lon)

    var_p = [_calc_variance(df[f], lat_col, lon_col, mean_lat, mean_lon)
             for f in range(nfrag)]
    var = merge_reduce(_merge_variance, var_p)
    var = compss_wait_on(var)

    t = int(np.sqrt(nfrag))
    if abs(var[0]-var[1]) <= 0.04:  # precision
        div = [t, t]
    elif var[0] < var[1]:
        div = [t+1, t]
    else:
        div = [t, t+1]

    div_lat = np.sqrt((max_lat - min_lat)**2)/div[0]
    div_lon = np.sqrt((max_lon - min_lon)**2)/div[1]

    init_lat = min_lat
    for ilat in range(div[0]):
        end_lat = init_lat + div_lat  # *(ilat+1)
        init_lon = min_lon
        for ilon in range(div[1]):
            end_lon = init_lon + div_lon  # *(ilon+1)
            g = [round(init_lat, 5), round(init_lon, 5),
                 round(end_lat, 5), round(end_lon, 5)]
            init_lon = end_lon
            grids.append(g)
        init_lat = end_lat

    return grids, div


@task(returns=list)
def _partitionize(df, settings, grid, frag):
    """Select points that belongs to each grid."""
    if len(df) > 0:
        spatial_threshold = settings.get('spatial_threshold', 100)
        lat_col = settings['lat_col']
        lon_col = settings['lon_col']

        init_lat,  init_lon,  end_lat, end_lon = grid

        # Note:
        # new_latitude  = latitude+(dy/r_earth)*(180/pi)
        # new_longitude = longitude+(dx/r_earth)*(180/pi)/cos(latitude*pi/180)

        dist = 2 * spatial_threshold * 0.0000089
        new_end_lat = end_lat + dist
        new_end_lon = end_lon + dist / np.cos(new_end_lat * 0.018)
        new_init_lat = init_lat - dist
        new_init_lon = init_lon - dist / np.cos(new_init_lat * 0.018)

        def check_bounds(point):
            return all([point[lat_col] >= new_init_lat,
                        point[lat_col] <= new_end_lat,
                        point[lon_col] >= new_init_lon,
                        point[lon_col] <= new_end_lon])

        tmp = df.apply(lambda point: check_bounds(point), axis=1)
        tmp = df.loc[tmp]

        if len(frag) > 0:
            frag = pd.concat([frag, tmp])
        else:
            frag = tmp

    return frag


@task(returns=list)
def _partial_stdbscan(df, settings, sufix):
    """Perform a partial stdbscan."""
    stack = []
    cluster_label = 0
    UNMARKED = -1
    NOISE = -999999

    df.reset_index(drop=True, inplace=True)
    spatial_threshold = settings['spatial_threshold']
    temporal_threshold = settings['temporal_threshold']
    minPts = settings['minPts']

    # columns
    lat_col = settings['lat_col']
    lon_col = settings['lon_col']
    dt_col = settings['datetime']
    clusterCol = settings['predCol']

    # creating a new tmp_id
    cols = df.columns
    idCol = 'tmp_stdbscan'
    i = 0
    while idCol in cols:
        idCol = 'tmp_stdbscan_{}'.format(i)
        i += 1
    settings['idCol'] = idCol

    df[idCol] = ['id{}_{}'.format(sufix, j) for j in range(len(df))]
    num_ids = len(df)
    C_UNMARKED = "p{}{}".format(sufix, UNMARKED)
    C_NOISE = "p{}{}".format(sufix, NOISE)
    df[clusterCol] = [C_UNMARKED for j in range(num_ids)]

    for index in range(num_ids):
        point = df.loc[index]
        CLUSTER = point[clusterCol]
        if CLUSTER == C_UNMARKED:
            X = _retrieve_neighbors(df, index, point, spatial_threshold,
                                    temporal_threshold,
                                    lat_col, lon_col, dt_col)

            if len(X) < minPts:
                df.loc[df.index[index], clusterCol] = C_NOISE
            else:  # found a core point, assign a label to core point
                cluster_label += 1
                df.loc[df.index[index], clusterCol] = sufix + str(cluster_label)

                for new_index in X:  # assign core's label to its neighborhood
                    df.loc[df.index[new_index],
                           clusterCol] = sufix + str(cluster_label)

                    if new_index not in stack:
                        stack.append(new_index)  # append neighborhood to stack

                    while len(stack) > 0:
                        # find new neighbors from core point neighborhood
                        newest_index = stack.pop()
                        new_point = df.loc[newest_index]
                        Y = _retrieve_neighbors(df, newest_index,
                                                new_point, spatial_threshold,
                                                temporal_threshold, lat_col,
                                                lon_col, dt_col)

                        if len(Y) >= minPts:  # current_point is a new core
                            for new_index_neig in Y:
                                neig_cluster = \
                                    df.loc[new_index_neig][clusterCol]
                                if neig_cluster == C_UNMARKED:
                                    df.loc[df.index[new_index_neig],
                                           clusterCol] \
                                        = sufix + str(cluster_label)
                                    if new_index_neig not in stack:
                                        stack.append(new_index_neig)

    settings['clusters'] = df[clusterCol].unique()
    return [df, settings]


def _retrieve_neighbors(df, i_point, point, spatial_threshold,
                        temporal_threshold, lat_col, lon_col, dt_col):
    """Retrieve a list of neighbors."""
    neigborhood = []
    timestamp = point[dt_col]
    min_time = timestamp - timedelta(minutes=temporal_threshold)
    max_time = timestamp + timedelta(minutes=temporal_threshold)
    df = df[(df[dt_col] >= min_time) & (df[dt_col] <= max_time)]
    for index, row in df.iterrows():
        if index != i_point:
                distance = _great_circle((point[lat_col], point[lon_col]),
                                         (row[lat_col], row[lon_col]))
                if distance <= spatial_threshold:
                    neigborhood.append(index)

    return neigborhood


def _great_circle(a, b):
    """Great-circle.

    The great-circle distance or orthodromic distance is the shortest
    distance between two points on the surface of a sphere, measured
    along the surface of the sphere (as opposed to a straight line
    through the sphere's interior).

    :Note: use cython in the future
    :returns: distance in meters.
    """
    import math
    earth_radius = 6371.009
    lat1, lng1 = np.radians(a[0]), np.radians(a[1])
    lat2, lng2 = np.radians(b[0]), np.radians(b[1])

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = math.atan2(np.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                           (cos_lat1 * sin_lat2 -
                           sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                   sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return (earth_radius * d) * 1000

# -----------------------------------------------------------------------------
#
#                   #stage3: combining clusters
#
# -----------------------------------------------------------------------------


@task(returns=dict)
def _combine_clusters(p1, p2):
    """Identify which points are duplicated in each grid pair."""
    df1, settings = p1
    df2 = p2[0]

    primary_key = settings['idCol']
    cluster_col = settings['predCol']
    a = settings['clusters']
    b = p2[1]['clusters']
    unique_c = np.unique(np.concatenate((a, b), 0))
    links = []

    if len(df1) > 0 and len(df2) > 0:
        merged = pd.merge(df1, df2, how='inner', on=[primary_key])

        for index, point in merged.iterrows():
            cluster_in_df1 = point[cluster_col+"_x"]
            cluster_in_df2 = point[cluster_col+"_y"]
            link = [cluster_in_df1, cluster_in_df2]
            if link not in links:
                links.append(link)

    result = dict()
    result['cluster'] = unique_c
    result['links'] = links
    print result
    return result


@task(returns=dict)
def _merge_mapper(mapper1, mapper2):
    """Merge all the duplicated points list."""
    if len(mapper1) > 0:
        if len(mapper2) > 0:
            clusters1 = mapper1['cluster']
            clusters2 = mapper2['cluster']
            clusters = np.unique(np.concatenate((clusters1, clusters2), 0))

            mapper1['cluster'] = clusters
            mapper1['links'] += mapper2['links']
    else:
        mapper1 = mapper2
    return mapper1


@task(returns=dict)
def _find_components(merged_mapper):
    """Find the list of components presents in the duplicated points."""
    AllClusters = merged_mapper['cluster']
    links = merged_mapper['links']
    i = 0
    G = nx.Graph()
    for line in links:
        G.add_edge(line[0], line[1])

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    oldC_newC = dict()

    for component in components:
        hascluster = any(["-9" not in c for c in component])
        if hascluster:
            i += 1
            for c in component:
                if "-9" not in c:
                    oldC_newC[c] = i

    # rename others clusters
    for c in AllClusters:
        if c not in oldC_newC:
            if "-9" not in c:
                i += 1
                oldC_newC[c] = i

    return oldC_newC

# -----------------------------------------------------------------------------
#
#                   #stage4: _update_cluster
#
# ------------------------------------------------------------------------------


@task(returns=list)
def _update_cluster(partial, oldC_newC, grid):
    """Update the information about the cluster."""
    df1, settings = partial
    clusters = settings['clusters']
    primary_key = settings['idCol']
    clusterCol = settings['predCol']
    lat_col = settings['lat_col']
    lon_col = settings['lon_col']

    init_lat,  init_lon,  end_lat, end_lon = grid

    if len(df1) > 0:
        f = lambda point: all([round(point[lat_col], 5) >= init_lat,
                               round(point[lon_col], 5) >= init_lon,
                               round(point[lat_col], 5) < (end_lat+0.00001),
                               round(point[lon_col], 5) < (end_lon+0.00001)])
        tmp = df1.apply(f, axis=1)
        df1 = df1.loc[tmp]
        df1.drop_duplicates([primary_key], inplace=False)
        df1.reset_index(drop=True, inplace=True)
        for key in oldC_newC:
            if key in clusters:
                df1.loc[df1[clusterCol] == key, clusterCol] = oldC_newC[key]

        df1.loc[df1[clusterCol].str.contains("-9", na=False), clusterCol] = -1
        df1 = df1.drop([primary_key], axis=1)

    return df1
