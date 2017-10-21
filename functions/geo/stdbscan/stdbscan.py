#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import time
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from math import atan, tan, sin, cos, pi, sqrt, atan2, asin, radians

#COMPSs's imports
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce



class STDBSCAN(object):

    def fit_predict(self, df, settings, numFrag):
        """
        fit():
        Perform STDBSCAN clustering from the features column.

        :param df:       A list with numFrag pandas's dataframe.
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

        :param numFrag:  Number of fragments;
        :return:     Returns the same list of dataframe with the cluster column.
        """
        from pycompss.api.api import compss_wait_on

        if not all(['datetime' in settings,
                    'lat_col'  in settings,
                    'lon_col'  in settings]):
            raise Exception("Please inform, at least, the fields: "
                            "`lat_col`, `lon_col` and `datetime`")

        #settings
        minPts             = settings.get('minPts', 15)
        spatial_threshold  = settings.get('spatial_threshold', 1000) #meters
        temporal_threshold = settings.get('temporal_threshold', 60) #minutes

        #columns
        lat_col = settings['lat_col']
        lon_col = settings['lon_col']
        dt_col  = settings['datetime']
        predCol = settings.get('predCol', "cluster")


        grids, divs= self.fragment(df, numFrag, lat_col, lon_col)
        print "[INFO] - Matrix: {}x{}".format(divs[0],divs[1])
        nlat,nlon = divs

        #stage1 and stage2: partitionize and local dbscan
        t = 0
        partial =  [ [] for f in range(nlat*nlon)]
        for l in range(nlat):
            for c in range(nlon):
                frag = []
                for f in range(numFrag):
                   frag   =  self.partitionize(df[f], settings, grids[t], frag)

                partial[t]=self.partial_dbscan(frag,settings, "p_{}".format(t))
                t+=1

        #stage3: combining clusters
        n_iters_diagonal   = (nlat-1)*(nlon-1)*2
        n_iters_horizontal = (nlat-1)*nlon
        n_iters_vertial    = (nlon-1)*nlat
        n_inters_total =  n_iters_vertial+n_iters_horizontal+n_iters_diagonal
        mapper  =  [ [] for f in range(n_inters_total) ]
        m = 0
        for t in range(nlat*nlon):
            i = t % nlon #which column
            j = t / nlon #which line
            if i < (nlon-1): #horizontal
                #print "M={} | t:{} and (t+1):{}".format(m, t, t+1)
                mapper[m] = self.CombineClusters(partial[t], partial[t+1])
                m+=1

            if j < (nlat-1):
                #print "M={} | t:{} and (t+nlat):{}".format(m, t, t+nlon)
                mapper[m] = self.CombineClusters(partial[t], partial[t+nlon])
                m+=1

            if i < (nlon-1) and j < (nlat-1):
                #print "M={} | t:{} and (t+div+1):{}".format(m, t, t+nlon+1)
                mapper[m] = self.CombineClusters(partial[t], partial[t+nlon+1])
                m+=1

            if i>0 and j < (nlat-1):
                #print "M={} | t:{} and (t+div+1):{}".format(m, t, t+nlon-1)
                mapper[m] = self.CombineClusters(partial[t], partial[t+nlon-1])
                m+=1

        merged_mapper = mergeReduce(self.MergeMapper, mapper)
        components    = self.findComponents(merged_mapper, minPts)

        #stage4: updateClusters
        t = 0
        result =  [ [] for f in range(nlat*nlon)]
        for t in range(nlat*nlon):
            result[t]  =  self.updateClusters(partial[t], components, grids[t])


        return result

#-------------------------------------------------------------------------------
#
#       stage1 and stage2: partitionize in 2dim and local dbscan
#
#-------------------------------------------------------------------------------

    @task(returns=list, isModifier = False)
    def get_bounds(self, df, lat_col, lon_col):
        mins = df[[lat_col,lon_col]].min(axis=0).values
        maxs = df[[lat_col,lon_col]].max(axis=0).values
        sums = df[[lat_col,lon_col]].sum(axis=0).values
        n = len(df) if len(df) > 0 else -1
        return [mins,maxs,sums,len(df)]

    @task(returns=list, isModifier = False)
    def mergeBounds(self, b1, b2):
        mins1, maxs1,sums1,n1 = b1
        mins2, maxs2,sums2,n2 = b2

        if n1 > 0:
            if n2>0:

                min_lat =  min([mins1[0], mins2[0]])
                min_lon =  min([mins1[1], mins2[1]])
                max_lat =  max([maxs1[0], maxs2[0]])
                max_lon =  max([maxs1[1], maxs2[1]])
                sums = [sums1[0]+sums2[0],sums1[1]+sums2[1]]
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
        return [ mins, maxs , sums, n]

    @task(returns=list, isModifier = False)
    def calc_var(self, df, lat_col, lon_col, mean_lat, mean_lon):
        if len(df)>0:
            sum_lat = df.\
                apply(lambda row: (row[lat_col]-mean_lat)**2,axis=1).sum()
            sum_lon = df.\
                apply(lambda row: (row[lon_col]-mean_lon)**2,axis=1).sum()
        else:
            sum_lat= -1
            sum_lon= -1
        return [sum_lat,sum_lon]

    @task(returns=list, isModifier = False)
    def mergevar(self,var1,var2):
        if var1[0]>0:
            if var2[0]>0:
                var = [var1[0]+var2[0],var1[1]+var2[1]]
            else:
                var = var1
        else:
            var = var2
        return var

    def fragment(self, df, numFrag, lat_col, lon_col):
        from pycompss.api.api import compss_wait_on
        grids   =  []
        #retrieve the boundbox
        minmax  = [ self.get_bounds(df[f], lat_col,lon_col)
                        for f in range(numFrag) ]
        minmax  = mergeReduce(self.mergeBounds,minmax)
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
        """.format(min_lat,max_lat,min_lon,max_lon)

        var_p = [self.calc_var(df[f], lat_col, lon_col, mean_lat, mean_lon)
                    for f in range(numFrag) ]
        var = mergeReduce(self.mergevar, var_p)
        var  = compss_wait_on(var)

        t = int(np.sqrt(numFrag))
        if abs(var[0]-var[1])<=0.04:
            div = [t,t]
        elif var[0]<var[1]:
            div = [t+1,t]
        else:
            div = [t,t+1]

        div_lat = np.sqrt((max_lat - min_lat)**2)/div[0]
        div_lon = np.sqrt((max_lon - min_lon)**2)/div[1]

        init_lat = min_lat
        for ilat in range(div[0]):
            end_lat = init_lat + div_lat#*(ilat+1)
            init_lon = min_lon
            for ilon in range(div[1]):
                end_lon = init_lon + div_lon#*(ilon+1)
                g = [ round(init_lat,5),  round(init_lon,5),
                      round(end_lat,5), round(end_lon,5)]
                init_lon = end_lon
                grids.append(g)
            init_lat = end_lat

        return grids,div


    def haversine(self, grid):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lat1, lon1, lat2, lon2 = grid
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6367 * c
        return km


    @task(returns=list, isModifier = False)
    def partitionize(self, df, settings, grid, frag):
        if len(df)>0:
            spatial_threshold  = settings.get('spatial_threshold',100)
            lat_col = settings['lat_col']
            lon_col = settings['lon_col']

            init_lat,  init_lon,  end_lat, end_lon = grid

            #Note:
            #new_latitude  = latitude+(dy/r_earth)*(180/pi)
            #new_longitude = longitude+(dx/r_earth)*(180/pi)/cos(latitude*pi/180)

            dist = 2 * spatial_threshold * 0.0000089
            new_end_lat = end_lat + dist
            new_end_lon = end_lon + dist / np.cos(new_end_lat * 0.018)
            new_init_lat = init_lat - dist
            new_init_lon = init_lon - dist / np.cos(new_init_lat * 0.018)

            f = lambda point: all([ point[lat_col] >=  new_init_lat  ,
                                    point[lat_col] <=  new_end_lat ,
                                    point[lon_col] >=  new_init_lon ,
                                    point[lon_col] <=  new_end_lon
                                ])

            tmp =  df.apply(f, axis=1)
            tmp = df.loc[tmp]

            if len(frag)>0:
                frag = pd.concat([frag, tmp])
            else:
                frag = tmp

        return frag



    @task(returns=list, isModifier = False)
    def partial_dbscan(self,df, settings, sufix):

        stack = []
        cluster_label = 0
        cluster_noise = 0
        UNMARKED      = -1
        NOISE         = -999999

        df.reset_index(drop=True,inplace=True)

        #settings
        spatial_threshold   = settings['spatial_threshold']
        temporal_threshold  = settings['temporal_threshold']
        minPts              = settings['minPts']

        #columns
        lat_col     = settings['lat_col']
        lon_col     = settings['lon_col']
        dt_col      = settings['datetime']
        clusterCol  = settings['predCol']

        #creating a new tmp_id
        cols = df.columns
        idCol = 'tmp_stbscan'
        i = 0
        while idCol in cols:
            idCol = 'tmp_stbscan_{}'.format(i)
            i+=1
        df[idCol] = df.index
        settings['idCol'] = idCol

        num_ids = len(df)
        C_UNMARKED  = "{}{}".format(sufix,UNMARKED)
        C_NOISE     = "{}{}".format(sufix,NOISE)
        df[clusterCol]   = [ C_UNMARKED for i in range(num_ids)]
        cores = list()
        noise = dict()
        for index in range(num_ids):
            point = df.loc[index]
            CLUSTER = point[clusterCol]

            if CLUSTER == C_UNMARKED:

                X = self.retrieve_neighbors( df, index, point,
                                                    spatial_threshold,
                                                    temporal_threshold,
                                                    lat_col, lon_col,dt_col )

                if len(X) < minPts:
                    #cluster_noise += 1
                    df.set_value(index, clusterCol, C_NOISE)
                    # tmp = []
                    # for x in X:
                    #     tmp.append(df.loc[x][idCol])
                    # noise[str(point[idCol])] = tmp
                else: # found a core point
                    cluster_label += 1
                    # assign a label to core point
                    df.set_value(index,clusterCol, sufix + str(cluster_label))
                    cores.append(point[idCol])

                    for new_index in X: # assign core's label to its neighborhood
                        df.set_value(new_index, clusterCol,
                                        sufix + str(cluster_label))
                        if new_index not in stack:
                            stack.append(new_index)#append neighborhood to stack

                        while len(stack) > 0:
                            #find new neighbors from core point neighborhood
                            newest_index  = stack.pop()
                            new_point = df.loc[newest_index]
                            Y = self.retrieve_neighbors(df,newest_index,
                                    new_point, spatial_threshold,
                                    temporal_threshold, lat_col,
                                    lon_col, dt_col )

                            if len(Y) >= minPts: # current_point is a new core
                                cores.append(df.loc[newest_index][idCol])
                                for new_index_neig in Y:
                                    neig_cluster = \
                                        df.loc[new_index_neig][clusterCol]
                                    if (neig_cluster == C_UNMARKED):
                                        df.set_value(new_index_neig, clusterCol,
                                                    sufix + str(cluster_label))
                                        if new_index_neig not in stack:
                                            stack.append(new_index_neig)

        settings['clusters'] = df[clusterCol].unique()
        settings['noise'] = noise
        settings['cores'] = cores
        print df
        return [df, settings]

    def retrieve_neighbors(self, df, i_point, point, spatial_threshold,
                                temporal_threshold, lat_col, lon_col, dt_col):

        neigborhood = []
        DATATIME    = point[dt_col]

        min_time = DATATIME - timedelta(minutes = temporal_threshold)
        max_time = DATATIME + timedelta(minutes = temporal_threshold)
        df = df[(df[dt_col] >= min_time) & (df[dt_col] <= max_time)]
        for index, row in df.iterrows():
            if index != i_point:
                    distance = self.great_circle(
                                            (point[lat_col], point[lon_col]),
                                            (row[lat_col], row[lon_col])
                                )
                    if distance <= spatial_threshold:
                        neigborhood.append(index)

        return neigborhood




    def great_circle(self, a, b):
        """
            The great-circle distance or orthodromic distance is the shortest
            distance between two points on the surface of a sphere, measured
            along the surface of the sphere (as opposed to a straight line
            through the sphere's interior).

            :Note: use cython in the future
            :returns: distance in meters.
        """

        EARTH_RADIUS = 6371.009
        lat1, lng1 = radians(a[0]), radians(a[1])
        lat2, lng2 = radians(b[0]), radians(b[1])

        sin_lat1, cos_lat1 = sin(lat1), cos(lat1)
        sin_lat2, cos_lat2 = sin(lat2), cos(lat2)

        delta_lng = lng2 - lng1
        cos_delta_lng, sin_delta_lng = cos(delta_lng), sin(delta_lng)

        d = atan2(sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                       (cos_lat1 * sin_lat2 -
                        sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                  sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

        return (EARTH_RADIUS * d)*1000

#-------------------------------------------------------------------------------
#
#                   #stage3: combining clusters
#
#-------------------------------------------------------------------------------


    @task(returns=dict, isModifier = False)
    def CombineClusters(self, p1, p2):

        df1, settings = p1
        df2           = p2[0]

        primary_key = settings['idCol']
        clusterCol  = settings['predCol']
        minPts      = settings['minPts']
        a = settings['clusters']
        b = p2[1]['clusters']
        unique_c = np.unique(np.concatenate((a,b),0))
        noise1 = settings['noise']
        noise2 = p2[1]['noise']
        links = []
        noise_r = dict()
        cores = settings['cores'] + list(   set(p2[1]['cores']) -
                                            set(settings['cores']))


        if len(df1)>0 and len(df2)>0:
            merged = pd.merge(df1, df2, how='inner', on=[primary_key] )

            for index,point in merged.iterrows():
                CLUSTER_DF1 = point[clusterCol+"_x"]
                CLUSTER_DF2 = point[clusterCol+"_y"]
                l = [CLUSTER_DF1, CLUSTER_DF2]
                if l not in links:
                    links.append(l)


                # if all(["-99" for c in CLUSTER_DF1]):
                #         key = point[primary_key]
                #         n = list(set(noise1[str(key)] + noise2[str(key)]))
                #         noise_r[str(key)] = n

        result = dict()
        result['cluster']  = unique_c
        result['links']    = links
        result['noise']    = noise_r
        result['cores']    = cores

        return result

    @task(returns=dict, isModifier = False)
    def MergeMapper(self,mapper1,mapper2):

        clusters1 = mapper1['cluster']
        clusters2 = mapper2['cluster']
        clusters  = np.unique(np.concatenate((clusters1,clusters2),0))

        noise1 = mapper1['noise']
        noise2 = mapper2['noise']
        for k in noise2:
            if k not in noise1:
                noise1[k] = noise2[k]
            else:
                noise1[k] = list(set(noise1[k] + noise2[k]))

        mapper1['cluster'] = clusters
        mapper1['cores']  += mapper2['cores']
        mapper1['links']  += mapper2['links']
        mapper1['noise']   = noise1

        return mapper1


    @task(returns=list, isModifier = False)
    def findComponents(self, merged_mapper, minPts):

        import networkx as nx

        AllClusters = merged_mapper['cluster']
        links       = merged_mapper['links']
        noise       = merged_mapper['noise']
        cores       = merged_mapper['cores']
        i = 0

        G = nx.Graph()
        for line in links:
            G.add_edge(line[0], line[1])

        components = sorted(nx.connected_components(G), key = len, reverse=True)

        oldC_newC = dict()
        id_newC = dict()


        for  component in components:
            hascluster = any(["-999" not in c for c in component])
            #hascore  = any([c in cores for c in component])

            if hascluster:
                i+=1
                for c in component:
                    if "-999" not in c:
                        oldC_newC[c] = i
            # else:
            #     i+=1
            #     notnoise = [c for c in component if "-999" not in c ]
            #     print notnoise
            #     for c in notnoise:
            #         oldC_newC[c] = i


        #rename others clusters
        for  c in AllClusters:
            if c not in oldC_newC:
                if "-99" not  in c:
                    i+=1
                    oldC_newC[c]=i


        # G = nx.Graph()
        # for key in noise:
        #     neighbors = noise[key]
        #     if len(neighbors) > minPts: #new core
        #         for v in neighbors:
        #             G.add_edge(key, v)
        #
        # components = sorted(nx.connected_components(G), key = len, reverse=True)
        # print components
        #
        #
        # for  component in components:
        #     i+=1
        #     for c in component:
        #         id_newC[str(c)] = i

        #
        # print "----------------------"
        # print "oldC_newC",oldC_newC
        # print "----------------------"
        # print "id_newC",id_newC
        # print "----------------------"

        return [oldC_newC,id_newC]




#-------------------------------------------------------------------------------
#
#                   #stage4: updateClusters
#
#-------------------------------------------------------------------------------

    @task(returns=list, isModifier = False)
    def updateClusters(self, partial, dicts, grid):
        df1, settings = partial
        clusters      = settings['clusters']
        primary_key   = settings['idCol']
        clusterCol  = settings['predCol']
        lat_col = settings['lat_col']
        lon_col = settings['lon_col']

        oldC_newC, id_newC = dicts
        init_lat,  init_lon,  end_lat, end_lon = grid

        if len(df1)>0:

            f = lambda point: all([ round(point[lat_col],5) >= init_lat ,
                                    round(point[lon_col],5) >= init_lon ,
                                    round(point[lat_col],5) < (end_lat+0.00001),
                                    round(point[lon_col],5) < (end_lon+0.00001),

                                ])
            tmp =  df1.apply(f, axis=1)
            df1 =  df1.loc[tmp]

            df1.drop_duplicates([primary_key],inplace=False)

            df1.reset_index(drop=True, inplace=True)

            for key in oldC_newC:
                if key in clusters:
                    df1.ix[df1[clusterCol] == key, clusterCol] = oldC_newC[key]

            df1.ix[df1[clusterCol].str.contains("_-9",na=False),clusterCol] = -1
            df1.drop(primary_key, inplace=True)

        return df1
