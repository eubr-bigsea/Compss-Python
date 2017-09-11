#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import argparse
import time
import pandas as pd
import numpy as np



#COMPSs's imports
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data import chunks
from pycompss.api.api import compss_wait_on



#---------------------------------------------------------------------
def plotter_p(result_df, windows, fase):
    result_df = compss_wait_on(result_df)

    import itertools
    import matplotlib.pyplot as plt


    colors = itertools.cycle(["r", "b", "g",'k','c','m','y'])
    for p in range(len(result_df)):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.axis((0,1, 0, 1))
        plt.grid(True)
        for w in windows:
            init, end, end2 = w
            plt.axvline(end[1], color='blue')
            plt.axhline(end[0], color='blue')
            plt.axvline(init[1], color='black')
            plt.axhline(init[0], color='black')


        Lat  = result_df[p][0]['LONGITUDE'].tolist()
        Long = result_df[p][0]['LATITUDE'].tolist()
        #COR = [ "C{}".format(p+1) for i in range(len(Lat))]
        plt.scatter(Lat,Long,c = next(colors), s=30)


        plt.savefig("sample_fase_{}_p{}.png".format(fase,p))


def round_up_to_even(f):
    import math
    return math.ceil(f / 2.) * 2

class DBSCAN(object):


    def fragment(self,div,eps):
        windows = []


        for lat in range(div):
            for log in range(div):
                init = [ (1.0 / div)*lat           ,  (1.0 / div)*log]
                end  = [ (1.0 / div)*(lat+1) + eps , (1.0 / div)*(log+1) + eps ]
                end2 = [ (1.0 / div)*(lat+1)       , (1.0 / div)*(log+1)]
                windows.append([init,end,end2])
        return windows



    def fit_predict(self, df, settings, numFrag):
        """
        fit():
        Perform DBSCAN clustering from the features column.

        :param df:       A list with numFrag pandas's dataframe.
        :param settings: A dictionary that contains:
            - feature:   Column name of the features in the test data;
            - idCol:     Column name to be a primary key of the dataframe;
            - predCol:   Alias to the new column with the labels predicted;
            - minPts:    The number of samples in a neighborhood for a point
                         to be considered as a core point;
                         This includes the point itself. (int, default: 15)
            - eps:       The maximum distance between two samples for them to
                         be considered as in the same neighborhood.
                         (float, default: 0.1)
        :param numFrag:  Number of even fragments;
        :return:         Returns the same list of dataframe with the cluster column.
        """


        minPts  = settings.get('minPts', 15)
        eps     = settings.get('eps', 0.1)
        predCol = settings.get('predCol', "Cluster")
        settings['eps']     = eps
        settings['minPts']  = minPts
        settings['predCol'] = predCol
        if 'column' not in settings or 'idCol' not in settings:
            return df

        div = int(float(numFrag)/2)
        windows = self.fragment(div,eps)

        #stage1 and stage2: partitionize and local dbscan
        t = 0
        partial =  [ [] for f in range(numFrag)]
        for l in range(div):
            for c in range(div):
                frag = []
                for f in range(numFrag):
                    frag   =  self.partitionize(df[f], settings, windows[t], frag)

                partial[t] = self.partial_dbscan(frag, settings, "p_{}_".format(t))
                t+=1


        #plotter_p(partial,windows,1)


        #stage3: combining clusters
        intersections =  2*(div-1)*div  + (div-1)**2
        mapper  =  [ [] for f in range(intersections) ]
        m = 0
        for t in range(numFrag):
            i = t % div
            if i < (div-1):
                #print "M={} | t:{} and (t+1):{}".format(m, t, t+1)
                mapper[m] = self.CombineClusters(partial[t], partial[t+1])
                m+=1
            j = t / div
            if j < (div-1):
                #print "M={} | t:{} and (t+div):{}".format(m, t, t+div)
                mapper[m] = self.CombineClusters(partial[t], partial[t+div])
                m+=1

            if i < (div-1) and j < (div-1):
                #print "M={} | t:{} and (t+div+1):{}".format(m, t, t+div+1)
                mapper[m] = self.CombineClusters(partial[t], partial[t+div+1])
                m+=1

        merged_mapper = mergeReduce(self.MergeMapper, mapper)
        #merged_mapper = compss_wait_on(merged_mapper)
        components    =  self.findComponents(merged_mapper, minPts)
        #partial = compss_wait_on(partial)
        #stage4: updateClusters
        t = 0
        result =  [ [] for f in range(numFrag)]
        for t in range(numFrag):
            result[t]  =  self.updateClusters(partial[t], components, windows[t])


        return result

#-------------------------------------------------------------------------------
#
#       stage1 and stage2: partitionize in 2dim and local dbscan
#
#-------------------------------------------------------------------------------


    def inblock(self,row, column, init,end):
        return  all([   row[column][0]>=init[0],
                        row[column][1]>=init[1],
                        row[column][0]<= end[0],
                        row[column][1]<= end[1]
                    ])


    @task(returns=list, isModifier = False)
    def partitionize(self,df, settings, windows, frag):

        column  = settings['column']

        #print "init: {} --> end: {}".format(init, end)
        init, end, end2 = windows
        f = lambda row: self.inblock(row,column,init,end) #>=lat_i and row[column][1]>=log_i  and  row[column][0]<=lat_f and row[column][1]<=log_f
        tmp =  df.apply(f, axis=1)

        tmp = df.loc[tmp]

        if len(frag)>0:
            frag = pd.concat([frag,tmp])
        else:
            frag = tmp

        return frag




    @task(returns=list, isModifier = False)
    def partial_dbscan(self,df, settings, sufix):
        cluster_label = 0
        cluster_noise = 0
        UNMARKED = -1

        stack = []
        NOISE   = -999999

        #CORE    = -51
        #BOARD   = -42
        #print df
        #pd.set_option('display.expand_frame_repr', False)

        df = df.reset_index(drop=True)
        num_ids = len(df)
        eps         = settings['eps']
        minPts      = settings['minPts']
        columns     = settings['column']
        clusterCol  = settings['predCol']
        idCol       = settings['idCol']

        C_UNMARKED = "{}{}".format(sufix,UNMARKED)
        C_NOISE = "{}{}".format(sufix,NOISE)
        df[clusterCol]   = pd.Series([ C_UNMARKED for i in range(num_ids)], index=df.index)

        noise = dict()
        for index in range(num_ids):
            point = df.loc[index]
            CLUSTER = point[clusterCol]

            if CLUSTER == C_UNMARKED:

                X = self.retrieve_neighbors( df, index, point, eps, columns )

                if len(X) < minPts:
                    #cluster_noise += 1
                    df.set_value(index, clusterCol, C_NOISE) #+"_" + str(cluster_noise)
                    tmp = []
                    for x in X:
                        tmp.append(df.loc[x][idCol])
                    noise[str(point[idCol])] = tmp
                else: # found a core point
                    cluster_label += 1
                    df.set_value(index,clusterCol, sufix + str(cluster_label)) # assign a label to core point

                    for new_index in X: # assign core's label to its neighborhood
                        df.set_value(new_index,clusterCol, sufix + str(cluster_label))
                        if new_index not in stack:
                            stack.append(new_index) # append neighborhood to stack

                        while len(stack) > 0: # find new neighbors from core point neighborhood
                            newest_index  = stack.pop()
                            new_point = df.loc[newest_index]
                            Y = self.retrieve_neighbors(df,newest_index,new_point, eps, columns)

                            if len(Y) >= minPts: # current_point is a new core
                                for new_index_neig in Y:
                                    neig_cluster = df.loc[new_index_neig][clusterCol]
                                    if (neig_cluster == C_UNMARKED):
                                        df.set_value(new_index_neig,clusterCol,  sufix + str(cluster_label))
                                        if new_index_neig not in stack:
                                            stack.append(new_index_neig)

        settings['clusters'] = df[clusterCol].unique()
        settings['noise'] = noise
        #print df
        return [df, settings]

    def retrieve_neighbors(self, df, i_point, point, eps, column):

        neigborhood = []

        for index, row in df.iterrows():
            if index != i_point:
                    a = np.array(point[column])
                    b =  np.array([row[column]])
                    distance = np.linalg.norm(a-b)
                    if distance <= eps:
                        neigborhood.append(index)

        return neigborhood

#-------------------------------------------------------------------------------
#
#                   #stage3: combining clusters
#
#-------------------------------------------------------------------------------


    @task(returns=list, isModifier = False)
    def CombineClusters(self, p1, p2):

        df1, settings = p1
        df2           = p2[0]
        columns     = settings['column']
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
        if len(df1)>0 and len(df2)>0:

            merged = pd.merge(df1, df2, how='inner', on=[primary_key] )

            for index,point in merged.iterrows():
                CLUSTER_DF1 = point[clusterCol+"_x"]
                CLUSTER_DF2 = point[clusterCol+"_y"]
                l = [CLUSTER_DF1, CLUSTER_DF2]
                if l not in links:
                    links.append(l)
                if all(["-99" for c in CLUSTER_DF1]):
                        key = point[primary_key]
                        print noise1[str(key)]
                        print noise2[str(key)]
                        n = list(set(noise1[str(key)] + noise2[str(key)]))
                        noise_r[str(key)] = n

            return [unique_c,links, noise_r]
        else:
            return [unique_c,links, noise_r]

    @task(returns=list, isModifier = False)
    def MergeMapper(self,mapper1,mapper2):
        a1,l1,d1 = mapper1
        a2,l2,d2 = mapper2
        unique_c = np.unique(np.concatenate((a1,a2),0))

        for k in d2:
            if k not in d1:
                d1[k] = d2[k]
            else:
                d1[k] = list(set(d1[k] + d2[k]))

        return [unique_c, l1 + l2, d1]


    @task(returns=list, isModifier = False)
    def findComponents(self, merged_mapper, minPts):

        import networkx as nx

        AllClusters, byReplicatedPoints, d1 =  merged_mapper
        i = 0

        G = nx.Graph()
        for line in byReplicatedPoints:
            G.add_edge(line[0], line[1])

        components =   sorted(nx.connected_components(G), key = len, reverse=True)


        oldC_newC = dict()

        for  component in components:
            has_cluster = any(["-999" not in c for c in component])
            if has_cluster:
                i+=1
                for c in component:
                    oldC_newC[c] = i


        for  c in AllClusters:
            if c not in oldC_newC:
                if "-99" not  in c:
                    i+=1
                    oldC_newC[c]=i

        G = nx.Graph()
        for key in d1:
            neighbors = d1[key]
            if len(neighbors) > minPts: #new core
                for v in neighbors:
                    G.add_edge(key, v)

        components =   sorted(nx.connected_components(G), key = len, reverse=True)
        #print components

        id_newC = dict()
        for  component in components:
            i+=1
            for c in component:
                id_newC[str(c)] = i


        # print "----------------------"
        # print oldC_newC
        # print "----------------------"
        # print id_newC
        # print "----------------------"

        return [oldC_newC,id_newC]




#-------------------------------------------------------------------------------
#
#                   #stage4: updateClusters
#
#-------------------------------------------------------------------------------

    @task(returns=list, isModifier = False)
    def updateClusters(self, partial, dicts, windows):
        df1, settings = partial
        clusters      = settings['clusters']
        primary_key   = settings['idCol']
        clusterCol  = settings['predCol']
        column  = settings['column']
        oldC_newC, id_newC = dicts
        init, end, end2 = windows

        if len(df1)>0:
            f = lambda row: self.inblock(row,column,init,end2)
            tmp =  df1.apply(f, axis=1)
            df1 =  df1.loc[tmp]

            df1.drop_duplicates([primary_key],inplace=False)

            df1 = df1.reset_index(drop=True)
            #print df1
            for key in oldC_newC:
                if key in clusters:
                    df1.ix[df1[clusterCol] == key, clusterCol] = oldC_newC[key]

            #print df1

            if len(id_newC)>0:
                for index, point in df1.iterrows():
                    key = point[primary_key]
                    if str(key) in id_newC:
                        #df1.set_value(index,clusters,id_newC[str(key)])
                        df1.ix[index, clusterCol] = id_newC[str(key)]

            print df1
            df1.ix[df1[clusterCol].str.contains("_-9", na=False), clusterCol] = -1

        return df1
