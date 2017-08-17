#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.api.api import compss_wait_on

"""
outlink,inlink
url_1,url_1
url_1,url_2
url_2,url_1
url_2,url_3
url_3,url_2

"""


class PageRank(object):


    def create_AdjList(self,data,inlink,outlink,numFrag):
        """1º Load all URL's from the data and initialize their neighbors """
        """2º Initialize each page’s rank to 1.0 """

        adjlist = [[] for i in range(numFrag)]
        rankslist = [[] for i in range(numFrag)]
        counts_in = [[] for i in range(numFrag)]

        for i in range(numFrag):
            adjlist[i]    =   self.partial_AdjList(data[i],inlink, outlink)
            rankslist[i]  =   self.partial_RankList(data[i],inlink, outlink)
            counts_in[i] =    self.counts_inlinks(adjlist[i])

        counts_in = mergeReduce(self.merge_counts,counts_in)
        adjlist =  [self.update_AdjList(adjlist[i], counts_in) for i in range(numFrag)]

        return adjlist,rankslist


    @task(returns=dict,isModifier = False)
    def partial_RankList(self,data,inlink,outlink):
        ranks = {}
        for link in data[[outlink,inlink]].values:
            #print link
            v_out = link[0]
            v_in  = link[1]
            if v_out not in ranks:
                ranks[v_out] = 1.0 # Rank, contributions, main

            if v_in not in ranks:
                ranks[v_in] = 1.0

        return ranks


    @task(returns=dict,isModifier = False)
    def partial_AdjList(self,data,inlink,outlink):
        adj = {}

        for link in data[[outlink,inlink]].values:
            v_out = link[0]
            v_in  = link[1]
            if v_out in adj:
                adj[v_out][0].append(v_in)
                adj[v_out][1]+=1
            else:
                adj[v_out] = [[v_in],1]

        return adj

    @task(returns=dict,isModifier = False)
    def counts_inlinks(self, adjlist1):
        counts_in = {}
        for v_out in adjlist1:
            counts_in[v_out] = adjlist1[v_out][1]
        return counts_in


    @task(returns=dict,isModifier = False)
    def merge_counts(self, counts1, counts2):
        for v_out in counts2:
            if v_out in counts1:
                counts1[v_out] += counts2[v_out]
            else:
                counts1[v_out] = counts2[v_out]
        return counts1



    @task(returns=dict,isModifier = False)
    def update_AdjList(self,adj1, counts_in):

        for key in adj1:
            adj1[key][1] = counts_in[key]
            #print "output",adjlist1
        return adj1

    @task(returns=dict,isModifier = False)
    def calcContribuitions(self,adj,ranks):

        contrib = {}
        for key in adj:
            urls = adj[key][0]
            numNeighbors = adj[key][1]
            rank = ranks[key]
            for url in urls:
                if url not in contrib:
                    #  destino  =  contrib
                    contrib[url] = rank/numNeighbors
                else:
                    contrib[url] += rank/numNeighbors

        #print "---------- calc contributions -----------"
        #print contrib
        #print "---------- cal contributions -----------"
        return contrib


    @task(returns=dict,isModifier = False)
    def mergeContribs(self,contrib1,contrib2):
        #print "----------contributions -----------"
        #print contrib1
        #print contrib2


        for k2 in contrib2:
            if k2 in contrib1:
                contrib1[k2] += contrib2[k2]
            else:
                contrib1[k2] = contrib2[k2]


        #print contrib1
        #print "----------merged contributions -----------"
        return contrib1

    @task(returns=dict,isModifier = False)
    def updateRank_p(self,ranks,contrib,factor):
        bo = 1.0 - factor

        for key in contrib:
            if key in ranks:
                #print "{} teve {} de contribution = {}".format(key,contrib[key], 0.15 + 0.85*contrib[key])
                ranks[key] = bo + factor*contrib[key]

        return ranks



    def runPageRank(self,data,settings,numFrag):
        inlink = settings['inlink']
        outlink = settings['outlink']
        factor = settings['damping_factor']
        maxIterations = settings['maxIters']
        col1 = settings['col1']
        col2 = settings['col2']

        adj, rank = self.create_AdjList(data,inlink,outlink,numFrag)

        #adj = compss_wait_on(adj)
        #print adj
        for iteration in xrange(maxIterations):
            contributions = [ self.calcContribuitions(adj[i],rank[i])  for i in range(numFrag) ]
            merged_c = mergeReduce(self.mergeContribs,contributions)
            rank =  [ self.updateRank_p(rank[i],merged_c,factor)   for i in range(numFrag) ]
            #rank = compss_wait_on(rank)
            #print rank

        table = [ self.printingResults(rank[i],col1,col2) for i in range(numFrag)]
        merged_table = mergeReduce(self.mergeRanks,table)
        return merged_table


    @task(returns=list,isModifier = False)
    def printingResults(self,ranks,c1,c2):

        Links = []
        Ranks = []
        for v in ranks:
            Links.append(v)
            Ranks.append(ranks[v])

        data = pd.DataFrame()
        data[c1] = Links
        data[c2] = Ranks

        return data

    @task(returns=list,isModifier = False)
    def mergeRanks(self,df1,df2):
        result = pd.concat([df1,df2], ignore_index=True).drop_duplicates()
        return result
