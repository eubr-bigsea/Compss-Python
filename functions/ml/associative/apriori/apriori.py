# -*- coding: utf-8 -*-
#!/usr/bin/env python


import sys

from itertools import chain, combinations
from collections import defaultdict

from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce  import mergeReduce
from pycompss.api.api import compss_wait_on



class Apriori(object):

    def runApriori(self,data, settings,numFrag):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
         - items (tuple, support)
         - rules ((pretuple, posttuple), confidence)
        """
        col         = settings['col']
        minSupport  = settings.get('minSupport', 0.5)
        largeSet    = []

        currentCSet_reduced = self.getFirstItemsWithMinSupport(data, col, minSupport, numFrag) # Candidate pruning
        currentCSet_merged  = mergeReduce(self.mergeSetstoLocal,currentCSet_reduced) # localset and ntotal
        currentCSet_merged  = compss_wait_on(currentCSet_merged)
        nTotal = currentCSet_merged[1]
        k = 2

        while(currentCSet_merged[0] != {}):
            largeSet.append(currentCSet_merged[0])
            currentLSet = self.joinSet(currentCSet_reduced , k ,numFrag)  # Candidate generation
            currentCSet_reduced = self.getItemsWithMinSupport(currentLSet, data, col,  minSupport, numFrag)  # Candidate pruning
            currentCSet_merged = mergeReduce(self.mergeSetstoLocal,currentCSet_reduced) # localset and ntotal
            currentCSet_merged = compss_wait_on(currentCSet_merged)
            k += 1

        import pandas as pd
        #print largeSet
        largeSet_df = []
        for l in largeSet:
            k = [[ list(x), l[x]]  for x in l]
            largeSet_df.append( pd.DataFrame(k,columns=['transaction','support']) )
            #print l
        #print largeSet_df
        return largeSet_df, nTotal


    def getItemsWithMinSupport(self,candidates, data, t, minSupport, numFrag):
        "Returns all candidates that meets a minimum support level"


        itemSet_local  = [self.returnItemsWithMinSupport_count(candidates[i], data[i], t) for i in range(numFrag)]
        itemSet_global = mergeReduce(self.returnItemsWithMinSupport_merge, itemSet_local) # localset and ntotal
        C_reduced = [self.returnItemsWithMinSupport_reduce(data[i], t, itemSet_global, minSupport) for i in range(numFrag)]# localset,  ntotal


        return C_reduced

    @task(returns=list,isModifier = False)
    def returnItemsWithMinSupport_count(self,itemSet,data,t):
        """count the frequency of an item"""

        localSet = defaultdict(int)

        for transaction in data[t].values:
            for item in itemSet:
                if item.issubset(transaction):
                    localSet[item]  += 1

        return [localSet, len(data)]



    def getFirstItemsWithMinSupport(self, data, col, minSupport, numFrag):
        "Returns all candidates that meets a minimum support level"

        itemSet_local  = [self.returnFirstItemsWithMinSupport_count(data[i], col) for i in range(numFrag)]
        itemSet_global = mergeReduce(self.returnItemsWithMinSupport_merge,itemSet_local) # localset and ntotal
        C_reduced = [self.returnItemsWithMinSupport_reduce(data[i],col,itemSet_global, minSupport) for i in range(numFrag)]# localset,  ntotal

        return C_reduced

    @task(returns=list,isModifier = False)
    def returnFirstItemsWithMinSupport_count(self,data, t):
        """count the frequency of an item"""

        itemSet = set()
        for record in data[t].values:
            for item in record:
                itemSet.add(frozenset([item]))

        localSet = defaultdict(int)

        for transaction in data[t].values:
            for item in itemSet:
                if item.issubset(transaction):
                    localSet[item]  += 1
                    #print "satisfies= item {} in t:{}".format(item,transaction)


        return [localSet, len(data)]

    @task(returns=list,isModifier = False)
    def returnItemsWithMinSupport_merge(self,data1,data2):
            localSet1,n1 = data1
            localSet2,n2 = data2

            for freq in localSet2:
                if freq in localSet1:
                    localSet1[freq] += localSet2[freq]
                else:
                    localSet1[freq] = localSet2[freq]

            return [localSet1,n1+n2]

    @task(returns=list,isModifier = False)
    def returnItemsWithMinSupport_reduce(self,data, t,freqSet, minSupport):
            """calculates the support for items in the itemSet and returns a subset
           of the itemSet each of whose elements satisfies the minimum support"""

            GlobalSet,N = freqSet
            freqTmp = defaultdict(int)

            for transaction in data[t].values:
                for item in GlobalSet:
                    if item.issubset(transaction):
                        support = float(GlobalSet[item])/N
                        if support >= minSupport:
                            freqTmp[item] = support

            return [freqTmp, N]



    def joinSet(self,itemSets, length, numFrag):
            """Join a set with itself and returns the n-element itemsets"""

            joined = [[] for i in range(numFrag)]
            for i in range(numFrag):
                tmp = [ self.joiner(itemSets[i],itemSets[j],length) for j in range(numFrag) if i!=j]
                joined[i] = mergeReduce(self.mergeSets,tmp)

            return joined

    @task(returns=list,isModifier = False)
    def mergeSets(self,itemSet1,itemSet2):
        return itemSet1 | itemSet2


    @task(returns=list,isModifier = False)
    def mergeSetstoLocal(self,itemSet1,itemSet2):

        for k,v in itemSet2[0].items():
            itemSet1[0][k] = v

        return [itemSet1[0], itemSet1[1]]


    @task(returns=list,isModifier = False)
    def joiner(self,itemSets_local1,itemSets_local2, length):
        sets1 = set(itemSets_local1[0].keys())
        sets2 = set(itemSets_local2[0].keys())
        itemSets_global =  sets1 | sets2
        joined = set([i.union(j) for i in itemSets_global for j in itemSets_global if len(i.union(j)) == length])

        return joined


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def generateRules(self, L, settings,nTotal):

        min_confidence = settings.get('confidence',0.5)
        if nTotal <= 0:
            n_p = [ self.count_transations(L[f]) for f in range(len(L))]
            nTotal = mergeReduce(self.mergeRules,n_p)

        toRetRules = [ self.getRules( L[i], L, min_confidence, nTotal ) for i in range(1, len(L)) ]
        toRetRules = mergeReduce(self.mergeRules, toRetRules)
        return toRetRules

    @task(returns=list,isModifier = False)
    def getRules(self, Li, L, minConfidence, nTotal):

        toRetRules = []
        for index, row in Li.iterrows():
            item    = row['transaction']
            support = row['support']
            _subsets = [list(x) for x in self.subsets(item)] #map(frozenset, [x for x in self.subsets(item)])

            for element in _subsets:
                remain = list(set(item).difference(element))

                if len(remain) > 0:
                    num = float(support) / nTotal
                    den = self.getSupport(element,L)/nTotal
                    if den == 0.0: den = 0.000001
                    confidence = num/den
                    #print confidence
                    if confidence > minConfidence:
                        r = [element, remain, confidence]
                        toRetRules.append(r)


        import pandas as pd
        rules = pd.DataFrame(toRetRules,columns=['Pre-Rule','Post-Rule','confidence'])

        return rules



    def subsets(self,arr):
        """ Returns non empty subsets of arr"""
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    def getSupport(self,element, L):
        for df in L:
            for t,s  in zip(df['transaction'].values, df['support'].values):
                if element == t:
                    return s
        return 0.0

    @task(returns=int,isModifier = False)
    def count_transations(self,L):
        return len(L)

    @task(returns=list,isModifier = False)
    def mergeRules(self,rules1,rules2):
        return rules1 + rules2
