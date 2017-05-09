#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from ListAdj import *


#def createInitalRank(links,urls,numFrag):


def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls: yield (url, rank / num_urls)



inlink_file_name =  "data"
#damping factor
d=0.85
inlink_file = open(inlink_file_name, 'r')
outlink_count = {}
inlinks = {}
oldpagerank = {}
newpagerank = {}
docids = {}
dangling_docs = {}

lines = []
for line in inlink_file:
    row = [i.replace("\n","") for i in line.split(" ")]
    lines.append(row)

print lines

links = [['MapR', ('Baidu', 'Blogger')], ['Google', 'MapR'], ['Blogger', ('Baidu', 'Google')], ['Baidu', 'MapR']]
print links



ranks = createInitalRank(links) # serial
print ranks

maxIterations = 15
for iteration in xrange(maxIterations):
    # Calculates URL contributions to the rank of other URLs.
    contribs = links.join(ranks).flatMap(lambda (url, (urls, rank)): computeContribs(urls, rank))
    contribs = [computeContribs(urls, rank) for (url, (urls, rank) in  ]
    # Re-calculates URL ranks based on neighbor contributions.
    ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)
# Collects all URL ranks and dump them to console.
for (link, rank) in ranks.collect():
    print "%s has rank: %s." % (link, rank)
