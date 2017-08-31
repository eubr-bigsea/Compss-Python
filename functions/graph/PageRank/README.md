# PageRank

PageRank is one of the methods Google uses to determine a page’s relevance or importance. The idea that Page Rank brought up was that, the importance of any web page can be judged by looking at the pages that link to it.
PageRank can be utilized in others domains, for example, may also be used as a methodology to measure the apparent impact of a community or of a protein chain.

See more at: http://www.cs.princeton.edu/~chazelle/courses/BIB/pagerank.htm

## Instructions:

Use the method `runPageRank()` to run PageRank algorithm implementation for a fixed number of iterations returning a graph with vertex attributes containing the PageRank.

All parameters are explained below:

* :param data:        A list with numFrag pandas's dataframe.
* :param settings:    A dictionary that contains:
    - inlink:  column name of the inlinks vertex;
    - outlink: column name of the outlinks vertex;
    - damping_factor: the coeficent of the damping factor [0,1];
    - maxIters: The number of iterations;
    - col1: alias of the vertex column;
    - col2: alias of the ranking column.
* :param numFrag:     A number of fragments
* :return:            A list of pandas's dataframe with the ranking of each vertex in the dataset.

## Example:


```sh
from functions.graph.PageRank.pagerank import *

numFrag = 4
settings = dict()
settings['inlink']         = 'Column1'
settings['outlink']        = 'Column2'
settings['maxIters']       = 100
settings['damping_factor'] = 0.85
pr = PageRank()
result = pr.runPageRank(data,settings,numFrag)
```                              
