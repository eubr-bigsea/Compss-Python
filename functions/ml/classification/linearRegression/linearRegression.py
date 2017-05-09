#
#  Copyright 2.02-2016 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
@author: scorella

PyCOMPSs Mathematical Library: Classification: Linear Regression
================================================================
    This file contains the linear regression algorithm.

    # yi = alpha + betaxi + epsiloni
    # goal: y=alpha + betax
"""


from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce
import math


@task(returns=int)
def _add(x):
    return sum(x)


@task(returns=int)
def reduce_add(x, y):
    return x+y


@task(returns=float)
def _mean(X, n):
    return sum(X)/float(n)


def mean(X, wait=False):
    # chunked data
    n = len(X)*len(X[0])
    result = mergeReduce(reduce_add, [_mean(x, n) for x in X])
    if wait:
        from pycompss.api.api import compss_wait_on
        result = compss_wait_on(result)
    return result


@task(returns=list)
def _norm(X, m):
    return [x-m for x in X]


@task(returns=list)
def _pow(X, p=2):
    return [pow(x, 2) for x in X]


@task(returns=float)
def _mul(x, y):
    return x*y


def std(X, m, wait=False):
    xs = [_norm(x, m) for x in X]
    xp = [_pow(x, 2) for x in xs]
    n = len(X)*len(X[0])
    suma = mergeReduce(reduce_add, [_mean(x, n) for x in xp])
    if wait:
        from pycompss.api.api import compss_wait_on
        suma = compss_wait_on(suma)
    return suma


@task(returns=float)
def op_task(sum_x, sum_y, suma):
    return suma/float(math.sqrt(sum_x*sum_y))


def pearson(X, Y, mx, my):
    from pycompss.api.api import compss_wait_on
    xs  = [_norm(x, mx) for x in X]
    ys  = [_norm(y, my) for y in Y]
    xxs = [_pow(x, 2) for x in xs]
    yys = [_pow(y, 2) for y in ys]

    xs = compss_wait_on(xs)
    ys = compss_wait_on(ys)

    aux = [zip(a, b) for (a, b) in [(x, y) for (x, y) in zip(xs, ys)]]
    suma = mergeReduce(
        reduce_add,
        [mergeReduce(reduce_add, [_mul(a, b) for (a, b) in p]) for p in aux])

    sum_x = mergeReduce(reduce_add, map(_add, xxs))
    sum_y = mergeReduce(reduce_add, map(_add, yys))
    r = op_task(sum_x, sum_y, suma)
    return r


def fit(X, Y):
    from pycompss.api.api import compss_wait_on
    mx = mean(X)  # mx future object
    my = mean(Y)  # my future object
    r = pearson(X, Y, mx, my)
    stdx = std(X, mx)
    stdy = std(Y, mx)
    stdx = compss_wait_on(stdx)
    stdy = compss_wait_on(stdy)
    r = compss_wait_on(r)

    b = r * (math.sqrt(stdy) / math.sqrt(stdx))

    mx = compss_wait_on(mx)
    my = compss_wait_on(my)

    A = my - b*mx

    def line(x):
        return b*x+A

    return line
