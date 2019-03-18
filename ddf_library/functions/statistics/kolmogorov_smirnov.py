#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"


from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
from pycompss.api.local import local

import numpy as np
from scipy.stats import distributions


def kolmogorov_smirnov_one_sample(data, settings):
        """
        Perform the Kolmogorov-Smirnov test for goodness of fit. This
        implementation of Kolmogorov–Smirnov test is a two-sided test
        for the null hypothesis that the sample is drawn from a continuous
        distribution.

        :param data: A list with of pandas's DataFrame;
        :param settings: A dictionary that contains:
            - 'col': sample column name;
            - 'distribution': Name of distribution (default is 'norm');
            - 'args': A tuple of distribution parameters;
            - 'mode': Defines the distribution used for calculating the p-value.

              - 'approx': use approximation to exact distribution
              - 'asymp': use asymptotic distribution of test statistic
        :return: KS statistic and p-value

        .. seealso:: Visit this `link <https://docs.scipy.org/doc/scipy-0.14.0/
         reference/stats.html#module-scipy.stats>`__ to see all supported
         distributions.
        """

        col = settings['col']
        mode = settings.get('mode', 'asymp')

        if mode not in ['asymp', 'asymp']:
            raise Exception("Only supported `approx` or `asymp` mode.")

        if not isinstance(col, list):
            col = [col]

        nfrag = len(data)

        for f in range(nfrag):
            data[f], _ = _select(data[f], col)

        from ..etl.sort import SortOperation
        settings_sort = {
            'columns': col,
            'ascending': [True for _ in range(len(col))]
             }

        sort_output = SortOperation().transform(data, settings_sort)
        data, info = sort_output['data'], sort_output['info']
        info = merge_reduce(merge_schema, info)

        for f in range(nfrag):
            data[f] = _ks_theorical_dist(data[f], info, f, settings)
        data = merge_reduce(_ks_merge, data)

        data = _ks_d_critical(data, mode)
        return data


@task(returns=2)
def _select(data, col):
    from ..etl.select import select
    data, info = select(data, col)
    return data, info


@task(returns=1)
def _ks_theorical_dist(data, info, f, settings):
    col = settings['col']
    distribution = settings.get('distribution', 'norm')
    args = settings.get('args', ())
    _, _, n = info
    if f == 0:
        n1 = 0
    else:
        n1 = sum(n[0:f])

    n1 = float(n1)
    n2 = n1 + n[f]
    n = sum(n)
    error = []

    if len(data) > 0:

        values = data[col].values.flatten()

        cdf = getattr(distributions, distribution).cdf
        cdf_vals = cdf(values, *args)

        d_plus = (np.arange(n1+1.0, n2+1.0)/n - cdf_vals).max()
        d_min = (cdf_vals - np.arange(n1, n2)/n).max()

        error = [d_plus, d_min]

    return [error, n]


@task(returns=1)
def _ks_merge(info1, info2):
    error1, n = info1
    error2, _ = info2
    error1 = error1 + error2

    return [[max(error1)], n]


@local
def _ks_d_critical(info, mode):
    ks_stat, n = info
    ks_stat = ks_stat[0]

    if mode == 'asymp':
        pvalue = distributions.kstwobign.sf(ks_stat * np.sqrt(n))
    else:
        pval_two = distributions.kstwobign.sf(ks_stat * np.sqrt(n))
        if n > 2666 or pval_two > 0.80 - n * 0.3 / 1000.0:
            pvalue = pval_two
        else:
            pvalue = distributions.ksone.sf(ks_stat, n) * 2

    return ks_stat, pvalue


@task(returns=1)
def merge_schema(schema1, schema2):

    columns1, dtypes1, p1 = schema1
    columns2, dtypes2, p2 = schema2

    schema = [columns1, dtypes1, p1 + p2]
    return schema

