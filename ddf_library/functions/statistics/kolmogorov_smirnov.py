#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import merge_info
from ..etl.select import select
from ..etl.sort import SortOperation

from pycompss.api.task import task
from pycompss.functions.reduce import merge_reduce
# from pycompss.api.local import local # guppy module isnt available in python3
from pycompss.api.api import compss_delete_object, compss_wait_on

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
            - col: sample column name;
            - distribution: Name of distribution (default is 'norm');
            - args: A tuple of distribution parameters;
            - mode: Defines the distribution used for calculating the p-value,
             'approx' to use approximation to exact distribution or 'asymp' to
             use asymptotic distribution of test statistic
        :return: KS statistic and p-value

        .. seealso:: Visit this `link <https://docs.scipy.org/doc/scipy-0.14.0/
         reference/stats.html#module-scipy.stats>`__ to see all supported
         distributions.

        """

        nfrag = len(data)
        col = settings['col']
        mode = settings.get('mode', 'asymp')

        if mode not in ['asymp', 'asymp']:
            raise Exception("Only supported `approx` or `asymp` mode.")

        if not isinstance(col, list):
            col = [col]

        info = [0 for _ in range(nfrag)]
        for f in range(nfrag):
            params = {'columns': col, 'id_frag':f}
            data[f], info[f] = _select(data[f], params)
        info = merge_info(info)
        info = compss_wait_on(info)

        params = {
            'columns': col,
            'ascending': [True for _ in col],
            'info': [info]
        }

        sort_output = SortOperation().transform(data, params)
        sort_data = sort_output['data']
        compss_delete_object(data)

        for f in range(nfrag):
            sort_data[f] = _ks_theorical_dist(sort_data[f], info, f, settings)
        info = merge_reduce(_ks_merge, sort_data)
        compss_delete_object(sort_data)

        info = _ks_d_critical(info, mode)
        return info


@task(returns=2)
def _select(data, col):
    data, info = select(data, col)
    return data, info


@task(returns=1)
def _ks_theorical_dist(data, info, f, settings):
    col = settings['col']
    distribution = settings.get('distribution', 'norm')
    args = settings.get('args', ())
    n = info['size']

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


# @local
def _ks_d_critical(info, mode):

    info = compss_wait_on(info)
    ks_stat, n = info
    ks_stat = ks_stat[0]

    if mode == 'asymp':
        p_value = distributions.kstwobign.sf(ks_stat * np.sqrt(n))
    else:
        p_val_two = distributions.kstwobign.sf(ks_stat * np.sqrt(n))
        if n > 2666 or p_val_two > 0.80 - n * 0.3 / 1000.0:
            p_value = p_val_two
        else:
            p_value = distributions.ksone.sf(ks_stat, n) * 2

    p_value = round(p_value, 10)
    ks_stat = round(ks_stat, 10)
    return ks_stat, p_value

