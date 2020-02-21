#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from ddf_library.utils import merge_info, read_stage_file, generate_info
from ddf_library.functions.etl.sort import sort_stage_1, sort_stage_2

from pycompss.api.parameter import FILE_IN, COLLECTION_IN
from pycompss.api.task import task
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

    col = settings['col']
    mode = settings.get('mode', 'asymp')

    if mode not in ['approx', 'asymp']:
        raise Exception("Only supported `approx` or `asymp` mode.")

    if not isinstance(col, list):
        col = [col]

    settings['columns'] = col
    settings['ascending'] = [True for _ in col]
    settings['return_info'] = True
    settings['only_key_columns'] = True

    # range_partition used in sort operation can create less partitions
    sort_data, info, _ = sort_stage_1(data, settings)

    info = merge_info(info)
    nfrag = len(sort_data)

    for f in range(nfrag):
        settings['id_frag'] = f
        sort_data[f] = _ks_theoretical_dist(sort_data[f], info, settings.copy())

    info = _ks_merge_and_critical(sort_data, mode)
    info = compss_wait_on(info)
    return info


@task(returns=2, data_input=FILE_IN)
def _select(data_input, params):
    df = read_stage_file(data_input, params['columns'])
    info = generate_info(df, params['id_frag'])
    return df, info


@task(returns=1, data=FILE_IN)
def _ks_theoretical_dist(data, info, settings):
    col = settings['col']
    data = read_stage_file(data, col)
    f = settings['id_frag']
    distribution = settings.get('distribution', 'norm')
    args = settings.get('args', ())
    n = info['size']

    data, _ = sort_stage_2(data, settings)

    if f == 0:
        n1 = 0
    else:
        n1 = sum(n[0:f])

    n1 = float(n1)
    n2 = n1 + n[f]
    n = sum(n)
    error = []

    if len(data) > 0:

        values = data[col].values
        del data

        cdf = getattr(distributions, distribution).cdf
        cdf_values = cdf(values, *args)

        x = np.arange(n1, n2)
        d_min = (cdf_values - x / n).max()
        d_plus = ((x + 1)/n - cdf_values).max()

        error = [d_plus, d_min]

    return [error, n]


@task(returns=1, infos=COLLECTION_IN)
def _ks_merge_and_critical(infos, mode):
    n = 0
    error = []
    for info in infos:
        error += info[0]
        n += info[1]

    ks_stat = max(error)

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
