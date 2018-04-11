#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
import pandas as pd


class SortOperation(object):
    """Sort Operation.

    Returns a DataFrame sorted by the specified column(s).
    """

    def transform(self, data, settings, numFrag):
        """transform.

        :param data: A list with numFrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - algorithm:
                * 'odd-even', to sort using Odd-Even Sort (default);
                * 'bitonic', to sort using Bitonic Sort
                  (only if numFrag is power of 2);
            - columns: The list of columns to be sorted.
            - ascending: A list indicating whether the sort order
                is ascending (True) for each column.
        :param numFrag: The number of fragments;
        :return: A list with numFrag pandas's dataframe

        Note: the list of columns should have the same size of the list
        of boolean to indicating if it is ascending sorting.
        """
        algorithm = self.validate(settings, numFrag)

        if algorithm == "bitonic":
            result = self.sort_by_bittonic(data, settings, numFrag)
        else:
            result = self.sort_by_oddeven(data, settings, numFrag)

        return result

    def validate(self, settings, numFrag):
        """Check all settings."""
        cols1 = settings.get('columns', [])
        asc = settings.get('ascending', [])
        if any([len(cols1) == 0,
                len(asc) == 0,
                len(cols1) != len(asc)]):
            raise Exception('The list of `columns` ans `ascending` '
                            'should have equal lenght (and diffent '
                            'form zero).')

        def is_power2(num):
            return ((num & (num - 1)) == 0) and num != 0

        algorithm = settings.get('algorithm', 'odd-even')
        if not is_power2(numFrag):
            algorithm == 'odd-even'

        return algorithm

    def sort_by_bittonic(self, data, settings, numFrag):
        """Sort by Bittonic Sort.

        Given an unordered sequence of size 2*fragments, exactly
        log2 (fragments) stages of merging are required to produce
        a completely ordered list.
        """
        data = self.bitonic_sort(data, settings)
        return data

    def sort_by_oddeven(self, data, settings, numFrag):
        """Sort by Odd-Even Sort."""
        for f in range(numFrag):
            data[f] = self._sort(data[f], settings)

        f = 0
        nsorted = True
        from pycompss.api.api import compss_wait_on
        while nsorted:
            signals = [0 for i in range(numFrag)]
            if (f % 2 == 0):
                for i in range(numFrag):
                    if (i % 2 == 0):
                        signals[i] = self.mergesort(data[i],
                                                    data[i+1], settings)

            else:
                for i in range(numFrag-1):
                    if (i % 2 != 0):
                        signals[i] = self.mergesort(data[i],
                                                    data[i+1], settings)

            if f > 2:
                signals = compss_wait_on(signals)
                nsorted = any([i == -1 for i in signals])
                # nsorted = False
            f += 1
        return data

    def bitonic_sort(self, x, settings):
        """Recursive method of the Bittonic sort."""
        if len(x) <= 1:
            return x
        else:
            first = self.bitonic_sort(x[:len(x) // 2], settings)
            second = self.bitonic_sort(x[len(x) // 2:], settings)
            return self.bitonic_merge(first + second, settings)

    def bitonic_merge(self, x, settings):
        """Recursive method of the Bittonic sort."""
        if len(x) == 1:
            return x
        else:
            self.bitonic_compare(x, settings)
            first = self.bitonic_merge(x[:len(x) // 2], settings)
            second = self.bitonic_merge(x[len(x) // 2:], settings)
            return first + second

    def bitonic_compare(self, x, settings):
        """Sort the elements in two fragments at time."""
        dist = len(x) // 2
        for i in range(dist):
            self.mergesort(x[i], x[i+dist], settings)
            from pycompss.api.api import compss_wait_on
            x = compss_wait_on(x)
            print x[i]['CRSDepTime'].head(5)
            print x[i+dist]['CRSDepTime'].head(5)

    @task(isModifier=False, returns=list)
    def _sort(self, data, settings):
        """Perform a partial sort."""
        col = settings['columns']
        order = settings['ascending']
        data.sort_values(col, ascending=order, inplace=True)
        data = data.reset_index(drop=True)
        return data

    @task(isModifier=False, data1=INOUT, data2=INOUT, returns=int)
    def mergesort(self, data1, data2, settings):
        """Return 1 if [data1, data2] is sorted, otherwise is -1."""
        col = settings['columns']
        order = settings['ascending']
        n1 = len(data1)
        n2 = len(data2)
        nsorted = 1

        if n1 == 0 or n2 == 0:
            return nsorted

        print data1['CRSDepTime'].head(5)
        print data2['CRSDepTime'].head(5)

        data = pd.concat([data1, data2])
        data.reset_index(drop=True, inplace=True)
        indexes = data.index
        data.sort_values(col, ascending=order, inplace=True)
        if any(data.index != indexes):
            nsorted = -1

        data = data.reset_index(drop=True)
        data1.iloc[0:, :] = data.iloc[:n1, :].values

        data = data[data.index >= n1]
        data = data.reset_index(drop=True)
        data2.iloc[0:, :] = data.iloc[:, :].values

        print data1['CRSDepTime'].head(5)
        print data2['CRSDepTime'].head(5)
        print n1, n2
        print len(data1), len(data2)

        return nsorted
