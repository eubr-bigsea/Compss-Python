#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task


class FilterOperation(object):

    def transform(self, data, settings):
        """
        Filters rows using the given condition.

        :param data: A list with nfrag pandas's dataframe;
        :param settings: A dictionary that contains:
            - 'query': A valid query.
        :param nfrag: The number of fragments;
        :return: Returns a list with nfrag pandas's dataframe.

        Note: Visit the link bellow to more information about the query.
        https://pandas.pydata.org/pandas-docs/stable/generated/
        pandas.DataFrame.query.html

        example:
            settings['query'] = "(VEIC == 'CARX')" to rows where VEIC is CARX
            settings['query'] = "(VEIC == VEIC) and (YEAR > 2000)" to
                rows where VEIC is not NaN and YEAR is greater than 2000
        """
        nfrag = len(data)
        result = [[] for _ in range(nfrag)]
        query = self.preprocessing(settings)
        for i in range(nfrag):
            result[i] = _filter(data[i], query)
        return result

    def preprocessing(self, settings):
        query = settings.get('query', "")
        if len(query) == 0:
            raise Exception("You should pass at least one query.")
        return query

    def transform_serial(self, data, query):
        """Perform partial filter."""
        return _filter_(data, query)


@task(returns=list)
def _filter(data, query):
    """Perform partial filter."""
    return _filter_(data, query)


def _filter_(data, query):
    """Perform partial filter."""
    data.query(query, inplace=True)
    return data


