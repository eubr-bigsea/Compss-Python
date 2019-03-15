#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import context
from .ddf import DDF


class GroupedDDF(DDF):

    """
    A set of methods for aggregations on a DDF, created by DDF.group_by().

    The available aggregate functions are:

    - avg: Computes average values for each numeric columns for each group;
    - mean: Alias for avg;
    - count: Counts the number of records for each group;
    - first': Returns the first element of group;
    - last': Returns the last element of group;
    - max': Computes the max value for each numeric columns for each group;
    - min': Computes the min value for each numeric column for each group;
    - sum': Computes the sum for each numeric columns for each group;
    - list': Returns a list of objects with duplicates;
    - set': Returns a set of objects with duplicate elements
    """

    def __init__(self, ddf_var):
        self.last_uuid = ddf_var.last_uuid
        self.ddf_var = ddf_var
        self.parameters = context.COMPSsContext() \
            .get_task_function(self.last_uuid)[1]

        super(GroupedDDF, self).__init__(task_list=ddf_var.task_list,
                                         last_uuid=self.last_uuid)

    def agg(self, exprs, alias=None):
        """
        Compute aggregates and returns the result as a DDF.

        :param exprs: A single dict mapping from string to string, where the
         key is the column to perform aggregation on, and the value is a list
         of aggregation functions.
        :param alias: A single dict mapping from string to string, where the
         key is the old column name to perform aggregation on, and the value
         is a list of aliases for each aggregation function.

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).agg({'col_2': ['sum', 'last']})
        """
        self.parameters['operation'] = exprs
        if alias is None:
            alias = {}
            for col in exprs:
                alias[col] = []
                for f in exprs[col]:
                    alias[col].append("{}({})".format(f, col))

        self.parameters['aliases'] = alias
        context.COMPSsContext.tasks_map[self.last_uuid]['function'][1] = \
            self.parameters

        return self.ddf_var

    def avg(self, cols, alias=None):
        """
        Computes average values for each numeric columns for each group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'mean', alias)
        return self

    def count(self, cols, alias=None):
        """
        Counts the number of records for each group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).count(['col_2'])
        """
        self._apply_agg(cols, 'count', alias)
        return self

    def first(self, cols, alias=None):
        """
        Returns the first element of group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'first', alias)
        return self

    def last(self, cols, alias=None):
        """
        Returns the last element of group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'last', alias)
        return self

    def list(self, cols, alias=None):
        """
        Returns a list of objects with duplicates;

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'list', alias)
        return self

    def max(self, cols, alias=None):
        """
        Computes the max value for each numeric columns for each group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).avg(['col_2'])
        """
        self._apply_agg(cols, 'max', alias)
        return self

    def mean(self, cols, alias=None):
        """
        Alias for avg. Computes average values for each numeric columns for
        each group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).mean(['col_2'])
        """
        self._apply_agg(cols, 'mean', alias)
        return self

    def min(self, cols, alias=None):
        """
        Computes the min value for each numeric column for each group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).min(['col_2'])
        """
        self._apply_agg(cols, 'min', alias)
        return self

    def set(self, cols, alias=None):
        """
        Returns a set of objects with duplicate elements.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).set(['col_2'])
        """
        self._apply_agg(cols, 'set', alias)
        return self

    def sum(self, cols, alias=None):
        """
        Computes the sum for each numeric columns for each group.

        :param cols: String or a list of columns names
        :param alias: String or a list of aliases

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).sum(['col_2'])
        """
        self._apply_agg(cols, 'sum', alias)
        return self

    def _apply_agg(self, cols, func, new_alias):
        exprs = self.parameters['operation']
        aliases = self.parameters.get('aliases', {})
        groupby = self.parameters['groupby'][0]

        if not isinstance(cols, list):
            cols = [cols]
        if not isinstance(new_alias, list):
            new_alias = [new_alias]

        diff = len(new_alias) - len(cols)
        if diff > 0:
            new_alias = new_alias[:diff]
        if diff < 0:
            new_alias = new_alias + [None for _ in range(diff+1)]

        for col, alias in zip(cols, new_alias):

            if alias is None:
                alias = "{}({})".format(func, col)

            if col == '*':
                col = groupby

            if col not in exprs:
                exprs[col] = []
            exprs[col].append(func)

            if col not in aliases:
                aliases[col] = []
            aliases[col].append(alias)

        self.parameters['operation'] = exprs
        self.parameters['aliases'] = aliases

        context.COMPSsContext.tasks_map[self.last_uuid]['function'][1] = \
            self.parameters

