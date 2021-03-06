#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.

Public classes:

  - :class:`DDF`:
      Distributed DataFrame (DDF), the abstraction of this library.
"""

from ddf_library.bases.metadata import OPTGroup, Status
from pycompss.api.api import compss_open, compss_delete_file
from ddf_library.bases.ddf_base import DDFSketch
import ddf_library.bases.data_saver as ds
from ddf_library.bases.context_base import ContextBase
from ddf_library.utils import concatenate_pandas

import pandas as pd


__all__ = ['DDF']


class DDF(DDFSketch):
    """
    Distributed DataFrame Handler.
    """

    def __init__(self, **kwargs):
        super(DDF, self).__init__()

        # parent uuid operation
        self.last_uuid = kwargs.get('last_uuid', 'init')
        self.partitions = list()
        self.save = ds.Save()
        ds.last_uuid = self.last_uuid

    def __str__(self):
        return "DDF object."

    def cache(self):
        # noinspection PyUnresolvedReferences
        """
        Currently it is only an alias for persist().

        :return: DDF

        :Example:

        >>> ddf1.cache()
        """
        return self.persist()

    def crst_transform(self, lat_col, lon_col, src_epsg, dst_epsg,
                       lat_alias=None, lon_alias=None):
        # noinspection PyUnresolvedReferences
        """
        Given a source EPSG code, and target EPSG code, convert the Spatial
        Reference System / Coordinate Reference System.

        :param lat_col: Latitude column name;
        :param lon_col: Longitude column name;
        :param src_epsg: Coordinate Reference System used in the source points;
        :param dst_epsg: Target coordinate Reference System;
        :param lat_alias: Latitude column alias (default, replace the input);
        :param lon_alias: Longitude column alias (default, replace the input);

        :return: DDF

        :Example:

        >>> ddf1.crst_transform('latitude', 'longitude',
        >>>                     src_epsg=4326, dst_epsg=32633)
        """
        settings = {'lat_col': lat_col, 'lon_col': lon_col,
                    'src_epsg': src_epsg, 'dst_epsg': dst_epsg,
                    'lat_alias': lat_alias, 'lon_alias': lon_alias}

        from ddf_library.bases.optimizer.operations import CRSTTransform

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=CRSTTransform(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def num_of_partitions(self):
        # noinspection PyUnresolvedReferences
        """
        Returns the number of data partitions (Task parallelism).

        :return: integer

        :Example:

        >>> print(ddf1.num_of_partitions())
        """

        info = self._get_info()
        size = len(info['size'])
        return size

    def balancer(self, forced=False):
        # noinspection PyUnresolvedReferences
        """
        Repartition data in order to balance the distributed data between nodes.

        :return: DDF

        :Example:

        >>> ddf1.balancer(force=True)
        """

        from ddf_library.bases.optimizer.operations import WorkloadBalancer
        settings = {'forced': forced}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=WorkloadBalancer(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def count_rows(self, total=True):
        # noinspection PyUnresolvedReferences
        """
        Return the number of rows in this DDF.

        :param total: Show the total number (default) or the number over
         the fragments.
        :return: integer

        :Example:

        >>> print(ddf1.count_rows())
        """

        info = self._get_info()
        size = info['size']

        if total:
            size = sum(size)

        return size

    def cast(self, column, cast):
        """
        Change the data's type of some columns.

        Is it a Lazy function: Yes

        :param column: String or list of strings with columns to cast;
        :param cast: String or list of string with the supported types:
         'integer', 'string', 'decimal', 'date', 'date/time';
        :return: DDF
        """

        from ddf_library.bases.optimizer.operations import WithColumn
        settings = {'column': column, 'cast': cast}
        new_state_uuid = ContextBase \
            .ddf_add_task(operation=WithColumn(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def add_column(self, data2, suffixes=None):
        # noinspection PyUnresolvedReferences
        """
        Merge two DDF, column-wise.

        Is it a Lazy function: No

        :param data2: The second DDF;
        :param suffixes: Suffixes in case of duplicated columns name
            (default, ["_l","_r"]);
        :return: DDF

        :Example:

        >>> ddf1.add_column(ddf2)
        """

        if suffixes is None:
            suffixes = ['_l', '_r']

        settings = {'suffixes': suffixes}

        from ddf_library.bases.optimizer.operations import AddColumn

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=AddColumn(settings),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def group_by(self, group_by):
        # noinspection PyUnresolvedReferences
        """
        Computes aggregates and returns the result as a DDF.

        Is it a Lazy function: No

        :param group_by: A list of columns to be grouped;
        :return: A GroupedDFF with a set of methods for aggregations on a DDF

        :Example:

        >>> ddf1.group_by(group_by=['col_1']).mean(['col_2']).first(['col_2'])
        """
        settings = {'groupby': group_by, 'operation': []}
        from ddf_library.bases.groupby import GroupedDDF
        from ddf_library.bases.optimizer.operations import Aggregation

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Aggregation(settings),
                          parent=[self.last_uuid])

        tmp = DDF(last_uuid=new_state_uuid)
        return GroupedDDF(tmp)

    def fillna(self, subset=None, mode='VALUE', value=None):
        # noinspection PyUnresolvedReferences
        """
        Replace missing rows or columns by mean, median, value or mode.

        Is it a Lazy function: Yes, only if mode *"VALUE"*

        :param subset: A list of attributes to evaluate;
        :param mode: action in case of missing values: *"VALUE"* to replace by
         parameter "value" (default); *"MEDIAN"* to replace by median value;
         *"MODE"* to replace by  mode value; *"MEAN"* to replace by mean value;
        :param value: Value to be replaced (only if mode is *"VALUE"*)
        :return: DDF

        :Example:

        >>> ddf1.fillna(['col_1'], value=42)
        """

        settings = {'value': value, 'attributes': subset,
                    'cleaning_mode': mode}

        if mode is 'VALUE':
            from ddf_library.bases.optimizer.operations import FillNaByValue

            if not value:
                raise Exception("It is necessary a value "
                                "when using `VALUE` mode.")

            new_state_uuid = ContextBase \
                .ddf_add_task(operation=FillNaByValue(settings),
                              parent=[self.last_uuid])

            return DDF(last_uuid=new_state_uuid)

        else:
            from ddf_library.bases.optimizer.operations import FillNan

            new_state_uuid = ContextBase \
                .ddf_add_task(operation=FillNan(settings),
                              parent=[self.last_uuid])

            return DDF(last_uuid=new_state_uuid)

    def columns(self):
        """
        Returns the columns name in the current DDF.

        :return: A list of strings
        """

        columns = self._get_info()['cols']
        return columns

    def correlation(self, col1, col2):
        """
        Calculate the Pearson Correlation Coefficient.

        When one of the standard deviations is zero so the correlation is
        undefined (NaN).

        :param col1: The name of the first column
        :param col2: The name of the second column
        :return: The result of sample covariance
        """

        if not isinstance(col1, str):
            raise ValueError("col1 should be a string.")
        if not isinstance(col2, str):
            raise ValueError("col2 should be a string.")

        df, nfrag, tmp = self._ddf_initial_setup(self)

        from ddf_library.functions.statistics.correlation import correlation

        params = {'col1': col1, 'col2': col2}
        result = correlation(df, params)

        return result

    def covariance(self, col1, col2):
        """
        Calculate the sample covariance for the given columns, specified by
        their names, as a double value.

        :param col1: The name of the first column
        :param col2: The name of the second column
        :return: The result of sample covariance
        """

        if not isinstance(col1, str):
            raise ValueError("col1 should be a string.")
        if not isinstance(col2, str):
            raise ValueError("col2 should be a string.")

        df, nfrag, tmp = self._ddf_initial_setup(self)

        from ddf_library.functions.statistics.covariance import covariance

        params = {'col1': col1, 'col2': col2}
        result = covariance(df, params)

        return result

    def cross_join(self, data2):
        # noinspection PyUnresolvedReferences
        """
        Returns the cartesian product with another DDF.

        :param data2: Right side of the cartesian product;
        :return: DDF.

        :Example:

        >>> ddf1.cross_join(ddf2)
        """
        from ddf_library.bases.optimizer.operations import CrossJoin

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=CrossJoin(dict()),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def cross_tab(self, col1, col2):
        # noinspection PyUnresolvedReferences
        """
        Computes a pair-wise frequency table of the given columns. Also known as
        a contingency table.  The number of distinct values for each column
        should be less than 1e4. At most 1e6 non-zero pair frequencies will be
        returned.

        Is it a Lazy function: No

        :param col1: The name of the first column
        :param col2: The name of the second column
        :return: DDF

        :Example:

        >>> ddf1.cross_tab(col1='col_1', col2='col_2')
        """
        from ddf_library.bases.optimizer.operations import CrossTab
        if any([not isinstance(col1, str), not isinstance(col2, str)]):
            raise Exception('Columns must be a string (column names).')

        settings = {'col1': col1, 'col2': col2}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=CrossTab(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def describe(self, columns=None):
        # noinspection PyUnresolvedReferences
        """
        Computes basic statistics for numeric and string columns. This include
        count, mean, number of NaN, stddev, min, and max.

        Is it a Lazy function: No

        :param columns: A list of columns, if no columns are given,
         this function computes statistics for all numerical or string columns;
        :return: A pandas DataFrame

        :Example:

        >>> ddf1.describe(['col_1'])
        """

        df, nfrag, tmp = self._ddf_initial_setup(self)

        from ddf_library.functions.statistics.describe import describe

        if not columns:
            columns = []

        result = describe(df, columns)

        return result

    def export_ddf(self):
        """
        Export ddf data.

        :return: A list of Pandas's DataFrame
        """

        self._check_stored()
        return self.partitions

    def freq_items(self, col, support=0.01):
        # noinspection PyUnresolvedReferences
        """
        Finding frequent items for columns, possibly with false positives.
        Using the frequent element count algorithm described in
        “http://dx.doi.org/10.1145/762471.762473, proposed by Karp, Schenker,
        and Papadimitriou”

        Is it a Lazy function: No

        :param col: Names of the columns to calculate frequent items
        :param support: The frequency with which to consider an item 'frequent'.
         Default is 1%. The support must be greater than 1e-4.
        :return: DDF

        :Example:

        >>> ddf1.freq_items(col='col_1', support=0.01)
        """
        from ddf_library.functions.statistics.freq_items import freq_items
        settings = {'col': col, 'support': support}

        df, nfrag, tmp = self._ddf_initial_setup(self)

        result = freq_items(df, settings)

        return result

    def subtract(self, data2):
        # noinspection PyUnresolvedReferences
        """
        Returns a new DDF with containing rows in the first frame but not
        in the second one. This is equivalent to EXCEPT in SQL.

        Is it a Lazy function: No

        :param data2: second DDF;
        :return: DDF

        :Example:

        >>> ddf1.subtract(ddf2)
        """
        from ddf_library.bases.optimizer.operations import Subtract

        settings = {}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Subtract(settings),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def except_all(self, data2):
        # noinspection PyUnresolvedReferences
        """
        Returns a new DDF with containing rows in the first frame but not
        in the second one while preserving duplicates. This is equivalent to
        EXCEPT ALL in SQL.

        Is it a Lazy function: No

        :param data2: second DDF;
        :return: DDF

        :Example:

        >>> ddf1.except_all(ddf2)
        """
        from ddf_library.bases.optimizer.operations import ExceptAll

        settings = {}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=ExceptAll(settings),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def distinct(self, cols, opt=True):
        # noinspection PyUnresolvedReferences
        """
        Returns a new DDF containing the distinct rows in this DDF.

        Is it a Lazy function: No

        :param cols: subset of columns;
        :param opt: Tries to reduce partial output size before shuffle;
        :return: DDF

        :Example:

        >>> ddf1.distinct('col_1')
        """

        settings = {'columns': cols, 'opt_function': opt}
        from ddf_library.bases.optimizer.operations import Distinct

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Distinct(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def drop(self, columns):
        # noinspection PyUnresolvedReferences
        """
        Remove some columns from DDF.

        Is it a Lazy function: Yes

        :param columns: A list of columns names to be removed;
        :return: DDF

        :Example:

        >>> ddf1.drop(['col_1', 'col_2'])
        """

        settings = {'columns': columns}

        from ddf_library.bases.optimizer.operations import DropColumns
        
        new_state_uuid = ContextBase\
            .ddf_add_task(operation=DropColumns(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def drop_duplicates(self, cols):
        # noinspection PyUnresolvedReferences
        """
        Alias for distinct.

        :Example:

        >>> ddf1.drop_duplicates('col_1')
        """
        return self.distinct(cols)

    def dropna(self, subset=None, mode='REMOVE_ROW', how='any', thresh=None):
        # noinspection PyUnresolvedReferences
        """
        Cleans missing rows or columns fields.

        Is it a Lazy function: Yes, if mode is *"REMOVE_ROW"*, otherwise is No

        :param subset: A list of attributes to evaluate;
        :param mode: *"REMOVE_ROW"** to remove entire row (default) or
         **"REMOVE_COLUMN"** to remove a column.
        :param thresh: int, default None If specified, drop rows that have less
          than thresh non-null values. This overwrites the how parameter.
        :param how: 'any' or 'all'. If 'any', drop a row if it contains any
          nulls. If 'all', drop a row only if all its values are null.
        :return: DDF

        :Example:

        >>> ddf1.dropna(['col_1'], mode='REMOVE_ROW')
        """

        settings = {'attributes': subset, 'how': how, 'thresh': thresh,
                    'cleaning_mode': mode}

        if mode is 'REMOVE_ROW':
            from ddf_library.bases.optimizer.operations import DropNaRows

            new_state_uuid = ContextBase \
                .ddf_add_task(operation=DropNaRows(settings),
                              parent=[self.last_uuid])

            return DDF(last_uuid=new_state_uuid)

        else:
            from ddf_library.bases.optimizer.operations import DropNaColumns

            new_state_uuid = ContextBase \
                .ddf_add_task(operation=DropNaColumns(settings),
                              parent=[self.last_uuid])

            return DDF(last_uuid=new_state_uuid)

    def explode(self, column):
        # noinspection PyUnresolvedReferences
        """
        Returns a new row for each element in the given array.

        Is it a Lazy function: Yes

        :param column: Column name to be unnest;
        :return: DDF

        :Example:

        >>> ddf1.explode('col_1')
        """

        settings = {'column': column}

        from ddf_library.bases.optimizer.operations import Explode

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Explode(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def filter(self, expr):
        # noinspection PyUnresolvedReferences
        """
        Filters rows using the given condition.

        Is it a Lazy function: Yes

        :param expr: A filtering function;
        :return: DDF

        .. seealso:: Visit this `link <https://pandas.pydata.org/pandas-docs/
         stable/generated/pandas.DataFrame.query.html>`__ to more
         information about query options.

        :Example:

        >>> ddf1.filter("(col_1 == 'male') and (col_3 > 42)")
        """
        from ddf_library.bases.optimizer.operations import Filter

        settings = {'query': expr}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Filter(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def geo_within(self, shp_object, lat_col, lon_col, polygon,
                   attributes=None, suffix='_shp'):
        # noinspection PyUnresolvedReferences
        """
        Returns the sectors that the each point belongs.

        Is it a Lazy function: No

        :param shp_object: The DDF with the shapefile information;
        :param lat_col: Column which represents the Latitude field in the data;
        :param lon_col: Column which represents the Longitude field in the data;
        :param polygon: Field in shp_object where is store the
            coordinates of each sector;
        :param attributes: Attributes list to retrieve from shapefile,
            empty to all (default, None);
        :param suffix: Shapefile attributes suffix (default, *'_shp'*);
        :return: DDF

        :Example:

        >>> ddf2.geo_within(ddf1, 'LATITUDE', 'LONGITUDE', 'points')
        """

        from ddf_library.bases.optimizer.operations import GeoWithin

        settings = {'lat_col': lat_col, 'lon_col': lon_col,
                    'polygon': polygon, 'alias': suffix,
                    'attributes': attributes}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=GeoWithin(settings),
                          parent=[self.last_uuid, shp_object.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def hash_partition(self, columns, nfrag=None):
        # noinspection PyUnresolvedReferences
        """
        Hash partitioning is a partitioning technique where data
        is stored separately in different fragments by a hash function.

        Is it a Lazy function: No

        :param columns: Columns to be used as key in a hash function;
        :param nfrag: Number of fragments (default, keep the input nfrag).

        :Example:

        >>> ddf2 = ddf1.hash_partition(columns=['col1', col2])
        """

        from ddf_library.bases.optimizer.operations import HashPartition

        if not isinstance(columns, list):
            columns = [columns]

        settings = {'columns': columns}

        if nfrag is not None:
            settings['nfrag'] = nfrag

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=HashPartition(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def intersect(self, data2):
        # noinspection PyUnresolvedReferences
        """
        Returns a new DDF containing rows in both DDF. This is equivalent to
        INTERSECT in SQL.

        Is it a Lazy function: No

        :param data2: DDF
        :return: DDF

        :Example:

        >>> ddf2.intersect(ddf1)
        """

        from ddf_library.bases.optimizer.operations import Intersect

        settings = {'distinct': True}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Intersect(settings),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def intersect_all(self, data2):
        # noinspection PyUnresolvedReferences
        """
        Returns a new DDF containing rows in both DDF while preserving
        duplicates. This is equivalent to INTERSECT ALL in SQL.

        Is it a Lazy function: No

        :param data2: DDF
        :return: DDF

        :Example:

        >>> ddf2.intersect_all(ddf1)
        """

        from ddf_library.bases.optimizer.operations import Intersect

        settings = {'distinct': False}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Intersect(settings),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def join(self, data2, key1=None, key2=None, mode='inner',
             suffixes=None, keep_keys=False, case=True):
        # noinspection PyUnresolvedReferences
        """
        Joins two DDF using the given join expression.

        Is it a Lazy function: No

        :param data2: Second DDF;
        :param key1: List of keys of first DDF;
        :param key2: List of keys of second DDF;
        :param mode: How to handle the operation of the two objects. {‘left’,
         ‘right’, ‘inner’}, default ‘inner’
        :param suffixes: A list of suffix to be used in overlapping columns.
         Default is ['_l', '_r'];
        :param keep_keys: True to keep keys of second DDF (default is False);
        :param case: True to keep the keys as case sensitive;
        :return: DDF

        :Example:

        >>> ddf1.join(ddf2, key1=['col_1'], key2=['col_1'], mode='inner')
        """

        if key1 is None:
            key1 = []

        if key2 is None:
            key2 = []

        if suffixes is None:
            suffixes = ['_l', '_r']

        from ddf_library.bases.optimizer.operations import Join

        settings = {'key1': key1,
                    'key2': key2,
                    'option': mode,
                    'keep_keys': keep_keys,
                    'case': case,
                    'suffixes': suffixes}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Join(settings),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def kolmogorov_smirnov_one_sample(self, col, distribution='norm',
                                      mode='asymp', args=None):
        # noinspection PyUnresolvedReferences
        """
        Perform the Kolmogorov-Smirnov test for goodness of fit. This
        implementation of Kolmogorov–Smirnov test is a two-sided test
        for the null hypothesis that the sample is drawn from a continuous
        distribution.

        Is it a Lazy function: No

         :param col: sample column name;
         :param distribution: Name of distribution (default is 'norm');
         :param mode: Defines the distribution used for calculating the p-value.
            - 'approx' : use approximation to exact distribution
            - 'asymp' : use asymptotic distribution of test statistic

        :param args: A tuple of distribution parameters. Default is (0,1);
        :return: KS statistic and two-tailed p-value

        .. seealso:: Visit this `link <https://docs.scipy.org/doc/scipy-0.14.0/
         reference/stats.html#module-scipy.stats>`__ to see all supported
         distributions.

        .. note:: The KS statistic is the absolute max distance between
         the CDFs of the two samples. The closer this number is to 0 the
         more likely it is that the two samples were drawn from the
         same distribution.

         The p-value returned by the KS test has the same interpretation
         as other p-values. You reject the null hypothesis that the two
         samples were drawn from the same distribution if the p-value is
         less than your significance level.

        :Example:

        >>> ddf1.kolmogorov_smirnov_one_sample(col='col_1')
        """
        from ddf_library.functions.statistics.kolmogorov_smirnov \
            import kolmogorov_smirnov_one_sample

        if not isinstance(col, str):
            raise Exception('Column name (col) must be a string.')

        df, nfrag, tmp, info = self._ddf_initial_setup(self, info=True)
        settings = {'col': col, 'distribution': distribution, 'mode': mode,
                    'schema': [info]}
        if args is not None:
            settings['args'] = args

        result = kolmogorov_smirnov_one_sample(df, settings)

        return result

    def map(self, f, alias):
        # noinspection PyUnresolvedReferences
        """
        Apply a function to each row of this DDF.

        Is it a Lazy function: Yes

        :param f: Lambda function that will take each element of this data
         set as a parameter;
        :param alias: name of column to put the result;
        :return: DDF

        :Example:

        >>> from ddf_library.columns import col
        >>> from ddf_library.types import DataType
        >>> ddf1.map(col('col_0').cast(DataType.INT), 'col_0_new')
        """
        settings = {'function': f, 'alias': alias}
        from ddf_library.bases.optimizer.operations import Map

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Map(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def persist(self):
        # noinspection PyUnresolvedReferences
        """
        Compute the current flow and keep in disk.

        :return: DDF

        :Example:

        >>> ddf1.persist()
        """

        status = ContextBase.catalog_tasks.get_task_status(self.last_uuid)

        if status in [Status.STATUS_WAIT, Status.STATUS_DELETED]:
            self.last_uuid = ContextBase().run_workflow(self.last_uuid)

        ContextBase.update_status([self.last_uuid], Status.STATUS_PERSISTED)
        return self

    def range_partition(self, columns, ascending=None, nfrag=None):
        # noinspection PyUnresolvedReferences
        """
        Range partitioning is a partitioning technique where ranges of data
        is stored separately in different fragments.

        Is it a Lazy function: No

        :param columns: Columns to be used as key;
        :param ascending: Order of each key (True to ascending order);
        :param nfrag: Number of fragments (default, keep the input nfrag).

        :Example:

        >>> ddf2 = ddf1.range_partition(columns=['col1', col2],
        >>>                             ascending=[True, False])
        """

        from ddf_library.bases.optimizer.operations import RangePartition

        if not isinstance(columns, list):
            columns = [columns]

        if ascending is None:
            ascending = True

        if not isinstance(ascending, list):
            ascending = [ascending for _ in columns]

        settings = {'columns': columns, 'ascending': ascending}

        if nfrag is not None:
            settings['nfrag'] = nfrag

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=RangePartition(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def repartition(self, nfrag=-1, distribution=None):
        # noinspection PyUnresolvedReferences
        """
        Repartition a distributed data based in a fixed number of partitions or
        based on a distribution list.

        :param nfrag: Optional, if used, the data will be partitioned in
         nfrag fragments.
        :param distribution: Optional, a list of integers where each
         element will represent the amount of data in this index.
        :return: DDF
        """

        from ddf_library.bases.optimizer.operations import Repartition

        settings = {'nfrag': nfrag}

        if distribution is not None:
            settings['distribution'] = distribution

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Repartition(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def replace(self, replaces, subset=None, regex=False):
        # noinspection PyUnresolvedReferences
        """
        Replace one or more values to new ones.

        Is it a Lazy function: Yes

        :param replaces: dict-like `to_replace`;
        :param subset: A list of columns to be applied (default is
         None to applies in all columns);
        :param regex: Whether to interpret to_replace and/or value as regular
         expressions. If this is True then replaces must be a dictionary.
        :return: DDF

        :Example:

        >>> ddf1.replace({0: 'No', 1: 'Yes'}, subset=['col_1'])
        """

        from ddf_library.bases.optimizer.operations import Replace

        settings = {'replaces': replaces, 'regex': regex}
        if subset is not None:
            settings['subset'] = subset

        obj = Replace(settings)

        settings = obj.settings

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Replace(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def sample(self, value=None, seed=None):
        # noinspection PyUnresolvedReferences
        """
        Returns a sampled subset.

        Is it a Lazy function: No

        :param value: None to sample a random amount of records (default),
            a integer or float N to sample a N random records;
        :param seed: optional, seed for the random operation.
        :return: DDF

        :Example:

        >>> ddf1.sample(10)  # to sample 10 rows
        >>> ddf1.sample(0.5) # to sample half of the elements
        >>> ddf1.sample()  # a random sample
        """

        from ddf_library.bases.optimizer.operations import Sample
        settings = dict()
        settings['seed'] = seed

        if value:
            """Sample a N random records"""
            settings['type'] = 'value'
            settings['value'] = value

        else:
            """Sample a random amount of records"""
            settings['type'] = 'percent'

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Sample(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def schema(self):
        # noinspection PyUnresolvedReferences
        """
        Returns a schema table where each row contains the name
        columns and its data types of the current DDF.

        :return: a Pandas's DataFrame
        """

        info = self._get_info()
        tmp = pd.DataFrame.from_dict({'columns': info['cols'],
                                      'dtypes': info['dtypes']})
        return tmp

    def dtypes(self):
        # noinspection PyUnresolvedReferences
        """
        Returns a list of dtypes of each column on the current DDF.

        :return: a list
        """

        info = self._get_info()['dtypes']
        return info

    def select(self, columns):
        # noinspection PyUnresolvedReferences
        """
        Projects a set of expressions and returns a new DDF.

        Is it a Lazy function: Yes

        :param columns: list of column names (string);
        :return: DDF

        :Example:

        >>> ddf1.select(['col_1', 'col_2'])
        """

        settings = {'columns': columns}

        from ddf_library.bases.optimizer.operations import Select

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Select(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def select_expression(self, *exprs):
        # noinspection PyUnresolvedReferences
        """
        Projects a set of SQL expressions and returns a new DDF.
        This is a variant of select() that accepts SQL expressions.

        Is it a Lazy function: Yes

        :param exprs: SQL expressions.
        :return: DDF

        .. note:: These operations are supported by select_exprs:

          * Arithmetic operations except for the left shift (<<) and
            right shift (>>) operators, e.g., 'col' + 2 * pi / s ** 4 % 42
            - the_golden_ratio

          * list and tuple literals, e.g., [1, 2] or (1, 2)
          * Math functions: sin, cos, exp, log, abs, log10, ...
          * You must explicitly reference any local variable that you want to
            use in an expression by placing the @ character in front of the
            name.
          * This Python syntax is not allowed:

           - Function calls other than math functions.
           - is/is not operations
           - if expressions
           - lambda expressions
           - list/set/dict comprehensions
           - Literal dict and set expressions
           - yield expressions
           - Generator expressions
           - Boolean expressions consisting of only scalar values
           - Statements: Neither simple nor compound statements are allowed.

        .. seealso:: Visit this `link <https://pandas-docs.github.io/pandas-docs
            -travis/reference/api/pandas.eval.html#pandas.eval>`__ to more
            information about eval options.

        :Example:

        >>> ddf1.select_exprs('col1 = age * 2', "abs(age)")
        """

        from ddf_library.bases.optimizer.operations import SelectExprs

        settings = {'exprs': exprs}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=SelectExprs(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def show(self, n=20):
        # noinspection PyUnresolvedReferences
        """
        Print the DDF contents in a concatenated pandas's DataFrame.

        :param n: A number of rows in the result (default is 20);
        :return: DataFrame in stdout

        :Example:

        >>> ddf1.show()
        """

        self._check_stored()

        n_rows_frags = self.count_rows(False)

        from .functions.etl.take import take
        conf = {'value': n,
                'balancer': False,
                'schema': [{'size': n_rows_frags}]
                }
        res = take(self.partitions, conf)['data']

        df = [0 for _ in range(len(res))]
        for i, f in enumerate(res):
            df[i] = pd.read_parquet(compss_open(f, mode='rb'))
            compss_delete_file(f)
        df = concatenate_pandas(df)
        print(df)
        return self

    def sort(self, cols,  ascending=None):
        # noinspection PyUnresolvedReferences
        """
        Returns a sorted DDF by the specified column(s).

        Is it a Lazy function: No

        :param cols: list of columns to be sorted;
        :param ascending: list indicating whether the sort order
            is ascending (True) for each column (Default, True);
        :return: DDF

        :Example:

        >>> dd1.sort(['col_1', 'col_2'], ascending=[True, False])
        """

        from ddf_library.bases.optimizer.operations import Sort

        settings = {'columns': cols, 'ascending': ascending}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Sort(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def split(self, percentage=0.5, seed=None):
        # noinspection PyUnresolvedReferences
        """
        Randomly splits a DDF into two DDF.

        Is it a Lazy function: No

        :param percentage: percentage to split the data (default, 0.5);
        :param seed: optional, seed in case of deterministic random operation;
        :return: DDF

        :Example:

        >>> ddf2a, ddf2b = ddf1.split(0.5)
        """
        from ddf_library.bases.optimizer.operations import RandomSplit
        settings = {'percentage': percentage, 'seed': seed}

        split_out1_uuid, split_out2_uuid = ContextBase \
            .ddf_add_task(operation=RandomSplit(settings),
                          parent=[self.last_uuid],
                          n_output=2)

        out1 = DDF(last_uuid=split_out1_uuid)
        out2 = DDF(last_uuid=split_out2_uuid)
        return out1, out2

    def take(self, num):
        # noinspection PyUnresolvedReferences
        """
        Returns the first num rows.

        Is it a Lazy function: No

        :param num: number of rows to retrieve;
        :return: DDF

        :Example:

        >>> ddf1.take(10)
        """

        from ddf_library.bases.optimizer.operations import Take
        settings = {'value': num, 'balancer': True}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Take(settings),
                          parent=[self.last_uuid])
        result = DDF(last_uuid=new_state_uuid)

        if settings['balancer']:
            result = result.balancer(True)
        return result

    def to_df(self, columns=None, split=False):
        # noinspection PyUnresolvedReferences
        """
        Returns the DDF contents as a pandas's DataFrame.

        :param columns: Optional, A column name or list of column names;
        :param split: True to keep data in partitions (default, False);
        :return: Pandas's DataFrame

        :Example:

        >>> df = ddf1.to_df(['col_1', 'col_2'])
        """

        self._check_stored()

        res = [compss_open(f, mode='rb') for f in self.partitions]

        if isinstance(columns, str):
            columns = [columns]
        if split:
            df = [pd.read_parquet(f, columns=columns) for f in res]
        else:
            df = concatenate_pandas([pd.read_parquet(f, columns=columns)
                                     for f in res])
            df.reset_index(drop=True, inplace=True)

        return df

    def unpersist(self):
        ContextBase.update_status([self.last_uuid], Status.STATUS_COMPLETED)
        return self

    def union(self, data2):
        # noinspection PyUnresolvedReferences
        """
        Combine this data set with some other DDF. Also as standard in SQL,
        this function resolves columns by position (not by name). Union can
        only be performed on tables with the same number of columns.

        Is it a Lazy function: No

        :param data2:
        :return: DDF

        :Example:

        >>> ddf1.union(ddf2)
        """

        from ddf_library.bases.optimizer.operations import Union

        settings = {'by_name': False}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Union(settings),
                          parent=[self.last_uuid, data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def union_by_name(self, data2):
        # noinspection PyUnresolvedReferences
        """
        Combine this data set with some other DDF. This function resolves
         columns by name (not by position).

        Is it a Lazy function: No

        :param data2:
        :return: DDF

        :Example:

        >>> ddf1.union_by_name(ddf2)
        """

        from ddf_library.bases.optimizer.operations import Union

        settings = {'by_name': True}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=Union(settings),
                          parent=[self.last_uuid,  data2.last_uuid])

        return DDF(last_uuid=new_state_uuid)

    def rename(self, old_column, new_column):
        # noinspection PyUnresolvedReferences
        """
        Returns a new DDF by renaming an existing column. This is a no-op if
        schema does not contain the given column name.

        Is it a Lazy function: Yes

        :param old_column: String or list of strings with columns to rename;
        :param new_column: String or list of strings with new names.

        :return: DDF
        """

        from ddf_library.bases.optimizer.operations import WithColumnRenamed

        settings = {'old_column': old_column, 'new_column': new_column}

        new_state_uuid = ContextBase\
            .ddf_add_task(operation=WithColumnRenamed(settings),
                          parent=[self.last_uuid])

        return DDF(last_uuid=new_state_uuid)
