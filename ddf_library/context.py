#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""


from ddf_library.bases.context_base import ContextBase
from ddf_library.ddf import DDF
from ddf_library.utils import check_serialization, delete_result, _gen_uuid
import ddf_library.bases.data_reader as dr

import networkx as nx
from prettytable import PrettyTable


class COMPSsContext(object):
    """
    Controls the DDF tasks executions
    """

    def __init__(self):
        if ContextBase.started:
            print('COMPSsContext is already started.')
        else:
            import os
            folder = '/tmp/ddf_' + _gen_uuid()
            while os.path.isdir(folder):
                folder = '/tmp/ddf_' + _gen_uuid()
            os.mkdir(folder)
            ContextBase.app_folder = folder
            ContextBase.DEBUG = False
            self.read = dr.DataReader()

    @staticmethod
    def stop():
        """
        Stop the DDF environment. It is important to stop at end of an
        application in order to avoid that COMPSs sends back all partial
        result at end.
        """
        import shutil
        import os
        for id_task in list(ContextBase.catalog_tasks.keys()):
            data = ContextBase.catalog_tasks[id_task].get('result', [])

            if check_serialization(data):
                delete_result(data)

        ContextBase.catalog_schemas = dict()
        ContextBase.catalog_tasks = dict()
        ContextBase.dag = nx.DiGraph()
        ContextBase.catalog_tasks = list()

        if ContextBase.monitor:
            ContextBase.monitor.stop()
            os.remove('/tmp/ddf_dash_object.pickle')

        # TEMPORARY
        import glob
        files_sync = glob.glob(ContextBase.app_folder+'/*')
        if len(files_sync) > 0:
            raise Exception('Partial files were synchronized.')

        shutil.rmtree(ContextBase.app_folder)

    def start_monitor(self):
        """
        Start a web service monitor that informs the environment current status.
        The process will be shown in http://127.0.0.1:58227/.
        :return:
        """
        ContextBase.start_monitor()
        return self

    def show_tasks(self):
        """
        Show all tasks in the current code. Only to debug.
        :return:
        """
        print("\nList of all tasks:")
        t = PrettyTable(['uuid', 'Task name', 'STATUS', 'Example result'])

        for uuid in ContextBase.catalog_tasks:
            r = ContextBase.catalog_tasks[uuid].get('result', [])
            if isinstance(r, list):
                if len(r) > 0:
                    r = r[0]
                else:
                    r = ''
            t.add_row([uuid[:8],
                       ContextBase.catalog_tasks[uuid]['name'],
                       ContextBase.catalog_tasks[uuid]['status'],
                       r])
        print(t)
        print('\n')
        return self

    def set_log(self, enabled=True):
        """
        Set the log level.

        :param enabled: True to debug, False to off.
        :return:
        """
        ContextBase.DEBUG = enabled
        return self

    @staticmethod
    def context_status():
        """
        Generates a DAG (in dot file) and some information on screen about
        the status process.

        :return:
        """
        table = ContextBase.gen_status()
        t = PrettyTable(['Metric', 'Value'])
        for row in table:
            t.add_row(row)

        print("\nContext status:\n{}\n".format(t))
        ContextBase.plot_graph()

    @staticmethod
    def parallelize(df, num_of_parts='*'):
        """
        Distributes a DataFrame into DDF.

        :param df: DataFrame input
        :param num_of_parts: number of partitions (default, '*' meaning all
         cores available in master CPU);
        :return: DDF

        :Example:

        >>> cc = COMPSsContext()
        >>> ddf1 = cc.parallelize(df)
        """

        from .functions.etl.parallelize import parallelize
        if isinstance(num_of_parts, str):
            import multiprocessing
            num_of_parts = multiprocessing.cpu_count()

        def task_parallelize(_, params):
            return parallelize(df, num_of_parts)

        # result, info = parallelize(df, num_of_parts)
        # ContextBase.catalog_schemas[new_state_uuid] = info
        first_uuid = ContextBase.create_init()
        new_state_uuid = ContextBase \
            .ddf_add_task('parallelize',
                          opt=ContextBase.OPT_OTHER,
                          function=[task_parallelize, {}],
                          parent=[first_uuid],
                          n_input=0)

        return DDF(task_list=[first_uuid], last_uuid=new_state_uuid)

    @staticmethod
    def import_data(df_list, info=None, parquet=False):
        # noinspection PyUnresolvedReferences
        """
        Import a previous Pandas DataFrame list into DDF abstraction.

        :param df_list: DataFrame input
        :param parquet: if data is saved as list of parquet files
        :param info: (Optional) A list of columns names, data types and size
         in each partition;
        :return: DDF

        :Example:

        >>> cc = COMPSsContext()
        >>> ddf1 = cc.import_partitions(df_list)
        """

        from .functions.etl.parallelize import import_to_ddf

        def task_import_to_ddf(x, y):
            return import_to_ddf(df_list, parquet=parquet, schema=info)

        result, info = import_to_ddf(df_list, parquet=parquet, schema=info)

        first_uuid = ContextBase.create_init()

        new_state_uuid = ContextBase \
            .ddf_add_task('import_data',
                          status=ContextBase.STATUS_COMPLETED,
                          opt=ContextBase.OPT_OTHER,
                          function=[task_import_to_ddf, {}],
                          parent=[first_uuid],
                          result=result,
                          n_input=0,
                          info_data=info)

        return DDF(task_list=[first_uuid], last_uuid=new_state_uuid)
