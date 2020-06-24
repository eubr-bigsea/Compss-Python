#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

"""
DDF is a Library for PyCOMPSs.
"""

from ddf_library.bases.metadata import Status, OPTGroup
from ddf_library.bases.context_base import ContextBase
from ddf_library.ddf import DDF
from ddf_library.utils import _gen_uuid
import ddf_library.bases.data_reader as dr

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

        ContextBase.catalog_tasks.clear()

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
        ContextBase.catalog_tasks.show_tasks()
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

        from ddf_library.bases.optimizer.operations import Parallelize
        if isinstance(num_of_parts, str):
            import multiprocessing
            num_of_parts = multiprocessing.cpu_count()

        settings = {'nfrag': num_of_parts, 'input data': df}

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=Parallelize(settings))

        return DDF(last_uuid=new_state_uuid)

    @staticmethod
    def import_compss_data(df_list, schema=None, parquet=False):
        # noinspection PyUnresolvedReferences
        """
        Import a previous Pandas DataFrame list into DDF abstraction.

        :param df_list: DataFrame input
        :param parquet: if data is saved as list of parquet files
        :param schema: (Optional) A list of columns names, data types and size
         in each partition;
        :return: DDF

        :Example:

        >>> cc = COMPSsContext()
        >>> ddf1 = cc.import_compss_data(df_list)
        """

        settings = {'parquet': parquet, 'schema': schema, 'input_data': df_list}
        from ddf_library.bases.optimizer.operations import ImportCOMPSsData

        new_state_uuid = ContextBase \
            .ddf_add_task(operation=ImportCOMPSsData(settings),
                          status=Status.STATUS_WAIT)

        return DDF(last_uuid=new_state_uuid)
