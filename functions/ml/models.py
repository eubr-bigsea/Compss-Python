#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Methods that can be used in models.

- SaveModelToHDFS: Save a machine learning model in HDFS.
- LoadModelFromHDFS: Load a machine learning model from a HDFS source.
- SaveModel: Save a machine learning model into a file.
- LoadModel: Load a machine learning model from a file.
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT
import cPickle as pickle


def SaveModelOperation(model, settings):
    """
    Save a machine learning model in HDFS or in commom FS.
    :param model: The model to be saved
    :param settings:
    :return:
    """
    storage = settings.get('storage', 'hdfs')
    if storage == 'hdfs':
        return save_model_hdfs(model, settings)
    else:
        return save_model_fs(model, settings)


@task(returns=list)
def save_model_hdfs(model, settings):
    """SaveModelToHDFS.
    :param settings:  A dictionary with:
        - path:       The path of the file from the '/' of the HDFS;
        - host:       The host of the Namenode HDFS; (default, 'default')
        - port:       Port of the Namenode HDFS; (default, 0)
        - overwrite:  True if overwrite in case of colision name,
                      False to raise a error.
    """
    from hdfspycompss.HDFS import HDFS
    host = settings.get('host', 'localhost')
    port = settings.get('port', 9000)
    dfs = HDFS(host=host, port=port)

    overwrite = settings.get('overwrite', True)

    if dfs.ExistFile(settings) and not overwrite:
        raise Exception("File already exists in this source.")

    to_save = pickle.dumps(model, 0)
    success, dfs = dfs.writeBlock(settings, to_save, None, False)
    return [success]


@task(filename=FILE_OUT)
def save_model_fs(model, filename):
    """SaveModel.

    Save a machine learning model into a file.
    :param filename: Absolute path of the file;
    :param model: The model to be saved
    """
    with open(filename, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def LoadModelOperation(settings):
    """
    Load a machine learning model in HDFS or in commom FS.
    :param settings:
    :return:
    """
    storage = settings.get('storage', 'hdfs')
    if storage == 'hdfs':
        return load_model_hdfs(settings)
    else:
        filename = settings['filename']
        return load_model_fs(filename)


@task(returns=dict)
def load_model_hdfs(settings):
    """LoadModelFromHDFS.

    Load a machine learning model from a HDFS source.
    :param settings: A dictionary with:
        - path:      The path of the file from the / of the HDFS;
        - host:      The host of the Namenode HDFS; (default, 'default')
        - port:      Port of the Namenode HDFS; (default, 0)
    :return:         Returns a model (a dictionary)
    """
    from hdfspycompss.HDFS import HDFS
    from hdfspycompss.Block import Block
    host = settings.get('host', 'localhost')
    port = settings.get('port', 9000)
    filename = settings['filename']

    dfs = HDFS(host=host, port=port)
    blk = dfs.findNBlocks(filename, 1)
    to_load = Block(blk).readBinary()
    model = None
    if len(to_load) > 0:
        model = pickle.loads(to_load)
    return model


@task(returns=dict, filename=FILE_IN)
def load_model_fs(filename):
    """LoadModel.

    Load a machine learning model from a file.
    :param filename: Absolute path of the file;
    :return:         Returns a model (a dictionary).
    """
    with open(filename, 'rb') as input:
        model = pickle.load(input)
    return model
