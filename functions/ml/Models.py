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

from pycompss.api.parameter import *
from pycompss.api.task import task
import cPickle as pickle


@task(returns=list)
def SaveModelToHDFS(model, settings):
    """SaveModelToHDFS.

    Save a machine learning model in HDFS.
    :param settings:  A dictionary with:
        - path:       The path of the file from the '/' of the HDFS;
        - host:       The host of the Namenode HDFS; (default, 'default')
        - port:       Port of the Namenode HDFS; (default, 0)
        - overwrite:  True if overwrite in case of colision name,
                      False to raise a error.
    """
    import hdfs_pycompss.hdfsConnector as hdfs

    overwrite = settings.get('overwrite', True)

    if hdfs.ExistFile(settings) and not overwrite:
        raise Exception("File already exists in this source.")

    to_save = pickle.dumps(model, 0)
    success, dfs = hdfs.writeBlock(settings, to_save, None, False)
    return [success]


@task(returns=dict)
def LoadModelFromHDFS(settings):
    """LoadModelFromHDFS.

    Load a machine learning model from a HDFS source.
    :param settings: A dictionary with:
        - path:      The path of the file from the / of the HDFS;
        - host:      The host of the Namenode HDFS; (default, 'default')
        - port:      Port of the Namenode HDFS; (default, 0)
    :return:         Returns a model (a dictionary)
    """
    import hdfs_pycompss.hdfsConnector as hdfs

    to_load = hdfs.readAllBytes(settings)
    model = None
    if len(to_load) > 0:
        model = pickle.loads(to_load)
    return model


@task(filename=FILE_OUT)
def SaveModel(model, filename):
    """SaveModel.

    Save a machine learning model into a file.
    :param filename: Absolute path of the file;
    """
    with open(filename, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


@task(returns=dict, filename=FILE_IN)
def LoadModel(filename):
    """LoadModel.

    Load a machine learning model from a file.
    :param filename: Absolute path of the file;
    :return:         Returns a model (a dictionary).
    """
    with open(filename, 'rb') as input:
        model = pickle.load(input)
    return model
