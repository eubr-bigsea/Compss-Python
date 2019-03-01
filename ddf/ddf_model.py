#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import cPickle as pickle
from ddf import DDFSketch


class ModelDDF(DDFSketch):
    """

    """

    def __init__(self):
        super(ModelDDF, self).__init__()

        self.settings = dict()
        self.model = {}
        self.name = ''

    def save_model(self, filepath, storage='hdfs', overwrite=True,
                   namenode='localhost', port=9000):
        """
        Save a machine learning model as a binary file in a storage.

        :param filepath: The output absolute path name;
        :param storage: *'hdfs'* to save in HDFS storage or *'fs'* to save in
         common file system;
        :param overwrite: Overwrite if file already exists (default, True);
        :param namenode: IP or DNS address to NameNode (default, *'localhost'*);
        :param port: NameNode port (default, 9000);
        :return: self

        :Example:

        >>> ml_model.save_model('/trained_model')
        """
        if storage not in ['hdfs', 'fs']:
            raise Exception('Only `hdfs` and `fs` storage are supported.')

        if storage == 'hdfs':
            save_model_hdfs(self.model, filepath, namenode,
                            port, overwrite)
        else:
            save_model_fs(self.model, self.settings)

        return self

    def load_model(self, filepath, storage='hdfs', namenode='localhost',
                   port=9000):
        """
        Load a machine learning model from a binary file in a storage.

        :param filepath: The absolute path name;
        :param storage: *'hdfs'* to load from HDFS storage or *'fs'* to load
         from common file system;
        :param storage: *'hdfs'* to save in HDFS storage or *'fs'* to save in
         common file system;
        :param namenode: IP or DNS address to NameNode (default, *'localhost'*).
         Note: Only if storage is *'hdfs'*;
        :param port: NameNode port (default, 9000). Note: Only if storage is
         *'hdfs'*;
        :return: self

        :Example:

        >>> ml_model = ml_algorithm().load_model('/saved_model')
        """
        if storage not in ['hdfs', 'fs']:
            raise Exception('Only `hdfs` and `fs` storage are supported.')

        if storage == 'hdfs':
            self.model = load_model_hdfs(filepath, namenode, port)
        else:
            self.model = load_model_fs(filepath)

        return self


def save_model_hdfs(model, path, namenode='localhost', port=9000,
                    overwrite=True):
    """
    Save a machine learning model as a binary file in a HDFS storage.

    :param model: Model to be storaged in HDFS;
    :param path: The path of the file from the '/' of the HDFS;
    :param namenode: The host of the Namenode HDFS; (default, 'localhost')
    :param port: NameNode port (default, 9000).
    :param overwrite: Overwrite if file already exists (default, True);
    """
    from hdfspycompss.HDFS import HDFS
    dfs = HDFS(host=namenode, port=port)

    if dfs.exist(path) and not overwrite:
        raise Exception("File already exists in this source.")

    to_save = pickle.dumps(model, 0)
    dfs.writeBlock(path, to_save, append=False, overwrite=True)
    return [-1]


def save_model_fs(model, filepath):
    """
    Save a machine learning model as a binary file in a common file system.

    Save a machine learning model into a file.
    :param filepath: Absolute file path;
    :param model: The model to be saved
    """
    with open(filepath, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load_model_hdfs(filepath, namenode='localhost', port=9000):
    """
    Load a machine learning model from a HDFS source.

    :param filepath: The path of the file from the '/' of the HDFS;
    :param namenode: The host of the Namenode HDFS; (default, 'localhost')
    :param port: NameNode port (default, 9000).
    :return: Returns a model
    """
    from hdfspycompss.HDFS import HDFS
    from hdfspycompss.Block import Block

    dfs = HDFS(host=namenode, port=port)
    blk = dfs.findNBlocks(filepath, 1)
    to_load = Block(blk).readBinary()
    model = None
    if len(to_load) > 0:
        model = pickle.loads(to_load)
    return model


def load_model_fs(filepath):
    """
    Load a machine learning model from a common file system.

    :param filepath: Absolute file path;
    :return: Returns a model.
    """
    with open(filepath, 'rb') as data:
        model = pickle.load(data)
    return model
