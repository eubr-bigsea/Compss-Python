#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import _pickle as pickle
from .ddf_base import DDFSketch


class ModelDDF(DDFSketch):
    """
    Class base of model DDF algorithms
    """

    def __init__(self):
        super(ModelDDF, self).__init__()

        self.model = dict()
        self.name = self.__class__.__name__
        self.output_col = None
        self.input_col = None
        self.remove = False

    def check_fitted_model(self):
        if self.model.get('algorithm') != self.name:
            raise Exception("Model is not fitted by {}".format(self.name))

    def save_model(self, filepath, overwrite=True):
        # noinspection PyUnresolvedReferences
        """
        Save a machine learning model as a binary file in a storage.

        :param filepath: The output absolute path name;
        :param overwrite: Overwrite if file already exists (default, True);
        :return: self

        :Example:

        >>> cls = KMeans().fit(dataset, input_col='features')
        >>> cls.save_model('hdfs://localhost:9000/trained_model')
        """
        host, port = None, None
        import re
        if re.match(r"hdfs:\/\/+", filepath):
            storage = 'hdfs'
            host, filename = filepath[7:].split(':')
            port, filename = filename.split('/', 1)
            filename = '/' + filename
            port = int(port)
        elif re.match(r"file:\/\/+", filepath):
            storage = 'file'
            filename = filepath[7:]
        else:
            raise Exception('`hdfs:` and `file:` storage are supported.')

        if storage == 'hdfs':
            from hdfspycompss.hdfs import HDFS
            dfs = HDFS(host=host, port=port)

            if dfs.exist(filename) and not overwrite:
                raise Exception("File already exists in this source.")

            to_save = pickle.dumps(self.__dict__, 0)
            dfs.write_block(filename, to_save, append=False,
                            overwrite=True, binary=True)

        else:
            with open(filepath, 'wb') as output:
                pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

        return self

    def load_model(self, filepath):
        # noinspection PyUnresolvedReferences
        """
        Load a machine learning model from a binary file in a storage.

        :param filepath: The absolute path name;
        :return: self

        :Example:

        >>> ml_model = Kmeans().load_model('hdfs://localhost:9000/model')
        """
        host, port = None, None
        import re
        if re.match(r"hdfs:\/\/+", filepath):
            storage = 'hdfs'
            host, filename = filepath[7:].split(':')
            port, filename = filename.split('/', 1)
            filename = '/' + filename
            port = int(port)
        elif re.match(r"file:\/\/+", filepath):
            storage = 'file'
            filename = filepath[7:]
        else:
            raise Exception('`hdfs:` and `file:` storage are supported.')

        if storage == 'hdfs':
            from hdfspycompss.hdfs import HDFS
            from hdfspycompss.block import Block

            blk = HDFS(host=host, port=port)\
                .find_n_blocks(filename, 1)
            to_load = Block(blk).read_binary()

            if len(to_load) > 0:
                self.__dict__ = pickle.loads(to_load)
        else:
            with open(filename, 'rb') as data:
                self.__dict__ = pickle.load(data)

        return self
