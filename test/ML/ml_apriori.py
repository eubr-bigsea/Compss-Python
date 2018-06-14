#!/usr/bin/python

from pycompss.api.api import compss_wait_on
from functions.data.read_data import ReadOperationHDFS
from functions.etl.transform import TransformOperation
from functions.ml.associative.apriori.apriori import Apriori
from functions.ml.associative.AssociationRules import AssociationRulesOperation
import pandas as pd


if __name__ == '__main__':
    """Test Apriori and Association Rules function."""
    numFrag = 4
    settings = dict()
    settings['port'] = 9000
    settings['host'] = 'localhost'
    filename = '/transactions.txt'
    settings['header'] = False
    settings['separator'] = '\n'
    data0 = ReadOperationHDFS().transform(filename, settings, numFrag)

    data0 = compss_wait_on(data0)
    print data0

    settings = dict()
    settings['functions'] = \
            [['col_0', "lambda col: col['col_0'].split(',')", '']]

    data0 = TransformOperation().transform(data0, settings, numFrag)
    data0 = compss_wait_on(data0)

    settings = dict()
    # settings['column'] = "col_0"
    settings['confidence'] = 0.10  # 0.68
    settings['minSupport'] = 0.05  # 0.17

    pfreq = Apriori()
    freqSet = pfreq.runApriori(data0, settings, numFrag)
    # rules = pfreq.generateRules(freqSet,settings)

    # or:
    rules = AssociationRulesOperation(freqSet, settings)

    rules = compss_wait_on(rules)
    rules = pd.concat(rules, axis=0)
    print rules
