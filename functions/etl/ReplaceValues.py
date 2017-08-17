# -*- coding: utf-8 -*-
#!/usr/bin/env python


from pycompss.functions.data import chunks
from pycompss.api.task          import task
from pycompss.api.parameter     import *
from pycompss.functions.reduce import mergeReduce

import numpy as np
import math
import pickle
import pandas as pd


def  ReplaceValuesOperation (data,settings,numFrag):
