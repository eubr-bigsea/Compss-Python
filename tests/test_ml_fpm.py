#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf_library.ddf import DDF
import pandas as pd


def ml_fpm_fpgrowth():
    from ddf_library.functions.ml.fpm import AssociationRules, FPGrowth

    """
    Transaction id	Items
    t1	{1, 3, 4}
    t2	{2, 3, 5}
    t3	{1, 2, 3, 5}
    t4	{2, 5}
    t5	{1, 2, 3, 5}
    
    itemsets	 support

    {2}	            4
    {3}     	    4
    {5}	            4
    {2, 5}	        4
    {1}	            3
    {1, 3}	        3
    {2, 3}  	    3
    {3, 5}	        3
    {2, 3, 5}	    3
    {1, 2}	        2
    {1, 5}  	    2
    {1, 2, 3}	    2
    {1, 2, 5}	    2
    {1, 3, 5}	    2
    {1, 2, 3, 5}	2
    """
    data = pd.DataFrame([['1,3,4'],
                         ['2,3,5'],
                         ['1,2,3,5'],
                         ['2,5'],
                         ['1,2,3,5']], columns=['col_0'])

    # data_set = DDF() \
    #     .parallelize(data, 2) \
    #     .map(lambda row: row['col_0'].split(','), 'col_0')

    data_set = DDF()\
        .load_text('/transactions.csv', num_of_parts=4, header=False,
                   sep='\n')\
        .map(lambda row: row['col_0'].split(' ')[:-1], 'col_0')

    """
    when support is 2.84%  (284 in 10000):
                       items  support
        0               (39)     5489
        1               (48)     4312
        2           (39, 48)     2907
        3               (41)     2663
        4           (39, 41)     1973
        5               (32)     1828
        6               (38)     1722
        7           (48, 41)     1473
        8       (39, 48, 41)     1183
        9           (39, 38)     1105
        10          (32, 39)     1003
        11          (32, 48)      947
        12          (38, 48)      775
        13          (38, 41)      697
        14      (32, 39, 48)      605
        15          (32, 41)      597
        16      (39, 38, 48)      583
        17      (39, 38, 41)      530
        18      (32, 39, 41)      427
        19              (65)      393
        20             (170)      391
        21              (89)      387
        22         (38, 170)      382
        23            (1327)      380
        24      (38, 48, 41)      377
        25      (32, 48, 41)      362
        26             (310)      360
        27             (225)      351
        28             (352)      343
        29             (604)      338
        30             (237)      329
        31          (32, 38)      324
        32              (36)      321
        33  (39, 38, 48, 41)      315
        34          (38, 36)      308
        35             (475)      304
        36              (60)      293
        37             (110)      291
    """

    fp = FPGrowth(min_support=0.0284)
    item_set = fp.fit_transform(data_set, column='col_0')

    rules = AssociationRules(confidence=0.7) \
        .fit_transform(item_set) \
        .select(['Pre-Rule', 'Post-Rule', 'confidence']).cache()

    print('RESULT item set:')
    item_set.show()
    print("RESULT rules:")
    rules.show()


if __name__ == '__main__':
    print("_____Testing Frequent Pattern Mining_____")
    ml_fpm_fpgrowth()
