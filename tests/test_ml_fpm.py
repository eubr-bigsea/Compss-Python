#!/usr/bin/python
# -*- coding: utf-8 -*-

from ddf.ddf import DDF


def ml_fpm():
    from ddf.functions.ml.fpm import AssociationRules, Apriori
    dataset = DDF()\
        .load_text('/transactions.txt', num_of_parts=4, header=False, sep='\n')\
        .transform(lambda row: row['col_0'].split(','), 'col_0')

    apriori = Apriori(column='col_0', min_support=0.10).run(dataset)

    itemset = apriori.get_frequent_itemsets()

    ar = AssociationRules(confidence=0.10)\
        .run(itemset)\
        .select(['Pre-Rule', 'Post-Rule', 'confidence']).cache()

    rules1 = apriori.generate_association_rules(confidence=0.1).cache()

    itemset = itemset.select(['items', 'support'])

    rules1 = rules1.show()[:20]
    itemset = itemset.cache().show()[:20]
    rules2 = ar.show()[:20]
    print 'RESULT itemset:', itemset
    print "RESULT rules1:", rules1
    print "RESULT rules2:", rules2


if __name__ == '__main__':
    print "_____Testing Frequent Pattern Mining_____"
    ml_fpm()
