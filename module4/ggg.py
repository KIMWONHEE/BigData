from apyori import apriori

import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter
from orangecontrib.associate.fpgrowth import *
# from orangecontrib.associate.fpgrowth import *
# import Orange
from scipy.stats._continuous_distns import frechet_l_gen


def size(obj):
    return "{0:.2f} MB".format(sys.getsizeof(obj) / (1000 * 1000))

orders = pd.read_csv('/home/machine/Desktop/data/train.csv', header=None)
orders.columns = ["user_id","product_id","add_to_cart_order","reordered"]

orderproduct = orders.groupby(["user_id"])["product_id"].apply(list)

# data = Orange.data.Table("lenses"
# rules = Orange.associate.AssociationRulesInducer(data, support=0.3)
# for r in rules:
    # print "%5.3f  %5.3f  %s" % (r.support, r.confidence, r)
# itemset = frequent_itemsets(list(orderproduct),0.01)
# itemset = frequent_itemsets(list(orderproduct),4)
# print(list(itemset))

# class_items = {item for item,var, _ in OneHot.decode(mapping,data,mapping)
#                if var is data
# }

# rules = [(P,Q, supp, conf)
#          for P,Q, supp, conf in association_rules(itemset, .7)
#          if len(Q) == 1 and Q & class_items]


# Item = frequent_itemsets(orderproduct,2)
# print(list(Item))

# rules = Orange.associate.AssociationRulesInducer(orderproduct, support=0.3)
rules = list(apriori(orderproduct, min_support = 0.3, min_confidence = 0.01))

# for r in rules:
#     print("%5.3f  %5.3f  %s" % (r.support, r.confidence, r))

f = open("/home/machine/Desktop/output3.txt",'w')
for i in rules:
        line=str(i)
        f.write(line + '\n')