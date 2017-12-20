import re

f = open("/home/machine/Desktop/output2.txt", 'r')
itemlist = []
baselist=[]
addlist=[]
conflist=[]

while True:
    line = f.readline()
    if not line: break
    itembase_s=line.find("items_base")
    itembase_e=line.find(")",itembase_s)
    itembase=re.findall('\d+',line[itembase_s:itembase_e])
    if(len(itembase)==0):
        itembase='0'
    else:
        itembase=",".join(itembase)
    # print(re.findall('\d+',itembase))
    itemadd_s = line.find("items_add")
    itemadd_e = line.find("}",itemadd_s)
    itemadd = ",".join(re.findall('\d+',line[itemadd_s:itemadd_e]))
    confi_s=line.find("confidence")
    confi_e=line.find(",",confi_s)
    confi=",".join(re.findall('\d+',line[confi_s:confi_e]))
    baselist.append(itembase)
    addlist.append(itemadd)
    conflist.append(confi)
    item=[]
    item.append(itembase)
    item.append(itemadd)
    item.append(confi)
    itemlist.append(item)
f.close()

import collections

dict = collections.defaultdict(list)
for k in itemlist:
    pair = (k[1],k[2])
    dict[k[0]].append(pair)

# for i in dict.keys():
#     for j in range(len(dict[str(i)])):
#         print(dict[str(i)][j][0] + '  ', end='')
#     print('\n')
import operator
for i in dict.keys():
    dict[i].sort(key=operator.itemgetter(1),reverse=True)

# for i in dict.keys():
#     print(i.split(","))

    # print(i)
# for k in sorted(dict, key=len, reverse=True): # Through keys sorted by length
#         text = text.replace(k, dict[k])
# import pandas as pd
f = open('/home/machine/Desktop/data/test.txt','r')
testlist=[]
while True:
    line = f.readline()
    if not line: break
    line = line.replace('\n',"")
    testline=line.split(",")
    testlist.append(testline)
f.close()

# for i in testlist:
#     print(i)


recommand = [[] for x in range(1000)]
t=0

for i in testlist:
    for j in dict.keys():
        base=j.split(",")
        if(set(base).issubset(set(i))):
            for k in range(len(dict[str(j)])):
                if(not set(dict[str(j)][k][0]).issubset(set(i))):
                    recommand[t].append(dict[str(j)][k])
                    # print(dict[str(j)][k][0], end='')
                    # print(' : ', end='')
                    # print(i, end='')
                    # print()
                    # print(t)
    t+=1

# for i in range(1000):
#     for j in recommand[i]:
#         if(len(j)<5):
#             for k in range(len(j),5):


for i in range(1000):
    recommand[i].sort(key=operator.itemgetter(1),reverse=True)
    # for j in recommand[i]:
    # dict[i].sort(key=operator.itemgetter(1),reverse=True)


recommandlist = [[] for x in range(1000)]

for i in range(1000):
    for j in recommand[i]:
        if j[0] not in recommandlist[i]:
            recommandlist[i].append(j[0])

# for i in dict[str(0)]:
#     print(i)


for i in range(1000):
    if(len(recommandlist[i])<5):
        for j in dict[str(0)]:
            if j[0] not in recommandlist[i]:
                recommandlist[i].append(j[0])

for i in range(1000):
    print(str(i) + ' : ', end='')
    for j in recommandlist[i]:
        print(j,end='')
        print(' ',end='')
    print()
#
f = open('/home/machine/Desktop/data/out.txt', 'w')
print(recommand[999])
for i in range(1000):
    for j in range(5):
        f.write(str(recommandlist[i][j]) + ',')
    f.write('\n')

# print(len(addlist))
# print(baselist[0])
# print(len(baselist))
#
# f = open("/home/machine/Desktop/output.txt", 'w')
# for i in baselist:
#     f.write(str(i) + '\n')