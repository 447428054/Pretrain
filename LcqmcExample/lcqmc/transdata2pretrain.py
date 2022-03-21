#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： JMXGODLZZ
# datetime： 2022/3/18 下午2:43 
# ide： PyCharm

import os
import matplotlib.pyplot as plt

datadir = './data'
savepath = 'lcqmc4pretrain.txt'
fw = open(savepath, encoding='utf-8', mode='w')
lengdict = {}
for filename in os.listdir(datadir):
    datapath = os.path.join(datadir, filename)
    lines = open(datapath, encoding='utf-8').readlines()
    for line in lines:
        line = line.strip()
        l1, l2, label = line.split('\t')
        lg = len(l1)
        lg2 = len(l2)
        lengdict.setdefault(lg, 0)
        lengdict.setdefault(lg2, 0)
        lengdict[lg] += 1
        lengdict[lg2] += 1
        if label == '1':
            fw.write('{}\n{}\n\n'.format(l1, l2))
        else:
            fw.write('{}\n\n{}\n\n'.format(l1, l2))

lengdict = sorted(lengdict.items(), key=lambda item: item[0])
x = [item[0] for item in lengdict]
y = [item[1] for item in lengdict]

fig = plt.figure()
plt.title('LCQMC Sentence Length')
plt.plot(x, y)
plt.savefig('lcqmc_leng.png')