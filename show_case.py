# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:03:15 2023

@author: nkliu
"""

import pickle
import numpy as np


npz_data = np.load('C:/Users/nkliu/Desktop/interaction/ntu120/NTU120_CSub.npz')
label = np.where(npz_data['y_test'] > 0)[1]


fr = open("results/baseline.pkl",'rb')
inf = pickle.load(fr)
fr.close()
feature = np.zeros((11660, 26))
test_dict = {"task": "recognition"}
for ind, item in enumerate(inf):
    feature[ind, :] = inf[item]
pred1 = feature.argmax(1)
    

fr = open("results/ours.pkl",'rb')
inf = pickle.load(fr)
fr.close()
feature = np.zeros((11660, 26))
test_dict = {"task": "recognition"}
for ind, item in enumerate(inf):
    feature[ind, :] = inf[item]
pred2 = feature.argmax(1)



for ind, (p1, p2, lb) in enumerate(zip(pred1, pred2, label)):
    if p1 != lb:
        if p2 == lb:
            if lb == 0:
                print(ind)

    

# result1 = np.zeros(26)
# result2 = np.zeros(26)

# for p1, p2, lb in zip(pred1, pred2, label):
#     result1[lb] += (p1 == lb)
#     result2[lb] += (p2 == lb)
    