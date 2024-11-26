# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:49:36 2023

@author: nkliu
"""

import numpy as np
import pickle

train_x = np.load('train_data_joint_200.npy')
f = open('train_label.pkl', 'rb')
train_y = pickle.load(f)[1]

test_x = np.load('test_data_joint_200.npy')
test_y =(np.zeros(len(test_x))).tolist()


np.savez('assemble101.npz', x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

    
# train_x = np.load('train_data_joint_200.npy')
# f = open('train_label.pkl', 'rb')
# train_y = pickle.load(f)[1]

# test_x = np.load('validation_data_joint_200.npy')
# f = open('validation_label.pkl', 'rb')
# test_y = pickle.load(f)[1]


# np.savez('assemble101.npz', x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
