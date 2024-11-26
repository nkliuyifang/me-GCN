# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:03:15 2023

@author: nkliu
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 26.),
                 fontdict={'weight': 'bold', 'size': 20})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.axis('off')
    return fig


fr = open("results/mte.pkl",'rb')
inf = pickle.load(fr)
fr.close()

feature = np.zeros((11660, 26))

test_dict = {"task": "recognition"}
for ind, item in enumerate(inf):
    feature[ind, :] = inf[item]

pred = feature.argmax(1)
    
npz_data = np.load('C:/Users/nkliu/Desktop/interaction/ntu120/NTU120_CSub.npz')
label = np.where(npz_data['y_test'] > 0)[1]

acc = (pred==label).sum() / len(label)
print(acc)

#feature = feature[0:100, :]
#label = label[0:100]

tsne = TSNE(n_components=2, init='pca', random_state=0)
result = tsne.fit_transform(feature)
print('result.shape',result.shape)
fig = plot_embedding(result, label,
                     ' ')
