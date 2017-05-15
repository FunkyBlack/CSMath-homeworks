# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:24:22 2017

@author: FunkyBlack
"""
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=123)

color = {}
for i, label in enumerate(y):
    color[i] = 'brown' if label == 0 else 'blue'
plt.figure(figsize=(10,10))
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Generated data with two informative features, one cluster per class")
plt.scatter(X[:,0], X[:,1], marker='o', c=list(color.values()), edgecolor='black')
plt.show()
plt.savefig('Generated_data.png')

fname = ('dataset.txt')

with open(fname, 'w') as f:
    for feature, label in zip(X, y):
        label = -1 if label > 0 else 1
        write_str = '%f\t%f\t%d\n'%(feature[0], feature[1], label)
        f.writelines(write_str)
    f.close()
