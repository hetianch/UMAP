# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:44:06 2014

@author: 071cht
"""


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

#import data and labels
path='/Users/071cht/Desktop/ANN_test/'
data = np.loadtxt(path+'scaled_x.txt',delimiter= ',') 
labels = np.loadtxt(path+'label_vector.txt')

intLabels=labels.astype(int)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)

pca = PCA(n_components=25)
reduction=pca.fit(data_nonmissing)
X = reduction.transform(data_nonmissing)

filename='x_pca.txt'

np.savetxt(path+filename,X)