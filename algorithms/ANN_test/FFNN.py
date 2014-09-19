# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:59:28 2014

@author: Hetian Chen
"""
import neurolab as nl
import numpy as np
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn import cross_validation

#import data and labels
path='/Users/071cht/Desktop/chase/sample/'
data = np.loadtxt(path+'window_matrix_preprocess_scale.txt',delimiter= ',') 
labels = np.loadtxt(path+'label_vector_preprocess.txt')

intLabels=labels.astype(int)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)

pca = PCA(n_components='mle')
reduction=pca.fit(data_nonmissing)
X = reduction.transform(data_nonmissing)

size=len(X)
inp=X
#==============================================================================
# y_temp = labels.reshape(size,1)
# y_temp2= np.abs(y_temp-1)
# y=np.column_stack((y_temp2,y_temp))
#==============================================================================
y=labels.reshape(size,1)
"""clear unused variables"""
#==============================================================================
# data=None
# labels=None
# intLabels=None
# imp=None
# pca=None
# reduction=None
#==============================================================================


cvfold=3
kfold = cross_validation.KFold(len(X), n_folds=cvfold)

ACC=list()
error=list()

for i,(train,test) in enumerate(kfold):
    
    inp=X[train]
    tar=y[train]
    inp_test=X[test]
    tar_test=y[test]
    #create neural net
    hidden_nodes=(1+inp.shape[1])*1/2
    net = nl.net.newff([[0, 1]]*X.shape[1],[hidden_nodes, 1])
   # print inp.shape, tar.shape, net.ci,net.co
   # the value of feature range from 0 to 1.
   # input number = X.shape[1]
   # hidden nodes number= hidden_nodes
   # output number = 1
    
    net.trainf=nl.train.train_bfgs
    error_elm = net.train(inp, tar, epochs=1500, show=100, goal=0.00001)
    out = net.sim(inp_test)
#==============================================================================
#     pred_temp=out[:,1]
#     pred=np.where(pred_temp<0.5,0,pred_temp)
#     pred=np.where(pred_temp>=0.5,1,pred)
#     true=tar_test[:,0]
#==============================================================================
    pred=np.where(out<0.5,0,out)
    pred=np.where(pred>0.5,1,pred)
    true=tar_test    
    
    ACC_elm= 1-sum(np.abs(pred-true))/len(true)
    ACC.append(ACC_elm)
    error.append(error_elm)
    print 'ACC_elm', ACC_elm

print 'ACC=', str(sum(ACC)/cvfold) 


# pl.plot(0)
# #pl.plot(error[0][150:],'-',error[1][150:],'.',error[2][150:],'p')
# pl.plot(error[0][:1000],'-',error[1][:1000],'.',error[2][:1000],'p')
# pl.legend(['cv fold 1', 'cv fold 2', 'cv fold 3'])
# pl.ylabel('error (default SSE)')
# pl.xlabel('Epoch number')
# pl.show()
