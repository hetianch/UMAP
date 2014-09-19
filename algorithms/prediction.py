import numpy as np
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
# import data and labels using python array handling package numpy
data = np.loadtxt('/Users/071cht/Desktop/for_conference/data/original/output/ori_cv_X_scale.txt',delimiter= ',') 
labels = np.loadtxt("/Users/071cht/Desktop/for_conference/data/original/output/ori_cv_Y.txt")
data_whole_contents1= sio.loadmat('/Users/071cht/Desktop/for_conference/data/original/output/window_scale1.mat')
data_whole_contents2= sio.loadmat('/Users/071cht/Desktop/for_conference/data/original/output/window_scale2.mat')


data_whole1=data_whole_contents1['window_scale1']  #'window' is the variable name in this window2.mat
data_whole2=data_whole_contents2['window_scale2']
# check dataset
#print data,labels
#print type(data), type(labels)
#print data.shape,labels.shape,data.dtype,labels.dtype
#print 'data',data.shape
#print 'data_whole',data_whole.shape
#print 'data_whole[0,0]',data_whole[0,0].shape
#print 'data_whole[0,1]',data_whole[0,1].shape
"""somehow labels are loaded as float so we need to change it back to int in order to implement classifier"""
intLabels=labels.astype(int)
#print intLabels.dtype
#assert not numpy.any(numpy.isnan(data) | numpy.isinf(data))
#print numpy.isinf(data),numpy.isnan(data)
#print 1 in numpy.isinf(data) # if returns true, means there're inf in data
#print 1 in numpy.isnan(data) # if returns true, means there're nan in data



"""before imputation, we must drop colomns that only contain missing values for data and data_whole first because ,when inputating
it will automatically discard colomns with only NaN before transform. This will make the imputation of data and data_whole different 
in colomn size!!! We did this by using the scaled data"""




#the following code retures an np.array, data_nonmissing. It is a non-missing value version of data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)

data_whole1_nonmissing=np.empty((data_whole1.shape[1],),dtype=np.object)
for i in range(0,data_whole1.shape[1]):
	imp.fit(data_whole1[0,i]) 
	data_whole1_nonmissing[i]=imp.transform(data_whole1[0,i])

data_whole2_nonmissing=np.empty((data_whole2.shape[1],),dtype=np.object)
for i in range(0,data_whole2.shape[1]):
	imp.fit(data_whole2[0,i]) 
	data_whole2_nonmissing[i]=imp.transform(data_whole2[0,i])


#check dimention
for i in range(0,data_whole1.shape[1]):
	if data_whole1_nonmissing[i].shape[1] !=data_nonmissing.shape[1] :
		print 'data_whole1_nonmissing colomns not equal to', data_nonmissing.shape[1],i

#check dimention
for i in range(0,data_whole2.shape[1]):
	if data_whole2_nonmissing[i].shape[1] !=data_nonmissing.shape[1]:	
		print 'data_whole2_nonmissing colomns not equal to', data_nonmissing.shape[1],i




#for i in range(0,data_whole.shape[1]):
#
#	print 1 in np.isinf(X_data_whole[0,i])
#	print 1 in np.isnan(X_data_whole[0,i])


#rename data and intLables for model fitting convenience
X_data=data_nonmissing
X_data_whole1=data_whole1_nonmissing
X_data_whole2=data_whole2_nonmissing
Y=intLabels

from sklearn.decomposition import PCA
pca = PCA(n_components=150)
reduction=pca.fit(X_data)
X = reduction.transform(X_data)

X_whole1=np.empty((X_data_whole1.shape[0],),dtype=np.object)
for i in range(0,X_data_whole1.shape[0]):
	X_whole1[i]=reduction.transform(X_data_whole1[i])

X_whole2=np.empty((X_data_whole2.shape[0],),dtype=np.object)
for i in range(0,X_data_whole2.shape[0]):
	X_whole2[i]=reduction.transform(X_data_whole2[i])

#train classifier by cv data. 
svm_gaussian_params={'kernel': 'rbf', 'probability': 1}
svm_gaussian=SVC(**svm_gaussian_params)
curr_classifier= svm_gaussian.fit(X,Y)
#save model to desktop
joblib.dump(curr_classifier,'/Users/071cht/Desktop/svm_gaussian_scaled_mean.pkl')

#output whole frame prediction to desktop
y_whole_pred1=np.empty((X_data_whole1.shape[0],),dtype=np.object)
y_whole_pred2=np.empty((X_data_whole2.shape[0],),dtype=np.object)

for i in range(0,X_data_whole1.shape[0]):

	y_whole_pred1[i]=curr_classifier.predict_proba(X_whole1[i])[:,1]

sio.savemat('/Users/071cht/Desktop/scores_whole1.mat',{'scores1':y_whole_pred1})

for i in range(0,X_data_whole2.shape[0]):

	y_whole_pred2[i]=curr_classifier.predict_proba(X_whole2[i])[:,1]

sio.savemat('/Users/071cht/Desktop/scores_whole2.mat',{'scores2':y_whole_pred2})






