import numpy as np
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
# import data and labels using python array handling package numpy
labels = np.loadtxt("/Users/071cht/Desktop/for_conference/data/original/output/ori_cv_Y.txt")
data_whole_contents= sio.loadmat('/Users/071cht/Desktop/for_conference/data/original/output/cv_wholeframe_scale.mat')


data_whole=data_whole_contents['cv_wholeframe_scale']  #'cv_wholeframe_scale' is the variable name in this cv_wholeframe_scale.mat
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
imp.fit(data_whole)
data_nonmissing=imp.transform(data_whole)


#for i in range(0,data_whole.shape[1]):
#
#	print 1 in np.isinf(X_data_whole[0,i])
#	print 1 in np.isnan(X_data_whole[0,i])


#rename data and intLables for model fitting convenience

X_data_whole=data_whole_nonmissing
Y=intLabels

from sklearn.decomposition import PCA
pca = PCA(n_components=150)
reduction=pca.fit(X_data_whole)
X = reduction.transform(X_data_whole)



#train classifier by cv data. 
boost_params = {'n_estimators': 100, 'max_depth': 1,'learning_rate': 1.0, 'random_state': 0}
boost= GradientBoostingClassifier(**boost_params)
boost_classifier= boost.fit(X,Y)

svm_gaussian_params={'kernel': 'rbf', 'probability': 1}
svm_gaussian=SVC(**svm_gaussian_params)
curr_classifier= svm_gaussian.fit(X,Y)
#save model to desktop
joblib.dump(curr_classifier,'/Users/071cht/Desktop/boost_scaled_mean.pkl')

#output whole frame prediction to desktop
y_whole_pred1=np.empty((X_data_whole1.shape[0],),dtype=np.object)
y_whole_pred2=np.empty((X_data_whole2.shape[0],),dtype=np.object)

for i in range(0,X_data_whole1.shape[0]):

	y_whole_pred1[i]=curr_classifier.predict_proba(X_whole1[i])

sio.savemat('/Users/071cht/Desktop/scores_whole1.mat',{'scores1':y_whole_pred1})

for i in range(0,X_data_whole2.shape[0]):

	y_whole_pred2[i]=curr_classifier.predict_proba(X_whole2[i])

sio.savemat('/Users/071cht/Desktop/scores_whole2.mat',{'scores2':y_whole_pred2})






