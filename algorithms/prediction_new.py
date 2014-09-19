import numpy as np
import scipy.io as sio
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import neighbors
# import data and labels using python array handling package numpy
filepath='/Users/071cht/Desktop/Lab/publications/tracks/1/'
labels = np.loadtxt(filepath+'labels.txt')

# Since .mat fild is too big and has to store with -v7.3 and sio cannot read -7.3 so let's use results in .txt format.
#data_whole_contents= sio.loadmat('/Users/071cht/Desktop/for_conference/data/original/output/cv_wholeframe_scale.mat')
data_whole=np.loadtxt(filepath+'cv_wholeframe.txt', delimiter=',')

### labels=1 which is the second row (1) of pred_proba is having behavior.
#data_whole=data_whole_contents['cv_wholeframe_scale']  #'cv_wholeframe_scale' is the variable name in this cv_wholeframe_scale.mat
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
data_whole_nonmissing=imp.transform(data_whole)


#for i in range(0,data_whole.shape[1]):
#
#	print 1 in np.isinf(X_data_whole[0,i])
#	print 1 in np.isnan(X_data_whole[0,i])


#rename data and intLables for model fitting convenience
X_data_whole=data_whole_nonmissing
Y=intLabels


from sklearn.decomposition import PCA
pca = PCA(n_components='mle')
reduction=pca.fit(X_data_whole)
X = reduction.transform(X_data_whole)

boost_params = {'n_estimators': 100, 'max_depth': 1,'learning_rate': 1.0, 'random_state': 0}
logit_params={'C':1e5}
svm_linear_params={'kernel': 'linear','probability':1}
svm_gaussian_params={'kernel': 'rbf', 'probability': 1}
knn_params={'n_neighbors':5, 'weights': 'distance'}
params=dict([('boost', boost_params),('logit',logit_params),('svm_linear',svm_linear_params),('svm_gaussian',svm_gaussian_params),('knn',knn_params)])

#set up classifiers
boost= GradientBoostingClassifier(**params['boost'])
logit= linear_model.LogisticRegression(**params['logit'])
svm_linear=SVC(**params['svm_linear'])
svm_gaussian=SVC(**params['svm_gaussian'])
knn= neighbors.KNeighborsClassifier(**params['knn'])

algorithms=dict([('boosting',boost),('logistic_regression',logit),('svm_with_linear_kernel',svm_linear),('svm_with_gaussian_kernel',svm_gaussian),('k_nearest_neighbors',knn)])
#algorithms_name=['boosting','logistic_regression','svm_with_linear_kernel','svm_with_gaussian_kernel','k_nearest_neighbors']
algorithms_name=['svm_with_linear_kernel']

for name in algorithms_name:

	algorighm=algorithms[name]
	curr_classifier=algorighm.fit(X[0:Y.shape[0],],Y)
	
	#save model to desktop
	#joblib.dump(curr_classifier,'/Users/071cht/Desktop/boost_scaled_mean.pkl')

	#output whole frame prediction to desktop
	#y_whole_pred=curr_classifier.predict_proba(X[Y.shape[0]:,])[:,1]
	#y_whole_pred=curr_classifier.predict_proba(X[Y.shape[0]:,]) # we need to know the prob of each category for multiclass classifier
	y_whole_pred=curr_classifier.predict(X[Y.shape[0]:,])
	filename=name+'_python_scores.txt'
	#sio.savemat('/Users/071cht/Desktop/'+filename+'.mat',{filename:y_whole_pred})
	np.savetxt(filepath+filename, y_whole_pred)







