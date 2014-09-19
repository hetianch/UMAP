import numpy as np
import scipy.io as sio
# import data and labels using python array handling package numpy
data = np.loadtxt('/Users/071cht/Desktop/Lab/lab_data/new_test/test2/data_nonmissing_scale.txt',delimiter= ',') 
labels= np.loadtxt('/Users/071cht/Desktop/Lab/lab_data/new_test/test2/labels.txt')
data_whole_contents1= sio.loadmat('/Users/071cht/Desktop/Lab/lab_data/new_test/test2/window_nonmissing_scale1.mat')
data_whole_contents2= sio.loadmat('/Users/071cht/Desktop/Lab/lab_data/new_test/test2/window_nonmissing_scale2.mat')


data_whole1=data_whole_contents1['window_nonmissing_scale1']  #'window' is the variable name in this window2.mat
data_whole2=data_whole_contents2['window_nonmissing_scale2']
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
"""We must pay special attention to missing values, because scikit cann't use dataset containing missing values
Solutions:
1.discard entire rows and columns containing missing values
2.imputation"""

# scikit has a imputer class which provide basic strategies for imputing missing values, using mean/median/or the most frequent values of the row or column
# in which the missing values are located. 

"""before imputation, we must drop colomns that only contain missing values for data and data_whole first because ,when inputating
it will automatically discard colomns with only NaN before transform. This will make the imputation of data and data_whole different 
in colomn size!!!"""
"""
##handle data
nan_value=np.isnan(data)
colomn= [sum(x) for x in zip(*nan_value)]
delete_idx= [x/data.shape[0] for x in colomn]
delete_list=[index for index,value in enumerate(delete_idx) if value>0.999]
print delete_list

##handle data_whole
delete_list= np.empty((data_whole.shape[1],),dtype=np.object)
for i in range(0,data_whole.shape[1]):
	nan_value=np.isnan(data_whole[0,i])
	colomn= [sum(x) for x in zip(*nan_value)]
	delete_idx= [x/data_whole[0,1].shape[0] for x in colomn]
	delete_list[i]=[index for index,value in enumerate(delete_idx) if value>0.999]
	
print data_whole[0,1].shape[0]
print '1/3800',1/3800

print 'nan_value shape', nan_value.shape
print 'colomn', colomn
print 'delete_idx', delete_idx
print delete_list"""



"""
#the following code retures an np.array, data_nonmissing. It is a non-missing value version of data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)
data_whole_nonmissing=np.empty((data_whole.shape[1],),dtype=np.object)
for i in range(0,data_whole.shape[1]):
	imp.fit(data_whole[0,i]) 
	data_whole_nonmissing[i]=imp.transform(data_whole[0,i])

print 'data_nonmissing',data_nonmissing.shape
print 'data_whole',data_whole_nonmissing[0].shape,data_whole_nonmissing[1].shape"""



###HOWEVER:
## From my aspect, it doesn't make much sense to impute the value according to the mean of the entire colomn
## So I decide to adopt the first solution in the future.

""" I need to figure out how to work with numpy array"""


#rename data and intLables to chase_X and chase_y for model fitting convenience
X_data=data
X_data_whole1=data_whole1
X_data_whole2=data_whole2
y=intLabels
#print X_data_whole[0,0].shape,X_data_whole[0,1].shape,X_data.shape


#for i in range(0,data_whole.shape[1]):
#
#	print 1 in np.isinf(X_data_whole[0,i])
#	print 1 in np.isnan(X_data_whole[0,i])

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
reduction=pca.fit(X_data)
X = reduction.transform(X_data)

X_whole1=np.empty((data_whole1.shape[1],),dtype=np.object)
for i in range(0,data_whole1.shape[1]):
	X_whole1[i]=reduction.transform(X_data_whole1[0,i])

X_whole2=np.empty((data_whole2.shape[1],),dtype=np.object)
for i in range(0,data_whole2.shape[1]):
	X_whole2[i]=reduction.transform(X_data_whole2[0,i])	

#print 'after pca x_data', X_data.shape
#print 'after pca x_whole', X_whole[1].shape
#Plot the PCA spectrum
import pylab as pl
pl.figure(0)
pl.axes([.2, .2, .7, .7])
pl.plot(pca.explained_variance_, linewidth=2)
pl.axis('tight')
pl.xlim([0.0, 80])
pl.ylim([0.0, 30000])
pl.xlabel('n_components')
pl.ylabel('explained_variance_')



#sebset data for training and testing by random seed and use 60% to train and 40% to test
"""turn off random seed
numpy.random.seed(0) #Seed the random number generator with a fixed value, so that it return the same array each time.
indices = numpy.random.permutation(len(X))
seperateIdx= len(X)*60/100
X_train = X[indices[0:seperateIdx]] 
y_train = y[indices[0:seperateIdx]] 
X_test  = X[indices[seperateIdx:]]
y_test  = y[indices[seperateIdx:]] """




""" Note:Random seed return a score around 0.9, which is much higher than KFold crossvalidation (around 0.5). I DON'T KNOW WHY!!!!
The only difference is that the sample from K-fold crossvalidation is consecutive in time and highly correlated with adjecent data points. 
But random seed shuffle the data. Will this make difference? If so, we need to shuffle the data before crossvalidate"""




# Try cross-validation(We did a 3 fold crossvalidation and arbitaryly assigning the following paremeters)
"""bug: the following code arise an warning: overflow encountered in exp. Check out the discussion about it later http://comments.gmane.org/gmane.comp.python.scikit-learn/3730"""

# 3-fold cross validation means:
# Ex: >>X = np.array([2, 3, 1, 0,12,10,22,11,22,111,23,12]),
#	  >>kfold = cross_validation.KFold(len(X), n_folds=3)  3-fold means 2/3 train, 1/3 test
#     ###ytrain [ 12  10  22  11  22 111  23  12] ytest [2 3 1 0]
	  ###ytrain [  2   3   1   0  22 111  23  12]  ytest [12 10 22 11]
	  ###ytrain [ 2  3  1  0 12 10 22 11]          ytest [ 22 111  23  12]
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
kfold = cross_validation.KFold(len(X), n_folds=3)
gbc= GradientBoostingClassifier(n_estimators=100, max_depth=1, learning_rate=1.0,random_state=0)
crossScores=[gbc.fit(X[train], y[train]).score(X[test], y[test]) for train, test in kfold]
print crossScores, 'average crossvalidation score ='+str(sum(crossScores)/3)   ##return : 0.55

#train a gradient boosting classifier based on decision stumps
"""hardcode all the paremeters here (use default setting for all except max_deph. We need to optimize all of them in the future"""

seperateIdx= len(X)*60/100
X_train = X[0:seperateIdx]
y_train = y[0:seperateIdx] 
X_test  = X[seperateIdx:]
y_test  = y[seperateIdx:]

from sklearn.ensemble import GradientBoostingClassifier
params = {'n_estimators': 100, 'max_depth': 1,'learning_rate': 1.0, 'random_state': 0}
boost= GradientBoostingClassifier(**params)
boosting_classifier = boost.fit(X_train, y_train)
y_pred = boosting_classifier.predict(X_test)
#print boosting_classifier 
#print y_pred


"""output whole frame prediction"""
y_whole_pred1=np.empty((data_whole1.shape[1],),dtype=np.object)
y_whole_pred2=np.empty((data_whole2.shape[1],),dtype=np.object)

for i in range(0,data_whole1.shape[1]):

	y_whole_pred1[i]=boosting_classifier.predict_proba(X_whole1[i])[:,1]

sio.savemat('/Users/071cht/Desktop/Lab/lab_data/new_test/test2/scores_whole1.mat',{'scores1':y_whole_pred1})

for i in range(0,data_whole2.shape[1]):

	y_whole_pred2[i]=boosting_classifier.predict_proba(X_whole2[i])[:,1]

sio.savemat('/Users/071cht/Desktop/Lab/lab_data/new_test/test2/scores_whole2.mat',{'scores2':y_whole_pred2})




"""Algorithm evaluation starts here!!!!"""

#ACC
ACC=boosting_classifier.score(X_test,y_test)
print 'boosting_ACC:' + ' '+ str(ACC) 
	#Another way to calcuate Accuracy:
		# from sklearn.metrics import accuracy_score
		# ACC=accuracy_score(y_test, y_pred, normalize=True) 
		# normalize==False returns number of correctly classified samiples (TP+TN); otherwise, return the fraction


#RMSE
from sklearn.metrics import mean_squared_error
y_pred_prob=boosting_classifier.predict_proba(X_test)
RMSE = mean_squared_error(y_test, y_pred_prob[:,1])
print 'boosting_RMSE:' + ' '+ str(RMSE) 

##Prc,Rec,F
from sklearn.metrics import precision_recall_fscore_support
P_R_F=precision_recall_fscore_support(y_test, y_pred, average='weighted') # 'weighted' take into acount for label imbalance
print 'boosting_Precision:'+ ' '+ str(P_R_F[0])
print 'boosting_Recall and sensitivity of the model:'+ ' '+ str(P_R_F[1])
print 'boosting_F:'+ ' '+ str(P_R_F[2])
print 'number of 1 in y_test' + ' ' +str(P_R_F[3]) # (TP+FN)

## FPR
from sklearn.metrics import accuracy_score
TP=P_R_F[1] * P_R_F[3]
TN = accuracy_score(y_test, y_pred, normalize=False) - TP
#FN = P_R_F[3]-TP
FP= TP/P_R_F[0] - TP
FPR = FP/ (FP+TN)
sepecificity = 1-FPR
print 'boosting_FPR:'+ ' '+ str(FPR)
print 'specificity of the model:' + ' ' + str(sepecificity)

#ROC
from sklearn.metrics import roc_curve,auc
import pylab as pl
y_pred_prob = boosting_classifier.predict_proba(X_test) 
# return an n (number of samples) by 2 (number of classes) array with probability of the sample for 
# that class. The columns correspond to the classes in sorted order (0,1...)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

## Plot ROC curve

pl.figure(0)
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0.5, 1], 'k--')
pl.xlim([0.0, 1])
pl.ylim([0.5, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic for boost')
pl.legend(loc="lower right")

#Training and testing error by number of estimators or (number of iterations)
from sklearn.metrics import zero_one_loss

boost_test_err=np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred_stage in enumerate(boosting_classifier.staged_predict(X_test)):
    boost_test_err[i] = zero_one_loss(y_test, y_pred_stage, normalize=True) 
# if normalize = False it returns the number rather than the fraction of misclassification

boost_train_err = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred_stage in enumerate(boosting_classifier.staged_predict(X_train)):
    boost_train_err[i] = zero_one_loss(y_train, y_pred_stage,normalize=True) 																  

#fig = pl.figure()
#ax = fig.add_subplot(111)
#ax.plot(numpy.arange(params['n_estimators']) + 1, boost_test_err , label='Boost Test Error', color='red')
#ax.plot(numpy.arange(params['n_estimators']) + 1, boost_train_err, label='Boost Train Error', color='blue')
#ax.set_xlabel('number of estimators')
#ax.set_ylabel('error rate')
#pl.title('training and testing error by number of estimators')
#leg = ax.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(1.0)

pl.figure(1)
pl.title('training and testing error by number of estimators')
pl.plot(np.arange(params['n_estimators']) + 1, boost_test_err , label='Boost Test Error', color='red')
pl.plot(np.arange(params['n_estimators']) + 1, boost_train_err, label='Boost Train Error', color='blue')
pl.xlabel('number of estimators')
pl.ylabel('error rate')
pl.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(1.0)


#Plot feature importance
pl.figure(2)
feature_importance = boosting_classifier.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx_toplot=sorted_idx[-10:]
pos = np.arange(sorted_idx_toplot.shape[0]) + .5
pl.barh(pos, feature_importance[sorted_idx_toplot], align='center')
pl.yticks(pos, sorted_idx_toplot[-10:]) # gives the feature_names in numpy.ndarray: ['f1' 'f2' 'f3' 'f4' ....]
pl.xlabel('Relative Importance')
pl.title('Variable Importance')

#plot confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

for i,(train, test) in  enumerate (kfold):
	
	#from sklearn.cross_validation import train_test_split     We don't use this function because we have subseted train and test by random seed
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	labels= ['not multifly', 'multifly']
	boosting_classifier=boost.fit(X[train],y[train])
	y_pred = boosting_classifier.predict(X[test])
	cm=confusion_matrix(y[test],y_pred)
	cm +=cm

#plt.clf()
fig = plt.figure(3)
ax=fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
##Add annotations in the 2 by 2 table 
width=len(cm)
height=len(cm[0])
for x in xrange(width):
	for y in xrange(height):
		ax.annotate(str(cm[x][y]),xy=(y,x), horizontalalignment = 'center', verticalalignment = 'center', weight= 'extra bold',size = 'large')

ax.set_xticklabels(['']+labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('predicted')
plt.ylabel('True')
plt.title('confusion matrix for boosting with window radius 0')	

    
pl.show()

