import pylab as pl
import numpy as np
import pickle
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


# import data and labels using python array handling package np
path='/Users/071cht/Desktop/chase/test0/ori/'
data = np.loadtxt(path+'cv.txt',delimiter= ',')
labels = np.loadtxt(path+'labels.txt')
behavior_value=1 #labels=1 is having behavior, labels=0 is not having behavior
behavior_col=behavior_value

#length_before_preprocess=1777


# check dataset
#print data,labels
#print type(data), type(labels)
#print data.shape,labels.shape,data.dtype,labels.dtype
"""somehow labels are loaded as float so we need to change it back to int in order to implement classifier"""
intLabels=labels.astype(int)
#print intLabels.dtype
#assert not np.any(np.isnan(data) | np.isinf(data))
#print np.isinf(data),np.isnan(data)
#print 1 in np.isinf(data) # if returns true, means there're inf in data
#print 1 in np.isnan(data) # if returns true, means there're nan in data
"""We must pay special attention to missing values, because scikit cann't use dataset containing missing values
Solutions:
1.discard entire rows and columns containing missing values
2.imputation"""

# scikit has a imputer class which provide basic strategies for imputing missing values, using mean/median/or the most frequent values of the row or column
# in which the missing values are located.

#the following code retures an np.array, data_nonmissing. It is a non-missing value version of data

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)
#print data_nonmissing.shape
#print data.shape


#rename data and intLables for model fitting convenience
X_data=data_nonmissing
Y=intLabels


"""

#PCA arbitary number of components
explained_variance=list()
components=np.linspace(10,1000,50)
for i in components:
	pca = PCA(n_components=i.astype(int))
	reduction=pca.fit(X_data)
	X = reduction.transform(X_data)
	explained_variance_elm=pca.explained_variance_[-1]
	explained_variance.append(explained_variance_elm)


#plot the PCA spectrum
pl.figure(0)
pl.plot(components,explained_variance, linewidth=2)
pl.xlim([0.0, 500])
pl.ylim([0.0, 50])
pl.xlabel('n_components')
pl.ylabel('explained_variance increment by adding components')
#pl.axvline(150,linestyle=':', label='n_components chosen')
pl.legend(prop=dict(size=12))"""


pca = PCA(n_components='mle')
reduction=pca.fit(X_data)
X = reduction.transform(X_data)

""" comment out automatic dimension choice for now it runs very slow!
#Automatic Choice of Dimensionality for PCA
pca=PCA(n_components='mle')
reduction=pca.fit(X_data)
X = reduction.transform(X_data)

#save model to desktop
joblib.dump(reduction,'/Users/071cht/Desktop/pca.pkl')

n_components_pca_mle=pca.explained_variance_ratio_.shape[0]
print("best n_components by PCA MLE = %d" % n_components_pca_mle)

#plot the PCA spectrum
pl.figure(0)
pl.plot(pca.explained_variance_, linewidth=2)
pl.xlim([0.0, 2000])
pl.ylim([0.0, 1000])
pl.xlabel('n_components')
pl.ylabel('explained_variance increment by adding components')
pl.axvline(n_components_pca_mle,linestyle=':', label='n_components chosen')"""


# setting cross validation and learing algorithms
#X=X_whole[0:Y.shape[0],]

cvfold=3
kfold = cross_validation.KFold(len(X), n_folds=cvfold)
#kfold = cross_validation.StratifiedKFold(Y, n_folds=cvfold)
print X.shape, Y.shape

#set up params
#boost_params = {'n_estimators': 100, 'max_depth': 1,'learning_rate': 1.0, 'random_state': 0}
boost_params = {'n_estimators': 500, 'max_depth': 4,'learning_rate': 0.5, 'random_state': 0}
logit_params={'C':1e5}
svm_linear_params={'kernel': 'linear','probability':1}
svm_gaussian_params={'kernel': 'rbf', 'probability': 1}
knn_params={'n_neighbors':15, 'weights': 'distance'}
params=dict([('boost', boost_params),('logit',logit_params),('svm_linear',svm_linear_params),('svm_gaussian',svm_gaussian_params),('knn',knn_params)])

#set up classifiers
boost= GradientBoostingClassifier(**params['boost'])
logit= linear_model.LogisticRegression(**params['logit'])
svm_linear=SVC(**params['svm_linear'])
svm_gaussian=SVC(**params['svm_gaussian'])
knn= neighbors.KNeighborsClassifier(**params['knn'])

algorithms=dict([('boosting',boost),('logistic regression',logit),('svm with linear kernel',svm_linear),('svm with gaussian kernel',svm_gaussian),('k-nearest neighbors',knn)])
algorithms_name=['boosting','logistic regression','svm with linear kernel','svm with gaussian kernel','k-nearest neighbors']

for name in algorithms_name:

	ACC=list()
	RMSE=list()
	Precision=list()
	Recall=list()
	F=list()
	Num_1_inytest=list()
	FPR=list()
	Specificity=list()
	ROC_AUC=list()
	mean_tpr=0.0
	mean_fpr=np.linspace(0,1,100)
	all_tpr=[]
	cm=np.array([[0,0],[0,0]])
	algorighm=algorithms[name]

	for i,(train, test) in  enumerate (kfold):
		curr_classifier=algorighm.fit(X[train],Y[train])
		#print 'curr_classifier' ,curr_classifier
		y_pred = curr_classifier.predict(X[test])
		y_pred_prob=curr_classifier.predict_proba(X[test])
		# return an n (number of samples) by 2 (number of classes) array with probability of the sample for
	    # that class. The columns correspond to the classes in sorted order (0,1...)

		#ACC
		ACC_elm=curr_classifier.score(X[test], Y[test])
		ACC.append(ACC_elm)


		#RMSE
		"""RMSE_elm = mean_squared_error(Y[test], y_pred_prob[:,behavior_col])
		RMSE.append(RMSE_elm)"""

		"""##Prc,Rec,F
		P_R_F=precision_recall_fscore_support(Y[test], y_pred, average='weighted') # 'weighted' take into acount for label imbalance
		Precision_elm=P_R_F[0]
		Recall_elm=P_R_F[1]
		F_elm=P_R_F[2]
		Num_1_inytest_elm=P_R_F[3]
		Precision.append(Precision_elm)
		Recall.append(Recall_elm)
		F.append(F_elm)
		Num_1_inytest.append(Num_1_inytest_elm)

		## FPR
		TP=P_R_F[1] * P_R_F[3]
		TN = accuracy_score(Y[test], y_pred, normalize=False) - TP
		#FN = P_R_F[3]-TP
		FP= TP/P_R_F[0] - TP
		FPR_elm = FP/ (FP+TN)
		Specificity_elm = 1-FPR_elm
		FPR.append(FPR_elm)
		Specificity.append(Specificity_elm)"""

		#compute for roc
		fpr, tpr, thresholds = roc_curve(Y[test], y_pred_prob[:, behavior_col])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		ROC_AUC_elm = roc_auc
		ROC_AUC.append(ROC_AUC_elm)
		pl.figure(1)
		pl.plot(fpr, tpr, lw=2, label='ROC fold %d (area = %0.4f)' % (i, roc_auc))

		#compute for confusion matrix
		labels=['single fly','multi fly']
		cm_elm=confusion_matrix(Y[test],y_pred)
		cm =cm+cm_elm
	#plot ROC mean curve
	mean_tpr /= cvfold
	mean_tpr[-1]=1.0
	mean_auc = auc(mean_fpr,mean_tpr)
	pl.plot(mean_fpr,mean_tpr, 'k--', label='Mean ROC (area = % 0.4f)' % mean_auc,lw=2)

	#plot ROC luck curve
	pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	#basic setting of plot
	pl.xlim([-0.05, 1.05])
	pl.ylim([0.5, 1.05])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic for %s' % name)
	pl.legend(loc="lower right",prop={'size':8})

	#plot confusion matrix
	#from sklearn.cross_validation import train_test_split     We don't use this function because we have subseted train and test by random seed
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	fig = pl.figure(2)
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
	pl.xlabel('predicted')
	pl.ylabel('True')
	pl.title('confusion matrix for %s' % name)


	#plot mean ROC for differnt algorithms
	pl.figure(3)
	pl.plot(mean_fpr,mean_tpr, lw=1, label='Mean ROC of %s (area = % 0.4f)' % (name, mean_auc))

	pl.xlim([-0.05, 1.05])
	pl.ylim([0.8, 1.00])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic')
	pl.legend(loc="lower right",prop={'size':10})

	#output model evaluation measures
	print name+' '+'ACC='+str(sum(ACC)/cvfold)
	#print name+'pre_process ACC=' + str((sum(ACC)/cvfold * X.shape[0]+ length_before_preprocess-X.shape[0])/ length_before_preprocess)
	#print name+' '+'RMSE='+str(sum(RMSE)/cvfold)
	#print name+' '+'Specificity='+str(sum(Specificity)/cvfold)
	#print name+' '+'Precision='+str(sum(Precision)/cvfold)
	#print name+' '+'Recall='+str(sum(Recall)/cvfold)
	#print name+' '+'F='+str(sum(F)/cvfold)
	#print name+' '+'Num_1_inytest='+str(sum(Num_1_inytest)/cvfold)
	#print name+' '+'FPR='+str(sum(FPR)/cvfold)
	#print name+' '+'AUC='+str(sum(ROC_AUC)/cvfold)
	pl.show()

#pl.show()	 #for plot mean roc for all algorithms









